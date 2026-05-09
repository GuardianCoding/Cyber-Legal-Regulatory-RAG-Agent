"""
Ingest PDFs from data/raw/ → chunk → embed → upsert to ChromaDB.

Run:  python src/ingest.py
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from openai import OpenAI

# Allow `python src/ingest.py` from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OPENAI_API_KEY,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    CHROMA_PATH,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

# ── Tuning constants ──────────────────────────────────────────────────────────

EMBED_BATCH = 96          # well under OpenAI's 2048-input limit
HEADING_SIZE_RATIO = 1.2  # span is heading candidate if size > modal_body * this
MIN_CHUNK_CHARS = 60      # discard fragments shorter than this

# ── Document-type heuristics ──────────────────────────────────────────────────

_DOCTYPE_RULES: list[tuple[list[str], str]] = [
    (["privacy act", "act 1988", "act 2022", "schedule", "legislative instrument",
      "statutory", "regulation"], "legislation"),
    (["essential eight", "asd", "acsc", "nist", "framework", "maturity model"], "framework"),
    (["gdpr", "general data protection"], "legislation"),
    (["lecture", "week", "slide", "notes", "module", "cybr"], "lecture_notes"),
]


def _detect_doc_type(filename: str) -> str:
    lower = filename.lower()
    for keywords, dtype in _DOCTYPE_RULES:
        if any(kw in lower for kw in keywords):
            return dtype
    return "document"


# ── PyMuPDF helpers ───────────────────────────────────────────────────────────

def _modal_font_size(doc: fitz.Document) -> float:
    """Return the most common (modal) font size — a proxy for body-text size."""
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    if not sizes:
        return 12.0
    return Counter(sizes).most_common(1)[0][0]


def _span_is_heading(span: dict, body_size: float) -> bool:
    text = span["text"].strip()
    if not text or len(text) > 120:
        return False
    ratio = span["size"] / body_size if body_size else 1.0
    is_large = ratio >= HEADING_SIZE_RATIO
    # PyMuPDF font flags: bit 4 (value 16) = bold; also check font name as fallback
    is_bold = bool(span["flags"] & 16) or "bold" in span["font"].lower()
    return is_large or (is_bold and ratio >= 1.05)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Character-window split with sentence-boundary preference and overlap."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Prefer to break at the last sentence end within the window
        boundary = text.rfind(". ", start, end)
        if boundary <= start:
            boundary = end
        else:
            boundary += 1  # keep the period in this chunk
        chunks.append(text[start:boundary].strip())
        start = boundary - overlap
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def _parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse one PDF into chunk dicts: { id, text, metadata }.

    Phase 1 — heading walk: accumulate body text between heading boundaries.
    Phase 2 — size guard:   split any section > CHUNK_SIZE with _split_fixed().
    Fallback — if no headings detected, split the full document text directly.
    """
    doc_type = _detect_doc_type(pdf_path.name)
    source = pdf_path.name

    doc = fitz.open(str(pdf_path))
    body_size = _modal_font_size(doc)

    # Phase 1: walk pages and detect heading boundaries
    sections: list[dict] = []
    current_topic = "Introduction"
    current_text = ""
    current_page = 1
    heading_found = False

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = ""
                line_is_heading = False
                for span in line["spans"]:
                    line_text += span["text"]
                    if _span_is_heading(span, body_size):
                        line_is_heading = True

                line_text = line_text.strip()
                if not line_text:
                    continue

                if line_is_heading and len(line_text) <= 120:
                    if current_text.strip():
                        sections.append({
                            "topic": current_topic,
                            "text": current_text.strip(),
                            "page_start": current_page,
                        })
                    current_topic = re.sub(r"\s+", " ", line_text)
                    current_text = ""
                    current_page = page_num
                    heading_found = True
                else:
                    current_text += line_text + " "

    if current_text.strip():
        sections.append({
            "topic": current_topic,
            "text": current_text.strip(),
            "page_start": current_page,
        })

    doc.close()

    # Fallback: no headings found → treat full text as a single section
    if not heading_found:
        full_text = " ".join(s["text"] for s in sections)
        sections = [{"topic": pdf_path.stem, "text": full_text, "page_start": 1}]

    # Phase 2: split oversized sections
    raw_chunks: list[dict] = []
    for section in sections:
        for sub in _split_fixed(section["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            raw_chunks.append({
                "text": sub,
                "metadata": {
                    "source": source,
                    "document_type": doc_type,
                    "topic": section["topic"],
                    "page_start": section["page_start"],
                },
            })

    # Stable per-source IDs → upsert is idempotent on re-ingestion
    for i, chunk in enumerate(raw_chunks):
        chunk["id"] = hashlib.md5(f"{source}::{i}".encode()).hexdigest()

    return raw_chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.05)
    return vectors


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert(chunks: list[dict], embeddings: list[list[float]]) -> None:
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    batch = 500
    for i in range(0, len(chunks), batch):
        sl = slice(i, i + batch)
        collection.upsert(
            ids=[c["id"] for c in chunks[sl]],
            embeddings=embeddings[sl],
            documents=[c["text"] for c in chunks[sl]],
            metadatas=[c["metadata"] for c in chunks[sl]],
        )
    print(f"  ChromaDB: upserted {len(chunks)} chunks → '{CHROMA_COLLECTION}'")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_RAW_DIR}. Drop source documents there and re-run.")
        return

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    all_chunks: list[dict] = []

    for pdf_path in pdfs:
        print(f"\nParsing  {pdf_path.name}")
        chunks = _parse_pdf(pdf_path)
        if not chunks:
            print("  No chunks extracted — PDF may be image-only.")
            continue
        dtype = chunks[0]["metadata"]["document_type"]
        topics = sorted({c["metadata"]["topic"] for c in chunks})
        print(f"  {len(chunks)} chunks | doc_type={dtype} | {len(topics)} topic(s)")
        for t in topics[:3]:
            print(f"    · {t}")
        if len(topics) > 3:
            print(f"    … and {len(topics) - 3} more")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced — check that PDFs contain extractable text.")
        return

    # Save for offline inspection and eval harness
    out = DATA_PROCESSED_DIR / "chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_chunks)} chunks → {out}")

    print(f"\nEmbedding with {EMBEDDING_MODEL}  (batch={EMBED_BATCH}) …")
    embeddings = _embed(openai_client, [c["text"] for c in all_chunks])
    print(f"  {len(embeddings)} vectors, dim={len(embeddings[0])}")

    print("\nUpserting to ChromaDB …")
    _upsert(all_chunks, embeddings)

    print(f"\nDone. {len(all_chunks)} chunks ingested.")


if __name__ == "__main__":
    main()
