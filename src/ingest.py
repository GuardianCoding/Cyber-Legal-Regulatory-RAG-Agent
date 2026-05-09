"""
Ingest PDFs from data/raw/ → chunk → embed → upsert to ChromaDB.

Run:  python src/ingest.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import chromadb

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Allow `python src/ingest.py` from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def _detect_doc_type(filename: str) -> str:
    lower = filename.lower()
    if any(kw in lower for kw in ["privacy act", "schedule 1", "legislation", "statutory"]):
        return "legislation"
    if any(kw in lower for kw in ["essential eight", "asd", "maturity model", "framework"]):
        return "framework"
    if any(kw in lower for kw in ["medibank", "mcclure", "fca", "class action"]):
        return "case_law"
    if any(kw in lower for kw in ["ndb", "data breach", "notifiable"]):
        return "legislation"
    if any(kw in lower for kw in ["cybr", "week", "lecture", "slide"]):
        return "lecture_notes"
    return "document"


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all PDFs
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_RAW_DIR),
        required_exts=[".pdf"],
        recursive=False,
    ).load_data(show_progress=True)

    if not documents:
        print(f"No PDFs found in {DATA_RAW_DIR}.")
        return

    print(f"Loaded {len(documents)} document page(s) from {DATA_RAW_DIR}.")

    # 2. Tag each document with document_type metadata based on filename
    for doc in documents:
        fname = doc.metadata.get("file_name", "")
        doc.metadata["document_type"] = _detect_doc_type(fname)

    # 3. Chunk with SentenceSplitter — gives us nodes we can inspect and save
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    print(f"Created {len(nodes)} node(s).")

    # 4. Embed + upsert via ChromaVectorStore
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY),
        show_progress=True,
    )

    # 5. Save nodes to chunks.json for offline inspection and eval harness
    records = [
        {
            "id": node.node_id,
            "text": node.get_content(),
            "source": node.metadata.get("file_name", ""),
            "document_type": node.metadata.get("document_type", "document"),
            "topic": node.metadata.get("topic", ""),
        }
        for node in nodes
    ]

    out = DATA_PROCESSED_DIR / "chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary
    type_counts = Counter(r["document_type"] for r in records)
    print(f"\n{'─' * 50}")
    print(f"Done. {len(records)} nodes ingested.")
    print(f"Saved → {out}")
    print(f"ChromaDB collection: '{CHROMA_COLLECTION}'")
    print("\nBreakdown by document_type:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")


if __name__ == "__main__":
    main()
