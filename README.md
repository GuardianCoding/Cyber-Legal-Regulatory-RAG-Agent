# Cyber Legal & Regulatory RAG Agent

A retrieval-augmented generation (RAG) agent for querying Australian cybersecurity law, policy, and governance documents. Ask plain-English questions and receive cited answers grounded strictly in the source material — legislation, regulatory frameworks, case law, and lecture notes.

---

## What It Does

- Accepts natural-language queries about Australian cybersecurity law and regulation
- Decomposes complex queries into targeted sub-queries
- Retrieves relevant chunks via hybrid search (vector similarity + BM25 keyword matching), then reranks with a cross-encoder
- Generates answers using Claude, with mandatory source citations
- Verifies that every answer is grounded in the retrieved chunks before returning it
- Exposes a Streamlit UI with source expanders and document-type filtering

---

## Architecture

```
User Query
    │
    ▼
[query_decomposer]   Claude splits the query into 1–3 targeted sub-queries
    │
    ▼
[retriever]          hybrid_search() per sub-query → deduplicate → top-5 chunks
    │                (VectorIndexRetriever + BM25Retriever → RRF fusion → cross-encoder rerank)
    │
    ▼
[generator]          Claude generates a cited answer from numbered source chunks
    │
    ▼
[verifier]           Claude self-checks: does the answer follow only from the sources?
    │                Sets grounded=True/False; never suppresses the answer
    ▼
{ answer, sources, grounded, chunks }
```

The LangGraph graph is compiled once at import time and reused for every query.

---

## Document Corpus

| Document | Type |
|---|---|
| CYBR7003 lecture slides (Weeks 1, 2, 3, 5, 7, 8, 9) | `lecture_notes` |
| Privacy Act 1988 — Schedule 1 (Australian Privacy Principles) | `legislation` |
| NDB Preparation and Response Guide (OAIC, June 2024) | `legislation` |
| NDB Scheme Overview (OAIC) | `legislation` |
| CTLR Privacy Amendments | `legislation` |
| ASD Essential Eight Maturity Model (November 2023) | `framework` |
| McClure v Medibank Private Limited [2025] FCA 167 | `case_law` |
| Medibank class action article (Baker McKenzie / Omni Bridgeway) | `case_law` |

Each chunk carries `{ source, document_type, topic }` metadata. The UI sidebar lets you filter by `document_type`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | Anthropic Claude (`claude-haiku-4-5` dev / `claude-sonnet-4-6` demo) |
| Ingestion | LlamaIndex — `SimpleDirectoryReader`, `SentenceSplitter`, `IngestionPipeline` |
| Embeddings | OpenAI `text-embedding-3-small` via LlamaIndex |
| Vector DB | ChromaDB (local persistent) via `ChromaVectorStore` |
| Sparse retrieval | LlamaIndex `BM25Retriever` |
| Hybrid fusion | LlamaIndex `QueryFusionRetriever` (reciprocal rank fusion) |
| Reranker | `SentenceTransformerRerank` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| PDF parsing | LlamaIndex `SimpleDirectoryReader` (backed by pypdf) |
| UI | Streamlit |

---

## Setup

### Prerequisites

- Python 3.11+
- An Anthropic API key
- An OpenAI API key (for embeddings only)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS: fix quarantine attributes if you get "zsh: killed"
xattr -r -d com.apple.quarantine .venv/
```

### Configure

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Ingest documents

Drop your PDFs into `data/raw/`, then run:

```bash
python src/ingest.py
```

This embeds all documents and writes them to `.chroma/` (ChromaDB). It also saves `data/processed/chunks.json` for offline inspection. Only needs to run once — subsequent queries load the persisted index.

---

## Usage

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The sidebar lets you filter by document type. The answer panel shows grounding status and expandable source chunks.

### Python API

```python
from src.agent import run_agent

result = run_agent("What are the notification obligations under the NDB scheme?")
print(result["answer"])
print(result["grounded"])   # True if answer is fully supported by sources
print(result["sources"])    # list of source document names
```

### Direct hybrid search

```python
from src.tools import hybrid_search
import json

chunks = hybrid_search("Privacy Act notification obligations", doc_type_filter="legislation")
print(json.dumps(chunks[:2], indent=2))
```

### Evaluation harness

```bash
python src/eval.py
```

Runs 10 hardcoded Q&A pairs and reports retrieval recall and grounding rate.

---

## Project Structure

```
.
├── src/
│   ├── config.py          # API keys, constants, model names, ChromaDB path
│   ├── ingest.py          # LlamaIndex ingestion pipeline
│   ├── tools.py           # Hybrid retriever + reranker, exposes hybrid_search()
│   ├── agent.py           # LangGraph agent: decompose → retrieve → generate → verify
│   └── eval.py            # Evaluation harness
├── app/
│   └── streamlit_app.py   # Demo UI
├── data/
│   ├── raw/               # Source PDFs (not committed)
│   └── processed/         # chunks.json (written by ingest.py)
├── tests/
│   └── test_retrieval.py
├── .chroma/               # ChromaDB persistent storage (auto-created, gitignored)
└── requirements.txt
```

---

## Design Decisions

**Hybrid search over vector-only retrieval.** Legal documents contain precise terminology — section numbers, defined terms, act citations — where exact keyword matching outperforms semantic similarity. BM25 captures these while vector search handles paraphrase. Reciprocal rank fusion combines both result lists without requiring score normalisation.

**Cross-encoder reranker.** Bi-encoder retrieval scores query and document independently, making relevance scoring approximate. The cross-encoder sees query and chunk together and produces a more accurate relevance signal at ~200ms extra latency per query — acceptable for a demo.

**LangGraph over a monolithic chain.** Explicit graph nodes (decompose → retrieve → generate → verify) make the control flow inspectable and each node independently testable. The verifier node catches post-hoc grounding failures without suppressing the answer — it flags uncertainty rather than hiding it.

**Local ChromaDB over managed vector DB.** The corpus is ~500–1000 chunks. A local persistent store removes all infra overhead and keeps the project fully self-contained. Pinecone or Weaviate would be the call for multi-user production serving.

**Mandatory citation system prompt.** Hallucination in a legal context has real consequences. The system prompt is the first line of defence: every claim must cite a document and section, and the model must flag when retrieved context is insufficient rather than speculating.

---

## What's Next

- **Semantic chunking** — `SemanticSplitter` instead of `SentenceSplitter` to better respect section boundaries in legislation
- **Streaming** — stream Claude's response tokens to the UI for perceived responsiveness
- **Feedback loop** — thumbs up/down on answers to log low-grounding queries for retrieval tuning
- **Metadata-aware retrieval** — use LlamaIndex metadata filters to push document-type filtering into the vector query rather than post-hoc
- **Persistent conversation** — stateful multi-turn sessions with message history passed into the generator
