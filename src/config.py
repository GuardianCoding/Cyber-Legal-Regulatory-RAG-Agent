import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")

# --- API Keys ---
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# --- Models ---
# Haiku for dev/testing (faster, cheaper); Sonnet for final demo (higher quality)
LLM_DEV = "claude-haiku-4-5-20251001"
LLM_PROD = "claude-sonnet-4-6"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- ChromaDB ---
CHROMA_COLLECTION = "cybr7003_rag"

# --- Chunking ---
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 100     # characters

# --- Retrieval ---
RETRIEVAL_TOP_K = 20    # candidates from dense + sparse before reranking
RERANK_TOP_N = 5        # final chunks passed to the LLM
BM25_WEIGHT = 0.4       # reciprocal rank fusion weight for sparse leg
DENSE_WEIGHT = 0.6      # reciprocal rank fusion weight for dense leg
