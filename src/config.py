from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- API Keys ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

_missing = [name for name, val in [("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY), ("OPENAI_API_KEY", OPENAI_API_KEY)] if not val]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Add them to your .env file."
    )

# --- Paths ---
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
CHROMA_PATH = ".chroma"

# --- ChromaDB ---
CHROMA_COLLECTION = "cybr7003"

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_DEV = "claude-haiku-4-5-20251001"
LLM_DEMO = "claude-sonnet-4-6"
