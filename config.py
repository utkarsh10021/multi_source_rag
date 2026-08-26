import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
FAISS_DIR = Path(os.getenv("FAISS_DIR", DATA_DIR / "faiss"))
LOG_DIR = Path(os.getenv("LOG_DIR", DATA_DIR / "logs"))

for directory in (DATA_DIR, UPLOAD_DIR, FAISS_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Gemini API Key
# ============================================================
#
# Priority:
# 1. Environment variable GOOGLE_API_KEY
# 2. Streamlit secret GEMINI_API_KEY
# 3. Streamlit secret GOOGLE_API_KEY
# 4. Local gemini_api_key.txt
#
# This allows the same project to work locally and on Streamlit.
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        import streamlit as st

        GOOGLE_API_KEY = (
            st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
        )
    except Exception:
        GOOGLE_API_KEY = None


# Local development fallback
if not GOOGLE_API_KEY:
    KEY_FILE = BASE_DIR / "gemini_api_key.txt"

    if KEY_FILE.exists():
        GOOGLE_API_KEY = KEY_FILE.read_text(
            encoding="utf-8"
        ).strip()


if not GOOGLE_API_KEY:
    raise ValueError(
        "Gemini API key not configured. "
        "Set GOOGLE_API_KEY as an environment variable, "
        "GEMINI_API_KEY/GOOGLE_API_KEY in Streamlit Secrets, "
        "or create gemini_api_key.txt locally."
    )


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemini-3.6-flash"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.4")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "2500")
)


# ============================================================
# Chunking configuration
# ============================================================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)


# ============================================================
# Website crawling configuration
# ============================================================

MAX_PAGES = int(
    os.getenv("MAX_PAGES", "5")
)

MAX_LINKS_PER_PAGE = int(
    os.getenv("MAX_LINKS_PER_PAGE", "30")
)


# ============================================================
# Retrieval configuration
# ============================================================

TOP_K = int(
    os.getenv("TOP_K", "20")
)

FINAL_K = int(
    os.getenv("FINAL_K", "5")
)

MEMORY_TURNS = int(
    os.getenv("MEMORY_TURNS", "5")
)


# ============================================================
# Supported files
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".csv"
}


# ============================================================
# Persistent FAISS files
# ============================================================

FAISS_INDEX_NAME = "index"

CHUNKS_FILE = FAISS_DIR / "chunks.json"

SOURCES_FILE = FAISS_DIR / "sources.json"