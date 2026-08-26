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

from pathlib import Path

KEY_FILE = Path(__file__).parent / "gemini_api_key.txt"

if not KEY_FILE.exists():
    raise FileNotFoundError(
        f"Gemini API key file not found: {KEY_FILE}"
    )

GOOGLE_API_KEY = KEY_FILE.read_text(encoding="utf-8").strip()

if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key file is empty.")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2500"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))
MAX_LINKS_PER_PAGE = int(os.getenv("MAX_LINKS_PER_PAGE", "30"))
TOP_K = int(os.getenv("TOP_K", "20"))
FINAL_K = int(os.getenv("FINAL_K", "5"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "5"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}

FAISS_INDEX_NAME = "index"
CHUNKS_FILE = FAISS_DIR / "chunks.json"
SOURCES_FILE = FAISS_DIR / "sources.json"
