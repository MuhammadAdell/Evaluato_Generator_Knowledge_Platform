from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
ENV = dotenv_values(ROOT / "app.env")

OPENROUTER_API_KEY = ENV["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = ENV.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
REDIS_URL = ENV.get("REDIS_URL", "redis://localhost:6379/0")

CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "external_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4
MAX_ITERATIONS = 4

CACHE_TTL = 60 * 60 * 24
WHISPER_MODEL = "base"

UPLOAD_DIR = ROOT / "uploads"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".md", ".pptx", ".wav",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go",
    ".rs", ".rb", ".php", ".sql", ".sh", ".html", ".css",
    ".json", ".yaml", ".yml", ".ipynb",
]
