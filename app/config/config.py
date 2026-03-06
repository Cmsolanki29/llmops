import os

# === LLM (Groq) configuration ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # set in .env
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# Other good options: "llama-3.1-8b-instant"

# === Vector store / data config ===
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50