import os
from dotenv import load_dotenv

load_dotenv()

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Ingestion
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 5

# Paths
PAPERS_DIR = "data/papers"
VECTORSTORE_DIR = "vectorstore"

# Verification
CONFIDENCE_THRESHOLD = 0.75