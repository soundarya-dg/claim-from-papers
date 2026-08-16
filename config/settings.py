import os
from dotenv import load_dotenv

load_dotenv()

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Ingestion
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 5

# Paths
PAPERS_DIR = "data/papers"
VECTORSTORE_DIR = "vectorstore"

# Verification
CONFIDENCE_THRESHOLD = 0.45

# Paper download configuration — edit these to change your topic and dataset
DOWNLOAD_TOPIC = "text classification BERT fine-tuning benchmark"
DOWNLOAD_NUM_PAPERS = 30
DOWNLOAD_START_YEAR = 2024
DOWNLOAD_END_YEAR = 2026