# Claim From Papers

This project is a proof-of-concept RAG system that not only answers questions based on research papers but also provides transparent claim verification for each part of the answer. It ensures that every claim in the generated answer is backed by real sources, giving users confidence in the information they receive.

### Motivation

With the rise of LLMs, there's a growing concern about hallucinations and misinformation.

A RAG pipeline reduces that risk, but does not eliminate it.

This project goes one step further: after the answer is generated, each atomic claim is independently re-verified against the vector store. The result is a transparent "Claim Grounding Rate" that quantifies exactly how much of the answer is traceable to the source material.


### How It Works

1. **Knowledge Base**: The app starts with a knowledge base of 30 arXiv papers on "Text Classification using Large Language Models" (topic can be changed based on user preference), processed into chunks and stored as BGE-M3 embeddings in ChromaDB.
2. **User Interaction**: Users can ask any question related to the topic, and the system will fetch relevant information from the papers to generate an answer. Also, users can upload their own PDFs to expand the knowledge base.
3. **Retrieval**: The user's question is embedded and semantically matched to the top-K most relevant chunks.
4. **Generation**: A prompt containing the retrieved chunks is sent to Llama 3.3 70B model (via Groq) to produce a grounded answer. A second LLM call then produces a concise summary (`short_answer`) that strips source citations for easy reading.
5. **Claim Extraction**: The answer is decomposed into atomic factual claims using the same LLM.
6. **Claim Verification**: Each claim is independently searched against ChromaDB. A high-similarity chunk triggers an LLM fact-check that labels the claim as Grounded, Unverified, or Contradicted.
7. **Claim Grounding Rate**: The fraction of grounded claims is returned alongside the answer as a single transparency metric.

## Technologies Used

- **Language**: Python 3.12
- **LLM API**: Groq (access to Llama 3.3 70B)
- **Embeddings**: BGE-M3 (local, no API cost)
- **Vector Database**: ChromaDB
- **Web Framework**: FastAPI
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF (fitz)
- **Paper Downloading**: arXiv API
- **Evaluation**: DeepEval


## Project Structure

```
claim-from-papers/
├── README.md                     # Project overview and instructions
├── requirements.txt              # Python dependencies
├── streamlit_app.py              # Streamlit UI
│
├── app/
│   ├── main.py                   # FastAPI application entry point
│   ├── api/
│   │   ├── documents.py          # POST /documents/upload, GET /documents/list
│   │   └── rag.py                # POST /rag/query, POST /rag/query-stream
│   ├── ingestion/
│   │   ├── ingest_and_vectorize.py   # Download papers + embed and index into ChromaDB
│   │   ├── downloader.py         # arXiv paper downloader
│   │   ├── parser.py             # PDF text extraction with page metadata
│   │   └── chunker.py            # Text chunking + ChromaDB VectorStore
│   ├── rag/
│   │   ├── retriever.py          # Semantic search against ChromaDB
│   │   ├── generator.py          # Groq LLM answer generation
│   │   ├── prompt_builder.py     # Prompt and message construction
│   │   └── pipeline.py           # End-to-end RAG orchestration
│   └── verification/
│       ├── claim_extractor.py    # Decompose answer into atomic claims
│       └── claim_verifier.py     # Verify each claim against vector store
│
├── config/
│   └── settings.py               # Application settings and constants
│
├── data/
│   └── papers/                   # Downloaded PDF files (research papers)
│
├── vectorstore/                  # ChromaDB persistent storage
│   └── chroma.sqlite3            # Vector embeddings database
│
├── evaluation/
│   ├── test_cases.py             # 15 test questions across 3 difficulty levels
│   └── evaluator.py              # Runs pipeline + measures all metrics
│
└── tests/                        # Unit and integration tests
    ├── conftest.py               # Shared pytest fixtures
    ├── test_ingestion.py         # Ingestion pipeline tests
    └── test_rag.py               # RAG pipeline tests
```

## Quick Start

### Prerequisites

- Python 3.12+
- Groq API key: create a free account at [Groq](https://www.groq.com/) and generate an API key to access Llama 3.3 70B

### Installation

```bash
# Navigate into the project
cd claim-from-papers

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Create .env with your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

### Run the Application

Before starting, ensure you have at least 30 papers in `data/papers/` or run the ingestion script to download and process them (see Vector Store section below).

**Terminal 1 — FastAPI backend:**
```bash
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit UI:**
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

- Streamlit UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs


## Vector Store

| Property | Value |
|---|---|
| Location | `vectorstore/` |
| Collection | `research_papers` |
| Embedding model | BGE-M3 (1024 dimensions) |
| Chunk size | 500 tokens with 50-token overlap |
| Source documents | 30 arXiv PDFs on text classification using LLMs (2024-2026) |

To rebuild from scratch:
```bash
python app/ingestion/ingest_and_vectorize.py                  # download + embed and store
python app/ingestion/ingest_and_vectorize.py --skip-download  # re-index only
```

## Evaluation

Runs 15 written test questions across three difficulty levels (easy, medium, hard) and measures:

- **Claim Grounding Rate** — fraction of claims verified as grounded (always computed)
- **Keyword Coverage** — fraction of expected keywords present in the answer
- **Answer Relevancy** — DeepEval metric using Groq as the LLM judge
- **Faithfulness** — DeepEval metric using Groq as the LLM judge

```bash
source venv/bin/activate
python evaluation/evaluator.py
```

Results are printed to stdout and saved to `evaluation/results.json`.


## API Reference

1. **POST /rag/query** — Ask a question and receive a grounded answer with claim verification.
2. **POST /rag/query-stream** — Ask a question and receive a streamed answer with real-time claim verification updates.
3. **POST /documents/upload** — Upload a PDF to expand the knowledge base.
4. **GET /documents/list** — List all PDFs in the papers directory.


## Tests

### Test files

| File | What it covers |
|---|---|
| `tests/conftest.py` | Session-scoped fixtures shared across all test files |
| `tests/test_ingestion.py` | `ArxivDownloader`, `PDFParser`, `TextChunker`, `VectorStore` |
| `tests/test_rag.py` | `Retriever`, `PromptBuilder`, `Generator`, `RAGPipeline` |

### How the fixtures work

`conftest.py` provides session-scoped fixtures so test data is prepared once and reused:

1. Attempts to download 5 arXiv papers into `data/test/` at the start of the session.
2. If arXiv is unavailable (rate-limited or returns an error), automatically falls back to copying 5 papers from `data/papers/` so tests can still run offline.
3. Shares those papers across every test class that needs them.
4. Deletes `data/test/` automatically after all tests complete.

Tests that depend on real PDFs are skipped gracefully (not failed) when no papers are available from either source.

### Running tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific file
pytest tests/test_rag.py -v

# Run a specific class
pytest tests/test_rag.py::TestRetriever -v
```