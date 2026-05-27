# Claim From Papers

This project is a proof-of-concept RAG system that not only answers questions based on research papers but also provides transparent claim verification for each part of the answer. It ensures that every claim in the generated answer is backed by real sources, giving users confidence in the information they receive.

### Motivation

With the rise of LLMs, there's a growing concern about hallucinations and misinformation.

How can we trust an answer if we don't know which parts are grounded in reality? - this project tackles that problem head-on by breaking down answers into atomic claims and verifying each one against the source documents. It's not just about getting an answer; it's about getting a **verified** answer.

### How It Works

1. **Knowledge Base**: The app starts with a knowledge base of 30 arXiv papers on "AI-Generated Text Detection" (topic can be changed based on user preference), processed into chunks and stored in ChromaDB.
2. **User Interaction**: Users can ask any question related to the topic, and the system will fetch relevant information from the papers to generate an answer. Also, users can upload their own PDFs to expand the knowledge base.
3. **Claim Extraction & Verification**: The generated answer is automatically broken down into atomic claims, and each claim is independently verified against the source documents.
4. **Verdict System**: Each claim gets a verdict of "Grounded", "Unverified", or "Contradicted", giving users a clear indication of the reliability of each part of the answer.
5. **Transparent Output**: The final answer is presented with a "Claim Grounding Rate" that indicates the percentage of claims backed by real sources, along with detailed verdicts for each claim.


## Technologies Used

- **Language**: Python 3.12
- **LLM API**: Groq (access to Llama 3.3 70B)
- **Embeddings**: BGE-M3 (local, free)
- **Vector Database**: ChromaDB
- **Web Framework**: FastAPI
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF
- **Evaluation**: DeepEval


## Project Structure

```
ClaimFromPapers/
├── README.md                          
├── requirements.txt                  # Python dependencies
├── streamlit_app.py                  # Streamlit UI
│
├── app/                              
│   ├── main.py                       # FastAPI application
│   │
│   ├── api/                          # API endpoints
│   │   ├── documents.py              # Document upload & management
│   │   └── rag.py                    # RAG query endpoints
│   │
│   ├── ingestion/                    # Document processing pipeline
│   │   ├── downloader.py             # Paper downloader
│   │   ├── parser.py                 # PDF text extraction with metadata
│   │   └── chunker.py                # Text chunking & vector storage
│   │
│   ├── rag/                          # RAG pipeline components
│   │   ├── retriever.py              # Semantic search & retrieval
│   │   ├── generator.py              # LLM answer generation
│   │   ├── prompt_builder.py         # Prompt templates
│   │   └── pipeline.py               # End-to-end RAG orchestration
│   │
│   ├── verification/                 # Claim verification system
│   │   ├── claim_extractor.py        # Extract atomic claims from answers
│   │   └── claim_verifier.py         # Verify claims against sources
│   │
│   └── middleware/                   
│       ├── logger.py                 # Logging configuration
│       └── error_handler.py          # Error handling utilities
│
├── config/                           
│   └── settings.py                   # Application settings & constants
│
├── data/                             # Data storage
│   └── papers/                       # Downloaded research papers (PDFs)
│       └── *.pdf                     
│
├── vectorstore/                      # ChromaDB persistence
│   ├── chroma.sqlite3                # Vector embeddings database
│   └── [collection_data]             
│
├── evaluation/                       # Evaluation & metrics
│   ├── evaluator.py                  # RAG & verification metrics
│   └── test_cases.py                 # Test queries & expected outputs
│
├── tests/                            # Unit & integration tests
│   ├── conftest.py                   # Shared pytest fixtures (session-scoped)
│   └── test_ingestion.py             # Tests for ingestion pipeline
│   ├── test_rag.py                   # RAG pipeline tests
│   └── test_verification.py          # Claim verification tests
│
└── docker/                           # Containerization
    ├── Dockerfile                    # Docker image definition
    └── docker-compose.yml            # Multi-container orchestration
```