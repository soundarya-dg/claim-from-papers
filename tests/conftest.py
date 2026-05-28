import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime
from app.ingestion.downloader import ArxivDownloader
from app.ingestion.parser import PDFParser
from app.ingestion.chunker import TextChunker, VectorStore


@pytest.fixture(scope="session")
def test_papers_dir():
    """
    Download test papers once for the entire test session.
    Creates "data/test/" folder, downloads 5 papers, and cleans up after all tests.
    """
    # Create test directory
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Download papers
    downloader = ArxivDownloader(
        output_dir=str(test_dir),
        max_results=5,
        start_year=2024,
        end_year=datetime.now().year
    )
    
    downloaded_files = downloader.download_papers(
        query="AI-Generated Text Detection",
        category="cs.CL"
    )
    
    # Provide the test directory to all tests
    yield test_dir
    
    # Cleanup after all tests complete
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def test_papers_list(test_papers_dir):
    """
    Get list of downloaded test PDF files.
    """
    pdf_files = list(test_papers_dir.glob("*.pdf"))
    return pdf_files


@pytest.fixture(scope="session")
def test_vector_store(test_papers_dir):
    """
    Create and populate a test vector store with chunks from test papers.
    Uses session scope so it's created once and shared across all RAG tests.
    """
    # Create test vector store with unique collection name
    vector_store = VectorStore(collection_name="test_rag_collection")
    vector_store.reset()  # Start fresh
    
    # Parse and chunk test papers
    parser = PDFParser()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    parsed_docs = parser.parse_directory(str(test_papers_dir))
    
    all_chunks = []
    for doc in parsed_docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    
    # Add chunks to vector store
    vector_store.add_chunks(all_chunks)
    
    yield vector_store
    
    # Cleanup - reset the test collection
    vector_store.reset()