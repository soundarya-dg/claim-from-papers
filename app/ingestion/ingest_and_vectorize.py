"""
Ingestion entry point.

Reads all configuration from config/settings.py, then:
    1. Downloads papers from arXiv
    2. Parses every PDF in the papers directory
    3. Chunks the text and stores embeddings in ChromaDB

Run from the project root:
    python app/ingestion/ingest_and_vectorize.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    PAPERS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOWNLOAD_TOPIC,
    DOWNLOAD_NUM_PAPERS,
    DOWNLOAD_START_YEAR,
    DOWNLOAD_END_YEAR,
)
from app.ingestion.downloader import ArxivDownloader
from app.ingestion.parser import PDFParser
from app.ingestion.chunker import TextChunker, VectorStore


def run_download() -> list:
    downloader = ArxivDownloader(
        output_dir=PAPERS_DIR,
        max_results=DOWNLOAD_NUM_PAPERS,
        start_year=DOWNLOAD_START_YEAR,
        end_year=DOWNLOAD_END_YEAR,
    )
    return downloader.download_papers(query=DOWNLOAD_TOPIC)


def run_ingestion():
    parser = PDFParser()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = VectorStore()

    parsed_docs = parser.parse_directory(PAPERS_DIR)
    if not parsed_docs:
        print(f"No PDF files found in '{PAPERS_DIR}'. Run download step first.")
        sys.exit(1)

    all_chunks = []
    for doc in parsed_docs:
        all_chunks.extend(chunker.chunk_document(doc))

    vector_store.add_chunks(all_chunks)
    stats = vector_store.get_stats()
    print(f"\nDone. Total chunks in vector store: {stats['total_chunks']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest research papers into the vector store.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and only (re-)index PDFs already in data/papers/",
    )
    args = parser.parse_args()

    if not args.skip_download:
        run_download()

    run_ingestion()