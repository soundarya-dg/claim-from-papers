from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import shutil
import tempfile
import os

from app.ingestion.parser import PDFParser
from app.ingestion.chunker import TextChunker, VectorStore
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, PAPERS_DIR


# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize components (singleton pattern)
_parser = None
_chunker = None
_vector_store = None


def get_parser() -> PDFParser:
    """Get or create PDF parser instance."""
    global _parser
    if _parser is None:
        _parser = PDFParser()
    return _parser


def get_chunker() -> TextChunker:
    """Get or create text chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return _chunker


def get_vector_store() -> VectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    save_to_papers: bool = False
) -> JSONResponse:
    """
    Upload and ingest a PDF document.
    
    Args:
        file: PDF file to upload
        save_to_papers: Whether to save the PDF to the papers directory
        
    Returns:
        JSON response with ingestion results
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file to save upload
    temp_file = None
    saved_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Get components
        parser = get_parser()
        chunker = get_chunker()
        vector_store = get_vector_store()
        
        # Parse the PDF
        try:
            parsed_doc = parser.parse_pdf(temp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse PDF: {str(e)}"
            )
        
        # Check if document has content
        if not parsed_doc['pages']:
            raise HTTPException(
                status_code=400,
                detail="No text content found in PDF"
            )
        
        # Chunk the document
        chunks = chunker.chunk_document(parsed_doc)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to create chunks from document"
            )
        
        # Add to vector store
        chunks_added = vector_store.add_chunks(chunks)
        
        # Optionally save to papers directory
        if save_to_papers:
            papers_dir = Path(PAPERS_DIR)
            papers_dir.mkdir(parents=True, exist_ok=True)
            saved_path = papers_dir / file.filename
            shutil.copy(temp_path, saved_path)
        
        # Prepare response
        response = {
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully",
            "document": {
                "filename": file.filename,
                "title": parsed_doc['title'],
                "total_pages": parsed_doc['total_pages'],
                "chunks_created": len(chunks),
                "chunks_added": chunks_added
            },
            "saved_to_papers": save_to_papers,
            "saved_path": str(saved_path) if saved_path else None
        }
        
        return JSONResponse(content=response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@router.get("/list")
async def list_papers() -> JSONResponse:
    """
    List all papers in the papers directory.
    
    Returns:
        JSON response with list of papers
    """
    try:
        papers_dir = Path(PAPERS_DIR)
        
        if not papers_dir.exists():
            return JSONResponse(
                content={
                    "papers": [],
                    "count": 0,
                    "directory": str(papers_dir)
                },
                status_code=200
            )
        
        # Get all PDF files
        pdf_files = sorted(papers_dir.glob("*.pdf"))
        
        papers = []
        for pdf_file in pdf_files:
            papers.append({
                "filename": pdf_file.name,
                "size_bytes": pdf_file.stat().st_size,
                "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2)
            })
        
        return JSONResponse(
            content={
                "papers": papers,
                "count": len(papers),
                "directory": str(papers_dir)
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list papers: {str(e)}"
        )