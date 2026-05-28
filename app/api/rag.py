from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.pipeline import RAGPipeline


# Create router
router = APIRouter(prefix="/rag", tags=["rag"])

# Initialize pipeline (singleton pattern)
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# Request/Response Models
class Message(BaseModel):
    """Conversation message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """RAG query request."""
    question: str = Field(..., description="User's question", min_length=1)
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Optional conversation history"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve",
        ge=1, # greater than or equal to 1
        le=20 # less than or equal to 20
    )
    temperature: Optional[float] = Field(
        default=None,
        description="LLM temperature",
        ge=0.0,
        le=2.0
    )
    include_sources: bool = Field(
        default=True,
        description="Include source information in response"
    )


class Source(BaseModel):
    """Source citation."""
    source_id: int
    title: str
    page: str
    filename: str


class ClaimResult(BaseModel):
    """Verification result for a single claim."""
    claim: str
    label: str
    confidence: float
    supporting_chunk: Optional[str] = None


class QueryResponse(BaseModel):
    """RAG query response."""
    question: str
    answer: str
    chunks_retrieved: int
    sources: Optional[List[Source]] = None
    claims: Optional[List[ClaimResult]] = None
    grounding_rate: Optional[float] = None
    metadata: Dict


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> JSONResponse:
    """
    Ask a question using RAG.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Answer with sources and metadata
    """
    try:
        # Get pipeline
        pipeline = get_pipeline()
        
        # Convert conversation history if provided
        history = None
        if request.conversation_history:
            history = [msg.model_dump() for msg in request.conversation_history]
        
        # Execute query
        result = pipeline.query(
            question=request.question,
            conversation_history=history,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources
        )
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {result['error']}"
            )
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/query-stream")
async def query_rag_stream(request: QueryRequest):
    """
    Ask a question using RAG with streaming response.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Streaming response with answer chunks
    """
    try:
        # Get pipeline
        pipeline = get_pipeline()
        
        # Convert conversation history if provided
        history = None
        if request.conversation_history:
            history = [msg.model_dump() for msg in request.conversation_history]
        
        # Create streaming generator
        def generate():
            for chunk in pipeline.query_streaming(
                question=request.question,
                conversation_history=history,
                top_k=request.top_k,
                temperature=request.temperature
            ):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )