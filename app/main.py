from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.api.documents import router as documents_router
from app.api.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    yield
    # Shutdown


# Create FastAPI app
app = FastAPI(
    title="Claim From Papers",
    description="Ask questions from research papers and verify exactly which parts of the answer are true.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(documents_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Claim From Papers",
        "version": "1.0.0",
        "description": "Ask questions from research papers and verify exactly which parts of the answer are true.",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "documents": {
                "upload": "POST /documents/upload",
                "list": "GET /documents/list"
            },
            "rag": {
                "query": "POST /rag/query",
                "query_stream": "POST /rag/query-stream"
            }
        }
    }


if __name__ == "__main__":
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )