import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.ingestion.chunker import VectorStore
from config.settings import TOP_K_CHUNKS


class Retriever:
    """Retrieves relevant document chunks from ChromaDB vector store using semantic search."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None, top_k: int = TOP_K_CHUNKS):
        """
        Args:
            vector_store: VectorStore instance (creates new if None)
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None, filter_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (overrides default)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = top_k or self.top_k
        
        # Query the vector store
        results = self.vector_store.query(
            query_text=query,
            top_k=k,
            filter_metadata=filter_metadata
        )
        
        # Format results
        retrieved_chunks = []
        
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]  # First query results
            metadatas = results['metadatas'][0] if 'metadatas' in results else []
            distances = results['distances'][0] if 'distances' in results else []
            
            for i, doc in enumerate(documents):
                chunk = {
                    'text': doc,
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'distance': distances[i] if i < len(distances) else None,
                    'rank': i + 1
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None, filter_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Retrieve chunks with detailed scoring information.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary with chunks and retrieval statistics
        """
        chunks = self.retrieve(query, top_k, filter_metadata)
        
        # Calculate statistics
        avg_distance = sum(c['distance'] for c in chunks if c['distance'] is not None) / len(chunks) if chunks else 0
        
        return {
            'query': query,
            'chunks': chunks,
            'total_retrieved': len(chunks),
            'avg_distance': avg_distance,
            'top_k': top_k or self.top_k
        }
    
    def get_unique_sources(self, chunks: List[Dict[str, any]]) -> List[Dict[str, str]]:
        """
        Extract unique source documents from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of unique sources with title and filename
        """
        sources = {}
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            if filename not in sources:
                sources[filename] = {
                    'title': metadata.get('title', 'Unknown'),
                    'filename': filename
                }
        
        return list(sources.values())