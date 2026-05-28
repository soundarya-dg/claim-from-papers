import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid
import re
from config.settings import (CHUNK_SIZE,CHUNK_OVERLAP,EMBEDDING_MODEL,VECTORSTORE_DIR)
from app.ingestion.parser import PDFParser
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# TextChunker - Cuts long text into smaller overlapping pieces
# VectorStore — Manages ChromaDB; stores and searches chunks

class TextChunker:
    """Chunks text into overlapping token-based pieces."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenization by splitting on whitespace.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        return tokens
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Chunk text into overlapping pieces.
    
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        tokens = self._tokenize(text)
        chunks = []
        
        # Create overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunk_size': len(chunk_tokens),
                    'start_token': start,
                    'end_token': min(end, len(tokens))
                }
            }
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def chunk_document(
        self,
        parsed_doc: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Chunk an entire parsed document page by page.
        
        Args:
            parsed_doc: Parsed document from PDFParser
            
        Returns:
            List of all chunks from all pages
        """
        all_chunks = []
        
        for page in parsed_doc.get('pages', []):
            page_text = page.get('text', '')
            page_metadata = page.get('metadata', {})
            
            # Chunk the page text
            chunks = self.chunk_text(page_text, page_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


class VectorStore:
    """Manages ChromaDB vector store for document chunks."""
    
    def __init__(self, collection_name: str = "research_papers", persist_directory: str = VECTORSTORE_DIR, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
            embedding_model: Name of the sentence-transformer model
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f" Model loaded: {embedding_model}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper chunks with BGE-M3 embeddings"}
        )
        
        print(f"  Connected to ChromaDB collection: {collection_name}")
        print(f"  Current document count: {self.collection.count()}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def add_chunks(self, chunks: List[Dict[str, any]], batch_size: int = 100) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        print(f"\nAdding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embed_batch(texts)
            
            # Generate unique IDs
            ids = [str(uuid.uuid4()) for _ in batch]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"  Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)")
        
        print(f"  Successfully added {len(chunks)} chunks")
        return len(chunks)
    
    def query(self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Query the vector store.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Query results
        """
        # Generate query embedding
        query_embedding = self.embed_text(query_text)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        return results
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Statistics dictionary
        """
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def reset(self):
        """Delete and recreate the collection."""
        print(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research paper chunks with BGE-M3 embeddings"}
        )
        print(f"  Collection reset")