import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from app.ingestion.parser import PDFParser
from app.ingestion.chunker import TextChunker, VectorStore


class TestArxivDownloader:
    """Tests for the ArxivDownloader class - uses session fixture from conftest.py"""
    
    def test_papers_downloaded(self, test_papers_dir, test_papers_list):
        """Test that papers were downloaded successfully."""
        assert test_papers_dir.exists()
        assert len(test_papers_list) >= 3  # At least 3 papers downloaded
        
        # Check all are PDFs
        for pdf_file in test_papers_list:
            assert pdf_file.suffix == '.pdf'
            assert pdf_file.exists()
            assert pdf_file.stat().st_size > 0  # Not empty


class TestPDFParser:
    """Tests for the PDFParser class - uses real downloaded test papers."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PDFParser()
    
    def test_initialization(self, parser):
        """Test parser initialization."""
        assert parser is not None
    
    def test_parse_directory(self, parser, test_papers_dir):
        """Test parsing directory of test PDFs."""
        parsed_docs = parser.parse_directory(str(test_papers_dir))
        
        assert isinstance(parsed_docs, list)
        assert len(parsed_docs) >= 3  # At least 3 papers parsed
        
        # Check structure of parsed documents
        for doc in parsed_docs:
            assert 'title' in doc
            assert 'pages' in doc
            assert 'total_pages' in doc
            assert 'filename' in doc
            assert isinstance(doc['pages'], list)
            assert len(doc['pages']) > 0
            
            # Check page structure
            for page in doc['pages']:
                assert 'page_number' in page
                assert 'text' in page
                assert 'metadata' in page
                assert isinstance(page['text'], str)
    
    def test_parse_single_pdf(self, parser, test_papers_list):
        """Test parsing a single PDF file from test papers."""
        # Parse first test PDF
        pdf_path = test_papers_list[0]
        parsed_doc = parser.parse_pdf(str(pdf_path))
        
        assert parsed_doc is not None
        assert 'title' in parsed_doc
        assert 'pages' in parsed_doc
        assert len(parsed_doc['pages']) > 0
        assert 'filename' in parsed_doc
        
        # Verify page content
        first_page = parsed_doc['pages'][0]
        assert 'text' in first_page
        assert len(first_page['text']) > 100  # Should have substantial text


class TestTextChunker:
    """Tests for the TextChunker class - uses real parsed documents."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return TextChunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PDFParser()
    
    def test_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
    
    def test_chunk_text(self, chunker):
        """Test chunking text."""
        text = "This is a test sentence. " * 50  # Create long text
        metadata = {'title': 'Test', 'page': 1, 'filename': 'test.pdf'}
        chunks = chunker.chunk_text(text, metadata=metadata)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Each chunk should be a dictionary with text and metadata
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert len(chunk['text']) > 0

            # Should have both passed metadata and chunk-specific metadata
            assert chunk['metadata']['title'] == 'Test'
            assert 'chunk_index' in chunk['metadata']
    
    def test_chunk_real_document(self, chunker, parser, test_papers_list):
        """Test chunking a real parsed PDF document."""
        # Parse first test paper
        pdf_path = test_papers_list[0]
        parsed_doc = parser.parse_pdf(str(pdf_path))
        
        # Chunk the document
        chunks = chunker.chunk_document(parsed_doc)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Check chunk structure
        for chunk in chunks:
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert 'title' in chunk['metadata']
            assert 'page' in chunk['metadata']
            assert 'filename' in chunk['metadata']
            assert len(chunk['text']) > 0
            # Should also have chunk-specific metadata
            assert 'chunk_index' in chunk['metadata']
            assert 'chunk_size' in chunk['metadata']


class TestVectorStore:
    """Tests for the VectorStore class - uses real chunked documents."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store instance with test collection."""
        # Use a test collection name
        vs = VectorStore(collection_name="test_collection")
        vs.reset()  # Start fresh
        return vs
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PDFParser()
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_initialization(self, vector_store):
        """Test vector store initialization."""
        assert vector_store is not None
        assert vector_store.collection_name == "test_collection"
    
    def test_add_and_query_real_chunks(self, vector_store, parser, chunker, test_papers_list):
        """Test adding real chunks from test papers and querying."""
        # Parse and chunk first test paper
        pdf_path = test_papers_list[0]
        parsed_doc = parser.parse_pdf(str(pdf_path))
        chunks = chunker.chunk_document(parsed_doc)
        
        # Take first 10 chunks to keep test fast
        test_chunks = chunks[:10]
        
        # Add chunks to vector store
        vector_store.add_chunks(test_chunks)
        
        # Query
        results = vector_store.query("AI-generated text detection", top_k=3)
        
        assert 'documents' in results
        assert 'metadatas' in results
        assert 'distances' in results
        assert len(results['documents'][0]) <= 3
        assert len(results['documents'][0]) > 0  # Should find matches
    
    def test_get_stats(self, vector_store, parser, chunker, test_papers_list):
        """Test getting vector store statistics with real data."""
        # Parse and chunk first test paper
        pdf_path = test_papers_list[0]
        parsed_doc = parser.parse_pdf(str(pdf_path))
        chunks = chunker.chunk_document(parsed_doc)
        
        # Add first 5 chunks
        vector_store.add_chunks(chunks[:5])
        
        stats = vector_store.get_stats()
        
        assert 'total_chunks' in stats
        assert 'collection_name' in stats
        assert stats['total_chunks'] >= 5
    
    def test_reset(self, vector_store):
        """Test resetting the collection."""
        # Add a sample chunk
        chunks = [
            {
                'text': 'Test chunk',
                'metadata': {
                    'title': 'Test',
                    'page': 1,
                    'filename': 'test.pdf'
                }
            }
        ]
        vector_store.add_chunks(chunks)
        
        # Reset
        vector_store.reset()
        
        # Stats should show 0 chunks
        stats = vector_store.get_stats()
        assert stats['total_chunks'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])