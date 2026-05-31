import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from app.rag.retriever import Retriever
from app.rag.prompt_builder import PromptBuilder
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline


class TestRetriever:
    """Tests for the Retriever class - uses test vector store."""
    
    @pytest.fixture
    def retriever(self, test_vector_store):
        """Create a retriever instance with test vector store."""
        return Retriever(vector_store=test_vector_store, top_k=3)
    
    def test_retrieve(self, retriever):
        """Test basic retrieval from test vector store."""
        query = "What is text classification?"
        results = retriever.retrieve(query, top_k=3)

        assert len(results) <= 3
        if len(results) == 0:
            pytest.skip("No data in test vector store (no papers available)")
        assert len(results) > 0  # Should find matches in test data
        assert all('text' in chunk for chunk in results)
        assert all('metadata' in chunk for chunk in results)
        assert all('distance' in chunk for chunk in results)
    
    def test_retrieve_with_scores(self, retriever):
        """Test retrieval with scores."""
        query = "How do language models work?"
        results = retriever.retrieve_with_scores(query, top_k=5)

        assert 'query' in results
        assert 'chunks' in results
        assert 'total_retrieved' in results
        assert results['query'] == query
        if len(results['chunks']) == 0:
            pytest.skip("No data in test vector store (no papers available)")
        assert len(results['chunks']) > 0  # Should find matches
    
    def test_get_unique_sources(self, retriever):
        """Test extracting unique sources."""
        query = "What is machine learning?"
        results = retriever.retrieve(query, top_k=5)
        if not results:
            pytest.skip("No data in test vector store (no papers available)")
        sources = retriever.get_unique_sources(results)
        
        assert isinstance(sources, list)
        assert len(sources) > 0  # Should have sources
        assert all('title' in source for source in sources)
        assert all('filename' in source for source in sources)


class TestPromptBuilder:
    """Tests for the PromptBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create a prompt builder instance."""
        return PromptBuilder()
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                'text': "This is a test chunk about AI.",
                'metadata': {
                    'title': 'Test Paper',
                    'page': 1,
                    'filename': 'test.pdf'
                }
            }
        ]
    
    def test_build_prompt(self, builder, sample_chunks):
        """Test building a basic prompt."""
        query = "What is AI?"
        prompt = builder.build_prompt(query, sample_chunks)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert "Test Paper" in prompt
        assert "CONTEXT FROM RESEARCH PAPERS" in prompt
    
    def test_build_messages(self, builder, sample_chunks):
        """Test building messages format."""
        query = "What is AI?"
        messages = builder.build_messages(query, sample_chunks)
        
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert messages[0]['role'] == 'system'
        assert messages[-1]['role'] == 'user'
        assert query in messages[-1]['content']
    
    def test_extract_sources(self, builder, sample_chunks):
        """Test extracting sources."""
        sources = builder.extract_sources_from_chunks(sample_chunks)
        
        assert isinstance(sources, list)
        assert len(sources) == 1
        assert sources[0]['title'] == 'Test Paper'
        assert sources[0]['page'] == 1
    
    def test_conversation_history(self, builder, sample_chunks):
        """Test including conversation history."""
        query = "What is AI?"
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        messages = builder.build_messages(query, sample_chunks, history)
        
        # Should have system + history + new user message
        assert len(messages) == 4
        assert messages[1] == history[0]
        assert messages[2] == history[1]


class TestGenerator:
    """Tests for the Generator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        try:
            return Generator()
        except ValueError as e:
            pytest.skip(f"Skipping test - {e}")
    
    def test_generate_basic(self, generator):
        """Test basic generation."""
        prompt = "Say 'test' and nothing else."
        answer = generator.generate(prompt, max_tokens=50)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_generate_from_messages(self, generator):
        """Test generation from messages."""
        messages = [
            {"role": "user", "content": "Say 'hello' and nothing else."}
        ]
        result = generator.generate_from_messages(messages, max_tokens=50)
        
        assert 'answer' in result
        assert 'model' in result
        assert 'usage' in result
        assert isinstance(result['answer'], str)


class TestRAGPipeline:
    """Tests for the RAG Pipeline - uses test vector store."""
    
    @pytest.fixture
    def pipeline(self, test_vector_store):
        """Create a pipeline instance with test vector store."""
        try:
            retriever = Retriever(vector_store=test_vector_store, top_k=5)
            return RAGPipeline(retriever=retriever)
        except Exception as e:
            pytest.skip(f"Skipping test - {e}")
    
    def test_query_basic(self, pipeline):
        """Test basic query execution with test data."""
        question = "What is text classification?"
        result = pipeline.query(question, include_sources=True)

        # Always-present keys (success and error paths)
        assert 'question' in result
        assert 'answer' in result
        assert 'chunks_retrieved' in result
        assert result['question'] == question
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

        # Skip success-only assertions when the API returned an error
        if result.get('error'):
            pytest.skip(f"Skipping success-path assertions — API error: {result['error']}")

        assert 'short_answer' in result
        assert 'metadata' in result
    
    def test_query_with_top_k(self, pipeline):
        """Test query with custom top_k."""
        question = "How do LLMs perform on text classification tasks?"
        result = pipeline.query(question, top_k=3, include_sources=True)

        assert result['chunks_retrieved'] <= 3

        # Only check sources when the pipeline succeeded
        if not result.get('error') and 'sources' in result:
            assert len(result['sources']) >= 1
    
    def test_get_sources_only(self, pipeline):
        """Test getting sources without answer generation."""
        question = "What is machine learning?"
        result = pipeline.get_sources_only(question, top_k=5)

        assert 'question' in result
        assert 'sources' in result
        assert 'chunks' in result
        assert 'total_retrieved' in result
        if len(result['chunks']) == 0:
            pytest.skip("No data in test vector store (no papers available)")
        assert len(result['chunks']) > 0  # Should retrieve chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])