import sys
from pathlib import Path
from typing import List, Dict, Optional
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.retriever import Retriever
from app.rag.prompt_builder import PromptBuilder
from app.rag.generator import Generator
from app.verification.claim_extractor import ClaimExtractor
from app.verification.claim_verifier import ClaimVerifier
from config.settings import TOP_K_CHUNKS


# RAG pipeline implementation - Orchestrates the entire flow in one place.
# This includes retrieval, prompt builder, generation, claim extraction, and verification.
# The API layer will call this pipeline to get answers to user questions.
class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""
    
    def __init__(self, retriever: Optional[Retriever] = None, prompt_builder: Optional[PromptBuilder] = None, generator: Optional[Generator] = None, top_k: int = TOP_K_CHUNKS):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance (creates new if None)
            prompt_builder: PromptBuilder instance (creates new if None)
            generator: Generator instance (creates new if None)
            top_k: Number of chunks to retrieve
        """
        self.retriever = retriever or Retriever(top_k=top_k)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generator = generator or Generator()
        self.top_k = top_k
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier(retriever=self.retriever)
    
    def query(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None, temperature: Optional[float] = None, include_sources: bool = True) -> Dict[str, any]:
        """
        Execute the complete RAG pipeline.
        
        Args:
            question: User's question
            conversation_history: Optional conversation history
            top_k: Override default number of chunks to retrieve
            temperature: Override default LLM temperature
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k or self.top_k
            )
            retrieval_time = time.time() - retrieval_start
            
            if not chunks:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in the papers to answer this question.",
                    'sources': [],
                    'chunks_retrieved': 0,
                    'claims': [],
                    'grounding_rate': 0.0,
                    'error': 'No relevant chunks found',
                    'metadata': {
                        'retrieval_time': retrieval_time,
                        'generation_time': 0,
                        'total_time': time.time() - start_time
                    }
                }
            
            # Step 2: Build prompt
            messages = self.prompt_builder.build_messages(
                query=question,
                chunks=chunks,
                conversation_history=conversation_history
            )
            
            # Step 3: Generate answer
            generation_start = time.time()
            result = self.generator.generate_from_messages(
                messages=messages,
                temperature=temperature
            )
            generation_time = time.time() - generation_start
            
            answer = result['answer']
            
            # Step 4: Claim extraction and verification
            try:
                claims = self.claim_extractor.extract(answer)
                verification_results = self.claim_verifier.verify_claims(claims)
                grounding_rate = self.claim_verifier.compute_grounding_rate(verification_results)
            except Exception:
                verification_results = []
                grounding_rate = 0.0

            # Step 5: Prepare response
            response = {
                'question': question,
                'answer': answer,
                'chunks_retrieved': len(chunks),
                'claims': verification_results,
                'grounding_rate': grounding_rate,
                'metadata': {
                    'model': result['model'],
                    'usage': result['usage'],
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'total_time': time.time() - start_time
                }
            }
            
            # Add sources if requested
            if include_sources:
                sources = self.prompt_builder.extract_sources_from_chunks(chunks)
                response['sources'] = sources
                response['retrieved_chunks'] = chunks
            
            return response
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'chunks_retrieved': 0,
                'claims': [],
                'grounding_rate': 0.0,
                'error': str(e),
                'metadata': {
                    'total_time': time.time() - start_time
                }
            }
    
    def query_streaming(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None, temperature: Optional[float] = None):
        """
        Execute RAG pipeline with streaming response.
        
        Args:
            question: User's question
            conversation_history: Optional conversation history
            top_k: Override default number of chunks to retrieve
            temperature: Override default LLM temperature
            
        Yields:
            Chunks of the answer as they are generated
        """
        try:
            # Retrieve chunks
            chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k or self.top_k
            )
            
            if not chunks:
                yield "I couldn't find any relevant information in the papers to answer this question."
                return
            
            # Build prompt
            messages = self.prompt_builder.build_messages(
                query=question,
                chunks=chunks,
                conversation_history=conversation_history
            )
            
            # Stream answer
            for chunk in self.generator.generate_streaming(
                messages=messages,
                temperature=temperature
            ):
                yield chunk
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_sources_only(self, question: str, top_k: Optional[int] = None) -> Dict[str, any]:
        """
        Retrieve and return only the sources without generating an answer.
        
        Args:
            question: User's question
            top_k: Number of sources to retrieve
            
        Returns:
            Dictionary with retrieved sources
        """
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k or self.top_k
        )
        
        sources = self.prompt_builder.extract_sources_from_chunks(chunks)
        
        return {
            'question': question,
            'sources': sources,
            'chunks': chunks,
            'total_retrieved': len(chunks)
        }
