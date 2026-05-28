from typing import List, Dict, Optional


class PromptBuilder:
    """Builds prompts for RAG-based question answering."""
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.system_prompt = self._get_system_prompt()
    
    # Get the system prompt that instructs the LLM
    def _get_system_prompt(self) -> str:

        return """You are a precise research assistant that answers questions based on provided research papers.

CRITICAL RULES:
1. Only answer using information from the provided context
2. If the context doesn't contain the answer, say "I cannot answer this based on the provided papers"
3. Be specific and cite which paper or section supports your answer
4. Do not make up information or use outside knowledge
5. If multiple papers disagree, mention both perspectives
6. Keep answers clear and concise

Your goal is accuracy and grounding in the source material.
"""

    def _format_chunks(self, chunks: List[Dict[str, any]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '')
            
            title = metadata.get('title', 'Unknown')
            page = metadata.get('page', 'N/A')
            
            context_parts.append(
                f"[Source {i}] {title} (Page {page}):\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    # formats past turns for multi-turn conversations
    def _format_conversation_history(self, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format conversation history for context.
        
        Args:
            conversation_history: List of {role, content} dicts
            
        Returns:
            Formatted history string
        """
        if not conversation_history:
            return ""
        
        history_parts = ["Previous conversation:"]
        
        for turn in conversation_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            
            if role == 'user':
                history_parts.append(f"User: {content}")
            elif role == 'assistant':
                history_parts.append(f"Assistant: {content}")
        
        history_parts.append("")  # Empty line separator
        return "\n".join(history_parts)
    
    def build_prompt(self, query: str, chunks: List[Dict[str, any]], conversation_history: Optional[List[Dict[str, str]]] = None, include_system: bool = True) -> str:
        """
        Build complete prompt for the LLM.
        
        Args:
            query: User's question
            chunks: Retrieved context chunks
            conversation_history: Optional conversation history
            include_system: Whether to include system prompt
            
        Returns:
            Complete prompt string
        """
        parts = []
        
        # Add system prompt
        if include_system:
            parts.append(self.system_prompt)
            parts.append("")
        
        # Add conversation history
        if conversation_history:
            history = self._format_conversation_history(conversation_history)
            if history:
                parts.append(history)
        
        # Add context
        parts.append("CONTEXT FROM RESEARCH PAPERS:")
        parts.append("-"*80)
        parts.append(self._format_chunks(chunks))
        parts.append("-"*80)
        parts.append("")
        
        # Add question
        parts.append(f"QUESTION: {query}")
        parts.append("")
        parts.append("ANSWER:")
        
        return "\n".join(parts)
    
    # Builds the messages list in format Groq expects: system message + history + user message with context injected
    def build_messages(self, query: str, chunks: List[Dict[str, any]], conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Build message format for chat-based LLMs.
        
        Args:
            query: User's question
            chunks: Retrieved context chunks
            conversation_history: Optional conversation history
            
        Returns:
            List of message dictionaries {role, content}
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history (skip if invalid)
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    if msg['role'] in ['user', 'assistant', 'system']:
                        messages.append(msg)
        
        # Build user message with context
        context = self._format_chunks(chunks)
        user_message = f"CONTEXT FROM RESEARCH PAPERS:\n{context}\n\nQUESTION: {query}\n\nPlease answer based only on the context provided above."
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def extract_sources_from_chunks(self, chunks: List[Dict[str, any]]) -> List[Dict[str, str]]:
        """
        Extract source citations from chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of source citations
        """
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            
            source = {
                'source_id': i,
                'title': metadata.get('title', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'filename': metadata.get('filename', 'Unknown')
            }
            sources.append(source)
        
        return sources