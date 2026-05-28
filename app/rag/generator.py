import sys
from pathlib import Path
from typing import List, Dict, Optional
from groq import Groq

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import GROQ_API_KEY, LLM_MODEL


# Generates answers using Groq LLM
class Generator:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize the generator.
        
        Args:
            api_key: Groq API key (uses env var if None)
            model: Model name
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or GROQ_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in .env file"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate answer from a prompt string.
        
        Args:
            prompt: Complete prompt text
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated answer string
        """
        try:
            # Convert prompt to messages format
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=False
            )
            
            answer = response.choices[0].message.content
            return answer.strip()
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    

    def generate_from_messages(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, any]:
        """
        Generate answer from messages format.
        
        Args:
            messages: List of {role, content} dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=False
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract metadata
            return {
                'answer': answer,
                'model': self.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    

    def generate_streaming(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Generate answer with streaming.
        
        Args:
            messages: List of {role, content} dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Chunks of the generated answer
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"Error in streaming generation: {str(e)}")