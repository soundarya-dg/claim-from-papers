import sys
import time
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
        messages = [{"role": "user", "content": prompt}]
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        last_exc: Exception = Exception("Unknown error")
        for attempt in range(4):  # up to 4 attempts with backoff
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    top_p=1,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if "tokens per day" in err_str:
                    break  # daily limit hit, stop immediately
                wait = 15 * (attempt + 1)
                print(f"[generate] Error: {err_str[:120]} — waiting {wait}s before retry {attempt + 1}/4...")
                time.sleep(wait)
        raise Exception(f"Error generating answer: {str(last_exc)}")
    

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
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        last_exc: Exception = Exception("Unknown error")
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    top_p=1,
                    stream=False
                )
                answer = response.choices[0].message.content.strip()
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
                last_exc = e
                err_str = str(e)
                if "tokens per day" in err_str:
                    break  # daily limit hit, stop immediately
                wait = 15 * (attempt + 1)
                print(f"[generate_from_messages] Error: {err_str[:120]} — waiting {wait}s before retry {attempt + 1}/4...")
                time.sleep(wait)
        raise Exception(f"Error generating answer: {str(last_exc)}")
    

    def simplify_answer(self, question: str, full_answer: str) -> Optional[str]:
        """
        Produce a short, plain-English 1-3 sentence summary of the full answer.

        Args:
            question: The original user question.
            full_answer: The detailed, source-cited answer from the RAG pipeline.

        Returns:
            A concise plain-English string, or None if generation fails.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that gives short, clear, plain-English answers. "
                    "Given a question and a detailed research-based answer, write a concise "
                    "1-3 sentence summary that directly answers the question in simple language. "
                    "Do not mention sources, page numbers, or citations. "
                    "Do not start with phrases like 'Based on the context' or 'According to'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Detailed answer:\n{full_answer}\n\n"
                    "Write a short, plain-English summary answer:"
                ),
            },
        ]
        last_exc: Exception = Exception("Unknown error")
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=256,
                    top_p=1,
                    stream=False,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if "tokens per day" in err_str:
                    break  # daily limit hit, stop immediately
                wait = 15 * (attempt + 1)
                print(f"[simplify_answer] Error: {err_str[:120]} — waiting {wait}s before retry {attempt + 1}/4...")
                time.sleep(wait)
        return None

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
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                top_p=1,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"Error in streaming generation: {str(e)}")