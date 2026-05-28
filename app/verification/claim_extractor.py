import sys
import json
import re
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from groq import Groq
from config.settings import GROQ_API_KEY, LLM_MODEL


_SYSTEM_PROMPT = (
    "You are a precise fact-extraction assistant. "
    "When given a passage of text, extract every individual atomic factual claim it makes. "
    "An atomic claim is a single, self-contained statement that asserts exactly one fact. "
    "Return ONLY a valid JSON array of strings — no explanation, no markdown, no extra text. "
    'Example output: ["Claim one.", "Claim two.", "Claim three."]'
)


class ClaimExtractor:
    """Breaks an LLM-generated answer into a list of atomic factual claims."""

    def __init__(self, api_key: Optional[str] = None, model: str = LLM_MODEL):
        api_key = api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY in .env file.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def extract(self, answer: str) -> List[str]:
        """
        Extract atomic claims from an LLM answer.

        Args:
            answer: The answer text to decompose.

        Returns:
            List of atomic claim strings.
        """
        if not answer or not answer.strip():
            return []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extract all atomic claims from the following text:\n\n{answer}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()

        # Attempt direct JSON parse
        try:
            claims = json.loads(raw)
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        except json.JSONDecodeError:
            pass

        # Attempt to extract embedded JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                claims = json.loads(match.group())
                if isinstance(claims, list):
                    return [str(c).strip() for c in claims if str(c).strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: treat each non-empty line as a claim
        lines = [
            line.strip().lstrip("-*•123456789. ").strip()
            for line in raw.splitlines()
        ]
        return [l for l in lines if l]
