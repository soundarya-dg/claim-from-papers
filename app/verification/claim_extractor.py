import sys
import json
import re
import time
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

        last_exc: Exception = Exception("Unknown error")
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=1024,
                    reasoning_effort="low",
                )
                raw = response.choices[0].message.content.strip()
                print(f"[claim_extractor] Received raw response ({len(raw)} chars): {raw[:200]}...")
                break
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if "tokens per day" in err_str:
                    print(f"[claim_extractor] Daily token limit reached")
                    return []  # daily limit hit, stop immediately
                wait = 15 * (attempt + 1)
                print(f"[claim_extractor] Error: {err_str[:120]} — waiting {wait}s before retry {attempt + 1}/4...")
                time.sleep(wait)
        else:
            print(f"[claim_extractor] All retries failed, returning empty list")
            return []

        # Attempt direct JSON parse
        try:
            claims = json.loads(raw)
            if isinstance(claims, list):
                result = [str(c).strip() for c in claims if str(c).strip()]
                print(f"[claim_extractor] Successfully parsed JSON, extracted {len(result)} claims")
                return result
        except json.JSONDecodeError as e:
            print(f"[claim_extractor] Direct JSON parse failed: {e}")

        # Attempt to extract embedded JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                claims = json.loads(match.group())
                if isinstance(claims, list):
                    result = [str(c).strip() for c in claims if str(c).strip()]
                    print(f"[claim_extractor] Extracted embedded JSON array, {len(result)} claims")
                    return result
            except json.JSONDecodeError as e:
                print(f"[claim_extractor] Embedded JSON parse failed: {e}")

        # Last resort: treat each non-empty line as a claim
        lines = [
            line.strip().lstrip("-*•123456789. ").strip()
            for line in raw.splitlines()
        ]
        result = [l for l in lines if l]
        print(f"[claim_extractor] Fallback line parsing, extracted {len(result)} claims")
        return result
