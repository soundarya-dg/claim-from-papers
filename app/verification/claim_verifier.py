import sys
from pathlib import Path
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from groq import Groq
from config.settings import GROQ_API_KEY, LLM_MODEL, CONFIDENCE_THRESHOLD
from app.rag.retriever import Retriever


_VERDICT_SYSTEM_PROMPT = (
    "You are a strict fact-checking assistant. "
    "Given a claim and a reference passage, determine whether the passage SUPPORTS or CONTRADICTS the claim. "
    "Reply with exactly one word: either 'supported' or 'contradicted'. No other output."
)

# Label constants
GROUNDED = "grounded"
UNVERIFIED = "unverified"
CONTRADICTED = "contradicted"


class ClaimVerifier:
    """
    Verifies each claim against ChromaDB independently.

    Labels:
        grounded     - a relevant chunk exists and supports the claim
        contradicted - a relevant chunk exists but contradicts the claim
        unverified   - no chunk with sufficient similarity was found
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        api_key = api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY in .env file.")
        self.retriever = retriever or Retriever(top_k=3)
        self.client = Groq(api_key=api_key)
        self.model = model
        self.confidence_threshold = confidence_threshold


    # Internal helpers

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB distance to a cosine similarity score in [0, 1].

        ChromaDB's default metric is squared L2. For unit-normalized embeddings
        (normalize_embeddings=True in SentenceTransformers):
            squared_L2 = 2 * (1 - cosine_similarity)
            cosine_similarity = 1 - squared_L2 / 2
        """
        return max(0.0, min(1.0, 1.0 - distance / 2.0))

    def _check_support(self, claim: str, chunk_text: str) -> str:
        """
        Ask the LLM whether the chunk supports or contradicts the claim.

        Returns 'supported' or 'contradicted'.
        """
        messages = [
            {"role": "system", "content": _VERDICT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: {claim}\n\nPassage: {chunk_text}"},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict


    # Public API

    def verify_claim(self, claim: str) -> Dict:
        """
        Verify a single claim against ChromaDB.

        Args:
            claim: Atomic claim string to verify.

        Returns:
            Dict with keys: claim, label, confidence, supporting_chunk.
        """
        chunks = self.retriever.retrieve(query=claim, top_k=3) # retrieves top 3 chunks for the claim

        if not chunks:
            return {
                "claim": claim,
                "label": UNVERIFIED,
                "confidence": 0.0,
                "supporting_chunk": None,
            }

        top_chunk = chunks[0]
        distance = top_chunk.get("distance") or 1.0
        similarity = self._distance_to_similarity(distance)

        if similarity >= self.confidence_threshold:
            verdict = self._check_support(claim, top_chunk["text"])
            label = GROUNDED if verdict == "supported" else CONTRADICTED
            supporting_chunk = top_chunk["text"]
        else:
            label = UNVERIFIED
            supporting_chunk = None

        return {
            "claim": claim,
            "label": label,
            "confidence": round(similarity, 4),
            "supporting_chunk": supporting_chunk,
        }

    def verify_claims(self, claims: List[str]) -> List[Dict]:
        """
        Verify a list of claims.

        Args:
            claims: List of atomic claim strings.

        Returns:
            List of verification result dicts (one per claim).
        """
        return [self.verify_claim(claim) for claim in claims]

    def compute_grounding_rate(self, verification_results: List[Dict]) -> float:
        """
        Compute the proportion of claims labelled as grounded.

        Args:
            verification_results: Output of verify_claims().

        Returns:
            Float between 0.0 and 1.0.
        """
        if not verification_results:
            return 0.0
        grounded_count = sum(
            1 for r in verification_results if r["label"] == GROUNDED
        )
        return round(grounded_count / len(verification_results), 4)