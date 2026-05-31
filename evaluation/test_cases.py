from typing import List, Dict

# Each test case:
#   id               - unique integer
#   level            - "easy" | "medium" | "hard"
#   category         - topic area
#   question         - query sent to the RAG pipeline
#   expected_keywords - key terms a correct answer should contain

TEST_CASES: List[Dict] = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "id": 1,
        "level": "easy",
        "category": "definition",
        "question": "What is text classification?",
        "expected_keywords": ["classification", "label", "category", "text", "assign"],
    },
    {
        "id": 2,
        "level": "easy",
        "category": "fine-tuning",
        "question": "What is fine-tuning in the context of text classification?",
        "expected_keywords": ["fine-tuning", "pre-trained", "adapt", "classification", "model"],
    },
    {
        "id": 3,
        "level": "easy",
        "category": "prompting",
        "question": "What is prompt-based text classification?",
        "expected_keywords": ["prompt", "template", "classification", "LLM", "instruction"],
    },
    {
        "id": 4,
        "level": "easy",
        "category": "datasets",
        "question": "What datasets are commonly used in text classification research?",
        "expected_keywords": ["dataset", "benchmark", "corpus", "label", "training"],
    },
    {
        "id": 5,
        "level": "easy",
        "category": "transfer-learning",
        "question": "What is transfer learning and how does it apply to text classification?",
        "expected_keywords": ["transfer learning", "pre-trained", "adapt", "classification", "task"],
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "id": 6,
        "level": "medium",
        "category": "comparison",
        "question": "How does fine-tuning compare to few-shot prompting for text classification?",
        "expected_keywords": ["fine-tuning", "few-shot", "prompting", "compare", "performance"],
    },
    {
        "id": 7,
        "level": "medium",
        "category": "limitations",
        "question": "What are the limitations of LLMs for domain-specific text classification?",
        "expected_keywords": ["limitation", "domain", "specific", "LLM", "classification"],
    },
    {
        "id": 8,
        "level": "medium",
        "category": "imbalanced-data",
        "question": "How do LLMs handle imbalanced datasets in text classification?",
        "expected_keywords": ["imbalanced", "dataset", "class", "LLM", "classification"],
    },
    {
        "id": 9,
        "level": "medium",
        "category": "prompt-sensitivity",
        "question": "How does the choice of prompt template affect text classification accuracy?",
        "expected_keywords": ["prompt", "template", "accuracy", "classification", "sensitivity"],
    },
    {
        "id": 10,
        "level": "medium",
        "category": "hierarchical",
        "question": "How do LLMs perform on hierarchical text classification tasks?",
        "expected_keywords": ["hierarchical", "classification", "label", "LLM", "performance"],
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    {
        "id": 11,
        "level": "hard",
        "category": "fine-tuning-vs-prompting",
        "question": (
            "Do papers agree on whether fine-tuning or prompting is better "
            "for text classification?"
        ),
        "expected_keywords": ["fine-tuning", "prompting", "disagree", "comparison", "classification"],
    },
    {
        "id": 12,
        "level": "hard",
        "category": "low-resource",
        "question": (
            "What conflicting findings exist about LLM performance on "
            "low-resource text classification?"
        ),
        "expected_keywords": ["low-resource", "conflicting", "performance", "LLM", "classification"],
    },
    {
        "id": 13,
        "level": "hard",
        "category": "chain-of-thought",
        "question": (
            "What are the disagreements about using chain-of-thought reasoning "
            "for classification tasks?"
        ),
        "expected_keywords": ["chain-of-thought", "reasoning", "disagreement", "classification", "performance"],
    },
    {
        "id": 14,
        "level": "hard",
        "category": "emergent-abilities",
        "question": (
            "What are the unresolved debates about emergent classification abilities "
            "in large language models?"
        ),
        "expected_keywords": ["emergent", "ability", "debate", "LLM", "classification"],
    },
    {
        "id": 15,
        "level": "hard",
        "category": "label-semantics",
        "question": (
            "What evidence exists for or against LLMs understanding label semantics "
            "in classification tasks?"
        ),
        "expected_keywords": ["label semantics", "understanding", "LLM", "classification", "evidence"],
    },
]


def get_cases_by_level(level: str) -> List[Dict]:
    """Return test cases filtered by difficulty level (easy / medium / hard)."""
    return [tc for tc in TEST_CASES if tc["level"] == level]


def get_all_questions() -> List[str]:
    """Return the question string for every test case."""
    return [tc["question"] for tc in TEST_CASES]