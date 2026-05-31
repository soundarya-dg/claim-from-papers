import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.pipeline import RAGPipeline
from config.settings import GROQ_API_KEY, LLM_MODEL
from evaluation.test_cases import TEST_CASES

# DeepEval imports — optional (gracefully skipped if unavailable or misconfigured)
_deepeval_available = False
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    from groq import Groq as _Groq
    _deepeval_available = True
except Exception:
    pass


# Custom Groq-backed LLM for DeepEval

if _deepeval_available:
    class _GroqEvalLLM(DeepEvalBaseLLM):
        """Wraps Groq so DeepEval metrics can use it as an LLM judge."""

        def __init__(self, model_name: str = LLM_MODEL, api_key: Optional[str] = None):
            self._model_name = model_name
            self._client = _Groq(api_key=api_key or GROQ_API_KEY)

        def load_model(self):
            return self._client

        def generate(self, prompt: str, *args, **kwargs) -> str:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        async def a_generate(self, prompt: str, *args, **kwargs) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self._model_name


# Keyword coverage helper (no external dependencies)

def _keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Return the fraction of expected keywords present in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return round(hits / len(expected_keywords), 4)


# Core evaluator

class Evaluator:
    """Runs the full evaluation suite against the RAG pipeline."""

    def __init__(self):
        self.pipeline = RAGPipeline()
        self._deepeval_llm = None
        self._answer_relevancy_metric = None
        self._faithfulness_metric = None

        if _deepeval_available and GROQ_API_KEY:
            try:
                self._deepeval_llm = _GroqEvalLLM()
                self._answer_relevancy_metric = AnswerRelevancyMetric(
                    threshold=0.5,
                    model=self._deepeval_llm,
                    include_reason=False,
                )
                self._faithfulness_metric = FaithfulnessMetric(
                    threshold=0.5,
                    model=self._deepeval_llm,
                    include_reason=False,
                )
            except Exception:
                self._deepeval_llm = None

    # Single test case

    def run_case(self, test_case: Dict) -> Dict[str, Any]:
        """Run one test case through the pipeline and compute all metrics."""
        question = test_case["question"]
        expected_keywords = test_case.get("expected_keywords", [])

        start = time.time()
        result = self.pipeline.query(question=question, include_sources=True)
        elapsed = round(time.time() - start, 3)

        answer = result.get("answer", "")
        short_answer = result.get("short_answer")
        claims = result.get("claims", [])
        grounding_rate = result.get("grounding_rate", 0.0)
        chunks_retrieved = result.get("chunks_retrieved", 0)
        sources = result.get("sources", [])
        pipeline_error = result.get("error")

        keyword_cov = _keyword_coverage(answer, expected_keywords)

        # DeepEval metrics
        answer_relevancy_score: Optional[float] = None
        faithfulness_score: Optional[float] = None
        deepeval_error: Optional[str] = None

        if self._deepeval_llm is not None and not pipeline_error:
            retrieval_context = [
                chunk.get("text", "")
                for chunk in result.get("retrieved_chunks", [])
                if chunk.get("text")
            ]
            try:
                deval_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=retrieval_context or ["No context retrieved."],
                )
                self._answer_relevancy_metric.measure(deval_case)
                answer_relevancy_score = round(
                    float(self._answer_relevancy_metric.score), 4
                )
            except Exception as exc:
                deepeval_error = str(exc)

            if retrieval_context:
                try:
                    deval_case2 = LLMTestCase(
                        input=question,
                        actual_output=answer,
                        retrieval_context=retrieval_context,
                    )
                    self._faithfulness_metric.measure(deval_case2)
                    faithfulness_score = round(
                        float(self._faithfulness_metric.score), 4
                    )
                except Exception as exc:
                    if not deepeval_error:
                        deepeval_error = str(exc)

        return {
            "id": test_case["id"],
            "level": test_case["level"],
            "category": test_case["category"],
            "question": question,
            "answer": answer,
            "short_answer": short_answer,
            "chunks_retrieved": chunks_retrieved,
            "total_claims": len(claims),
            "grounded_claims": sum(1 for c in claims if c.get("label") == "grounded"),
            "grounding_rate": grounding_rate,
            "keyword_coverage": keyword_cov,
            "answer_relevancy_score": answer_relevancy_score,
            "faithfulness_score": faithfulness_score,
            "sources_count": len(sources),
            "elapsed_seconds": elapsed,
            "pipeline_error": pipeline_error,
            "deepeval_error": deepeval_error,
        }

    # Full suite

    def run_all(self, test_cases: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Run all test cases and return per-case result dicts."""
        cases = test_cases or TEST_CASES
        results = []

        for i, tc in enumerate(cases):
            case_result = self.run_case(tc)
            results.append(case_result)
            if i < len(cases) - 1:
                time.sleep(30)

        return results

    # Summary statistics

    @staticmethod
    def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics over all results."""
        total = len(results)
        if total == 0:
            return {}

        def _avg(key: str) -> Optional[float]:
            vals = [r[key] for r in results if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        levels = ["easy", "medium", "hard"]
        per_level: Dict[str, Any] = {}
        for lvl in levels:
            subset = [r for r in results if r["level"] == lvl]
            if subset:
                gr_vals = [r["grounding_rate"] for r in subset]
                per_level[lvl] = {
                    "count": len(subset),
                    "avg_grounding_rate": round(sum(gr_vals) / len(gr_vals), 4),
                }

        return {
            "total_cases": total,
            "avg_grounding_rate": _avg("grounding_rate"),
            "avg_keyword_coverage": _avg("keyword_coverage"),
            "avg_answer_relevancy": _avg("answer_relevancy_score"),
            "avg_faithfulness": _avg("faithfulness_score"),
            "avg_chunks_retrieved": _avg("chunks_retrieved"),
            "avg_elapsed_seconds": _avg("elapsed_seconds"),
            "per_level": per_level,
        }


# Reporting helpers

def _print_row(r: Dict[str, Any]) -> None:
    gr = f"{r['grounding_rate'] * 100:.0f}%"
    kc = f"{r['keyword_coverage'] * 100:.0f}%"
    ar = f"{r['answer_relevancy_score']:.2f}" if r["answer_relevancy_score"] is not None else "N/A"
    fa = f"{r['faithfulness_score']:.2f}" if r["faithfulness_score"] is not None else "N/A"
    err = "[ERR]" if r.get("pipeline_error") else ""
    print(
        f"  {r['id']:>2}  {r['level']:<7}  {r['category']:<20}  "
        f"grnd={gr:<5}  kw={kc:<5}  rel={ar:<6}  faith={fa:<6}  "
        f"{r['elapsed_seconds']:.1f}s  {err}"
    )


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    separator = "-" * 90
    print(separator)
    print("EVALUATION REPORT")
    print(separator)
    print(
        f"  {'ID':>2}  {'Level':<7}  {'Category':<20}  "
        f"{'Grnd':<10}  {'KW':<10}  {'Relevancy':<10}  {'Faith':<10}  {'Time':<6}"
    )
    print(separator)
    for r in results:
        _print_row(r)
    print(separator)
    print("  SUMMARY")
    print(f"    Total cases        : {summary.get('total_cases', 0)}")
    print(f"    Avg Grounding Rate : {summary.get('avg_grounding_rate', 0) * 100:.1f}%")
    print(f"    Avg KW Coverage    : {summary.get('avg_keyword_coverage', 0) * 100:.1f}%")
    avg_rel = summary.get("avg_answer_relevancy")
    avg_fai = summary.get("avg_faithfulness")
    print(f"    Avg Ans Relevancy  : {f'{avg_rel:.4f}' if avg_rel is not None else 'N/A (DeepEval not configured)'}")
    print(f"    Avg Faithfulness   : {f'{avg_fai:.4f}' if avg_fai is not None else 'N/A (DeepEval not configured)'}")
    print(f"    Avg Elapsed (s)    : {summary.get('avg_elapsed_seconds', 0):.1f}")
    print("  Per Level:")
    for lvl, stats in summary.get("per_level", {}).items():
        print(
            f"    {lvl:<7}: {stats['count']} cases, "
            f"avg grounding rate = {stats['avg_grounding_rate'] * 100:.1f}%"
        )
    print(separator)


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    output_path = Path(__file__).parent / "results.json"
    payload = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


# Entry point
if __name__ == "__main__":
    # Which test cases to run: set to a list of IDs to run only those cases, e.g. [1, 2, 3]
    # Set to None (or []) to run all cases.
    RUN_IDS = [1,3,7,10,14]

    if RUN_IDS:
        cases_to_run = [tc for tc in TEST_CASES if tc["id"] in set(RUN_IDS)]
        print(f"Running {len(cases_to_run)} test case(s): {RUN_IDS}\n")
    else:
        cases_to_run = None  # run_all defaults to TEST_CASES
        print(f"Running all {len(TEST_CASES)} test cases.\n")

    evaluator = Evaluator()

    if not _deepeval_available or not evaluator._deepeval_llm:
        print(
            "Note: DeepEval answer relevancy and faithfulness metrics are unavailable. "
            "Claim Grounding Rate and keyword coverage will still be computed."
        )

    results = evaluator.run_all(test_cases=cases_to_run)
    summary = Evaluator.compute_summary(results)

    print_report(results, summary)
    saved = save_results(results, summary)
    print(f"\nFull results saved to: {saved}")