from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Model nhanh + rẻ cho RAGAS judge; có thể override qua .env.
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini")
RAGAS_EMBED_MODEL = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-3-small")
# Số async worker chạy song song. Đây là yếu tố quyết định tốc độ:
# RAGAS gọi ~vài LLM call cho mỗi (question × metric), tất cả là I/O-bound nên
# tăng concurrency là cách rút ngắn thời gian hiệu quả nhất (mặc định RAGAS = 16).
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "16"))
# Timeout (giây) cho mỗi HTTP request tới OpenAI — chặn call treo vô hạn.
RAGAS_REQUEST_TIMEOUT = float(os.getenv("RAGAS_REQUEST_TIMEOUT", "30"))

_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   max_workers: int = RAGAS_MAX_WORKERS) -> dict:
    """Run RAGAS evaluation."""
    # RAGAS cần OPENAI_API_KEY và Python 3.11+ → wrap try/except để pipeline
    # không crash khi thiếu key/dependency (trả về zeros để vẫn chạy được).
    zeros = {**{k: 0.0 for k in _METRIC_KEYS}, "per_question": []}

    # Fail-fast: thiếu key thì evaluate() sẽ retry với backoff hàng phút rồi mới
    # fail → check trước để trả zeros ngay, khỏi chờ.
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("  ⚠️  RAGAS skipped: OPENAI_API_KEY chưa được set.")
        return zeros

    try:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception as e:
            print(f"  ⚠️  Failed to apply nest_asyncio: {e}", flush=True)

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        # Pin model nhanh + temperature=0 (deterministic) thay vì để RAGAS tự
        # khởi tạo default → kiểm soát được tốc độ/chi phí.
        # QUAN TRỌNG: đặt timeout NGAY TRÊN client. Trên môi trường này LLM call
        # async có thể treo ở tầng HTTP (asyncio không hủy được → bar kẹt 0/80
        # vĩnh viễn). request_timeout buộc call tự bỏ sau N giây rồi retry/NaN.
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=RAGAS_LLM_MODEL, temperature=0.0,
            timeout=RAGAS_REQUEST_TIMEOUT, max_retries=2))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=RAGAS_EMBED_MODEL,
            timeout=RAGAS_REQUEST_TIMEOUT, max_retries=2))

        # max_workers cao = nhiều LLM call song song → nhanh hơn nhiều. timeout/
        # retry giữ để 1 call treo không kéo cả evaluate() kẹt vô hạn — call quá
        # hạn trả NaN (→ 0.0 ở _val) thay vì block mãi.
        run_config = RunConfig(timeout=90, max_retries=3, max_wait=20,
                               max_workers=max_workers)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, embeddings=embeddings, run_config=run_config)
        df = result.to_pandas()

        def _val(row, key):
            """RAGAS có thể trả NaN khi metric không tính được → coi như 0.0."""
            v = row.get(key, 0.0)
            try:
                v = float(v)
                return v if v == v else 0.0  # v != v → NaN
            except (TypeError, ValueError):
                return 0.0

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_val(row, "faithfulness"),
                answer_relevancy=_val(row, "answer_relevancy"),
                context_precision=_val(row, "context_precision"),
                context_recall=_val(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        n = len(per_question) or 1
        aggregate = {
            k: sum(getattr(r, k) for r in per_question) / n for k in _METRIC_KEYS
        }
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metric_keys = list(diagnostic_tree.keys())

    analyzed = []
    for r in eval_results:
        metrics = {k: getattr(r, k) for k in metric_keys}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda k: metrics[k])
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    # Sort theo avg tăng dần → lấy bottom_n câu tệ nhất.
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")