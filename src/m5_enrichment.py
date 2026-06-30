from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from functools import lru_cache
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Model rẻ + nhanh cho enrichment; override được qua .env.
ENRICH_MODEL = os.getenv("ENRICH_MODEL", "gpt-4o-mini")
# Số chunk enrich song song. Enrichment là I/O-bound (chờ API) nên chạy
# nhiều chunk cùng lúc rút ngắn thời gian gần như tuyến tính.
ENRICH_MAX_WORKERS = int(os.getenv("ENRICH_MAX_WORKERS", "8"))


@lru_cache(maxsize=1)
def _get_client():
    """OpenAI client dùng chung (tái sử dụng connection pool, tránh tạo lại
    mỗi call). lru_cache → khởi tạo 1 lần duy nhất cho cả module."""
    from openai import OpenAI
    return OpenAI()


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API): lấy 2 câu đầu.
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = (resp.choices[0].message.content or "").strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in raw if q.strip()][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback: biến các câu dài thành câu hỏi.
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            context = (resp.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Simple fallback: prepend tên tài liệu (vẫn giữ nguyên text gốc).
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                # JSON mode → đảm bảo output parse được, không cần regex dọn dẹp.
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150,
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=400,
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def _enrich_one(chunk: dict, methods: list[str], use_combined: bool) -> EnrichedChunk:
    """Enrich 1 chunk. Tách riêng để chạy song song được trong enrich_chunks."""
    text = chunk["text"]
    source = chunk.get("metadata", {}).get("source", "")

    if use_combined:
        result = _enrich_single_call(text, source)
        summary = result.get("summary", "")
        questions = result.get("questions", [])
        context_line = result.get("context", "")
        enriched_text = f"{context_line}\n\n{text}" if context_line else text
        auto_meta = result.get("metadata", {})
    else:
        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}

    return EnrichedChunk(
        original_text=text,
        enriched_text=enriched_text,
        summary=summary,
        hypothesis_questions=questions,
        auto_metadata={**chunk.get("metadata", {}), **auto_meta},
        method="+".join(methods),
    )


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    max_workers: int = ENRICH_MAX_WORKERS,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks (song song).

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
        max_workers: Số chunk enrich song song (I/O-bound → tăng = nhanh hơn).
    """
    if methods is None:
        methods = ["combined"]
    if not chunks:
        return []

    use_combined = "combined" in methods

    # Mỗi chunk = vài API call độc lập nhau → chạy song song qua thread pool.
    # Giữ nguyên thứ tự đầu vào bằng executor.map (results theo đúng thứ tự submit).
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(max_workers, len(chunks)))
    enriched: list[EnrichedChunk] = []
    done = 0
    total = len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ec in ex.map(lambda c: _enrich_one(c, methods, use_combined), chunks):
            enriched.append(ec)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Enriched {done}/{total} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
