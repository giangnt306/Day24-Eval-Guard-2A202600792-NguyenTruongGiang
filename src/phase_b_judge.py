from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí:
1. Độ chính xác (accuracy): có khớp với thực tế chính sách không?
2. Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
3. Tính súc tích (conciseness): có thừa / thiếu thông tin không?

Hãy chọn câu trả lời tốt hơn ("A" hoặc "B"), hoặc trả về "tie" nếu cả hai ngang nhau hoặc đều không tốt.
Trả về định dạng JSON (chỉ trả về JSON, không chứa markdown hay text khác ngoài JSON):
{{
  "winner": "A",
  "reasoning": "giải thích ngắn gọn lý do",
  "scores": {{"A": 0.9, "B": 0.5}}
}}
'''

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        
        winner = data.get("winner", "tie")
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        reasoning = data.get("reasoning", "")
        scores = data.get("scores", {"A": 0.0, "B": 0.0})
        if not isinstance(scores, dict):
            scores = {"A": 0.0, "B": 0.0}
        scores["A"] = float(scores.get("A", 0.0))
        scores["B"] = float(scores.get("B", 0.0))
        
        return {
            "winner": winner,
            "reasoning": reasoning,
            "scores": scores
        }
    except Exception as e:
        print(f"  ⚠️  Pairwise judge failed: {e}")
        return {
            "winner": "tie",
            "reasoning": f"Error: {e}",
            "scores": {"A": 0.0, "B": 0.0}
        }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    scores_pass2 = {
        "A": pass2_raw["scores"].get("B", 0.0),
        "B": pass2_raw["scores"].get("A", 0.0)
    }

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n

    j1 = judge_labels.count(1)
    j0 = judge_labels.count(0)
    h1 = human_labels.count(1)
    h0 = human_labels.count(0)

    p_e = ((j1 / n) * (h1 / n)) + ((j0 / n) * (h0 / n))

    if abs(p_e - 1.0) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0

    κ = (p_o - p_e) / (1 - p_e)
    return κ


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Không có dữ liệu đánh giá."
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner in ("A", "B"))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive
        },
        "interpretation": interpretation,
    }



# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.phase_a_ragas import load_test_set_50q
    test_set = load_test_set_50q()
    gt_map = {item["id"]: item["ground_truth"] for item in test_set}

    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Human labels loaded: {len(human_labels)} questions")

    pairwise_results = []
    judge_labels = []
    
    for item in human_data:
        qid = item["question_id"]
        q = item["question"]
        model_ans = item["model_answer"]
        gt = gt_map[qid]
        
        print(f"Evaluating QID {qid}...")
        res = swap_and_average(q, model_ans, gt)
        pairwise_results.append(res)
        
        # A is model answer, B is ground truth.
        # If A wins or tie, judge_label = 1. If B wins, judge_label = 0.
        j_label = 1 if res.final_winner in ("A", "tie") else 0
        judge_labels.append(j_label)
        
        print(f"  Final winner: {res.final_winner} (Judge label: {j_label}, Human: {item['human_label']})")

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ agreement: {kappa:.3f}")

    bias = bias_report(pairwise_results)
    print(f"\nBias report: {bias}")

    # Save report
    report = {
        "pairwise_runs": [
            {
                "question_id": h["question_id"],
                "question": h["question"],
                "model_answer": h["model_answer"],
                "ground_truth": gt_map[h["question_id"]],
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "position_consistent": r.position_consistent,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
                "human_label": h["human_label"],
                "judge_label": j
            }
            for h, r, j in zip(human_data, pairwise_results, judge_labels)
        ],
        "cohen_kappa": kappa,
        "bias_report": bias
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Saved reports/judge_results.json")

