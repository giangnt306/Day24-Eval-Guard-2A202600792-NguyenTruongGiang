# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Trường Giang
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~13ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~0.02ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Kết quả đo từ Task 12 — measure_p95_latency() chạy trên 20 queries adversarial)*

| Layer                 | P50 (ms) | P95 (ms)        | P99 (ms) | Budget           |
| --------------------- | -------- | --------------- | -------- | ---------------- |
| Presidio PII          | 11.50    | 12.96           | 13.50    | <10ms            |
| NeMo Input Rail       | 0.01     | 0.02            | 0.05     | <300ms           |
| RAG Pipeline          | -        | -               | -        | <2000ms          |
| NeMo Output Rail      | -        | -               | -        | <300ms           |
| **Total Guard** | 11.52    | **12.97** | 13.55    | **<500ms** |

**Budget OK?** [x] Yes / [ ] No
**Comment:** Presidio PII scan hơi vượt budget 10ms một chút (P95 là 12.96ms) do cơ chế Regex kết hợp model tiếng Việt của Presidio. Tuy nhiên, tổng latency của Guard Stack vẫn cực kỳ thấp (~13ms), nằm hoàn toàn trong giới hạn tổng 500ms cho phép.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric                            | Alert Threshold | Action                            |
| --------------------------------- | --------------- | --------------------------------- |
| RAGAS faithfulness (daily sample) | < 0.70          | Page on-call                      |
| Adversarial block rate            | < 80%           | Review new attack patterns        |
| Guard P95 latency                 | > 600ms         | Scale NeMo model / Optimize Regex |
| PII detected count                | spike >10/hour  | Security alert                    |

---

## Kết quả thực tế từ Lab

|                               | Kết quả        |
| ----------------------------- | ---------------- |
| RAGAS avg_score (50q)         | 75.6%            |
| Worst metric                  | answer_relevancy |
| Dominant failure distribution | factual          |
| Cohen's κ                    | 0.1379           |
| Adversarial pass rate         | 20 / 20          |
| Guard P95 latency             | 12.97 ms         |

---

## Nhận xét & Cải tiến

> Hệ thống hoạt động rất tốt đối với việc chặn đứng các cuộc tấn công Adversarial (chặn 20/20 trường hợp với latency rất thấp ~13ms). Tuy nhiên, độ tương đồng ý kiến giữa AI Judge và con người (Cohen's Kappa = 0.1379) còn khá thấp do AI Judge có hiện tượng Verbosity Bias mạnh (luôn ưu tiên các câu trả lời dài hơn từ Ground Truth). Để cải thiện khi deploy lên production, chúng ta cần:
>
> 1. Tối ưu hóa prompt của AI Judge để giảm thiểu Verbosity Bias và hướng dẫn chấm điểm dựa trên độ chính xác ngữ nghĩa thay vì độ dài.
> 2. Cải thiện chunking và prompt của RAG pipeline chính để nâng cao metric `answer_relevancy` ở phân phối `factual`.
> 3. Tối ưu hóa engine Microsoft Presidio bằng cách pre-compile các Regex tiếng Việt nhằm giảm latency P95 xuống dưới 10ms.

