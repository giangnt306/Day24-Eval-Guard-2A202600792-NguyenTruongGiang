# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Trường GIang
**Ngày:** 2026-06-30
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

| #  | Question (tóm tắt)                               | Winner | Reasoning tóm tắt                                                          |
| -- | -------------------------------------------------- | ------ | ---------------------------------------------------------------------------- |
| 1  | Số ngày nghỉ khi kết hôn                      | B      | B đầy đủ hơn khi nói rõ "không trừ vào phép năm".                |
| 5  | Phê duyệt mua thiết bị 55 triệu               | B      | B chính xác khi chỉ ra CEO phê duyệt do trên 50 triệu.                |
| 12 | Thưởng Tết tối thiểu nhân viên chính thức | B      | B chi tiết hơn về quy định cho nhân viên dưới 6 tháng.             |
| 21 | Senior 9 năm: phép năm & khoảng lương        | B      | B chi tiết hóa cách tính phép năm và cấp bậc lương.               |
| 23 | Hoàn trả phí đào tạo khi nghỉ trước hạn  | B      | B giải thích rõ cam kết 1 năm và mức hoàn trả 100%.                 |
| 29 | Tạm ứng 8 triệu quá hạn 30 ngày              | B      | B tính đúng phí phạt quá hạn 15 ngày pro-rata.                       |
| 33 | Manager 12 năm thâm niên: phụ cấp & phép     | B      | B giải thích cụ thể cách tính phụ cấp và phép thâm niên.         |
| 41 | Số ngày nghỉ phép năm mặc định             | B      | B chỉ ra chính sách v2024 (15 ngày) thay thế v2023 (12 ngày).          |
| 46 | Thử việc có phép năm không                   | tie    | Cả hai đều đúng; B có giải thích kỹ thuật thừa về negation trap. |
| 50 | WFH có dùng VPN cá nhân không                 | B      | B chính xác khi chỉ ra VPN cá nhân bị cấm theo v1.3.                  |

---

## 2. Swap-and-Average Results

| #  | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
| -- | ------------- | ------------- | ----- | -------------------- |
| 1  | B             | B             | B     | True                 |
| 5  | B             | B             | B     | True                 |
| 12 | B             | B             | B     | True                 |
| 21 | B             | B             | B     | True                 |
| 23 | B             | B             | B     | True                 |
| 29 | B             | B             | B     | True                 |
| 33 | B             | B             | B     | True                 |
| 41 | B             | B             | B     | True                 |
| 46 | A             | B             | tie   | False                |
| 50 | B             | B             | B     | True                 |

**Position bias rate:** 10.0% (1/10 cases)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)
**Judge labels:** [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

| Question ID | Human Label | Judge Label | Agree? |
| ----------- | ----------- | ----------- | ------ |
| 1           | 1           | 0           | No     |
| 5           | 0           | 0           | Yes    |
| 12          | 1           | 0           | No     |
| 21          | 1           | 0           | No     |
| 23          | 1           | 0           | No     |
| 29          | 0           | 0           | Yes    |
| 33          | 1           | 0           | No     |
| 41          | 0           | 0           | Yes    |
| 46          | 1           | 1           | Yes    |
| 50          | 0           | 0           | Yes    |

**Cohen's κ:** 0.138
**Interpretation:** slight agreement (đồng thuận thấp)

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):

- A thắng + A dài hơn B: 0 / 9 cases
- B thắng + B dài hơn A: 9 / 9 cases
- **Verbosity bias rate:** 100.0%

**Kết luận:**
LLM Judge (gpt-4o-mini) thể hiện xu hướng ưu ái cực đoan đối với các câu trả lời dài hơn và đầy đủ hơn (ở đây là B - Ground Truth). Điều này xảy ra do mô hình đánh giá cao độ đầy đủ (completeness) và độ chi tiết của thông tin so với tính súc tích (conciseness). Trong các bài kiểm tra thực tế, đây là một vấn đề nghiêm trọng vì LLM judge có thể đánh giá cao các câu trả lời lan man, lặp đi lặp lại hoặc "nhồi nhét" từ khóa nhưng lại hạ thấp các câu trả lời ngắn gọn, trực diện và chính xác của con người.

---

## 5. Nhận xét chung

- **Hệ số κ (0.138)** nằm ở mức đồng thuận thấp (slight agreement). Nguyên nhân chính là vì LLM Judge so sánh trực tiếp model_answer với Ground Truth, từ đó đánh giá khắt khe và chấm 0 (thông qua việc gán B thắng) cho các câu trả lời thiếu một vài chi tiết nhỏ (ví dụ QID 1, 12, 21), trong khi người đánh giá (human judge) vẫn chấp nhận chấm 1 (đạt yêu cầu) cho các câu trả lời này vì chúng đã cung cấp thông tin cốt lõi chính xác. Do đó, trong môi trường thực tế, LLM judge cần prompt mềm dẻo hơn hoặc ngưỡng chấm điểm linh hoạt hơn.
- **Position bias** ở mức 10.0%, khá thấp và không đáng lo ngại. Điều này chứng tỏ `gpt-4o-mini` khá ổn định về mặt vị trí trên tác vụ này.
- **Swap-and-average** thực sự giúp ích trong việc lọc ra các trường hợp không nhất quán về mặt vị trí (như QID 46), biến chúng thành kết quả "tie" để tránh đưa ra kết luận sai lệch từ một lượt chạy duy nhất.
- Trong môi trường production, nên sử dụng LLM Judge dưới dạng thang điểm chi tiết (Likert scale 1-5) kèm theo rubric rõ ràng cho từng tiêu chí, thay vì so sánh pairwise trực tiếp, hoặc kết hợp swap-and-average kết hợp chấm điểm độc lập để giảm thiểu tối đa verbosity bias và position bias.

