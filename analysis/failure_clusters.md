# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Trường Giang
**Ngày:** 2026-06-30

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric              | factual | multi_hop | adversarial |
| ------------------- | ------- | --------- | ----------- |
| faithfulness        | 0.9333  | 0.5738    | 0.9000      |
| answer_relevancy    | 0.4995  | 0.4156    | 0.4743      |
| context_precision   | 0.9667  | 0.9750    | 0.9250      |
| context_recall      | 0.8750  | 0.8458    | 0.6500      |
| **avg_score** | 0.8186  | 0.7026    | 0.7373      |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question                                                                                                                                                           | avg_score | worst_metric     |
| ---- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ---------------- |
| 1    | multi_hop    | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA.                                                      | 0.2500    | faithfulness     |
| 2    | multi_hop    | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?                                                      | 0.3333    | faithfulness     |
| 3    | adversarial  | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không?                                                         | 0.3333    | faithfulness     |
| 4    | factual      | Nghỉ phép không lương 20 ngày cần ai phê duyệt?                                                                                                           | 0.3750    | faithfulness     |
| 5    | multi_hop    | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu?                                      | 0.3750    | faithfulness     |
| 6    | multi_hop    | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn 15 ngày). Ai phê duyệt khoản này và phí phạt là bao nhiêu?                     | 0.6052    | faithfulness     |
| 7    | adversarial  | Bao lâu phải đổi mật khẩu một lần?                                                                                                                         | 0.6250    | answer_relevancy |
| 8    | multi_hop    | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố lớn, ở khách sạn 200 USD/đêm (3 đêm). Công ty thanh toán bao nhiêu tiền khách sạn? | 0.6352    | faithfulness     |
| 9    | multi_hop    | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?                                                                              | 0.6541    | faithfulness     |
| 10   | multi_hop    | Nhân viên đi công tác trong nước 2 ngày, ở khách sạn giá 1.500.000 VNĐ/đêm. Công ty thanh toán tối đa bao nhiêu cho tiền khách sạn?         | 0.6622    | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric      | factual | multi_hop | adversarial | Total |
| ----------------- | ------- | --------- | ----------- | ----- |
| faithfulness      | 1       | 8         | 1           | 10    |
| answer_relevancy  | 16      | 11        | 6           | 33    |
| context_precision | 0       | 0         | 1           | 1     |
| context_recall    | 3       | 1         | 2           | 6     |
| **Total**   | 20      | 20        | 10          | 50    |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual & multi_hop
**Dominant metric:** answer_relevancy

**Lý do phân tích:**

- **Answer Relevancy thấp**: RAGAS `answer_relevancy` đánh giá sự tương quan giữa câu trả lời sinh ra và câu hỏi thông qua việc generate ngược lại câu hỏi từ câu trả lời bằng LLM rồi so sánh embedding. Vì dữ liệu đầu vào và đầu ra là tiếng Việt, các Prompt mặc định của RAGAS thường hoạt động kém hiệu quả trên ngôn ngữ phi tiếng Anh, dẫn đến việc sinh câu hỏi ngược bị sai lệch hoặc không sát nghĩa, kéo thấp điểm số một cách thiếu khách quan ngay cả khi câu trả lời thực tế rất chính xác.
- **Faithfulness thấp ở Multi-hop**: Các câu hỏi multi-hop đòi hỏi tổng hợp thông tin từ nhiều nguồn hoặc thực hiện các bước tính toán phức tạp (như tính tiền phạt quá hạn, tính số ngày phép cộng thêm dựa trên thâm niên). Mô hình RAG cơ bản không được hướng dẫn chi tiết hoặc thiếu khả năng suy luận logic/toán học tốt dẫn đến sinh ra các số liệu sai lệch (hallucination) so với ngữ cảnh trích xuất.

---

## 5. Suggested Fixes

| Metric yếu       | Root cause                    | Suggested fix                                                                                                                                                                                                                                                |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| faithfulness      | LLM hallucinating             | Thắt chặt system prompt (system instructions), giảm temperature xuống 0.0 để tăng tính deterministic, yêu cầu LLM trích dẫn nguồn hoặc từ chối trả lời nếu thông tin không có trong ngữ cảnh.                                        |
| context_recall    | Missing relevant chunks       | Cải thiện chiến lược phân mảnh văn bản (chunking) bằng cách sử dụng semantic chunking hoặc overlap lớn hơn, đồng thời kết hợp tìm kiếm kết hợp (Hybrid Search: BM25 + Dense vector).                                                |
| context_precision | Too many irrelevant chunks    | Tích hợp thêm mô hình Reranker (như Cohere hoặc BGE-Reranker) để xếp hạng lại các đoạn văn bản, hoặc áp dụng các bộ lọc siêu dữ liệu (metadata filtering) trước khi đưa vào LLM.                                              |
| answer_relevancy  | Answer doesn't match question | Cải tiến Prompt Template cho LLM sinh câu trả lời ngắn gọn, tập trung thẳng vào trọng tâm câu hỏi, tránh lan man; đồng thời dịch hoặc tùy biến Prompt đánh giá của RAGAS sang tiếng Việt để LLM judge đánh giá đúng hơn. |

---

## 6. Nhận xét về Adversarial Distribution

- **So sánh avg_score:** Điểm số trung bình (`avg_score`) của `adversarial` (0.7373) thấp hơn đáng kể so với `factual` (0.8186), tuy nhiên lại cao hơn `multi_hop` (0.7026). Điều này cho thấy các câu hỏi adversarial (chứa bẫy mâu thuẫn phiên bản, negation trap) thực sự gây khó khăn cho hệ thống hơn là câu hỏi thực tế đơn giản.
- **Ảnh hưởng của Version Conflicts (v2023 vs v2024):** Có sự nhầm lẫn rõ rệt. Khi tài liệu v2023 và v2024 đều được lập chỉ mục, các kỹ thuật tìm kiếm vector thuần túy có xu hướng lấy cả hai vì độ tương đồng ngữ nghĩa cao. Nếu không có cơ chế lọc phiên bản cũ/mới hoặc prompt không chỉ định rõ việc ưu tiên tài liệu có hiệu lực mới nhất, LLM sẽ tổng hợp sai hoặc chọn nhầm phiên bản cũ (ví dụ: nhầm lẫn giữa 12 ngày phép của v2023 và 15 ngày của v2024).
- **Adversarial trong Bottom 10:** Câu hỏi ID 50 (*"Nhân viên Manager có thể dùng VPN cá nhân khi WFH..."*) và ID 44 (*"Bao lâu phải đổi mật khẩu một lần?"*) nằm trong bottom 10. Nguyên nhân là do hệ thống bị rơi vào bẫy phủ định (negation trap) và mâu thuẫn phiên bản mật khẩu (v1.0 yêu cầu 90 ngày, v2.0 yêu cầu 120 ngày) mà không phân biệt được đâu là quy định hiện hành đang có hiệu lực.

