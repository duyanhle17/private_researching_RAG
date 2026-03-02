# Báo cáo công việc ngày 04/03/2026

## 1. Tổng quan công việc
Trong ngày hôm nay, tôi đã thực hiện việc xây dựng và đánh giá hệ thống RAG (Retrieval-Augmented Generation) dựa trên dữ liệu từ bộ dữ liệu **SAT** (Source: `SAT/aligner/data/FB15k-237N`). Mục tiêu chính là kiểm tra khả năng truy vấn kết hợp giữa ngữ nghĩa (Semantic) và đồ thị kiến thức (Knowledge Graph) trên tập câu hỏi chuẩn `qa_eval.json`.

## 2. Quy trình xử lý dữ liệu từ SAT
Hệ thống đã trích xuất dữ liệu từ các file cấu trúc của SAT để xây dựng cơ sở tri thức:
- **Xây dựng Corpus văn bản (Chunks):** Lấy dữ liệu từ `id2text.txt`. Mỗi dòng đại diện cho một mô tả thực thể (entity description), được sử dụng làm đơn vị truy vấn (chunk) chính.
- **Ánh xạ thực thể:** Sử dụng `id2title.txt` để lấy tên thực thể và `mid2id.txt` để ánh xạ mã Freebase sang ID nội bộ.
- **Xây dựng Đồ thị Kiến thức (KG):** Nạp toàn bộ các bộ ba (triplets) từ các file `train.txt`, `valid.txt`, và `test.txt` để tạo danh sách lân cận (adjacency list). Điều này cho phép hệ thống hiểu được mối quan hệ giữa các thực thể (ví dụ: `Entity A` -> `relation` -> `Entity B`).

## 3. Cơ chế Truy vấn (Retrieval Strategy)
Tôi đã triển khai một cơ chế truy vấn lai (Hybrid Search) trong class `SATGraphRAG`:
- **Truy vấn Ngữ nghĩa (Semantic Search):** Sử dụng model `all-MiniLM-L6-v2` để tạo embedding cho các mô tả thực thể và lưu vào index **FAISS**.
- **Truy vấn theo Đồ thị (Graph Search):** 
    - Hệ thống tìm kiếm các thực thể xuất hiện trong câu hỏi (Entity Matching).
    - Áp dụng thuật toán cộng điểm: thực thể khớp trực tiếp nhận điểm cao nhất (1.0), các thực thể lân cận (1-hop neighbors) nhận điểm cộng thêm (0.3).
- **Kết hợp (Reranking):** Kết quả cuối cùng được tính bằng công thức:  
  `Score = alpha * Semantic_Score + (1 - alpha) * Graph_Score`  
  (Trong đó `alpha = 0.7` được sử dụng để ưu tiên ý nghĩa ngữ nghĩa nhưng vẫn giữ trọng số từ cấu trúc đồ thị).

## 4. Phương pháp thử nghiệm mới: Entity-First Retrieval (v2)
Để tối ưu hóa độ chính xác và giảm nhiễu, tôi đã phát triển thêm phương pháp **Entity-First Retrieval** (triển khai trong `run_sat_baseline_v2_with_entities.py`). 

### Ý tưởng cải tiến:
Thay vì tìm kiếm ngữ nghĩa trên toàn bộ 14,541 chunks rồi mới cộng điểm đồ thị, phương pháp này "neo" tri thức vào các thực thể cụ thể xuất hiện trong câu hỏi.

### Quy trình v2:
1.  **Xác định thực thể (Entity Extraction):** Sử dụng thuật toán so khớp tham lam (Greedy Longest Match) để tìm các tên thực thể từ `id2title.txt` có trong câu hỏi.
2.  **Truy xuất trực tiếp (Hard Retrieval):** 
    - Lấy ngay mô tả (chunk) của chính thực thể đó.
    - Lấy thêm các mô tả (chunks) của các thực thể lân cận (1-hop neighbors) trong Knowledge Graph.
3.  **Bổ sung ngữ nghĩa (Semantic Supplement):** Chạy FAISS search để lấy thêm một số lượng nhỏ các chunk liên quan khác để đảm bảo không bỏ sót thông tin nếu việc trích xuất thực thể thất bại.
4.  **Phôi hợp Context:** Gộp các nguồn trên theo thứ tự ưu tiên: *Entity Chunks > Neighbor Chunks > Semantic Chunks*.

### Ưu điểm so với v1:
- **Giảm nhiễu:** Loại bỏ việc LLM đọc các đoạn văn bản có nội dung tương tự nhưng nói về thực thể khác.
- **Tận dụng tối đa đồ thị:** Đảm bảo context luôn chứa các thực thể liên quan trực tiếp về mặt logic (theo cấu trúc FB15k-237N).

## 5. Thực thi và Kết quả (v1 & v2)
- **Dữ liệu đầu vào:** File `qa_eval.json` chứa danh sách các câu hỏi và đáp án tham chiếu.
- **Mô hình ngôn ngữ (LLM):** Sử dụng `moonshotai/kimi-k2-instruct-0905` thông qua NVIDIA API.
- **Kết quả đầu ra:** 
    - v1 lưu tại `sat_baseline_results.json`.
    - v2 lưu tại `sat_baseline_v2_entities_results.json` (bao gồm cả phân tích chi tiết chiến lược truy xuất cho từng câu).

## 6. Kết luận
Việc tích hợp dữ liệu từ SAT vào pipeline GraphRAG giúp hệ thống không chỉ hiểu nội dung văn bản mà còn tận dụng được các mối quan hệ logic giữa các thực thể. Phương pháp v2 (Entity-First) cho thấy tiềm năng lớn trong việc xử lý các câu hỏi yêu cầu độ chính xác cao về thực thể và các mối quan hệ đa tầng trong đồ thị kiến thức.
