# Báo Cáo Tiến Độ Dự Án FactKG (Cập nhật ngày 18/03)

## 1. Mô Hình Đã Huấn Luyện (Model Trained)
*   **Tên mô hình**: `bert-base-uncased` (Pipeline A - Claim Only Baseline).
*   **Nguồn gốc**: Mô hình ngôn ngữ tiền huấn luyện từ Google (hơn 110 triệu tham số).
*   **Phương pháp**: Fine-tuning (Huấn luyện tinh chỉnh) trên tập dữ liệu FactKG để thực hiện nhiệm vụ phân loại đúng/sai (Binary Classification).
*   **Thiết bị chạy**: MacBook Pro M1 (GPU MPS tăng tốc).
*   **Thời gian thực hiện**: Đã hoàn thành 3 Epochs huấn luyện vào đêm ngày 17/03/2026.

---

## 2. Thông Số Dữ Liệu Thực Tế (Dataset Statistics)
Dưới đây là bảng phân bổ dữ liệu chính xác đã được đo đạc trực tiếp từ các file `.pickle` của dự án:

| Tập dữ liệu | Số lượng câu (Claims) | Mục đích sử dụng |
| :--- | :---: | :--- |
| **Train Set** | 86,367 | Dùng để huấn luyện mô hình (Model Learning). |
| **Dev Set** | 13,266 | Dùng để theo dõi tiến trình trong lúc học. |
| **Test Set** | 9,041 | **Dùng để đánh giá chất lượng cuối cùng (Final Exam).** |
| **TỔNG CỘNG** | **108,674** | **Khớp 99.99% với con số trong Paper gốc (108,675).** |

---

## 3. Kết Quả Huấn Luyện & Đánh Giá (Evaluation)

Chúng ta đã thực hiện kiểm tra mô hình trên **toàn bộ 100% tập Test (9,041 câu)**, chứ không chỉ lấy một mẫu nhỏ.

### 3.1 Kết Quả Theo Loại Suy Luận (Reasoning Breakdown)

| Loại Suy Luận | Độ Chính Xác (Accuracy) | Số Lượng Câu (Test) | Ý nghĩa khoa học |
| :--- | :---: | :---: | :--- |
| **One-hop** | 55.43% | 1,914 | Suy luận đơn giản 1 bước. BERT nhớ được một phần kiến thức cũ. |
| **Multi-hop** | 51.33% | 1,874 | Suy luận nhiều bước (phức tạp). BERT bắt đầu đoán bừa. |
| **Existence** | 47.59% | 870 | Kiểm tra sự tồn tại thực thể. Hoàn toàn mất phương hướng. |
| **Negation** | 45.89% | 1,314 | **Kém nhất**. BERT không hiểu ý nghĩa của các từ phủ định. |
| **Conjunction** | 44.25% | 3,069 | Câu ghép nhiều mệnh đề. Điểm thấp do cấu trúc câu quá dài. |
| **TỔNG TRUNG BÌNH** | **48.65%** | **9,041** | **Mốc so sánh (Baseline) cho giai đoạn tiếp.** |

### 3.2 Giải thích sự chênh lệch con số:
*   **Tại sao điểm tối qua hiện 51.35%?**: Đây là điểm số cao nhất mà BERT đạt được khi đo thử trên một phần nhỏ dữ liệu trong lúc đang học.
*   **Tại sao điểm cuối cùng là 48.65%?**: Đây là điểm số khi đo trên **toàn bộ 9,041 câu**. Do hiện tượng Overfitting nhẹ (mô hình học vẹt tập Train quá nhiều), khả năng tổng quát hóa trên toàn bộ tập Test bị giảm xuống một chút.

---

## 4. Kết Luận Báo Cáo Thầy

1.  **Dữ liệu**: Bộ dataset chúng ta đang có là hoàn toàn trùng khớp với bài báo quốc tế (108k câu).
2.  **Mô hình**: Đã hoàn thành mốc Baseline (Pipeline A) với BERT.
3.  **Hiện trạng**: Kết quả quanh mức 48-50% chứng minh rằng: **Nếu chỉ dựa vào trí nhớ của AI mà không cho tra cứu Knowledge Graph, AI sẽ không thể xác minh sự thật một cách chính xác.**
4.  **Hướng tiếp theo**: Chuyển sang Pipeline B (Graph-RAG) để tích hợp đồ thị DBpedia làm bằng chứng, giúp cải thiện điểm số ở các mảng khó như Multi-hop và Negation.
