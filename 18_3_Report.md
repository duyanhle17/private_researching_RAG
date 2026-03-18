# Báo Cáo Tiến Độ Dự Án FactKG (Cập nhật ngày 18/03)

## 1. Mục Đích Cốt Lõi Của Bài Báo (Core Goal)
Bài báo giới thiệu **FACTKG**, một bộ dữ liệu chuẩn (benchmark) quy mô lớn (108k claims) nhằm đánh giá khả năng kiểm chứng sự thật của AI thông qua suy luận trực tiếp trên **Knowledge Graph (KG)**.
*   **Giải quyết lỗ hổng**: Thay thế các nghiên cứu cũ chỉ dùng văn bản thô hoặc chỉ suy luận 1-hop đơn giản.
*   **Thao trường logic**: Cung cấp 5 kiểu suy luận phức tạp để huấn luyện AI cách ánh xạ ngôn ngữ tự nhiên vào cấu trúc đồ thị (node/edge).
*   **Tính giải thích (Explainability)**: Sử dụng KG cho phép AI chỉ ra chính xác chuỗi logic (đường đi trên đồ thị) để kết luận Đúng/Sai thay vì chỉ đưa ra kết quả từ "hộp đen".

---

## 2. Phân Tích Hai Nhánh Tiếp Cận (Methodology)

Trong dự án FactKG, tác giả chia làm hai nhánh thí nghiệm chính với mục tiêu hoàn toàn khác nhau:

### 2.1 Nhánh "Claim Only" (Chỉ dùng Văn bản)
*   **Đối tượng**: BERT, BlueBERT (được fine-tune), Flan-T5 (chạy zero-shot).
*   **Bản chất**: Đây là bài toán **Phân loại nhị phân (Binary Classification)**. Mô hình học cách ánh xạ trực tiếp từ câu nhận định sang nhãn `SUPPORTED` (Đúng) hoặc `REFUTED` (Sai).
*   **Đặc điểm**: **Không truy xuất KG**. Mô hình chỉ dựa vào kiến thức có sẵn trong trọng số (pretrained) và các pattern học được từ tập train. Việc huấn luyện sinh ra Loss/Accuracy là quá trình mô hình "học thuộc" các đặc điểm văn bản của sự thật.

### 2.2 Nhánh "With Graphical Evidence" (Dùng Bằng chứng Đồ thị)
*   **Đối tượng**: Mô hình **GEAR** (Graph Evidence Aware Reasoning).
*   **Bản chất**: Đây là một hệ thống phức tạp chia làm nhiều bài toán con:
    1.  **Retriever**: Dự đoán quan hệ (Relation) và số bước (Hop) cần tìm trên KG.
    2.  **Verification**: Kết hợp câu nhận định với các "đường dẫn bằng chứng" (evidence paths) tìm được để đưa ra phán quyết.
*   **Sự khác biệt**: GEAR không phải là BERT thêm dữ liệu, mà là một quy trình **Graph-RAG** thực thụ (Truy xuất -> Suy luận -> Kết luận).

---

## 3. Thống Kê Dữ Liệu & Kết Quả Thực Nghiệm

### 3.1 Dataset Statistics
*   **Tổng số câu**: 108,674 (Train: 86k, Dev: 13k, Test: 9k).
*   **Phong cách**: Kết hợp cả văn nói (Colloquial) và văn viết (Written).

### 3.2 Bảng So Sánh Accuracy (%)

| Mô hình | One-hop | Conjunction | Existence | Multi-hop | Negation | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT (Paper)** | 69.64 | 63.31 | 61.84 | 70.06 | 63.62 | **65.20** |
| **BERT (M1 Pro)** | 55.43 | 44.25 | 47.59 | 51.33 | 45.89 | **48.65** |
| **GEAR (Paper)** | 83.23 | 77.68 | 81.61 | 68.84 | 79.41 | **77.65** |

**Phân tích**: 
*   Kết quả thực tế 48.65% của BERT trên máy cá nhân cho thấy mô hình khi "đọc chay" gần như chỉ đoán mò. 
*   GEAR vượt trội ở hầu hết các hạng mục (đặc biệt là Negation tăng +15%) nhờ có bằng chứng từ KG cứu vãn những chỗ logic văn bản bị rối.

---

## 4. Ý Nghĩa Đối Với Hệ Thống GraphRAG
FACTKG không chỉ là bài toán phân loại, mà còn là một **bộ dữ liệu huấn luyện lý tưởng cho GraphRAG**:
*   Cung cấp các **đường dẫn đồ thị chuẩn (ground-truth paths)** đi kèm mỗi câu hỏi.
*   Giúp đo lường chính xác khả năng trích xuất đồ thị con (subgraph extraction) và suy luận đa chặng (multi-hop) của các hệ thống RAG hiện đại.

---
*Người thực hiện: Duy Anh Le*
*Ngày báo cáo: 18/03/2026*
