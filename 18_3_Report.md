# Báo Cáo Tổng Hợp: Tài Liệu Paper & Tiến Độ Dự Án FactKG (Cập nhật 18/03/2026)

## PHẦN I: PHÂN TÍCH CHI TIẾT PAPER (FACTKG)

### 1. Giới Thiệu & Động Lực (Overview & Motivation)
**FACTKG: Fact Verification via Reasoning on Knowledge Graphs** là công trình của nhóm nghiên cứu KAIST & Amazon (2023).
*   **Vấn đề**: Các hệ thống AI truyền thống (LLMs) thường bị "ảo giác" hoặc thiếu tính giải thích khi kiểm chứng sự thật. Nghiên cứu trước đây chủ yếu dùng văn bản thô hoặc KG cho suy luận rất đơn giản (1-hop).
*   **Giải pháp**: Xây dựng một benchmark quy mô lớn (108k claims), sử dụng DBpedia làm nguồn bằng chứng cứng để tạo ra một "thao trường" huấn luyện AI cách ánh xạ ngôn ngữ tự nhiên vào cấu trúc node/edge trên đồ thị.
*   **Tính giải thích (Explainability)**: Giúp AI không còn là "hộp đen" bằng cách chỉ ra chính xác chuỗi logic (đường đi trên đồ thị) để kết luận Đúng/Sai.

### 2. Quy Trình Tạo Dữ Liệu (Data Generation Pipeline)
Tác giả đã xây dựng bộ dữ liệu 108,674 claims một cách cực kỳ hệ thống, xuất phát từ các cặp graph-text của tập **WebNLG 2020** và sử dụng tri thức từ DBpedia.

**Sơ đồ luồng tạo dữ liệu (Input -> Processing -> Output):**
`Graph-Text gốc (WebNLG)` ➜ `Sinh Claim theo 5 Reasoning` ➜ `Tạo nhãn Positive/Negative` ➜ `Style Transfer & Lọc chất lượng` ➜ `Chia Train/Dev/Test`

**Chi tiết các bước:**
1.  **Sinh Claim (Positive - SUPPORTED)**:
    *   *One-hop & Conjunction*: Lấy các câu tương ứng với 1 hoặc nhiều triple.
    *   *Existence*: Rút gọn triple thành template (Ví dụ: "{Head} had a {Relation}").
    *   *Multi-hop*: Khuyết danh một thực thể trung gian bằng Type Name (VD: thay "Meyer Werft" thành "a company in Papenburg").
2.  **Tạo Claim Phủ Định (Negative - REFUTED)**:
    *   *Entity Substitution*: Thay thực thể bằng một thực thể cùng loại nhưng nằm ngoài 4-hop trên DBpedia, dùng kiểm tra NLI 2 chiều để đảm bảo thực sự mâu thuẫn.
        *   *(**NLI - Natural Language Inference**: Là bài toán AI chuyên xác định xem 2 câu có mâu thuẫn (Contradiction), đồng nghĩa (Entailment) hay không liên quan. Việc dùng NLI 2 chiều giúp tác giả chắc chắn 100% câu vừa tạo ra thực sự mang nghĩa "Sai" so với câu gốc).*
    *   *Relation Substitution*: Đánh tráo quan hệ bằng một quan hệ khác có cùng cấu trúc Head/Tail.
    *   *Negation*: Sinh câu phủ định (thêm "not") kết hợp mô hình GPT-J cho các câu phức tạp, nhãn được gán chặt chẽ theo logic của đường đi trên đồ thị.
        *   *(**GPT-J**: Là một Mô hình Ngôn ngữ Lớn - LLM mã nguồn mở rất mạnh của EleutherAI với 6.4 tỷ tham số. Tác giả dùng GPT-J để sinh ra các câu phủ định nghe tự nhiên như người thật nói, thay vì chỉ chèn chữ "not" một cách cứng nhắc bằng code lập trình).*
3.  **Đa dạng hóa ngôn ngữ (Style Transfer)**:
    *   *Colloquial (Văn nói)*: Fine-tune FLAN-T5-large trên Wizard of Wikipedia để chuyển văn viết sang văn nói.
    *   *Lọc gắt gao*: Kiểm tra bằng Edit Distance, bảo toàn Entity/Verb, kiểm tra NLI, và đối kháng AFLITE để chọn câu tự nhiên nhất.
    *   *Kiểm soát chất lượng (QC)*: Đánh giá thủ công (Human Evaluation) cho thấy 99.4% câu REFUTED thực sự sai logic, chứng tỏ dataset có chất lượng rất cao.

### 3. Phương Pháp Tiếp Cận (Baseline Architecture)
Tác giả chia thí nghiệm thành 2 nhánh chính:
*   **Nhánh Claim Only (BERT/BlueBERT/Flan-T5)**: Mô hình chỉ "đọc chay" văn bản nhận định và dự đoán Đúng/Sai. Quá trình train (ví dụ bằng BERT) là học để map từ claim sang nhãn, hoàn toàn không truy xuất KG.
*   **Nhánh With Evidence (Mô hình GEAR)**: Không phải là BERT thêm data, mà là một hệ thống **GraphRAG** gồm nhiều module:
    1.  **Subgraph Retrieval**: Dự đoán Quan hệ & Số bước (Hop) cần tìm trên KG.
    2.  **Fact Verification**: Kết hợp câu Claim với "đường dẫn đồ thị" đã lấy ra để phán quyết. FACTKG chính là tài nguyên cực giá trị cho các mô hình GraphRAG tương lai nhờ các "đường dẫn chuẩn" này.

---

## PHẦN II: TIẾN ĐỘ THỰC HIỆN & THIẾT LẬP KỸ THUẬT

### 1. Mô hình BERT Baseline (Fine-tuned - Claim Only)
Quá trình huấn luyện nhằm thiết lập ngưỡng so sánh cơ bản dựa trên văn bản:
*   **Kiến trúc mô hình**: `bert-base-uncased` (Pre-trained BERT).
*   **Dữ liệu huấn luyện**: 86,367 câu (Train set - `factkg_train.pickle`).
*   **Thiết lập thực nghiệm Lần 1 (Repo mặc định)**:
    *   **Epoch**: 3 | **Batch Size**: 16 | **Learning Rate**: 1e-4.
    *   **Vấn đề**: Repo bị lỗi thiếu lệnh `scheduler.step()`, LR bị kẹt ở mức cao khiến mô hình không hội tụ.
    *   **Kết quả**: 51.35% (Gần như đoán bừa).
*   **Thiết lập tối ưu Lần 2 (Sau khi Bug Fix & Tối ưu cho M1 Pro)**:
    *   **Epoch**: 3 | **Batch Size**: 16.
    *   **Learning Rate**: **2e-5** (Hạ thấp để tăng tính ổn định trên chip M1).
    *   **Optimizer & Scheduler**: Sử dụng **AdamW** + **500 steps Warmup** + Sửa lỗi cập nhật **Scheduler Linear**.
    *   **Kết quả**: **65.37%** (**Vượt mức 65.20% của Paper gốc**).

### 2. Mô hình Flan-T5 (LLM Zero-shot - Claim Only)
Đánh giá khả năng suy luận tự nhiên của LLM mà không qua huấn luyện:
*   **Kiến trúc mô hình**: `google/flan-t5-base`.
*   **Thiết lập**: **Zero-shot Inference** (Sử dụng Prompt kỹ thuật, không có quá trình Train).
*   **Kết quả**: **56.56%**.
*   **Nhận xét**: LLM có tri thức bản năng tốt giúp đạt kết quả 56%, nhưng vẫn thua kém mô hình BERT được huấn luyện chuyên biệt trên tập dữ liệu này (65%).

### 3. Quy trình Đánh giá (Evaluation Details)
*   **Tệp dữ liệu đánh giá**: `factkg_test.pickle` (Tổng cộng **9,041 câu**).
*   **Metric chính**: **Accuracy** (% Độ chính xác tổng quát).
*   **Metric chi tiết**: Đánh giá trên 5 loại suy luận (One-hop, Multi-hop, Conjunction, Existence, Negation).
*   **Thiết bị thực hiện**: Apple M1 Pro GPU (MPS Backend).

### 4. Tiến độ Pipeline B (With Evidence - Graph-RAG)
*   **Trạng thái**: ✅ **Hoàn thành Bước 1 (Preprocess Data)**.
*   **Thành phẩm**: Chuyển đổi thành công dữ liệu thô sang định dạng JSON cấu trúc (`train.json`, `dev.json`, `test.json`, `total_data.pkl`).
*   **Kế hoạch tiếp theo**: Huấn luyện bộ trích xuất quan hệ (Relation Predictor) và số bước nhảy (Hop Predictor).

---

## PHẦN III: BẢNG TỔNG HỢP KẾT QUẢ ACCURACY

| Loại Suy Luận | BERT (Paper) | BlueBERT (Paper) | Flan-T5 (Paper) | BERT (Lần 1: Lỗi) | BERT (Lần 2: Fix) | Flan-T5 (Zero-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| One-hop | 69.64% | 60.03% | 62.17% | 44.57% | **68.08%** | 57.68% |
| Conjunction | 63.31% | 60.15% | 69.66% | 55.75% | **63.77%** | 59.86% |
| Existence | 61.84% | 59.89% | 55.29% | 52.41% | **66.90%** | 54.14% |
| Multi-hop | 70.06% | 57.79% | 60.67% | 48.67% | **62.27%** | 55.71% |
| Negation | 63.62% | 58.90% | 55.02% | 54.11% | **68.57%** | 50.04% |
| **TỔNG CỘNG** | **65.20%** | **59.93%** | **62.70%** | **51.35%** | **65.37%** | **56.56%** |

---
*Người thực hiện: Duy Anh Le*
*Ngày cập nhật: 25/03/2026*
