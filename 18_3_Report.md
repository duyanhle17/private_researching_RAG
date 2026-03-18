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
    *   *Entity Substitution*: Thay thực thể bằng một thực thể cùng loại nhưng nằm ngoài 4-hop trên DBpedia, dùng kiểm tra NLI 2 chiều để đảm bảo thực sự mâu thuẫn (Contradiction).
    *   *Relation Substitution*: Đánh tráo quan hệ bằng một quan hệ khác có cùng cấu trúc Head/Tail.
    *   *Negation*: Sinh câu phủ định (thêm "not") kết hợp mô hình GPT-J cho các câu phức tạp, nhãn được gán chặt chẽ theo logic của đường đi trên đồ thị.
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

## PHẦN II: TIẾN ĐỘ THỰC HIỆN THỰC TẾ (WORK PROGRESS)

### 1. Môi Trường & Thiết Bị (Environment)
*   **Thiết bị**: MacBook Pro M1 Pro (16GB RAM).
*   **Cấu hình tối ưu**: Sử dụng `mps` cho GPU và giảm Batch size xuống 16 để tránh lỗi bộ nhớ (OOM).

### 2. Kết Quả Huấn Luyện Pipeline A (BERT Baseline)
Chúng ta đã hoàn thành việc tái lập thí nghiệm Baseline thuộc nhóm **Claim Only**:
*   **Mô hình**: `bert-base-uncased`. Việc chạy file `bert_classification.py` chính là quá trình huấn luyện supervised bình thường để học cách phân loại Đúng/Sai từ văn bản.
*   **Quá trình**: Fine-tuning trên **86,367 câu** (tập Train) trong 3 Epochs.
*   **Dữ liệu đánh giá**: Sử dụng toàn bộ **9,041 câu** (100% tập Test) của tác giả để đo lường.

### 3. Đối Chiếu Kết Quả (Accuracy Comparison)

| Loại Suy Luận | Kết quả Paper (BERT) | Kết quả thực tế (M1 Pro) | GEAR (Có Evidence - Paper) |
| :--- | :---: | :---: | :---: |
| One-hop | 69.64% | **55.43%** | 83.23% |
| Conjunction | 63.31% | **44.25%** | 77.68% |
| Existence | 61.84% | **47.59%** | 81.61% |
| Multi-hop | 70.06% | **51.33%** | 68.84% |
| Negation | 63.62% | **45.89%** | 79.41% |
| **TỔNG CỘNG** | **65.20%** | **48.65%** | **77.65%** |

**Phân tích sự khác biệt**:
*   Kết quả thực tế **48.65%** cho thấy khi dùng mô hình bản `Base` trên văn bản thô, AI gần như phải đoán bừa. Lý do là BERT chỉ học xác suất của cụm từ mà không truy cập vào tri thức thực sự.
*   Sự chênh lệch giữa thực tế và Paper (65%) giải thích bởi cấu hình phần cứng hạng nặng và mô hình bản `Large` mà tác giả sử dụng. Tuy nhiên, xu hướng các loại suy luận khó (Negation/Conjunction) có điểm thấp nhất là tương tự nhau, phản ánh đúng giới hạn của việc không dùng Evidence (Claim Only).

---

## PHẦN III: KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

1.  **Mục tiêu**: Nâng cấp kết quả từ **48% (Pipeline A)** lên mức **>70% (Pipeline B)** bằng cách xây dựng hệ thống mô phỏng GEAR (GraphRAG).
2.  **Các bước cụ thể**:
    *   **Bước 1 - Preprocess Data**: Chạy tiền xử lý trên DBpedia để trích xuất cấu trúc đồ thị thành định dạng JSON/Pickle.
    *   **Bước 2 - Train Retriever**: Huấn luyện module phân loại Relation và Hop.
    *   **Bước 3 - Train Classifier**: Dùng bằng chứng đồ thị (Subgraphs) từ bước 2 ghép vào câu Claim để đưa ra dự đoán cuối cùng.
3.  **Mong đợi**: Minh chứng rõ rệt sức mạnh của phương pháp tiếp cận **With Graphical Evidence** so với nhánh **Claim Only**.

---
*Người thực hiện: Duy Anh Le*
*Ngày cập nhật: 18/03/2026*
