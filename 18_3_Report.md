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

*Các thông số quan trọng trong lúc chạy thí nghiệm:*
- Dữ liệu đánh giá: `factkg_test.pickle` (Tổng số câu: **9041 câu**)
- Metric đánh giá chính: **Accuracy** (Độ chính xác)
- Batch size khi đo BERT: 32 | Batch size Flan-T5 mô phỏng: 16

| Loại Suy Luận | Kết quả Paper (BERT) | BERT (M1 Pro - Chạy cấu hình lỗi) | BERT (M1 Pro - Cấu hình chuẩn) | Flan-T5 Zero-shot (M1 Pro) | GEAR (Có Evidence - Paper) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| One-hop | 69.64% | 44.57% | **68.08%** | 57.68% | 83.23% |
| Conjunction | 63.31% | 55.75% | **63.77%** | 59.86% | 77.68% |
| Existence | 61.84% | 52.41% | **66.90%** | 54.14% | 81.61% |
| Multi-hop | 70.06% | 48.67% | **62.27%** | 55.71% | 68.84% |
| Negation | 63.62% | 54.11% | **68.57%** | 50.04% | 79.41% |
| **TỔNG CỘNG** | **65.20%** | **51.35%** | **65.37%** | **56.56%** | **77.65%** |

**Phân tích & Lý giải chi tiết**:

**1. Về cơ chế Zero-shot của Flan-T5 (Tại sao không Train?)**
*   **Mục đích trong bài báo**: Trong paper, Flan-T5 được đưa vào nhánh *Claim Only* dưới dạng **Zero-shot**. Nghĩa là tác giả muốn đo lường xem: Nếu chúng ta lấy một mô hình LLM siêu khổng lồ, đã có lượng "kiến thức nền" (world knowledge) cực tốt từ trước nhưng lại **không được cung cấp bằng chứng (evidence) và không được huấn luyện trên dataset này**, thì nó có làm tốt việc xác thực sự thật (Fact Verification) hay không?
*   **BERT Baseline Hành Trình Đạt 65.37%**:
    - **Lần chạy đầu tiên (51.35%)**: Chúng ta đã sao chép chính xác từ Repo mã nguồn của tác giả. Tuy nhiên, kết quả thấp là do **Repo Bug** (tác giả quên không gọi `scheduler.step()`) và **Batch Size mismatch** (M1 Pro chỉ chạy được batch 16, nhưng cấu hình lại dùng cấu trúc Learning Rate 1e-4 của batch 64). Việc này làm thay đổi quá xa động lực học của Gradient, khiến tiến trình huấn luyện bị kẹt.
    - **Lần chạy thứ hai (65.37%)**: Sau khi phân tích "ngưỡng chết" của GPU, chúng ta đã tối ưu lại bằng bộ siêu tham số chuẩn mực cho BERT (`lr=2e-5`, `warmup_steps=500`, `AdamW`), đồng thời thêm lệnh gọi Scheduler. Kết quả chứng minh mô hình hoạt động ổn định và **khớp hoàn hảo với mức điểm mà Paper đưa ra (thậm chí vượt nhẹ 65.37% vs 65.20%)**. Các bài toán suy luận phức tạp như Negation (từ 54.11% -> 68.57%) và Conjunction (từ 55.75% -> 63.77%) được cải thiện mạnh mẽ.
*   **So sánh với Flan-T5 Zero-shot**: Khi chưa sửa lỗi, Flan-T5 nhìn có vẻ tốt (56.56%), nhưng với mô hình BERT được huấn luyện đúng cách (65.37%), ta thấy rõ: Mặc dù Flan-T5 có trí tuệ nhân tạo (kiến thức nền) rất mạnh, nó vẫn không thể bằng một con AI đơn giản hơn nhưng lại được trau dồi "đúng lộ trình" trên loại cấu trúc câu hỏi (dataset) riêng biệt này. Điều này cũng xác nhận lại hệ giá trị cốt lõi của bài báo.
*   Dù là BERT hay Flan-T5, ở giai đoạn **Claim Only**, tất cả mô hình đều bộc lộ giới hạn ở mức 60-65% Accuracy, đặc biệt yếu ớt khi đứng trước các câu khẳng định nhiều bẫy ngầm (Multi-hop). Do đó, sự cần thiết của kho tri thức **Knowledge Graph (Evidence)** như trong mô hình GEAR sắp tới (Dự kiến lên ~77%) là vô cùng rõ ràng.

**2. Giải thích con số tập dữ liệu (Total num: 9040)**
*   Theo Paper, đúng là FactKG có tổng cộng khoảng **108k Claims (hơn 108.000 nhận định)**. 
*   Tuy nhiên, theo quy chuẩn khoa học, dữ liệu này phải được chia theo tỷ lệ (Split Rate):
    *   **Train Set (~86k câu)**: Đã được dùng để chạy huấn luyện cho BERT Baseline trong mười mấy tiếng trước đây.
    *   **Validation/Dev Set (vài nghìn câu)**: Dành để hiệu chỉnh siêu tham số khi train.
    *   **Test Set (Chính xác là 9.041 câu)**: Dành để "đo lường" một cách công bằng.
*   Vì Flan-T5 đang làm nhiệm vụ ĐO LƯỜNG (Inference Evaluation) chứ không học (Train), nên script đã được code để load chính xác file `factkg_test.pickle`. Terminal hiện `9041/9041` chính là toàn bộ Test Set của bộ dữ liệu 108k câu này. Việc chạy qua hết hơn 9.000 câu bằng Large Language Model trong 1 phút 43 giây (87 vòng lặp/giây) là một tốc độ "óng ánh bão táp" chỉ có ở GPU xịn (MPS của Mac M1).

*   **BERT Baseline vs Flan-T5 Zero-shot**:
    *   BERT (M1 Pro - Base model): Học trên 86k câu Train, đo trên 9k câu Test → Kết quả 48.65%. Mặc dù đã cố gắng học bài nhưng do kích thước bộ não nhỏ cộng với không có evidence, AI gần như phải đoán bừa, cho ra kết quả bệt.
    *   Flan-T5 (M1 Pro - Base model): Không học gì cả (Zero-shot), dùng bản năng thế giới để đo trên 9k câu Test → Kết quả 56.56%. Khôn hơn BERT dù không cần học bài, nhưng khi đối diện với vấn đề đánh tráo khái niệm thì vẫn bó tay.

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
ok
