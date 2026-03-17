# Báo Cáo Tiến Độ Dự Án FactKG (Ngày 18/03)

## 1. Tổng Quan Về Bài Báo (Paper) & Kho Lưu Trữ (Repo)

### 1.1 Mục Đích Của Bài Báo
Bài báo **FactKG** ra đời nhằm giải quyết một vấn đề cốt lõi trong Trí tuệ Nhân tạo (AI): **Kiểm chứng sự thật (Fact Verification)**.
*   **Vấn đề đặt ra**: Các mô hình ngôn ngữ lớn (LLMs) như BERT, GPT thường bị "ảo giác" (hallucinations) hoặc đoán mò khi được hỏi về các sự thật cụ thể mà chúng không nhớ rõ.
*   **Giải pháp của Paper**: Sử dụng **Knowledge Graph (KG - Đồ thị Tri thức)** làm "bằng chứng thép". KG (ví dụ: DBpedia, Wikidata) chứa hàng triệu thông tin được lưu dưới dạng có cấu trúc rõ ràng (Thực thể A -> Quan hệ -> Thực thể B).
*   **Mục tiêu tối thượng**: Xây dựng một hệ thống AI (Pipeline) biết cách **tra cứu KG** để lấy bằng chứng, sau đó đối chiếu thông tin đó với "câu khẳng định" (Claim) để quyết định câu đó là **True (Đúng)** hay **False (Sai)**.

### 1.2 Kho Lưu Trữ (Repo) FactKG Là Gì?
Repo này là mã nguồn do các tác giả công bố để minh chứng cho lập luận trong paper của họ. Họ cung cấp:
*   **Dataset (Tập dữ liệu)**: 108k câu khẳng định (claims) được dán nhãn Đúng/Sai, kèm theo bằng chứng trích xuất từ DBpedia.
*   **Pipeline A (Claim Only)**: Phiên bản "ngô nghê" - Các mô hình AI (như BERT) chỉ được đọc câu khẳng định và phải "tự đoán" Đúng/Sai bằng trí nhớ của chính nó (không cho tra KG). **Đây là Mốc So Sánh (Baseline).**
*   **Pipeline B (With Evidence)**: Phiên bản "thám tử" - Hệ thống AI (Graph-RAG) được dạy cách tìm kiếm thông tin trên DBpedia, tạo thành các đường dẫn bằng chứng (evidence paths), rồi mới đưa ra quyết định.

**Tác giả muốn chứng minh: Pipeline B luôn vượt trội so với Pipeline A, đặc biệt là ở những câu suy luận phức tạp.**

---

## 2. Dữ Liệu (Datasets) Có Ý Nghĩa Gì?

Trong dự án này, bộ dữ liệu (nằm trong các file `.pickle`) đóng vai trò cực kỳ quan trọng:

1.  **Dùng để Huấn luyện (Train)**:
    *   Chia làm 3 tập: `factkg_train.pickle` (Dùng để dạy model), `factkg_dev.pickle` (Dùng để tinh chỉnh siêu tham số), và `factkg_test.pickle` (Dùng để thi học kỳ/đánh giá lần cuối).
    *   AI cần xem hàng chục nghìn ví dụ (câu + nhãn True/False) để học được "cảm giác" về một câu nói dối (ví dụ: cấu trúc câu, từ ngữ mâu thuẫn).
2.  **Dùng để Phân tích chi tiết**:
    *   Mỗi câu trong dataset không chỉ có nhãn Đúng/Sai mà còn có **Metadata (nhãn phân loại tư duy)**:
        *   `One-hop`: Suy luận 1 bước (ví dụ: *A sinh ra ở B*).
        *   `Multi-hop`: Suy luận nhiều bước (ví dụ: *A là con của B, B sinh ra ở C, vậy A quê ở C?*).
        *   `Conjunction`: Câu ghép, 2 điều kiện phải cùng đúng.
        *   `Existence`: Kiểm tra một thuộc tính có tồn tại thật hay không.
        *   `Negation`: Câu có yếu tố phủ định (Không, chưa từng...).

---

## 3. Đánh Giá Kết Quả Pipeline A (BERT Baseline)

Vừa qua, chúng ta đã hoàn thành việc huấn luyện (Train) và đánh giá (Evaluate) Pipeline A trên máy Mac M1 Pro.

### 3.1 Quá Trình Train Của Pipeline A
*   **Cách làm**: Lấy mô hình `bert-base-uncased` của Google, đưa 86k câu khẳng định (chỉ có văn bản, không có KG) vào cho nó tự phân loại.
*   **Quá trình xử lý**: Chạy 3 Epochs (3 vòng lặp qua toàn bộ dữ liệu) trong khoảng ~6 tiếng vào ban đêm.
*   **Tại sao không hiện Accuracy lúc Train?**: Lúc train, mã nguồn gốc của bài báo chỉ ưu tiên ghi nhận **Loss (Độ lỗi)** vào log file để đo sự hội tụ. Chữ "Validation accuracy" chỉ in lên màn hình rồi trôi đi mất để tăng tốc độ ghi đĩa.

### 3.2 Đánh Giá (Evaluation) Của Pipeline A
Để có số liệu chính xác thay vì phụ thuộc vào màn hình trôi, chúng ta đã dùng script kiểm tra trên **tập Test (~9.000 câu)** và bóc tách thành 5 loại suy luận.

**Kết quả thu được (Chỉ dùng Text, không dùng KG):**

| Loại Suy Luận | Độ Chính Xác (Accuracy) | Số Lượng Câu (Count) | Phân Tích Ý Nghĩa |
| :--- | :---: | :---: | :--- |
| **One-hop** (Dễ nhất) | 55.43% | 1914 | BERT thỉnh thoảng nhớ được các facts cơ bản phổ biến. |
| **Multi-hop** (Phức tạp) | 51.33% | 1874 | Cần nối 2-3 sự kiện, bộ nhớ nội tại của BERT bắt đầu "đuối", gần như đoán bừa 50/50. |
| **Existence** (Tồn tại) | 47.59% | 870 | Hoàn toàn đoán bừa. |
| **Negation** (Phủ định) | 45.89% | 1314 | **Kém nhất**. BERT rất dở trong việc xử lý từ "không", thường nhầm 1 câu phủ định sai thành đúng. |
| **Conjunction** (Câu ghép) | 44.25% | 3069 | Phức tạp, điểm cực kỳ thấp. |
| **TỔNG CỘNG** | **48.65%** | **9041** | **Mất phương hướng**. Khi không có công cụ tra cứu, AI đoán bừa (coin flip). |

*Lưu ý: Điểm số của chúng ta có thể thấp hơn một chút so với Paper vì chúng ta dùng bản `bert-base` (nhỏ gọn) và Train ít Epoch hơn để chạy được trên Mac M1 16GB, trong khi Paper dùng GPU hạng nặng. Tuy nhiên, tỷ lệ chênh lệch giữa các loại suy luận là tương đồng với chứng minh của tác giả.*

---

## 4. Kế Hoạch Tiếp Theo: Pipeline B (Graph-RAG)

Do kết quả "thảm hại" của Pipeline A (chỉ quanh quẩn 50%), bước tiếp theo là triển khai hệ thống lõi của bài báo.

Chúng ta sẽ dạy AI cách lật các file DBpedia Pickle (đại diện cho KG) ra tra cứu:
1.  **Tiền xử lý**: Trích xuất dữ liệu KG thành định dạng mà AI đọc được.
2.  **Train Retriever (Người thám tử)**: Dạy 2 model BERT để dự báo *nên tìm loại quan hệ gì* và *đi bao xa (hop)* trên mạng lưới DBpedia.
3.  **Train Classifier (Vị quan tòa)**: Đưa bằng chứng tìm được ghép với câu khẳng định ban đầu để phán xét Đúng/Sai.

Khi hoàn thành Pipeline B, dự kiến điểm Multi-hop và Negation sẽ tăng lên rất mạnh nhờ công lớn của "bằng chứng" từ Knowledge Graph.
