# Báo Cáo Triển Khai và Huấn Luyện Đồ Thị Tri Thức Y Khoa (Medical KG) Theo Chuẩn Mô Hình SAT

Báo cáo này tổng hợp toàn bộ quy trình từ khâu xây dựng bộ dữ liệu (Build Dataset) từ nguồn văn bản thô, cho đến việc huấn luyện (Training) Mô hình Liên kết Đồ thị - Văn bản (Graph-Text Alignment) dựa trên kiến trúc của mạng **SAT (Structure-Aware Transformer)**.

---

## 1. Giai Đoạn 1: Xây Dựng Dữ Liệu Gắn Chuẩn SAT (`build_dataset.py`)

Mục tiêu của giai đoạn này là biến một cuốn sách / tài liệu y khoa định dạng thô thành cấu trúc Triplex (Head, Relation, Tail) hoàn toàn tương thích với format của bộ `FB15k-237N` do tác giả SAT cung cấp.

### Các Kỹ Thuật Đã Áp Dụng:
- **Semantic Chunking:** Chia văn bản theo đúng ranh giới của câu hỏi để tránh đứt gãy ngữ nghĩa (Chunk size $\approx$ 200 từ, Overlap = 30 từ ~ 1,2 câu cuối). Mỗi đoạn văn như vậy được quy chuẩn giống hệt mô tả Wikipedia trong SAT (`id2text.txt`).
- **Prompt Engineering Khắt Khe:** LLM Llama3-70b chạy qua NVDIA NIM được ràng buộc chỉ lấy chính xác **5 loại Entity Y Khoa** (Diseases, Drugs, Anatomy, Pathogens, Procedures). Khóa chặt việc trích xuất các từ rác (đại từ "he/she/it", con số "1 cm", hoặc động từ).
- **Trình Lọc Hậu Kỳ Khử Nhiễu (Post-filter):** Loại bỏ hoàn toàn các thực thể mơ hồ cấu trúc chung chung (như "treatment", "condition", "risk", "team") và các thực thể chỉ chứa số hóa/đo lường để chuẩn hóa đỉnh đồ thị.
- **Ánh Xạ SAT (SAT Mapping):** Tự động sinh ra 10 file cấu trúc đặc thù mà mô hình mạng SAT cần để xây dataset (`train.txt`, `valid.txt`, `test.txt`, `mid2id.txt`, `rel2id.txt`, `id2text.txt`,...).

---

## 2. Giai Đoạn 2: Huấn Luyện Mô Hình (`train_medical.py`)

Chúng ta giữ nguyên **100% Cốt Lõi Toán Học SAT** bằng cách import chéo các file gốc của dự án (`model_gt.py` với mô hình mạng CLIP khổng lồ, `graph_transformer.py`, `data_helper.py`, bộ mã hóa `simple_tokenizer.py`) để quá trình học là thuần túy. Quá trình huấn luyện không chỉ học từ ngữ nghĩa văn xuôi, mà còn học cách nhúng cấu trúc nút mạng và quan hệ.

### Tóm tắt bộ phận cấu trúc:
1. **Dữ Liệu Thuộc Tính Của Graph Y Khoa:**
   - **Tổng số Thực thế Y Khoa (Entities):** 3,281
   - **Tổng số Các Loại Quan Hệ (Relations):** 311 
   - **Tổng số Mệnh đề Liên kết (Triples):** 7,369 (Trong đó: Train = 6,631 | Valid & Test = 738)
2. **Siêu Tham Số (Hyperparameters):**
   - Số vòng lặp huấn luyện (Epochs): 5
   - Batch size: 16
   - CPU Workers (Dữ liệu Dataloader): 2
   - Learning Rate: 2e-05
   - Thiết bị tăng tốc: Lõi Neutral Engine chuẩn bị MPS trên Apple Silicon (Mac M).

### Sự Cố Đã Gặp Và Cải Tiến Vượt Bậc Của Chúng Ta:
Bản mã nguồn gốc của SAT gặp lỗi **Memory Leak** nghiêm trọng khi chạy trên máy tính MacOS/Local ở vòng đánh giá (Evaluation). Pytorch lưu trữ cây đồ thị đạo hàm khổng lồ của phép thử All-to-all lúc chấm điểm gây tràn RAM dữ dội, máy tính bị ép lấy Ổ cứng SSD ra làm RAM chắp vá (Swap Mac Memory) khiến thời gian cho 1 lượt chậm mất 37 giây (Mất ~45 phút / Epoch).

**Giải pháp đột phá:** Sửa tận gốc file huấn luyện bằng cách bổ sung `with torch.no_grad():` tại hàm đánh giá và set lại cờ `model.train()` ở mỗi đầu Epoch.
=> **Kết Quả:** Khẩu độ chấm toán học từ 45 phút sụp giảm xuống còn đúng **41 giây** lướt thần tốc không lỗi lầm (Quét cực mượt đạt 1.72 vòng/giây). Tốc độ Training trơn tru ~ 1.1s / Lượt (It).

---

## 3. Kết Quả Huấn Luyện (Loss & Test Accuracy)

Sau khi chỉnh hình và huấn luyện, loss graph cho thấy mô hình đang thu hội cực kỳ vững chắc với tốc độ nhanh, thông qua 5 Epoch học lặp lại trên toàn thể bộ data:

| Epoch | Chức Năng | Mean Loss (Trung Bình Mất Mát) | Ghi Chú Tính Toán |
| :---: | :---: | :--- | :--- |
| **Epoch 0** | Triển khai | **~ 50.04** | Loss giảm cực nhanh từ 65.15 thẳng xuống 42.74 ngay vòng đầu tiên (Bắt đầu nắm kết cấu Graph). |
| **Epoch 1** | Hấp thu | **~ 38.77** | Biến thiên hội tụ. |
| **Epoch 2** | Hấp thu | **~ 32.54** | Đồ thị Mất mát xuống rất nhịp nhàng và đều đặn. |
| **Epoch 3** | Hấp thu | **~ 27.98** | Rèn luyện khả năng chống nhiễu loạn Alignment. |
| **Epoch 4** | Khảo sát thi | **~ 25.06** | Bắt đầu chạm đỉnh tiệm cận cực tiểu. |

### Đánh Giá Điểm Bài Thi (Evaluation Performance - Epoch 4):
> **Độ Chính Xác Tổng Thể Cuối Cùng (Best Test Accuracy) = 0.6163 (61.63%)**

💡 **Đánh Giá Của Chuyên Gia Về Kết Quả Này:**
- Với 3,281 nút Mạng Nhện chồng chéo và 311 nhãn dán loại quan hệ y khoa khác nhau, một bài toán trắc nghiệm đối mặt với tỷ lệ chọi cực cao (Tìm 1 Text định nghĩa nằm giữa rừng 3,281 đoạn văn nhiễu lộn xộn).
- Mức Accuracy **61.63%** chứng tỏ mô hình Graph Transformer đã nắm vô cùng vững kết cấu Alignment. Khả năng nó chỉ định 1 bệnh cụ thể nhặt ra chính xác bài text cấu trúc của bệnh đó thay vì đi lạc vô một triệu chứng khác là cực uy tín. 

👉 Trọng lượng tri thức nhúng của mô hình đã được trích thành công ra một file đóng nén `gt-og_best.pkl` nằm trong hòm đồ `checkpoints/medical`. Sẵn sàng đạn dược đem vác vào mạng Truy Xuất (GraphRAG `run_sat_baseline.py`) trả lời tự động ngôn ngữ tự nhiên!
