Tốt lắm, mình thấy bạn đang chạy BERT rồi! Để mình giải thích rõ từng bước cho bạn hiểu nhé:

---

## Pipeline A: Claim Only — "Đoán" chỉ bằng đọc câu

### A1: BERT Fine-tuning
- **Làm gì?**: Lấy mô hình BERT (đã được Google huấn luyện sẵn trên hàng tỉ câu tiếng Anh), rồi **dạy thêm** (fine-tune) nó trên 108k câu claim của FactKG để nó học cách phân biệt câu Đúng/Sai.
- **Input**: Chỉ có câu văn bản, ví dụ: *"Adam McQuaid was born in Wisconsin"*
- **Output**: True hoặc False
- **Ý nghĩa**: Đây là **Baseline thấp nhất**. Nó cho biết: nếu AI chỉ "đọc lướt" câu mà không tra cứu gì cả thì đúng được bao nhiêu %? Con số này thường khoảng 60-70%.

### A2: Flan-T5 Zero-shot
- **Làm gì?**: Dùng mô hình Flan-T5 của Google, **không huấn luyện gì cả** (zero-shot). Chỉ hỏi thẳng nó: *"Is this claim True or False?"*
- **Input**: Cùng câu claim như A1.
- **Output**: Model tự trả lời "True" hoặc "False".
- **Ý nghĩa**: Đo xem một mô hình ngôn ngữ lớn (LLM) có kiến thức tổng quát bao la, liệu từ "trí nhớ" của nó có thể kiểm chứng sự thật mà không cần được dạy trên dataset này không. Thường kết quả thấp hơn A1 ở nhiều loại suy luận phức tạp.

---

## Pipeline B: With Evidence — "Xác minh" bằng bằng chứng từ Knowledge Graph

### B1: Preprocess Data
- **Làm gì?**: Đọc các file pickle, trích xuất thông tin (câu claim, thực thể, quan hệ, số hop) rồi chuyển thành định dạng `.json` + `.pkl` để các bước sau sử dụng.
- **Ý nghĩa**: Bước chuẩn bị dữ liệu thuần túy, không có huấn luyện gì. Giống như "xắt rau, ướp gia vị" trước khi nấu.

### B2: Train Relation Predictor
- **Làm gì?**: Huấn luyện một model BERT để **dự đoán "nên tìm quan hệ gì"** trên KG.
- **Ví dụ**: Đọc câu *"Adam McQuaid was born in Wisconsin"* → model dự đoán nên tìm quan hệ `birthPlace`, `placeOfBirth` trên đồ thị DBpedia.
- **Ý nghĩa**: Đây là **bộ não thám tử thứ nhất** — nó biết phải tra cứu loại thông tin gì. Không có bước này thì KG có hàng nghìn loại quan hệ, sẽ không biết bắt đầu từ đâu.

### B3: Train Hop Predictor
- **Làm gì?**: Huấn luyện một model BERT khác để **dự đoán phải đi bao nhiêu bước (hop)** trên đồ thị.
- **Ví dụ**: Câu *"A's spouse is B"* → chỉ cần 1 hop (A → spouse → B). Nhưng câu *"A's child was born in the same city as B"* → cần 2-3 hop.
- **Ý nghĩa**: Đây là **bộ não thám tử thứ hai** — nó biết phải đi sâu bao nhiêu trên đồ thị. Tránh việc đi quá xa (tốn tài nguyên) hoặc quá gần (thiếu thông tin).

### B4: Train Final Classifier
- **Làm gì?**: Đây là **bộ não quyết định cuối cùng**. Nó nhận vào:
  1. Câu Claim gốc.
  2. Các đường đi (evidence paths) trên KG mà bước B2+B3 tìm được.
  
  Rồi ghép lại, đưa qua BERT + một lớp MLP để quyết định **True hay False**.
- **Ý nghĩa**: Bước này tương đương với việc thám tử đã thu thập đầy đủ bằng chứng, giờ ngồi lại **đối chiếu** xem bằng chứng có ủng hộ hay phản bác câu khẳng định.

---

## Cuối cùng so sánh cái gì?

So sánh **Accuracy** giữa Pipeline A và Pipeline B, đặc biệt chia theo 5 loại suy luận:

| Loại suy luận | Pipeline A (Chỉ đọc câu) | Pipeline B (Có tra KG) | Kỳ vọng |
|---|---|---|---|
| **One-hop** (đơn giản) | Khá cao | Rất cao | B hơn A một ít |
| **Multi-hop** (đa bước) | Thấp | Cao hơn nhiều | B vượt trội |
| **Conjunction** (phép hội) | Trung bình | Cao hơn | B hơn A đáng kể |
| **Existence** (sự tồn tại) | Trung bình | Cao | B hơn A |
| **Negation** (phủ định) | Rất thấp | Cao hơn | B vượt trội rõ rệt |

**Kết luận mà paper muốn chứng minh**: *Khi cho AI được "tra cứu" trên Knowledge Graph (Pipeline B), khả năng kiểm chứng thông tin tốt hơn hẳn so với khi nó chỉ "đoán" từ văn bản (Pipeline A), đặc biệt ở các dạng suy luận phức tạp như Multi-hop và Negation.*

Bạn đang chạy BERT trên terminal rồi đúng không? Muốn mình kiểm tra tiến trình nó đang train tới đâu không?