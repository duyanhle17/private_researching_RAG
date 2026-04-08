# Kế Hoạch Xây Dựng Benchmark Đồ Thị Tri Thức Mới & Đề Xuất Mô Hình

Tài liệu này trình bày định hướng mở rộng hệ thống đánh giá (benchmark) cho các mô hình ngôn ngữ lớn (LLMs) kết hợp Đồ thị Tri thức (Knowledge Graph - KG), vượt ra ngoài các giới hạn của bộ dữ liệu FactKG hiện hành. Kế hoạch tập trung đi sâu vào 3 thách thức đang là "điểm mù" của các hệ thống RAG (Retrieval-Augmented Generation) truyền thống.

---

## 1. Giải quyết Xung đột Tri thức (Knowledge Conflict Resolution)

### Bản chất vấn đề
Các mô hình ngôn ngữ (như GPT, LLaMA) được pre-train trên lượng dữ liệu khổng lồ, hình thành "trí nhớ nội tại" (Parametric Knowledge). Tuy nhiên, khi đối chiếu với các đồ thị tri thức đặc thù (Evidence), thường xảy ra xung đột thông tin (ví dụ: thông tin cập nhật mới, thông tin nội bộ doanh nghiệp). Khi đó, các LLM hiện tại có xu hướng "bảo thủ", phớt lờ Evidence được cung cấp và sinh ra câu trả lời theo trí nhớ (ảo giác - hallucination).

### Cách tích hợp vào Benchmark mới
*   **Dataset Counterfactual (Nghịch đảo thực tế):** Tạo ra các subgraph giả định và truy vấn trái với các sự thật thông thường.
    *   *Ví dụ:* Evidence là `[Tháp Eiffel] -> (Nằm tại) -> [Berlin]`. Câu hỏi: "Tháp Eiffel nằm ở đâu?".
*   **Tiêu chí chấm điểm:** Nếu mô hình trả lời "Paris" (theo Parametric Knowledge) thì bị tính điểm 0 dù thực tế là đúng. Mô hình chỉ được tính điểm tối đa nếu dựa chắc chắn vào Evidence đã cấp và trả lời "Berlin" (Evidence-grounded).

### Đề xuất Model giải quyết
*   **"Context-Grounding Attention Masking":** Xây dựng hoặc tinh chỉnh cấu trúc (fine-tune) mô hình để nó phân định rạch ròi tỷ trọng sự chú ý (Attention) nghiêng hẳn về Evidence thay vì Parametric context.
*   **Agentic Prompting / Self-Correction:** Yêu cầu một quá trình suy luận trung gian trong prompt: *"Chỉ sử dụng ngữ cảnh X được cung cấp, tôi bỏ qua kiến thức có sẵn..."* trước khi chốt đáp án cuối cùng.

---

## 2. Suy luận Định lượng và So sánh (Numerical & Comparative Reasoning)

### Bản chất vấn đề
Hầu hết các benchmark KG hiện nay chỉ đánh giá logic cấu trúc (A kề với B, C thuộc về A). Nhưng trong thực tế, Đồ thị Tri thức chứa đựng rất nhiều số liệu, thuộc tính định lượng, mốc thời gian. Các LLM chỉ giỏi suy luận ngữ nghĩa nhưng rất yếu kém trong việc thực hiện phép toán +, -, >, < hoặc đối chiếu định lượng chéo giữa các nodes cách xa nhau nhiều hop.

### Cách tích hợp vào Benchmark mới
*   **Câu hỏi định lượng phức tạp:** Vượt qua ranh giới tìm kiếm đơn giản (Retrival).
    *   *Ví dụ Định lượng:* "Sông A dài hơn tổng chiều dài của sông B và sông C cộng lại."
    *   *Ví dụ Thời gian:* "Chủ tịch X nhậm chức sau khi sự kiện Y diễn ra, nhưng trước khi công ty Z thành lập."
*   **Yêu cầu xử lý:** Mô hình phải vượt qua (traverse) biểu đồ đa bước để tìm đúng nodes, "nhặt" các thuộc tính (properties) cần thiết, và thực hiện việc so sánh toán học thực sự.

### Đề xuất Model giải quyết
*   **Neuro-Symbolic AI / Tool-Augmented LLM (ReAct, Toolformer):** LLM không nền tự tính toán thẳng bằng văn bản sinh ra. Mô hình sẽ được dạy để trích xuất Graph và sinh ra **mã giả/lệnh (ví dụ: SPARQL query mở rộng, Python code, hoặc API call)**.
    *   *Luồng xử lý ví dụ:* `Compare(Length(Node A), Add(Length(Node B), Length(Node C)))`. Phép toán này sẽ được giao cho một Symbolic Engine (như máy tính thực) xử lý để trả về boolean (True/False).

---

## 3. Kiểm tra Tính minh bạch đường đi (Explainability & Faithful Traversal Evaluation)

### Bản chất vấn đề
Hiện nay, điểm benchmark chủ yếu dựa vào kết quả đúng sai cuối cùng (Final Answer Accuracy). Việc lấy nguyên Graph nén thành văn bản khiến mô hình học vẹt tín hiệu (Spurious Correlations). Tức là mô hình trả lời đúng nhưng theo những "dấu hiệu nhận biết giả" chứ không thực sự tư duy đường đi A -> B -> C. Căn bệnh này gọi là Shortcut Learning.

### Cách tích hợp vào Benchmark mới
*   **Subgraph Overlap Score (Chấm điểm truy vết):** Đáp án "Vàng" nay phải kèm theo một "Gold Reasoning Path/Subgraph".
*   Khi dự đoán, mô hình bắt buộc xuất ra kết quả kèm theo đồ thị con (Reasoning Paths) mà nó vừa dùng. Hệ thống sẽ so sánh subgraph đó với Gold Subgraph. Nếu trả lời đúng kết quả nhưng sai đường biểu diễn hoặc sai lý do -> Không được công nhận (0 điểm).

### Đề xuất Model giải quyết
*   **Walk-based GNN / Generative Graph Traversal (như RoG - Reasoning on Graphs):** Mô hình không "đọc" toàn bộ Graph trong 1 lượt. Agent đóng vai trò người nhảy (Walker).
*   Từ Head Entity, mô hình sẽ step-by-step dự đoán bước nhảy tới Node/Edge kế tiếp. Kết thúc hành trình, nó lấy chính hành trình mình đã đi qua để đưa ra kết luận. Hành trình này tự động trở thành lời giải trình minh bạch tuyệt đối.

---

## Tóm Lược Thuyết Trình Đề Xuất

**Tuyên bố tầm nhìn:**
> *"Để vượt qua giới hạn của FactKG và tạo ra một benchmark thực tế khắt khe cho RAG doanh nghiệp, tập dữ liệu mới không đọ nhau ở khả năng đoán dựa vào văn bản ép phẳng. Benchmark sẽ tập trung vào 3 thử thách lớn nhất: (1) Tính trung thành tuyệt đối với Knowledge Graph bất chấp mâu thuẫn từ trí nhớ của LLM; (2) Có khả năng làm phép toán và so sánh thuộc tính siêu liên kết giữa các Node; (3) Buộc mô hình phải trả đúng nguyên mẫu đường đi thuật toán (Reasoning Path) thay vì đoán mò."*

**Tuyên bố công nghệ:**
> *"Hướng đi tiềm năng nhất không nằm ở Text Encoder truyền thống. Kiến trúc đề xuất sẽ là **Agentic Graph-RAG / Neuro-symbolic**, nơi mô hình có khả năng sinh công cụ (Tool-use) để làm phép toán, và tuần tự nhảy qua từng node thay vì đọc lướt thụ động."*
