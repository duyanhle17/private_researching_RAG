# Kế Hoạch Cải Tiến Mô Hình GEAR (Mục Tiêu Deadline: 15/04)

Tài liệu này được biên soạn để vạch ra lộ trình học tập, nghiên cứu và thử nghiệm mã nguồn nhằm tối ưu hóa khả năng suy luận đồ thị đa bước (Multi-hop Reasoning) trên mô hình GEAR (FactKG). Tập trung trực tiếp vào giải quyết giới hạn "chỉ ghép chuỗi String của BERT" và hiện tượng phân rã đồ thị trên chuỗi dài.

---

## 1. Ôn Tập Nhanh: Hiểu Luồng Chạy Của GEAR (With Evidence)

Để tác động đúng chỗ, ta cần nhớ hệ thống chia làm **2 giai đoạn độc lập**. Ta KHÔNG cần sửa toàn bộ mà chỉ "đánh tạt sườn" vào giai đoạn nạp dữ liệu và Classifier.

*   **Giai đoạn 1 (Retriever):** Các file `data_preprocess.py`, `relation_predict`, `hop_predict` đọc `factkg_train.pickle` và KG `dbpedia_...pickle` để sinh ra các **ứng viên đồ thị rác + thực**. Dữ liệu này được đóng gói cất vào `train_candid_paths.bin`, `dev_candid_paths.bin`, và `test_candid_paths_topX.bin`.
*   **Giai đoạn 2 (Classifier):** Mọi thí nghiệm sẽ diễn ra ở file `FactKG/with_evidence/classifier/baseline.py`. Nó load danh sách đồ thị ứng viên `.bin` kết hợp nhãn `.pickle` để train.

---

## 2. Điểm Nghẽn Tử Huyệt Trong Source Code Hiện Tại

Tại file `baseline.py`, phần hàm Tokenize Dữ liệu (tại DataCollator, dòng 116-125) đang xử lý rất thô sơ:
```python
seq_evidence = [f"{self.tokenizer.sep_token.join(evi)} {self.tokenizer.sep_token}" for evi in evidence]
tokenized_evidence = self.tokenizer(
    seq_evidence, 
    max_length=512-len(tokenized_claim["input_ids"][0]), ...
)
```
**Hậu quả:** 
Graph bị "đập bẹp" cấu trúc vòng và các mối quan hệ đa chiều, trở thành một câu text vô hồn được ghép bằng dấu ngang `[SEP]`. Chạm ngưỡng 512 token, thư viện xẻo cụt phần cuối - khiến các chặng ở đuôi của lập luận bị vứt sọt rác, gây ra lỗi nghiêm trọng đặc biệt cho Multi-hop dài.

---

## 3. Action Plan (Các Hướng Thử Nghiệm Tác Chiến Cho 15/04)

Dưới đây là 3 hướng thử nghiệm tính từ Dễ đến Khó. Giai đoạn này nên tập trung làm dứt điểm từng Pha để kiểm chứng độ chính xác.

### Pha 1 (Dễ): Data Pruning - Làm Sạch và Thêm Thắt Node 
Thay vì đổ hết file `train_candid_paths.bin` vào Classifier. Viết một Script trung gian làm bước lọc.
1.  **Lọc nhiễu (Pruning):** Đọc nội dung mảng ứng viên, sử dụng thuật toán TF-IDF, BM25, hoặc một vector nhỏ (như `sentence-transformers/all-MiniLM-L6-v2`) để tính toán độ tương quan Semantic giữa các Node trong Path so với câu Claim gốc. Xóa bỏ những đường đi quá lạc đề (< Threshold).
2.  **Khắc phục lỗi Multi-hop:** Việc lược bỏ bớt các node rác giúp rút ngắn chiều dài chuỗi khi `join` -> Tranh thủ nhường không gian trong định mức 512 token cho các node có ích.

### Pha 2 (Trung bình): Đổi định dạng Graph - Áp dụng K cấu trúc Triples
Mục tiêu: Đập bỏ việc ghép `.join()` toàn bộ Graph.
1.  **Sửa thuật toán Collator (baseline.py):** Thay vì ném mảng `candid_paths` nguyên xi thành 1 string text. Phân rã subgraph ra thành mảng array các **Triples Test** dạng `[CLS] Node A [SEP] Rel 1 [SEP] Node B`. 
2.  **Chấm điểm Triple Relevance Ranking:** So sánh từng đoạn Triples với Tokenzied Claim. Thay vì nhét toàn bộ vào, mô hình sẽ tính Similarity Score và chỉ chọn `Top K` Triples ghép lại nhồi vào Model. Lúc này cái đưa vào Text Encoder không phải là Subgraph tù mù mà là **K Sự thật kề sát nhất** có ý nghĩa trực tiếp.

### Pha 3 (Khó, Tính Tiên Phong Học Thuật): Triển khai thuật toán lai SAT
Đây là lúc thực hiện nâng cấp cấp độ Kiến trúc (Architecture level - Ghi điểm cực mạnh cho báo cáo).
1.  **Thay thế Baseline BERT:** Không dùng `ConcatClassifier` trong file `baseline.py`. Mà thiết kế lại một Multi-modal input.
2.  **Áp dụng Structure-Aware Mask:** Tích hợp cấu trúc SAT. Cho Claim đi qua một Encoding Text riêng. Cho Graph đi qua một GNN / Graph Attention riêng. Cấu trúc Graph Attention mask (một ma trận True/False) sẽ làm cho BERT hiểu được "Nhận thức cấu trúc liên kết": Nghĩa là Token chữ A chỉ được tính self-attention với Token chữ B nếu thực tế có một "Cạnh" giữa A và B trong KG.  

---

## 4. Tóm Lược Work-flow Cụ Thể Tuần Này Để Tập Trung

- [ ] **Bước 1:** Đọc và in thử ra màn hình cái ruột của file `train_candid_paths.bin` chứa gì. Xem thử cấu trúc Array / Subgraph trong đó.
- [ ] **Bước 2:** Chạy tệp `baseline.py` gốc 1 vòng để lấy lại ngưỡng Baseline cho cái GPU của cấu hình ta.
- [ ] **Bước 3:** Thử lập hàm gọt rác (Pruning - Cắt cụt mảng `evidence` trước dòng tokenize). Xem Accuracy có hồi mã thương được bao nhiêu %.
- [ ] **Bước 4:** Bắt tay viết lại class `Dataset / DataCollator` để chuyển rác String thành mảng Object (Ý tưởng Triples).
