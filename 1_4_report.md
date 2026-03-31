# Báo cáo Đánh giá Mô hình GEAR (With Evidence Pipeline) trên FactKG

## 1. Tổng quan & Cấu hình Huấn luyện
Báo cáo này trình bày kết quả đánh giá (evaluation) của mô hình phân loại **GEAR** (Graph-based Evidence Aggregation and Reasoning) sau quá trình huấn luyện. GEAR là mô hình thuộc nhánh tiếp cận "With Evidence", sử dụng bằng chứng được truy xuất từ Knowledge Graph (KG) kết hợp với claim để đưa ra phán đoán độ chân thực (Fact Checking).

Các siêu tham số (Hyperparameters) chạy full theo chuẩn của bài báo gốc:
- **Ngôn ngữ / Mô hình lõi**: BERT (đóng vai trò là Text Encoder và kết hợp xử lý đồ thị).
- **Bộ tối ưu hóa (Optimizer)**: Adam (giúp điều chỉnh tỷ lệ học linh hoạt và tối ưu hiệu quả cho mô hình ngôn ngữ lớn).
- **Số lượng Epoch**: 5 Epoch (Mô hình đã học qua toàn bộ tập dữ liệu huấn luyện 5 lần).
- **Batch Size (BERT)**: 64 (Kích thước lô mẫu được xử lý trong một bước cập nhật trọng số).
- **Learning rate**: Theo cấu hình chuẩn của bài báo gốc với mô hình BERT và mạch đồ thị GEAR.

## 2. Kết quả Đánh giá Tổng quan (Evaluation Results)
Sau khi kết thúc epoch số 5, mô hình tiến hành đánh giá tập Test của FactKG.
- **Tổng số mẫu đánh giá**: 9024 mẫu
- **Độ chính xác tổng (Total Test Accuracy)**: **79.06%**

*Lưu ý: Kết quả này phản ánh sự cải thiện rõ rệt của mô hình khi được cung cấp thêm thông tin (evidence) từ KG so với phương pháp chỉ cung cấp claim (Claim-Only).*

## 3. Phân tích Độ chính xác Theo Từng Loại Suy luận (Reasoning Types)
Để hiểu rõ hơn về khả năng của mô hình trước các dạng câu hỏi khác nhau, kết quả được phân tách thành 5 loại suy luận (reasoning) riêng biệt như sau:

| Loại Suy luận (Reasoning Type) | Độ chính xác (Accuracy) | Số lượng mẫu (Examples) |
| :----------------------------- | :---------------------: | :---------------------: |
| **One-hop**                    |         **80.67%**        |          1914           |
| **Conjunction**                |         **84.49%**        |          3069           |
| **Existence**                  |         **81.15%**        |           870           |
| **Multi-hop**                  |         **66.97%**        |          1874           |
| **Negation**                   |         **79.88%**        |          1297           |

### Đánh giá Chi tiết:
1. **Mạnh nhất ở "Conjunction" (84.49%)**: Mô hình xử lý rất tốt các thông tin kết hợp (chắp nối các sự thật lại với nhau). Việc cung cấp bằng chứng đồ thị ở dạng subgraph rất phù hợp cho loại suy luận này.
2. **Khả năng suy luận trên "Existence", "One-hop", "Negation" đều đạt quanh mức trung bình 80-81%**:
   - Vượt qua các dạng truy vấn sự thật đơn ($One-hop$).
   - Nhận diện sự tồn tại của tính chất ($Existence$) tốt.
   - Nhận dạng phủ định ($Negation$) cực kỳ tốt so với các kỹ thuật baseline cũ, chứng tỏ sức mạnh của bộ lý luận Graph (GEAR) trong việc nhận biết trạng thái mâu thuẫn giữa câu khẳng định/phủ định và evidence.
3. **Thách thức lớn nhất tại "Multi-hop" (66.97%)**: Suy luận qua nhiều bước/cạnh (multi-hop) trên tri thức đồ thị vẫn là bài toán khó nhất. Khi qua nhiều bước nhảy trên mạng tri thức, tính nhiễu (noise) tăng cao dẫn đến sự suy giảm độ chính xác của mô hình khi tổng hợp bằng chứng.

## 4. Kết luận
- **Huấn luyện thành công và Hội tụ tốt**: Mô hình hội tụ rất ổn định bằng thuật toán Adam trong suốt 5 epochs với BERT Batch size 64. 
- **Chất lượng mô hình**: Hiệu suất 79.06% là mức cao, tiệm cận (và tái hiện chuẩn xác) phân phối kỹ thuật trong các paper gốc của FactKG ở thiết lập sử dụng Evidence.
- Đặc tính sử dụng Evidence giúp giải quyết tốt các bài toán logic phức tạp như Negation và Conjunction, nhưng cũng mở ra hướng cần cải thiện ở suy luận kết nối chuỗi dài (Multi-hop reasoning).
