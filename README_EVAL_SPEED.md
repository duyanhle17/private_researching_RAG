# Làm thế nào để giải quyết vấn đề Evaluation quá chậm?

Trong `train_medical.py`, hàm `evaluate()` mất tới gần 30 phút vì tính chất "all-to-all matching" của mô hình CLIP trên hàng ngàn câu text dài. Vì quá trình đánh giá đang cực kỳ thắt cổ chai, tôi có 3 phương pháp để giải quyết việc này:

1. **Phương án 1 (Chỉ đánh giá ở Epoch cuối cùng):**
   Thay vì chạy Test Accuracy sau *mỗi* Epoch, ta có thể cài đặt cho nó chỉ chạy Validation 1 lần duy nhất ở Epoch cuối (Epoch số 5). Điều này cứu mạng bạn khỏi 2 tiếng chờ đợi dư thừa. Các file checkpint ở từng Epoch vẫn được lưu ra ổ đĩa bình thường.
   
2. **Phương án 2 (Thay đổi Tần suất Đánh giá):**
   Thay đổi dòng code thành `if epoch % 5 == 0:` (chỉ chạy đánh giá mỗi chu kỳ 5 epochs).
   
3. **Phương án 3 (Tắt hẳn tính năng Đánh giá / Evaluation):**
   Nếu bạn thật sự không quan tâm Điểm số test score lúc huấn luyện mà chỉ muốn mô hình nhúng và cất file weights để lát nữa kéo qua file `run_sat_baseline.py` xử dùng luôn, ta có thể bỏ hẳn lệnh gọi `evaluate()` trong vòng lặp và cứ mặc định lấy Epoch cuối làm file model chuẩn nhất.
