Trong những vòng Evaluation đầu tiên (lúc bạn bắt đầu hỏi tôi ở Epoch số 1): Tốc độ chạy test bị nghẹt ở mức **37 giây / 1 lượt (it)** và dự kiến cắn mất gần 30~50 phút cho một vòng Eval vì lỗi **Memory Leak**.

Sở dĩ ở đợt Epoch 4 này Evaluation kết thúc chỉ mât đúng **41 giây** (tức là tốc độ xử lý **0.5 giây / 1 lượt**) thần tốc như vậy hoàn toàn là công sức của câu thần chú `with torch.no_grad():` mà tôi đã thêm vào code cho bạn ở lượt sửa trước!

Khi thêm `torch.no_grad()`, mô hình Python không còn phải giữ lại lịch sử "Cây gradient đạo hàm" (Gradient Tree) của hàng chục triệu tham số toán học trong bộ nhớ RAM ở quá trình làm thi nữa. Vì tính chất `evaluate()` phải làm phép quét so khớp all-to-all (so Node 1 với 3200 Node kia xem ai hợp nhất) -> Máy tính chỉ việc chạy Forward (tính toán tiến) và xuất ra kết quả vứt luôn, RAM không bị đầy, Mac không phải chuyển sang cơ chế lấy SSD làm bộ nhớ mở rộng (Swap Memory). Máy bung tỏa 100% công lực C2 để xử lý.
