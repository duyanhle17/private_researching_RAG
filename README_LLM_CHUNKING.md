# Tại sao lại có chunk_words và overlap_words?

## 1. Giới hạn trí nhớ của LLM (Context Window)
- Mỗi mô hình LLM có giới hạn "trí nhớ ngắn hạn" nhất định (như Llama 3 70B trên NIM thường xử lý tốt nhất khoảng 4000-8000 tokens).
- Nếu ta ném CẢ một cuốn sách (100.000 từ) vào cùng lúc, AI sẽ bị "tràn bộ nhớ", báo lỗi hoặc trả lời linh tinh.

## 2. Giới hạn cấu trúc Prompt & Rút trích (Information Density)
- Nếu đưa một câu siêu dài (chứa hàng ngàn dữ kiện), LLM sẽ có xu hướng "lười" và bỏ sót thông tin, chỉ lấy được một vài Entities/Relations chính.
- Bằng cách chia nhỏ đoạn văn (~200 từ/đoạn), ta ép LLM phải soi thật kỹ từng chi tiết trong không gian hẹp, nhờ đó nó sẽ quét cạn kiệt (exhaustive extraction) mọi kiến thức KG.

## 3. Tại sao lại cần Overlap (Đoạn nối)?
- Giả sử câu: "Anh Duy sống ở London. Anh ta là bác sĩ."
- Nếu ta cắt "Anh Duy sống ở London." ở cuối chunk 1, và "Anh ta là bác sĩ." ở đầu chunk 2. LLM đọc chunk 2 sẽ không biết "Anh ta" là ai.
- Cắt với `overlap=20` (nhường lại 20 từ ghép vào đoạn sau), ta đảm bảo các câu nối không bị đứt gãy mạch ý nghĩa, duy trì được ngữ cảnh.
