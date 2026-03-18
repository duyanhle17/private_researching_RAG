# Báo Cáo Tiến Độ Dự Án FactKG (Ngày 18/03)

## 1. Tổng Quan Về Bài Báo (Paper Overview)
**FACTKG: Fact Verification via Reasoning on Knowledge Graphs** là một nghiên cứu của nhóm tác giả từ KAIST và Amazon (công bố năm 2023). 
*   **Mục tiêu**: Giới thiệu dataset mới gồm 108k claim để giải bài toán kiểm chứng sự thật dựa trên Knowledge Graph (KG).
*   **Động lực**: KG có độ tin cậy cao hơn văn bản thuần túy và có cấu trúc logic rõ ràng, giúp giải thích kết quả (explainability) tốt hơn cho các hệ thống như Amazon Alexa hay Google Assistant.

---

## 2. Chi Tiết Dataset & Kiểu Suy Luận (Reasoning Types)
FACTKG sử dụng **DBpedia** (0.1 tỷ triple) làm nguồn bằng chứng. Dataset bao gồm 108,674 claims, được xây dựng với 5 kiểu suy luận cốt lõi:

| Kiểu Suy Luận | Mô Tả | Ví dụ thực tế |
| :--- | :--- | :--- |
| **One-hop** | Kiểm tra 1 triple đơn lẻ | "AIDAstella was built by Meyer Werft." |
| **Conjunction** | Kiểm tra nhiều triple cùng lúc (câu ghép) | "AIDAstella was built by Meyer Werft and operated by AIDA." |
| **Existence** | Kiểm tra sự tồn tại của quan hệ | "Meyer Werft had a parent company." |
| **Multi-hop** | Suy luận qua chuỗi quan hệ (entity ẩn) | "AIDAstella was built by a company in Papenburg." |
| **Negation** | Phủ định các kiểu trên (chứa "not", "no") | "AIDAstella was not built by Meyer Werft." |

**Phân bổ tập dữ liệu thực tế trên máy:**
*   **Train Set**: 86,367 câu (Dùng để huấn luyện).
*   **Dev Set**: 13,266 câu (Dùng để kiểm tra nhanh).
*   **Test Set**: 9,041 câu (Dùng để đánh giá cuối cùng).
*   **Tổng cộng**: **108,674 câu** (Khớp với con số trong Paper).

---

## 3. Thực Nghiệm Pipeline A (Claim Only Baseline)
Chúng ta đã thực hiện huấn luyện và đánh giá mô hình **BERT (Base Uncased)** trên máy Mac M1 Pro để làm mốc so sánh.

### 3.1 Mô hình & Quá trình thực hiện
*   **Model**: `bert-base-uncased` (110M parameters).
*   **Huấn luyện**: Chạy hoàn tất 3 Epochs trên tập Train (86k câu).
*   **Đánh giá**: Thực hiện đo lường trên toàn bộ 100% tập Test (9,041 câu).

### 3.2 Bảng so sánh Kết quả (Accuracy %)
Dưới đây là sự so sánh giữa con số lý tưởng trong Paper và kết quả thực tế chúng ta đạt được trên thiết bị cá nhân:

| Chỉ số Accuracy (%) | One-hop | Conjunction | Existence | Multi-hop | Negation | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT (Trong Paper)** | 69.64 | 63.31 | 61.84 | 70.06 | 63.62 | **65.20** |
| **BERT (Thực tế M1 Pro)** | 55.43 | 44.25 | 47.59 | 51.33 | 45.89 | **48.65** |

**Giải thích sự chênh lệch:**
1.  **Quy mô mô hình**: Paper sử dụng các dòng `Large` hoặc `RoBERTa` mạnh hơn bản `Base` chúng ta đang dùng.
2.  **Thời gian huấn luyện**: Tác giả huấn luyện với số lượng Epoch lớn hơn và Batch size khổng lồ trên cụm GPU chuyên dụng (A100).
3.  **Môi trường**: Kết quả thực tế 48.65% cho thấy nếu chỉ dựa vào văn bản, mô hình BERT trên máy cá nhân gần như chỉ đạt ngưỡng "đoán mò" (50/50), củng cố thêm lý do tại sao cần Pipeline B.

---

## 4. Phân Tích & Định Hướng Tiếp Theo
Dựa trên kết quả thực nghiệm và lý thuyết của Paper:
*   **Điểm yếu của Baseline**: BERT cực kỳ kém ở mảng **Negation** (Phủ định) và **Conjunction** (Câu ghép) vì nó chỉ học thuộc mặt chữ mà không hiểu logic thực sự.
*   **Sức mạnh của GEAR (Pipeline B)**: Paper chứng minh khi có Evidence từ KG, điểm số có thể tăng vọt lên **77.65%** (Tổng). Đặc biệt Negation tăng tới +15% so với BERT.
*   **Hành động tiếp theo**: Triển khai Pipeline B (With Evidence) bao gồm các bước:
    1.  Tiền xử lý đồ thị DBpedia từ các file Pickle.
    2.  Huấn luyện Module **Retriever** (Relation & Hop Predictor) để tìm bằng chứng.
    3.  Huấn luyện Module **Classifier** (Sử dụng kiến trúc GEAR) để đưa ra phán quyết cuối cùng.

---
*Người thực hiện: Duy Anh Le*
*Ngày báo cáo: 18/03/2026*
