# Báo Cáo Tổng Hợp: Tài Liệu Paper & Tiến Độ Dự Án FactKG (18/03/2026)

## PHẦN I: PHÂN TÍCH CHI TIẾT PAPER (FACTKG)

### 1. Giới Thiệu & Động Lực (Overview & Motivation)
**FACTKG: Fact Verification via Reasoning on Knowledge Graphs** là công trình của nhóm nghiên cứu KAIST & Amazon (2023).
*   **Vấn đề**: Các hệ thống AI truyền thống (LLMs) thường bị "ảo giác" hoặc thiếu tính giải thích khi kiểm chứng sự thật.
*   **Giải pháp**: Sử dụng **Knowledge Graph (KG)** làm nguồn bằng chứng cứng. KG có độ tin cậy cao, cấu trúc logic rõ ràng (Node-Edge), dễ dàng truy xuất nguồn gốc của thông tin.

### 2. Bộ Dữ Liệu FACTKG (Dataset Construction)
Bộ dữ liệu gồm **108,675 claims** được xây dựng từ WebNLG 2020 trên nền tảng **DBpedia** (hơn 100 triệu bộ ba tri thức - triples).
*   **Kiến trúc 5 kiểu suy luận**:
    1.  **One-hop**: Kiểm tra 1 triple đơn giản.
    2.  **Conjunction**: Kiểm tra nhiều triple cùng lúc.
    3.  **Existence**: Kiểm tra sự tồn tại của quan hệ.
    4.  **Multi-hop**: Suy luận qua chuỗi quan hệ phức tạp (đặc biệt quan trọng cho GraphRAG).
    5.  **Negation**: Phủ định thông tin (thử thách lớn nhất cho AI).
*   **Phong cách ngôn ngữ**: Đa dạng với văn nói (Colloquial - sinh ra bởi FLAN-T5) và văn viết (Written).
*   **Kỹ thuật tạo dữ liệu giả (REFUTED)**: Sử dụng Entity Substitution và Relation Substitution kết hợp lọc dữ liệu Adversarial (AFLITE).

### 3. Phương Pháp Tiếp Cận (Baseline Architecture)
Tác giả chia thí nghiệm thành 2 nhánh chính để so sánh sức mạnh của tri thức có cấu trúc:
*   **Nhánh Claim Only (BERT/BlueBERT/Flan-T5)**: Mô hình chỉ "đọc chay" văn bản nhận định và dự đoán Đúng/Sai dựa trên kiến thức tiền huấn luyện. Đây là bài toán Phân loại văn bản nhị phân thông thường.
*   **Nhánh With Evidence (Mô hình GEAR)**: Đây là trái tim của bài báo. Hệ thống lấy câu Claim làm query, sau đó:
    1.  Dự đoán Quan hệ & Số bước (Hop) cần tìm trên KG.
    2.  Truy xuất các "đường dẫn bằng chứng" (evidence paths).
    3.  Kết hợp văn bản + bằng chứng đồ thị để đưa ra phán quyết cuối cùng.

---

## PHẦN II: TIẾN ĐỘ THỰC HIỆN THỰC TẾ (WORK PROGRESS)

### 1. Môi Trường & Thiết Bị (Environment)
*   **Thiết bị**: MacBook Pro M1 Pro (16GB RAM).
*   **Cấu hình tối ưu**: Sử dụng `mps` cho GPU và giảm Batch size xuống 16 để tránh lỗi bộ nhớ (OOM).

### 2. Kết Quả Huấn Luyện Pipeline A (BERT Baseline)
Chúng ta đã hoàn thành việc tái lập thí nghiệm Baseline đầu tiên của paper:
*   **Mô hình**: `bert-base-uncased`.
*   **Quá trình**: Fine-tuning supervised learning trên **86,367 câu** (tập Train) trong 3 Epochs.
*   **Dữ liệu đánh giá**: Sử dụng toàn bộ **9,041 câu** (100% tập Test) của tác giả.

### 3. Đối Chiếu Kết Quả (Accuracy Comparison)

| Loại Suy Luận | Kết quả Paper (BERT) | Kết quả thực tế (M1 Pro) | GEAR (Có Evidence - Paper) |
| :--- | :---: | :---: | :---: |
| One-hop | 69.64% | **55.43%** | 83.23% |
| Conjunction | 63.31% | **44.25%** | 77.68% |
| Existence | 61.84% | **47.59%** | 81.61% |
| Multi-hop | 70.06% | **51.33%** | 68.84% |
| Negation | 63.62% | **45.89%** | 79.41% |
| **TỔNG CỘNG** | **65.20%** | **48.65%** | **77.65%** |

**Phân tích sự khác biệt**:
*   Con số thực tế **48.65%** đạt được cho thấy khi dùng mô hình bản `Base` trên văn bản thô, AI gần như phải đoán bừa (xấp xỉ 50/50).
*   Sự chênh lệch giữa thực tế và Paper (65%) giải thích bởi cấu hình phần cứng hạng nặng và các mô hình bản `Large` mà tác giả sử dụng. Tuy nhiên, xu hướng các loại suy luận khó (Negation/Conjunction) có điểm thấp nhất là tương đồng.

---

## PHẦN III: KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

1.  **Mục tiêu**: Nâng cấp kết quả từ **48% (Pipeline A)** lên mức **>70% (Pipeline B)** bằng cách áp dụng mô hình GEAR.
2.  **Các bước cụ thể**:
    *   **Bước 1**: Xử lý tiền dữ liệu (Preprocess) sang định dạng JSON cho thám tử AI (Retriever).
    *   **Bước 2**: Huấn luyện bộ dự đoán Quan hệ và Số bước đi (Relation & Hop Predictor).
    *   **Bước 3**: Tích hợp đường dẫn đồ thị từ DBpedia để chạy Classifier cuối cùng.
3.  **Tầm quan trọng**: Bước này sẽ chứng minh rõ rệt giá trị của **GraphRAG**: AI có "bằng chứng" luôn thông minh hơn AI chỉ "nói suông".

---
*Người thực hiện: Duy Anh Le*
*Ngày cập nhật: 18/03/2026*
