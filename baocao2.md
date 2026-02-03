# 📊 Báo Cáo Giai Đoạn 2: Graph Transformer với KG Có Sẵn

> **Mục tiêu**: Dùng KG có sẵn + Graph Transformer để cải thiện QA

---

## 🎯 Tóm Tắt

| Thông tin | Giá trị |
|-----------|---------|
| **Nguồn dữ liệu** | FB15k-237N (KG có sẵn của SAT) |
| **Cách tạo KG** | KHÔNG tự build, dùng có sẵn |
| **Số entities** | 14,541 |
| **Số relations** | 237 |
| **Số triples** | 87,282 |
| **Kết quả QA** | 95.3% accuracy |

---

## 🔄 Khác Biệt Với Giai Đoạn 1

| | Giai đoạn 1 | Giai đoạn 2 |
|---|---|---|
| **KG** | Tự build | **Có sẵn** |
| **Extract entity** | Có (spaCy) | **Không cần** |
| **Extract relation** | Có (parsing) | **Không cần** |
| **Dùng KG khi retrieve** | ✅ Có | ❌ **Không** |

---

## 🔧 Những Gì Đã Làm

### 1. Dùng KG Có Sẵn (FB15k-237N)

**Đã làm gì:**
- Download dataset FB15k-237N từ SAT paper
- Dataset này đã có sẵn:
  - 14,541 thực thể với mô tả văn bản
  - 237 loại quan hệ
  - 87,282 bộ ba (triples)

**Mục đích:**
- Không cần tự extract entity/relation
- Dữ liệu chất lượng cao hơn

---

### 2. Graph Transformer - Học Embeddings

**Đã làm gì:**
- Cho Graph Transformer "nhìn" vào cấu trúc đồ thị (ai nối với ai)
- Học ra vector đặc trưng cho mỗi thực thể

**⚠️ Graph Transformer KHÔNG làm:**
- ❌ Không đọc văn bản
- ❌ Không nhận diện thực thể
- ❌ Không rút trích quan hệ
- ❌ Không xây KG mới

**Nó CHỈ làm:** Học embeddings từ KG đã có sẵn

---

### 3. Cải Thiện Prompt

**Prompt cũ:**
```
Nếu không có trong context, trả lời "not stated"
```
→ LLM quá nghiêm khắc, từ chối nhiều câu WHY

**Prompt mới:**
```
Với câu hỏi WHY/HOW: Suy luận từ context
Chỉ nói "not stated" nếu HOÀN TOÀN không có thông tin
```
→ LLM được khuyến khích suy luận

**Kết quả:** "Not stated" giảm từ 7 → 1 câu

---

## ⚠️ Vấn Đề Quan Trọng: ALPHA = 1.0

**Công thức hybrid:**
```
điểm = α × semantic + (1-α) × graph
```

**Khi α = 1.0:**
```
điểm = 1.0 × semantic + 0 × graph = CHỈ SEMANTIC
```

**Nghĩa là:**
- Kết quả 95.3% **KHÔNG DÙNG KG**
- Chỉ dùng semantic search thuần (RAG cơ bản)
- Graph Transformer embeddings đã tính nhưng **KHÔNG ĐƯỢC DÙNG**

---

## ❓ Tại Sao Không Dùng Graph Embeddings?

**Đã thử với α = 0.6 (hybrid):** Chỉ đạt 45.3% (tệ hơn nhiều!)

**Nguyên nhân:**
- Text embeddings học **ngữ nghĩa** (government ≈ state)
- Graph embeddings học **vị trí trong đồ thị** (ai gần ai)
- **2 không gian không cùng hệ quy chiếu** → kết hợp làm hỏng kết quả

**Giải pháp:** Cần train **Text-Graph Alignment** để 2 không gian khớp nhau

---

## 📂 Files Được Tạo

```
sat_kg_data/
├── text_embeddings.npy      # Vector văn bản (14541, 384)
├── node_embeddings.pt       # Vector từ Graph Transformer (14541, 128)
└── graph_data.pt            # Dữ liệu đồ thị
```

---

## ✅ Kết Luận Giai Đoạn 2

**Đã hoàn thành:**
- ✅ Fix Graph Transformer (không còn segfault)
- ✅ Tính embeddings cho 14,541 entities trong 0.27 giây
- ✅ Cải thiện prompt (giảm "not stated")
- ✅ Đạt 95.3% accuracy

**Thực tế:**
- 95.3% là từ **RAG thuần** (semantic search only)
- **KHÔNG dùng KG** khi retrieve (α = 1.0)
- Graph Transformer embeddings chưa được tận dụng

**Hướng tiếp theo:**
- [ ] Train Text-Graph Alignment (CLIP-style)
- [ ] Sau khi align, thử lại hybrid retrieval

---

## 📊 So Sánh Cuối Cùng

| | Giai đoạn 1 | Giai đoạn 2 |
|---|---|---|
| **KG** | Tự build | Có sẵn |
| **Số entities** | 5,088 | 14,541 |
| **Có dùng KG** | ✅ Có | ❌ Không |
| **Thực chất** | Hybrid RAG | Pure RAG |
| **Accuracy** | 67.2% | **95.3%** |

**Tại sao giai đoạn 2 cao hơn?**
- Không phải vì Graph Transformer
- Mà vì FB15k-237N có **mô tả văn bản đầy đủ** cho mỗi entity
- Semantic search tìm được context chính xác hơn

---

*Cập nhật: 03/02/2026*
