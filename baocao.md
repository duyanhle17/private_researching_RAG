# 📊 Báo Cáo: Enhanced GraphRAG với Structure-Aware Components

> **Tích hợp từ SAT Paper (Structure Aware Alignment and Tuning)**

---

## 📌 Tổng Quan

File `enhanced_graphrag.py` là phiên bản nâng cấp của `simple_graphrag.py`, được cải tiến bằng cách áp dụng các kỹ thuật từ **SAT paper** (Structure Aware Alignment and Tuning) để xây dựng Knowledge Graph (KG) hiệu quả hơn.


## SAT có gì

Những điểm nổi bật từ SAT code:
Graph Transformer (graph_transformer.py):

Sử dụng entity embeddings và relation embeddings học được
Positional encoding cho nodes
Multi-head attention trên graph structure
ConvE decoder cho knowledge graph completion
CLIP-style Alignment (model_gt.py):

Align graph embeddings với text embeddings
Sử dụng contrastive loss để học representation chung
Tokenizer riêng cho text (BPE-based)
Data Structure (data_helper.py):

Biểu diễn triples (head, relation, tail)
Edge normalization
One-hot label với label smoothing

---

## 🔄 So Sánh Simple GraphRAG vs Enhanced GraphRAG

| Feature | Simple GraphRAG | Enhanced GraphRAG |
|---------|-----------------|-------------------|
| **Node Embeddings** | Node2Vec (optional, static) | **Graph Transformer** (learnable, structure-aware) |
| **Relations** | Co-occurrence only | **Explicit relations + canonical mapping** |
| **Entity Storage** | Simple NetworkX dict | **ID mapping (mid2id, rel2id, id2text)** như SAT |
| **Text-Graph Bridge** | Không có | **Contrastive alignment (CLIP-style)** |
| **Triple Confidence** | Không có | **Có confidence score cho mỗi triple** |
| **Positional Encoding** | Không có | **Sinusoidal positional encoding cho nodes** |
| **Multi-head Attention** | Không có | **Có (trên graph structure)** |

---

## 🚀 Những Cải Tiến Chính

### 1. Graph Transformer (từ `graph_transformer.py`)

```python
class GraphTransformer(nn.Module):
    """
    Full Graph Transformer encoder for learning structure-aware node embeddings.
    """
```

**Đặc điểm:**
- ✅ Learnable entity & relation embeddings
- ✅ Positional encoding cho nodes (sinusoidal)
- ✅ Multi-head attention trên cấu trúc graph edges
- ✅ Layer normalization và residual connections
- ✅ Dropout để tránh overfitting

**Tham số chính:**
- `num_entities`: Số lượng entities
- `num_relations`: Số lượng relations
- `input_dim`: Kích thước input embedding (default: 128)
- `n_layers`: Số Graph Transformer layers (default: 3)
- `n_heads`: Số attention heads (default: 8)

---

### 2. Text-Graph Alignment Module (CLIP-style, từ `model_gt.py`)

```python
class TextGraphAligner(nn.Module):
    """
    Aligns text and graph embeddings using contrastive learning (CLIP-style).
    """
```

**Cách hoạt động:**
1. **Graph Encoder**: Encode nodes thành embeddings sử dụng Graph Transformer
2. **Text Encoder**: Encode text sử dụng Sentence Transformers + projection layer
3. **Contrastive Loss**: InfoNCE loss để align hai không gian embeddings

**Loss function:**
```
L = (L_graph→text + L_text→graph) / 2
```

---

### 3. Enhanced KG Builder (từ `data_helper.py`)

```python
class EnhancedKGBuilder:
    """
    Enhanced KG builder với:
    1. Better entity extraction
    2. Explicit relation extraction
    3. Triple scoring/confidence
    4. Entity/Relation ID mapping (SAT-style)
    """
```

**Cải tiến extraction:**
- **Entity extraction**: NER + Noun chunks filtering
- **Relation extraction**: Dependency parsing (subject-verb-object patterns)
- **Canonical relation mapping**: Map verbs về relations chuẩn

**Pre-defined relation patterns:**
```python
self.relation_patterns = {
    "treats": ["treat", "cure", "heal", "remedy"],
    "causes": ["cause", "lead to", "result in", "trigger"],
    "prevents": ["prevent", "avoid", "reduce risk"],
    "symptoms_of": ["symptom", "sign", "indicate"],
    "part_of": ["part of", "component", "include"],
    "affects": ["affect", "impact", "influence"],
    "associated_with": ["associate", "relate", "connect", "link"],
    "type_of": ["type of", "kind of", "form of", "is a"],
}
```

---

### 4. Data Format (SAT-compatible)

Enhanced GraphRAG lưu trữ KG data theo format tương thích SAT:

| File | Mô tả | Format |
|------|-------|--------|
| `entity2id.txt` | Entity → ID mapping | `entity_name\tid` |
| `relation2id.txt` | Relation → ID mapping | `relation_name\tid` |
| `id2text.txt` | ID → Entity text | `id\tentity_name` |
| `triples.txt` | KG triples | `head_id\trel_id\ttail_id\tconfidence` |

---

## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Import và Khởi tạo

```python
from enhanced_graphrag import EnhancedGraphRAG

# Khởi tạo với cấu hình mặc định
rag = EnhancedGraphRAG(
    embedding_model_name="all-MiniLM-L6-v2",  # Model cho text embeddings
    use_graph_transformer=True,                # Bật Graph Transformer
    graph_transformer_dim=128,                 # Kích thước embedding
    graph_transformer_layers=3,                # Số layers
    working_dir="./my_graphrag_cache"          # Thư mục lưu cache
)
```

### Bước 2: Thêm Documents

```python
# Chuẩn bị text chunks
chunks = [
    "Basal cell carcinoma (BCC) is the most common type of skin cancer...",
    "Treatment options for BCC include surgical excision...",
    "Fair skin and excessive sun exposure are major risk factors...",
    # ... thêm chunks khác
]

# Thêm vào hệ thống
rag.add_documents(chunks)
```

### Bước 3: Build Knowledge Graph

```python
# Build KG từ chunks
rag.build_kg(add_cooccurrence=True)

# Output:
# INFO - Building KG from 5 chunks...
# INFO - KG built: 15 entities, 8 relations, 23 triples
```

### Bước 4: Build Embeddings và Index

```python
# Build chunk embeddings + FAISS index
rag.build_embeddings(normalize=True, batch_size=32)

# Output:
# INFO - Computing chunk embeddings...
# INFO - FAISS index built with 5 vectors, dim=384
```

### Bước 5: Build Graph Transformer Embeddings

```python
# Compute node embeddings với Graph Transformer
rag.build_graph_transformer(device="cpu")  # hoặc "cuda"

# Output:
# INFO - Building Graph Transformer embeddings...
# INFO - Graph Transformer embeddings computed: torch.Size([15, 128])
```

### Bước 6: Query

```python
# Thực hiện query
result = rag.query(
    query="What are the treatments for skin cancer?",
    top_k=5,           # Số chunks trả về
    alpha=0.7,         # Trọng số semantic (1-alpha cho graph)
    include_kg_facts=True  # Có kèm KG facts không
)

# Kết quả
print(result["context"])      # Context tổng hợp
print(result["chunks"])       # List các chunks retrieved
print(result["kg_facts"])     # List các KG facts liên quan
print(result["retrieval_scores"])  # Scores chi tiết
```

### Bước 7: Save/Load

```python
# Lưu hệ thống
rag.save()

# Load lại sau này
rag_loaded = EnhancedGraphRAG()
rag_loaded.load("./my_graphrag_cache")
```

---

## 🔧 Cấu Hình Nâng Cao

### Custom Relation Patterns

```python
# Thêm relation patterns mới cho domain cụ thể
rag.kg_builder.relation_patterns["diagnoses"] = ["diagnose", "detect", "identify"]
rag.kg_builder.relation_patterns["inhibits"] = ["inhibit", "block", "suppress"]
```

### Điều chỉnh Graph Transformer

```python
# Cấu hình Graph Transformer chi tiết
rag = EnhancedGraphRAG(
    graph_transformer_dim=256,    # Tăng dimension
    graph_transformer_layers=4,   # Nhiều layers hơn
)
```

### Query với weights khác nhau

```python
# Ưu tiên semantic search
result = rag.query("...", alpha=0.9)

# Ưu tiên graph-based search
result = rag.query("...", alpha=0.3)

# Cân bằng
result = rag.query("...", alpha=0.5)
```

---

## 📊 Ví Dụ Output

### Query: "What causes skin cancer?"

```
================== QUERY RESULTS ==================
Query: What causes skin cancer?

Context:
Fair skin and excessive sun exposure are major risk factors 
for developing skin cancer, including BCC and melanoma.

Melanoma is a more aggressive form of skin cancer that can 
metastasize to other organs if not caught early.

[KG Fact] sun exposure causes skin cancer
[KG Fact] fair skin associated_with skin cancer
[KG Fact] uv radiation causes melanoma

KG Facts: ['sun exposure causes skin cancer', 
           'fair skin associated_with skin cancer', 
           'uv radiation causes melanoma']

Retrieval Scores:
  Chunk 2: combined=0.892, semantic=0.856, graph=0.980
  Chunk 3: combined=0.784, semantic=0.812, graph=0.720
  Chunk 4: combined=0.691, semantic=0.723, graph=0.620
===================================================
```

---

## 🎯 Khi Nào Sử Dụng Enhanced GraphRAG?

### ✅ Nên sử dụng khi:
- Dataset có cấu trúc quan hệ rõ ràng (medical, legal, scientific)
- Cần trích xuất facts từ KG
- Muốn combine semantic + structural retrieval
- Có đủ data để học Graph Transformer embeddings

### ❌ Không cần thiết khi:
- Dataset nhỏ, ít entities
- Query đơn giản, không cần KG reasoning
- Không có relations rõ ràng giữa entities

---

## 📚 Tham Khảo

- **SAT Paper**: Structure Aware Alignment and Tuning
- **Source files**:
  - `SAT/aligner/model/graph_transformer.py` - Graph Transformer architecture
  - `SAT/aligner/model/model_gt.py` - CLIP-style alignment
  - `SAT/aligner/model/data_helper.py` - Data processing utilities

---

## 🔜 Các Bước Tiếp Theo (TODO)

- [ ] Thêm training module cho Text-Graph Aligner
- [ ] Tích hợp ConvE decoder cho link prediction
- [ ] Support multi-hop reasoning trên KG
- [ ] Optimize cho large-scale KGs
- [ ] Thêm evaluation metrics (MRR, Hits@K)

---

*Báo cáo được tạo ngày: 31/01/2026*
