# 📊 Báo Cáo: Enhanced GraphRAG - Từ Simple GraphRAG đến SAT-Inspired

> **Tích hợp kỹ thuật từ SAT Paper (Structure Aware Alignment and Tuning)**

---

## 📌 Mục Tiêu

Cải tiến hệ thống **Simple GraphRAG** bằng cách áp dụng các kỹ thuật từ **SAT paper** để xây dựng Knowledge Graph (KG) hiệu quả hơn cho bài toán Question Answering.

---

## 🔍 Phân Tích Code Gốc

### Simple GraphRAG (`simple_graphrag.py`)

**Kiến trúc cơ bản:**
```
Text → Chunking → NER → Co-occurrence Graph → Node2Vec → FAISS → Query
```

**Đặc điểm:**
- ✅ Đơn giản, dễ hiểu
- ✅ Sử dụng spaCy NER để trích xuất entities
- ✅ Xây dựng graph dựa trên **co-occurrence** (các entities xuất hiện cùng chunk)
- ❌ **Không có explicit relations** - chỉ có edges vô hướng
- ❌ **Node2Vec embeddings tĩnh** - không học được từ structure
- ❌ **Không có entity/relation ID mapping** - khó scale

**Code snippet (entity extraction):**
```python
# Simple GraphRAG chỉ dùng NER cơ bản
doc = self.nlp(text)
entities = {ent.text.lower() for ent in doc.ents}
```

---

## 📚 SAT Paper - Những Gì Đã Học

### 1. Graph Transformer (`SAT/aligner/model/graph_transformer.py`)

**Ý tưởng chính:**
- Thay vì dùng Node2Vec (random walk), sử dụng **Transformer architecture** trên graph
- **Learnable embeddings** cho cả entities và relations
- **Multi-head attention** tính toán trên edges, không phải sequence

**Code từ SAT:**
```python
class GTLayer(nn.Module):
    """Graph Transformer Layer với sparse attention"""
    def forward(self, ent_emb, rel_emb, mr):
        # mr = (edge_indices, edge_types)
        # Tính attention trên cạnh của graph
```

### 2. CLIP-style Alignment (`SAT/aligner/model/model_gt.py`)

**Ý tưởng chính:**
- Align **graph embeddings** với **text embeddings** 
- Sử dụng **contrastive loss** (InfoNCE) để học representation chung
- Cho phép query bằng text nhưng tìm kiếm trên graph space

**Loss function:**
```
L = (L_graph→text + L_text→graph) / 2
```

### 3. Data Structure (`SAT/aligner/model/data_helper.py`)

**Ý tưởng chính:**
- **Entity2ID mapping**: `{entity_name: id}` - cho phép xử lý số học
- **Relation2ID mapping**: `{relation_name: id}` - standardize relations
- **Triples format**: `(head_id, relation_id, tail_id, confidence)`
- **One-hot labels với label smoothing** cho training

---

## 🚀 Enhanced GraphRAG - Những Gì Đã Áp Dụng

### So sánh tổng quan

| Feature | Simple GraphRAG | Enhanced GraphRAG | Nguồn từ SAT |
|---------|-----------------|-------------------|--------------|
| **Node Embeddings** | Node2Vec (static) | Graph Transformer (learnable) | `graph_transformer.py` |
| **Relations** | Co-occurrence only | Explicit relations + canonical mapping | `data_helper.py` |
| **Entity Storage** | NetworkX dict | ID mapping (entity2id, relation2id) | `data_helper.py` |
| **Text-Graph Bridge** | Không có | Contrastive alignment (CLIP-style) | `model_gt.py` |
| **Triple Format** | `(e1, e2)` | `(head, rel, tail, confidence)` | `data_helper.py` |
| **Positional Encoding** | Không có | Sinusoidal encoding | `graph_transformer.py` |
| **Entity Matching** | Exact match | Fuzzy matching + normalization | Cải tiến riêng |

---

### Cải tiến 1: Graph Transformer

**Từ SAT:**
```python
# SAT's GTLayer
class GTLayer(nn.Module):
    def __init__(self, args, use_norm=True):
        self.args = args
        self.use_norm = use_norm
        self.lin_Q = nn.Linear(args.emb_dim, args.emb_dim)
        self.lin_K = nn.Linear(args.emb_dim, args.emb_dim)
        self.lin_V = nn.Linear(args.emb_dim, args.emb_dim)
```

**Enhanced GraphRAG áp dụng:**
```python
class GraphTransformerLayer(nn.Module):
    """Inspired by SAT's GTLayer"""
    def __init__(self, d_model: int, n_heads: int, use_norm: bool = True):
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, node_embeds, edge_index):
        # Multi-head attention trên graph edges
        q = self.q_proj(src_embeds)
        k = self.k_proj(tgt_embeds)
        v = self.v_proj(tgt_embeds)
        att = torch.einsum("ehd,ehd->eh", q, k) / (self.head_dim ** 0.5)
```

**Lưu ý thực tế:** Graph Transformer bị **segfault** với KG lớn (>5000 entities), nên đã disable trong thực nghiệm.

---

### Cải tiến 2: Entity/Relation ID Mapping

**Từ SAT:**
```python
# SAT lưu trữ KG dưới dạng ID
mid2id = {"entity_name": 0, ...}
rel2id = {"relation_name": 0, ...}
id2text = {0: "entity_name", ...}
```

**Enhanced GraphRAG áp dụng:**
```python
class EnhancedKGBuilder:
    def __init__(self):
        self.entity2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        
    def _get_or_create_entity_id(self, entity: str) -> int:
        if entity not in self.entity2id:
            eid = len(self.entity2id)
            self.entity2id[entity] = eid
            self.id2entity[eid] = entity
        return self.entity2id[entity]
```

**Output files:**
- `entity2id.pkl` - Mapping entity → ID
- `relation2id.pkl` - Mapping relation → ID  
- `triples.json` - `[(head_id, rel_id, tail_id, confidence), ...]`

---

### Cải tiến 3: Explicit Relation Extraction

**Simple GraphRAG (chỉ co-occurrence):**
```python
# Chỉ tạo edge giữa entities cùng chunk
for e1, e2 in combinations(entities, 2):
    self.G.add_edge(e1, e2)  # Không có relation type
```

**Enhanced GraphRAG (explicit relations via dependency parsing):**
```python
# Trích xuất relation từ dependency tree
for token in doc:
    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
        subject = token.text
        verb = token.head.lemma_  # Relation
        # Tìm object
        for child in token.head.children:
            if child.dep_ in ("dobj", "attr", "prep"):
                object = child.text
                # → Triple: (subject, verb, object)
```

**Canonical relation mapping:**
```python
relation_patterns = {
    "treats": ["treat", "cure", "heal", "remedy"],
    "causes": ["cause", "lead to", "result in"],
    "part_of": ["part of", "component", "include"],
    "type_of": ["type of", "kind of", "is a"],
    "located_in": ["located", "in", "at", "based"],
}
```

---

### Cải tiến 4: Fuzzy Entity Matching

**Vấn đề phát hiện:**
- Query: `"satellite awards"` 
- KG entity: `"the satellite awards"`
- → Không match! → Trả về 0 KG facts

**Giải pháp (không từ SAT, cải tiến riêng):**
```python
def _normalize_query_entity(self, entity: str) -> str:
    """Remove articles (the, a, an) from entity"""
    entity = entity.lower().strip()
    for article in ["the ", "a ", "an "]:
        if entity.startswith(article):
            entity = entity[len(article):]
    return entity

def _fuzzy_entity_match(self, query_entity: str, kg_entities: Set[str]) -> Set[str]:
    """Find matching entities with fuzzy logic"""
    matches = set()
    q_norm = self._normalize_query_entity(query_entity)
    
    for kg_ent in kg_entities:
        kg_norm = self._normalize_query_entity(kg_ent)
        # Exact match after normalization
        if q_norm == kg_norm:
            matches.add(kg_ent)
        # Substring match
        elif q_norm in kg_norm or kg_norm in q_norm:
            matches.add(kg_ent)
        # Word overlap
        else:
            q_words = set(q_norm.split())
            kg_words = set(kg_norm.split())
            overlap = len(q_words & kg_words) / max(len(q_words), 1)
            if overlap >= 0.5:
                matches.add(kg_ent)
    return matches
```

**Kết quả:** Giảm từ **31 câu** xuống **20 câu** có 0 KG facts.

---

## 📊 Kết Quả Thực Nghiệm

### Dataset
- **Nguồn:** `data/medical_custom.json`
- **Số câu hỏi:** 64

### Build KG Statistics
```
📊 Build Complete!
   - Chunks: 173
   - Entities: 5088
   - Relations: 8 
   - Triples: 8452
```

### QA Evaluation Results

| Metric | Kết quả |
|--------|---------|
| **Strict Match** (GT substring in Answer) | 8/64 (12.5%) |
| **Fuzzy Match** (60% word overlap) | 43/64 (67.2%) |
| **"Not stated in text"** | 5/64 (7.8%) |

### Phân tích chi tiết

**Câu đúng (fuzzy match):**
```
Q: What institutional type is UCF?
A: "a public research university"
GT: "UCF is a public research university."
→ ✅ Đúng nội dung, chỉ khác format
```

**Câu sai (không tìm được context):**
```
Q: Why is the University of Essex called one of the 'original plate glass universities'?
A: "not stated in the text"
GT: "Because it is included among the group of universities..."
→ ❌ Context không chứa thông tin này
```

---

## 🔧 Cấu Hình Đã Sử Dụng

```python
# build_enhanced_kg_cache.py
rag = EnhancedGraphRAG(
    embedding_model_name="all-MiniLM-L6-v2",
    use_graph_transformer=False,  # Disabled do segfault
    graph_transformer_dim=128,
    graph_transformer_layers=3,
)

# run_enhanced_baseline.py
TOP_K = 10      # Số chunks retrieved
ALPHA = 0.6     # 60% semantic, 40% graph
```

---

## 📁 Files Được Tạo

| File | Mô tả |
|------|-------|
| `enhanced_graphrag.py` | Main module với tất cả components |
| `build_enhanced_kg_cache.py` | Script build KG cache |
| `run_enhanced_baseline.py` | Script chạy QA evaluation |
| `enhanced_sat_data/` | Folder chứa KG cache |
| `enhanced_results.json` | Kết quả QA (64 câu) |

### Cấu trúc `enhanced_sat_data/`:
```
enhanced_sat_data/
├── chunks.json          # List các text chunks
├── embeddings.npy       # Chunk embeddings (173, 384)
├── faiss.index          # FAISS index cho search
├── kg.pkl               # NetworkX graph
├── chunk_entities.pkl   # Entity mapping per chunk
├── entity2id.pkl        # Entity → ID (SAT-style)
├── relation2id.pkl      # Relation → ID (SAT-style)
├── triples.json         # [(head, rel, tail, conf), ...]
└── meta.json            # Metadata
```

---

## 🎯 Kết Luận

### Những gì đã học từ SAT:
1. ✅ **Graph Transformer architecture** - Multi-head attention trên graph
2. ✅ **Entity/Relation ID mapping** - SAT-style data format
3. ✅ **CLIP-style alignment concept** - Text-Graph bridge
4. ✅ **Triple with confidence** - Structured KG format

### Những gì đã cải tiến riêng:
1. ✅ **Fuzzy entity matching** - Giải quyết normalization issues
2. ✅ **Explicit relation extraction** - Dependency parsing
3. ✅ **Canonical relation mapping** - Standardize verbs → relations

### Hạn chế:
1. ❌ **Graph Transformer disabled** - Segfault với large KG
2. ❌ **Chưa train Text-Graph Aligner** - Chỉ dùng pre-computed embeddings
3. ❌ **5/64 câu không tìm được context** - Cần cải thiện chunking

### Hướng phát triển:
- [ ] Fix Graph Transformer cho large-scale KG (batching/sampling)
- [ ] Train Text-Graph Aligner với contrastive loss
- [ ] Thêm multi-hop reasoning
- [ ] Cải thiện relation extraction với LLM

---

*Báo cáo cập nhật: 02/02/2026*
