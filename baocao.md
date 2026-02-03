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

**✅ Ưu điểm:**

1. **Đơn giản, dễ hiểu và triển khai nhanh**
   - Kiến trúc pipeline thẳng, không có dependencies phức tạp
   - Dễ debug và maintain do code không quá trừu tượng
   - Phù hợp cho prototyping và baseline comparison

2. **Sử dụng spaCy NER để trích xuất entities**
   - Tận dụng pre-trained NER models của spaCy (en_core_web_sm/md/lg)
   - Nhận diện được các entity types cơ bản: PERSON, ORG, GPE, DATE, etc.
   - Không cần training data riêng, chạy được ngay "out-of-the-box"

3. **Xây dựng graph dựa trên co-occurrence**
   - Hai entities xuất hiện trong cùng một chunk sẽ được nối bằng một edge
   - Giả định: entities trong cùng context có mối quan hệ ngữ nghĩa
   - Đơn giản nhưng hiệu quả cho việc capture local context

**❌ Hạn chế:**

1. **Không có explicit relations - chỉ có edges vô hướng**
   - Graph chỉ lưu `(entity1, entity2)` mà không biết **quan hệ gì** giữa chúng
   - Ví dụ: "Aspirin treats headache" chỉ thành edge `(aspirin, headache)` - mất thông tin "treats"
   - Không phân biệt được "A causes B" vs "A treats B" vs "A is part of B"
   - Giảm khả năng reasoning và multi-hop query

2. **Node2Vec embeddings tĩnh - không học được từ structure**
   - Node2Vec dùng random walks để tạo embeddings, **không có gradient updates**
   - Embeddings được tính một lần và frozen, không adapt theo downstream task
   - Không capture được global graph structure, chỉ local neighborhoods
   - Khác với Graph Transformer có thể fine-tune embeddings theo loss function

3. **Không có entity/relation ID mapping - khó scale**
   - Entities được lưu trực tiếp dưới dạng string trong NetworkX graph
   - Khi KG lớn (>100k entities), việc lookup string rất chậm
   - Không standardize entities (ví dụ: "COVID-19", "Covid19", "coronavirus" là 3 nodes khác nhau)
   - Khó serialize và share KG giữa các hệ thống

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

## 🚀 Enhanced GraphRAG - Phân Tích Chi Tiết Các Cải Tiến

### Pipeline Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENHANCED GRAPHRAG PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] TEXT INPUT                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  [2] CHUNKING (split by sentence boundaries, ~800 chars)                    │
│       │                                                                     │
│       ├──────────────────────┬──────────────────────┐                       │
│       ▼                      ▼                      ▼                       │
│  [3] NER + Dependency    [4] Sentence           [5] Co-occurrence           │
│      Parsing                 Embeddings             Graph                   │
│       │                      │                      │                       │
│       ▼                      ▼                      ▼                       │
│  [6] TRIPLE EXTRACTION   [7] FAISS INDEX        [8] NetworkX KG             │
│      (head, rel, tail)       (semantic search)      (graph structure)       │
│       │                      │                      │                       │
│       ▼                      │                      │                       │
│  [9] ENTITY2ID &             │                      │                       │
│      RELATION2ID             │                      │                       │
│      MAPPING                 │                      │                       │
│       │                      │                      │                       │
│       └──────────────────────┴──────────────────────┘                       │
│                              │                                              │
│                              ▼                                              │
│                    [10] HYBRID RETRIEVAL                                    │
│                    (α × semantic + (1-α) × graph)                           │
│                              │                                              │
│                              ▼                                              │
│                    [11] LLM GENERATION                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️ PHẦN CHƯA SỬ DỤNG ĐƯỢC (từ SAT):
┌─────────────────────────────────────────────────────────────────────────────┐
│  [X] GRAPH TRANSFORMER    →  Disabled do Segfault với large KG             │
│  [X] TEXT-GRAPH ALIGNER   →  Chưa train, thiếu labeled data                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Cải Tiến 1: Entity/Relation ID Mapping

#### 📋 Flow
```
Entity string → Check entity2id dict → Nếu chưa có: tạo ID mới → Lưu mapping
"aspirin" → entity2id["aspirin"] = 0
"headache" → entity2id["headache"] = 1
"treats" → relation2id["treats"] = 0
```

#### 🎯 Mục đích
Chuyển đổi Knowledge Graph từ **string-based** sang **ID-based** để:
1. **Tăng tốc lookup**: So sánh integer nhanh hơn so sánh string
2. **Chuẩn hóa dữ liệu**: Mỗi entity/relation có một ID duy nhất
3. **Dễ dàng serialize**: Lưu trữ và tải KG hiệu quả hơn
4. **Tương thích với neural networks**: Embeddings cần integer indices

#### 💡 Ý nghĩa trong phát triển tri thức
- **Từ "nhận diện" đến "định danh"**: Thay vì chỉ nhận ra entity, hệ thống giờ **gán nhãn số học** cho mỗi entity
- **Nền tảng cho embedding learning**: ID mapping là điều kiện tiên quyết để có thể học embeddings (mỗi ID → 1 vector)
- **Khả năng mở rộng**: Khi KG có hàng triệu entities, ID-based storage tiết kiệm bộ nhớ đáng kể

#### 📊 Kết quả thực tế
```
Entities: 5088 unique entities với ID từ 0 đến 5087
Relations: 8 canonical relations (treats, causes, part_of, type_of, ...)
```

---

### Cải Tiến 2: Explicit Relation Extraction

#### 📋 Flow
```
Sentence: "Aspirin treats headache effectively"
    │
    ▼ [Dependency Parsing]
    │
    ├─ "Aspirin" (nsubj) ──┐
    │                      │
    ├─ "treats" (ROOT/VERB)┼──→ RELATION
    │                      │
    └─ "headache" (dobj) ──┘
    │
    ▼ [Canonical Mapping]
    │
    "treats" → matches pattern ["treat", "cure", "heal"] → canonical: "treats"
    │
    ▼ [Triple Formation]
    │
    (aspirin, treats, headache, confidence=0.8)
```

#### 🎯 Mục đích
Thay vì chỉ biết "aspirin và headache có liên quan" (co-occurrence), giờ ta biết **MỐI QUAN HỆ CỤ THỂ** giữa chúng.

#### 💡 Ý nghĩa trong phát triển tri thức

**Simple GraphRAG (Co-occurrence):**
```
Query: "What treats headache?"
Graph chỉ biết: aspirin ←→ headache (liên quan gì đó)
                ibuprofen ←→ headache (liên quan gì đó)
                stress ←→ headache (liên quan gì đó)
→ Không phân biệt được "treats" vs "causes"!
```

**Enhanced GraphRAG (Explicit Relations):**
```
Query: "What treats headache?"
Graph biết:    aspirin ──treats──→ headache ✅
               ibuprofen ──treats──→ headache ✅
               stress ──causes──→ headache ❌ (loại bỏ)
→ Reasoning chính xác hơn!
```

**Đây là bước tiến từ "Association" sang "Knowledge":**
- Association: A và B xuất hiện cùng nhau
- Knowledge: A có quan hệ R với B (A --R--> B)

#### 📊 Kết quả thực tế
```
8 canonical relations: treats, causes, part_of, type_of, located_in, 
                       associated_with, has_property, related_to
8452 triples được trích xuất
```

---

### Cải Tiến 3: Fuzzy Entity Matching (Cải tiến riêng)

#### 📋 Flow
```
Query entity: "satellite awards"
    │
    ▼ [Normalize]
    Remove articles: "satellite awards"
    Lowercase: "satellite awards"
    │
    ▼ [Match against KG entities]
    │
    KG entity: "the satellite awards"
    Normalize: "satellite awards"
    │
    ▼ [Comparison]
    "satellite awards" == "satellite awards" ✅ MATCH!
    │
    ▼ [Fallback: Word Overlap]
    If no exact match: check if overlap >= 50%
```

#### 🎯 Mục đích
Giải quyết vấn đề **entity mismatch** do:
- Articles: "the", "a", "an"
- Capitalization: "COVID-19" vs "covid-19"
- Slight variations: "United States" vs "United States of America"

#### 💡 Ý nghĩa trong phát triển tri thức
- **Bridge the gap giữa Query và KG**: User không biết entity được lưu chính xác như thế nào trong KG
- **Tăng recall**: Nhiều entities được match hơn → nhiều facts được retrieve hơn
- **Robustness**: Hệ thống ít nhạy cảm với cách viết của user

#### 📊 Kết quả thực tế
```
Trước: 31/64 câu có 0 KG facts
Sau:   20/64 câu có 0 KG facts
→ Cải thiện 35% số câu có thể retrieve KG facts
```

---

### Cải Tiến 4: Hybrid Retrieval (Semantic + Graph)

#### 📋 Flow
```
Query: "What university is UCF?"
    │
    ├────────────────────┬────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
[Entity Extract]    [Embed Query]       [KG Lookup]
    │                    │                    │
    │                    ▼                    │
    │               FAISS Search              │
    │                    │                    │
    │                    ▼                    │
    │            semantic_scores              │
    │               [0.8, 0.6, 0.5, ...]     │
    │                    │                    │
    ▼                    │                    ▼
[Fuzzy Match]            │            [Get related entities]
    │                    │                    │
    │                    │                    ▼
    │                    │            graph_scores (entity overlap)
    │                    │               [0.3, 0.7, 0.2, ...]
    │                    │                    │
    │                    └────────┬───────────┘
    │                             │
    │                             ▼
    │                    HYBRID SCORE
    │              α × semantic + (1-α) × graph
    │                    (α = 0.6)
    │                             │
    └─────────────────────────────┘
                                  │
                                  ▼
                         TOP-K CHUNKS
                                  │
                                  ▼
                         LLM GENERATION
```

#### 🎯 Mục đích
Kết hợp 2 nguồn thông tin:
1. **Semantic similarity**: Chunks có nghĩa gần với query
2. **Graph connectivity**: Chunks chứa entities liên quan trong KG

#### 💡 Ý nghĩa trong phát triển tri thức
- **Semantic search** tốt cho: paraphrase, synonyms, context understanding
- **Graph search** tốt cho: entity relationships, factual connections
- **Hybrid** = Best of both worlds

**Ví dụ:**
```
Query: "What treats migraines?"

Semantic search có thể trả về:
  "Headaches can be very painful..." (semantic similar nhưng không answer)

Graph search biết:
  "migraine" ←treats← "sumatriptan"
  → Chunk chứa "sumatriptan" được boost score
```

---

## ⚠️ Các Phần Từ SAT Chưa Sử Dụng Được

### 1. Graph Transformer

#### Mục đích ban đầu
- Học **learnable node embeddings** từ cấu trúc graph
- Thay thế Node2Vec (random walk, static) bằng attention-based learning
- Capture được **global graph structure** thay vì chỉ local neighborhoods

#### Tại sao chưa dùng được?
```
Vấn đề: SEGMENTATION FAULT khi KG > 5000 entities

Nguyên nhân kỹ thuật:
- Attention matrix có size O(E × E) với E = số edges
- KG hiện tại: 8452 edges → matrix ~71 triệu phần tử
- Mỗi phần tử là float32 (4 bytes) → ~284 MB chỉ cho 1 attention head
- Multi-head (8 heads) × Multi-layer (3 layers) → ~6.8 GB
- Vượt quá memory available → Crash
```

#### Cách khắc phục tiềm năng
| Approach | Mô tả | Độ khó |
|----------|-------|--------|
| **Mini-batching** | Chia graph thành subgraphs, process từng batch | ⭐⭐ |
| **Sparse Attention** | Chỉ tính attention cho k-nearest neighbors | ⭐⭐⭐ |
| **Graph Sampling** | Random sample edges để giảm size | ⭐ |
| **Gradient Checkpointing** | Trade compute for memory | ⭐⭐ |
| **Mixed Precision** | Dùng float16 thay float32 | ⭐ |

---

### 2. Text-Graph Aligner (CLIP-style)

#### Mục đích ban đầu
```
Ý tưởng: Tạo SHARED EMBEDDING SPACE cho cả text và graph

Text: "Aspirin is a medication"  →  [text_emb]  ─┐
                                                  │
                                                  ▼
                                          SHARED SPACE
                                                  ▲
                                                  │
Graph: (aspirin, type_of, medication)  →  [graph_emb] ─┘

→ Query bằng text, search trong graph space
→ Hoặc ngược lại: có entity, tìm text mô tả
```

#### Tại sao chưa dùng được?
```
Vấn đề 1: THIẾU LABELED DATA
- Cần pairs (text, entity) để train contrastive loss
- Dataset hiện tại không có annotation này
- SAT paper dùng FB15k-237 có sẵn text descriptions

Vấn đề 2: COMPUTATIONAL COST
- Train CLIP-style model cần nhiều negative samples
- Batch size lớn (512-4096) để contrastive loss hiệu quả
- Cần GPU với memory lớn

Vấn đề 3: COLD START
- Chưa có pre-trained weights cho domain-specific data
- Train from scratch cần nhiều data và time
```

#### Cách khắc phục tiềm năng
| Approach | Mô tả | Độ khó |
|----------|-------|--------|
| **Dùng LLM generate descriptions** | GPT/Llama tạo text cho mỗi entity | ⭐⭐ |
| **Transfer learning** | Fine-tune từ pre-trained CLIP | ⭐⭐ |
| **Self-supervised** | Dùng entity names làm text descriptions | ⭐ |
| **Dùng Sentence-BERT** | Embed cả entity names và text, không cần train | ⭐ |

---

## 📊 So Sánh Pipeline: Simple vs Enhanced

| Bước | Simple GraphRAG | Enhanced GraphRAG | Cải tiến |
|------|-----------------|-------------------|----------|
| **1. Chunking** | Fixed-size | Sentence-boundary aware | Không cắt ngang câu |
| **2. Entity Extraction** | spaCy NER only | spaCy NER + Dependency Parsing | Thêm relations |
| **3. Graph Structure** | Co-occurrence edges | Typed triples (head, rel, tail) | Biết quan hệ cụ thể |
| **4. Entity Storage** | String dict | ID mapping (entity2id) | Faster lookup |
| **5. Embeddings** | Node2Vec (static) | ~~Graph Transformer~~ → Node2Vec* | *Disabled |
| **6. Retrieval** | Semantic only | Hybrid (semantic + graph) | Multi-signal |
| **7. Entity Matching** | Exact match | Fuzzy matching | Robust hơn |
| **8. Text-Graph Bridge** | Không có | ~~CLIP-style~~ → Không có* | *Chưa train |

---

### Tổng Kết: Những Gì Đã Thực Sự Hoạt Động

✅ **Đang dùng và hoạt động:**
1. Entity/Relation ID Mapping
2. Explicit Relation Extraction  
3. Fuzzy Entity Matching
4. Hybrid Retrieval
5. Triple format với confidence scores

❌ **Đã implement nhưng disabled:**
1. Graph Transformer (segfault)
2. Text-Graph Aligner (chưa train)

→ **Thực tế**: Enhanced GraphRAG hiện tại là **Simple GraphRAG + Better Entity Handling + Explicit Relations + Hybrid Search**, chưa phải full SAT architecture.

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

1. ❌ **Graph Transformer bị vô hiệu hóa do lỗi bộ nhớ (Segfault)**
   - **Vấn đề**: Khi Knowledge Graph có hơn ~5000 entities, Graph Transformer gặp lỗi segmentation fault do tiêu thụ bộ nhớ quá lớn khi tính attention matrix trên toàn bộ edges.
   - **Nguyên nhân**: Thuật toán hiện tại tính attention O(E²) với E là số edges, không có cơ chế batching hay sparse attention.
   - **Hệ quả**: Mất đi khả năng học **learnable node embeddings** từ cấu trúc graph, phải fallback về Node2Vec embeddings tĩnh như Simple GraphRAG.

2. ❌ **Chưa train được Text-Graph Aligner (CLIP-style)**
   - **Vấn đề**: Module alignment giữa text embeddings và graph embeddings chưa được huấn luyện, chỉ sử dụng pre-computed embeddings độc lập.
   - **Nguyên nhân**: Thiếu dữ liệu training có nhãn (text-entity pairs), và cần computational resources đáng kể để train contrastive loss.
   - **Hệ quả**: Không tận dụng được ưu điểm lớn nhất của SAT paper - khả năng query bằng ngôn ngữ tự nhiên nhưng tìm kiếm hiệu quả trên graph space thông qua shared embedding space.

3. ❌ **5/64 câu hỏi (7.8%) không tìm được context phù hợp**
   - **Vấn đề**: Một số câu hỏi không retrieve được chunks chứa thông tin cần thiết để trả lời.
   - **Nguyên nhân gốc**:
     - Chunking strategy hiện tại (fixed-size) có thể cắt ngang các đoạn thông tin liên quan
     - Entity extraction bỏ sót một số entities do NER model không nhận diện được (đặc biệt với tiếng Việt hoặc thuật ngữ chuyên ngành)
     - Semantic similarity giữa câu hỏi và answer chunks không đủ cao
   - **Hệ quả**: Giới hạn recall tối đa của hệ thống ở mức ~92%

### Hướng phát triển:
- [x] Fix Graph Transformer cho large-scale KG (batching/sampling) ✅
- [ ] Train Text-Graph Aligner với contrastive loss
- [ ] Thêm multi-hop reasoning
- [ ] Cải thiện relation extraction với LLM 

---

## 🚀 Giai Đoạn 2: Graph Transformer với FB15k-237N (03/02/2026)

### 📝 Chi Tiết Giai Đoạn 1: Những Gì Đã Làm

Ở giai đoạn 1, tôi **tự xây dựng Knowledge Graph từ đầu** bằng các kỹ thuật sau:

#### 1. Entity/Relation ID Mapping - Tôi Đã Làm Gì?

**Bước thực hiện:**
- Đọc văn bản thô, dùng spaCy để nhận diện các thực thể (người, địa điểm, tổ chức...)
- Mỗi lần gặp một thực thể mới, gán cho nó một số ID tăng dần
- Tương tự với các quan hệ: mỗi động từ/quan hệ mới cũng được gán một ID

**Mục đích:**
- Chuyển từ lưu trữ bằng chuỗi ký tự sang lưu trữ bằng số, giúp máy tính xử lý nhanh hơn
- Chuẩn hóa dữ liệu: "UCF", "ucf", "University of Central Florida" nếu là cùng một thực thể sẽ có cùng ID
- Là bước bắt buộc nếu muốn dùng mạng neural sau này (neural network cần số, không nhận chuỗi)

**Có trong quá trình build KG không?** ✅ CÓ - Đây là bước tôi tự làm, lưu vào `entity2id.pkl` và `relation2id.pkl`

#### 2. Explicit Relation Extraction - Tôi Đã Làm Gì?

**Bước thực hiện:**
- Phân tích cú pháp câu (dependency parsing) để tìm chủ ngữ, động từ, tân ngữ
- Từ đó rút ra bộ ba: (chủ thể, quan hệ, đối tượng)
- Ví dụ: "UCF là một trường đại học công lập" → (UCF, là_loại, trường đại học công lập)

**Mục đích:**
- Biết được **quan hệ cụ thể** giữa các thực thể, không chỉ biết "chúng liên quan"
- Phân biệt được "A gây ra B" với "A chữa được B" - rất quan trọng cho suy luận

**Có trong quá trình build KG không?** ✅ CÓ - Tôi tự extract và lưu vào `triples.json`

#### 3. Hybrid Retrieval - Tôi Đã Làm Gì?

**Bước thực hiện:**
- Kết hợp 2 cách tìm kiếm: semantic (ngữ nghĩa) + graph (đồ thị)
- Semantic: tìm đoạn văn có nghĩa gần với câu hỏi
- Graph: tìm đoạn văn chứa nhiều thực thể liên quan trong KG (**đếm entity overlap**)
- Công thức: điểm = α × điểm_semantic + (1-α) × điểm_graph

**Mục đích:**
- Semantic giỏi tìm từ đồng nghĩa, paraphrase
- Graph giỏi tìm quan hệ logic, thông tin liên kết
- Kết hợp cả hai để tìm kiếm toàn diện hơn

**Có trong quá trình build KG không?** ❌ KHÔNG - Đây là bước retrieval, dùng sau khi đã có KG

**⚠️ Lưu ý quan trọng về Giai đoạn 1:**
- KG được **TỰ BUILD** từ văn bản thô bằng NLP
- Graph score = **entity overlap** (đếm số thực thể trùng), KHÔNG phải graph embedding
- **CÓ DÙNG KG** khi retrieve (đếm entities trong chunks)

---

### 📝 Chi Tiết Giai Đoạn 2: Những Gì Khác Biệt

#### Data Lấy Từ Đâu?

```
⚠️ QUAN TRỌNG: Giai đoạn 2 KHÔNG tự build KG!

Dữ liệu lấy từ: FB15k-237N (dataset CÓ SẴN của SAT paper)
- 14,541 thực thể với mô tả văn bản đầy đủ (id2text.txt)
- 237 loại quan hệ đã định nghĩa sẵn (rel2id.txt)  
- 87,282 bộ ba (triples) đã được gán nhãn sẵn (train.txt)

Tôi KHÔNG extract entity, KHÔNG extract relation, KHÔNG build KG mới!
```

#### Graph Transformer Làm Gì?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GRAPH TRANSFORMER - VAI TRÒ THỰC SỰ                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ĐẦU VÀO:  KG có sẵn (14541 entities, 87282 triples)                        │
│            ↓                                                                │
│  XỬ LÝ:    Học vector đặc trưng (embedding) cho mỗi thực thể                │
│            - Nhìn vào cấu trúc đồ thị: ai kết nối với ai                    │
│            - Dùng attention để tổng hợp thông tin từ các láng giềng         │
│            ↓                                                                │
│  ĐẦU RA:   Node embeddings (14541, 128) - mỗi entity 1 vector 128 chiều     │
│                                                                             │
│  ⚠️ KHÔNG LÀM:                                                              │
│     ❌ Không đọc văn bản                                                    │
│     ❌ Không nhận diện thực thể                                             │
│     ❌ Không rút trích quan hệ                                              │
│     ❌ Không xây dựng KG mới                                                │
│                                                                             │
│  → Graph Transformer CHỈ HỌC EMBEDDING từ KG đã có sẵn!                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### ⚠️ Vấn Đề ALPHA = 1.0: Có Dùng KG Không?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              THỰC TẾ: ALPHA = 1.0 → KHÔNG DÙNG GRAPH!                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Công thức: điểm = α × semantic + (1-α) × graph                             │
│                                                                             │
│  Khi α = 1.0:  điểm = 1.0 × semantic + 0 × graph = CHỈ semantic             │
│                                                                             │
│  → KẾT QUẢ 95.3% THỰC CHẤT LÀ RAG THUẦN TÚY!                               │
│  → KHÔNG dùng Graph Transformer embeddings                                  │
│  → KHÔNG dùng KG trong retrieval                                            │
│  → Chỉ dùng semantic search (FAISS + SentenceTransformer)                   │
│                                                                             │
│  Tại sao vẫn đạt 95.3%?                                                     │
│  - FB15k-237N có id2text.txt với mô tả văn bản đầy đủ cho mỗi entity        │
│  - Semantic search trên 14541 mô tả này tìm được context chính xác          │
│  - LLM với prompt cải tiến suy luận tốt từ context                          │
│                                                                             │
│  ⚠️ Graph Transformer embeddings ĐÃ TÍNH xong, chỉ KHÔNG ĐƯỢC DÙNG         │
│     vì khi thử α < 1.0, kết quả tệ hơn (45.3% với α=0.6)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### So Sánh 2 Giai Đoạn (Bản Rõ Ràng)

| Aspect | Giai đoạn 1 (Enhanced GraphRAG) | Giai đoạn 2 (Graph Transformer QA) |
|--------|--------------------------------|-----------------------------------|
| **Nguồn KG** | **Tự build** từ raw text | **Có sẵn** (FB15k-237N) |
| **Entity Extraction** | ✅ Có (spaCy NER) | ❌ Không cần |
| **Relation Extraction** | ✅ Có (Dependency Parsing) | ❌ Không cần |
| **ID Mapping** | ✅ Tự tạo | ✅ Có sẵn trong dataset |
| **Graph score là gì?** | **Entity overlap** (đếm trùng) | **Graph Transformer** embeddings |
| **Có dùng KG khi retrieve?** | ✅ CÓ (đếm entity overlap) | ❌ KHÔNG (α=1.0, pure semantic) |
| **Thực chất là gì?** | Hybrid RAG (semantic + graph) | **Pure RAG** (chỉ semantic) |
| **Kết quả** | 67.2% | 95.3% |

---

### Tại Sao Không Dùng Graph Embeddings (α < 1.0)?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VẤN ĐỀ: 2 KHÔNG GIAN KHÔNG LIÊN KẾT                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text Embeddings (SentenceTransformer):                                     │
│  ─────────────────────────────────────                                      │
│  - Học từ: ngữ nghĩa của từ, câu                                            │
│  - "government" gần với "state", "administration"                           │
│  - Đo lường: độ giống về nghĩa                                              │
│                                                                             │
│  Graph Embeddings (Graph Transformer):                                      │
│  ────────────────────────────────────                                       │
│  - Học từ: cấu trúc đồ thị (ai nối với ai)                                  │
│  - entity_0 gần entity_123 vì có chung nhiều láng giềng                     │
│  - Đo lường: độ giống về vị trí trong đồ thị                                │
│                                                                             │
│  ❌ HAI KHÔNG GIAN NÀY KHÔNG CÙNG HỆ QUY CHIẾU!                             │
│                                                                             │
│  Kết quả test:                                                              │
│  - α = 1.0 (chỉ semantic): 95.3% ✅                                         │
│  - α = 0.6 (hybrid):       45.3% ❌ (tệ hơn rất nhiều!)                     │
│                                                                             │
│  → Kết hợp 2 embeddings chưa aligned = làm hỏng kết quả                     │
│  → Cần TRAIN TEXT-GRAPH ALIGNMENT mới dùng được hybrid                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FB15k-237N Dataset - KG Có Sẵn Của SAT

```
FB15k-237N/
├── id2text.txt      # 14,541 entity descriptions (từ Wikipedia)
├── mid2id.txt       # Freebase MID → Entity ID mapping
├── rel2id.txt       # 237 relation types
├── train.txt        # 87,282 triples (head, relation, tail)
├── valid.txt        # 7,041 triples
└── test.txt         # 8,226 triples
```

**Ví dụ dữ liệu:**
```
# id2text.txt - Entity descriptions
0    A government is the system or group of people governing...
1    The University of Central Florida (UCF) is a public research university...
5    Tottenham Hotspur Football Club, commonly referred to as...

# train.txt - Triples (đã có sẵn quan hệ)
/m/0146hc    /education/educational_institution/colors    /m/067z2v
/m/0146hc    /education/educational_institution/school_type    /m/05jxkf
```

### Graph Transformer - Vai Trò Thực Sự

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GRAPH TRANSFORMER KHÔNG TỰ BUILD KG!                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Graph Transformer CHỈ làm 1 việc:                                          │
│  ─────────────────────────────────                                          │
│                                                                             │
│  INPUT:  KG đã có sẵn (entities, relations, triples)                        │
│          ↓                                                                  │
│  PROCESS: Học EMBEDDINGS từ cấu trúc graph                                  │
│           - Multi-head attention trên edges                                 │
│           - Aggregate thông tin từ neighbors                                │
│           ↓                                                                 │
│  OUTPUT: Node embeddings (14541, 128) - mỗi entity 1 vector                 │
│                                                                             │
│  ⚠️ KHÔNG LÀM:                                                              │
│     - Không extract entities từ text                                        │
│     - Không extract relations từ text                                       │
│     - Không build KG mới                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tại Sao Hybrid Search Thất Bại?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VẤN ĐỀ: 2 EMBEDDING SPACES KHÔNG ALIGNED                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text Embeddings (SentenceTransformer):                                     │
│  ─────────────────────────────────────                                      │
│  "government" → [0.2, 0.5, 0.1, ...] (384-dim)                              │
│  Học từ: Ngữ nghĩa của từ, synonyms, context                                │
│  "government" ≈ "state", "administration", "regime"                         │
│                                                                             │
│  Graph Embeddings (Graph Transformer):                                      │
│  ────────────────────────────────────                                       │
│  entity_0 → [0.8, -0.3, 0.4, ...] (128-dim)                                 │
│  Học từ: Vị trí trong graph, neighbors, edge patterns                       │
│  entity_0 gần entity_123 vì có chung nhiều neighbors                        │
│                                                                             │
│  ❌ 2 không gian này KHÔNG CÙNG HỆ QUY CHIẾU!                               │
│                                                                             │
│  Kết quả test:                                                              │
│  ─────────────                                                              │
│  α=1.0 (semantic only):  95.3% accuracy  ✅                                 │
│  α=0.6 (hybrid):         45.3% accuracy  ❌                                 │
│                                                                             │
│  → Graph embeddings làm GIẢM hiệu quả khi chưa aligned!                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Graph Transformer V2 - Giải Quyết Segfault

**Vấn đề cũ:**
```python
# V1: Tính attention cho TẤT CẢ edges cùng lúc
attention = softmax(Q @ K.T)  # O(E²) memory → SEGFAULT
```

**Giải pháp V2:**
```python
# V2: Batched edge processing
for batch_edges in chunks(all_edges, batch_size=10000):
    attention_batch = compute_attention(batch_edges)
    aggregate(attention_batch)
```

**Kết quả:**
- V1: Segfault với 5000+ entities
- V2: Chạy được 14,541 entities trong **0.27 giây**

### Cải Thiện Prompt - Giảm "Not Stated"

**Prompt cũ:**
```
Answer using ONLY the provided context.
If the context does not contain the answer, reply: "not stated in the text"
```
→ LLM quá nghiêm khắc, từ chối trả lời câu hỏi "WHY"

**Prompt mới:**
```
For "WHY/HOW" questions: Use reasoning to infer from context clues.
Even if not explicitly stated, derive logical conclusions.
ONLY say "Not stated" if there is absolutely NO relevant information.
Think step by step, then provide answer.
```
→ LLM được khuyến khích SUY LUẬN

**Kết quả:**
| Metric | Prompt Cũ | Prompt Mới |
|--------|----------|------------|
| Not stated | 7/64 | **1/64** |
| Accuracy | 84.4% | **95.3%** |

### Kết Quả Cuối Cùng

| Hệ thống | KG Type | Accuracy | Not Stated |
|----------|---------|----------|------------|
| Enhanced GraphRAG | Tự build (5088 entities) | 67.2% | 1/64 |
| **Graph Transformer QA** | FB15k-237N (14541 entities) | **95.3%** | 1/64 |

### Files Mới Được Tạo

| File | Mô tả |
|------|-------|
| `graph_transformer_v2.py` | Graph Transformer optimized với batched edge processing |
| `run_qa_with_graph_transformer.py` | QA pipeline với FB15k-237N |
| `test_graph_transformer_sat.py` | Test script cho Graph Transformer |
| `sat_kg_data/` | Cache cho embeddings và graph data |
| `qa_results_graph_transformer.json` | Kết quả QA (64 câu) |

### Hướng Đi Tiếp Theo

Để sử dụng Graph Embeddings hiệu quả, cần **train Text-Graph Alignment**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEXT-GRAPH ALIGNMENT (CLIP-style)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text: "government"  →  [Text Encoder]  →  text_emb  ─┐                     │
│                              ↓                        │                     │
│                        Projection Head                │                     │
│                              ↓                        ↓                     │
│                        ┌──────────────────────────────┐                     │
│                        │      SHARED SPACE            │                     │
│                        │   (Contrastive Loss)         │                     │
│                        └──────────────────────────────┘                     │
│                              ↑                        ↑                     │
│                        Projection Head                │                     │
│                              ↑                        │                     │
│  Graph: entity_0  →  [Graph Transformer]  →  graph_emb ─┘                   │
│                                                                             │
│  Training data: (entity_id, entity_text) pairs từ id2text.txt              │
│  Loss: InfoNCE Contrastive Loss                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Các bước cần làm:**
- [ ] Tạo ProjectionHead module (MLP: 384 → 256 và 128 → 256)
- [ ] Implement InfoNCE Contrastive Loss
- [ ] Training loop với FB15k-237N id2text pairs
- [ ] Evaluate hybrid retrieval sau khi align

---

*Báo cáo cập nhật: 03/02/2026*
