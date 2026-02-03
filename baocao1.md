# 📊 Báo Cáo Giai Đoạn 1: Tự Xây Dựng Knowledge Graph từ Văn Bản

> **Mục tiêu**: Xây dựng Knowledge Graph (KG) từ văn bản thô và sử dụng KG hỗ trợ hệ thống Question Answering

---

## 🎯 Tóm Tắt Kết Quả

| Thông số | Giá trị |
|----------|---------|
| **Nguồn dữ liệu** | Văn bản thô (Wikipedia articles) |
| **Số chunks** | 173 đoạn văn bản |
| **Số entities** | 5,088 thực thể |
| **Số relations** | 8 loại quan hệ |
| **Số triples (cạnh KG)** | 8,451 cạnh |
| **Kết quả QA** | 67.2% accuracy |

---

## 📚 Phần 1: Các Kỹ Thuật Từ Paper SAT Đã Áp Dụng

Paper SAT (Structure-Aware Alignment and Tuning) đề xuất nhiều kỹ thuật để liên kết text với knowledge graph. Trong giai đoạn này, tôi đã áp dụng **3 ý tưởng chính**:

### 1.1. ID Mapping (Ánh Xạ ID)

#### 🎯 Vấn Đề Cần Giải Quyết

Hãy tưởng tượng bạn có một **Đồ Thị Tri Thức** (Knowledge Graph) chứa hàng nghìn thực thể như: "Đại học Bách Khoa", "Thành phố Hồ Chí Minh", "Việt Nam"...

**Vấn đề:** Máy tính (đặc biệt là mạng nơ-ron) **không hiểu được chữ**, nó chỉ hiểu **số**.

Ví dụ: Bạn muốn máy tính học mối quan hệ `"Hà Nội" → nằm_tại → "Việt Nam"`, nhưng máy tính không thể tính toán với chuỗi ký tự "Hà Nội" hay "Việt Nam".

#### 💡 Giải Pháp: Đánh Số Cho Mọi Thứ

**Ánh xạ ID** là kỹ thuật **gán một số duy nhất cho mỗi thực thể và mỗi loại quan hệ**.

**Ý tưởng từ SAT:**
- SAT sử dụng file `mid2id.txt` để ánh xạ từ Freebase MID (ví dụ: `/m/01234`) sang số ID (ví dụ: `0, 1, 2, ...`)
- Mục đích: Chuyển đổi tên thực thể thành số để mạng nơ-ron xử lý được

#### 🔧 Cách Tôi Áp Dụng Vào Code

Trong code của tôi (file `enhanced_graphrag.py`), tôi **tự xây dựng KG từ văn bản thô**, nên **ID được tự động sinh ra** khi trích xuất thực thể từ văn bản.

**Quy trình cụ thể:**

```
Văn bản thô: "UCF is a public research university located in Florida..."
       ↓
   spaCy NER trích xuất thực thể: ["UCF", "Florida", "public research university"]
       ↓
   Mỗi thực thể được gán số ID tự động
       ↓
   entity2id = {"ucf": 0, "florida": 1, "public research university": 2, ...}
```

**Code thực tế trong lớp `EnhancedKGBuilder`:**

```python
# Khi gặp thực thể mới, tự động gán số ID tiếp theo
def _get_or_create_entity_id(self, entity: str) -> int:
    """Lấy ID của thực thể, nếu chưa có thì tạo mới"""
    entity = self._normalize_entity(entity)  # Chuẩn hóa: "UCF" → "ucf"
    
    if entity not in self.entity2id:
        # Thực thể mới → gán số ID tiếp theo
        idx = len(self.entity2id)  # Ví dụ: 0, 1, 2, ...
        self.entity2id[entity] = idx
        self.id2entity[idx] = entity  # Từ điển ngược để tra ngược
    
    return self.entity2id[entity]

# Tương tự cho quan hệ
def _get_or_create_relation_id(self, relation: str) -> int:
    relation = relation.lower().strip()
    if relation not in self.relation2id:
        idx = len(self.relation2id)
        self.relation2id[relation] = idx
        self.id2relation[idx] = relation
    return self.relation2id[relation]
```

**Ví dụ minh họa quá trình:**

| Bước | Thực thể gặp được | `entity2id` sau bước này |
|------|-------------------|--------------------------|
| 1 | "ucf" | `{"ucf": 0}` |
| 2 | "florida" | `{"ucf": 0, "florida": 1}` |
| 3 | "public research university" | `{"ucf": 0, "florida": 1, "public research university": 2}` |
| 4 | "ucf" (gặp lại) | Không thay đổi (đã có ID = 0) |

**Kết quả cuối cùng được lưu:**
- `entity2id.pkl`: Từ điển ánh xạ tên → số (5,088 thực thể)
- `relation2id.pkl`: Từ điển ánh xạ quan hệ → số (8 loại quan hệ)

#### 🎯 Mục Đích Của Bước Ánh Xạ ID Trong Dự Án

| Mục đích | Giải thích |
|----------|------------|
| **Chuẩn hóa tên gọi** | "UCF", "ucf", "Ucf" → đều trở thành `"ucf"` → cùng 1 số ID |
| **Chuyển KG sang dạng số** | Để đưa vào Graph Transformer (mạng nơ-ron trên đồ thị) |
| **Tạo cạnh dạng số** | Triple `("ucf", "co_occurs_with", "florida")` → `(0, 0, 1)` |
| **Tiết kiệm bộ nhớ** | Lưu số thay vì chuỗi ký tự |

#### 📊 Khác Biệt So Với SAT Gốc

| Tiêu chí | SAT gốc | Code của tôi |
|----------|---------|--------------|
| **Nguồn KG** | KG có sẵn (Freebase, FB15k-237) | Tự xây từ văn bản thô |
| **Cách đánh ID** | Đọc từ file `mid2id.txt` có sẵn | Tự động sinh khi trích xuất thực thể |
| **Số lượng thực thể** | Cố định theo KG gốc | Phụ thuộc vào văn bản đầu vào |
| **Chất lượng** | Cao (KG chuẩn, đã được kiểm duyệt) | Thấp hơn (phụ thuộc spaCy NER) |

---

### 1.2. Relation Extraction (Rút Trích Quan Hệ)

#### 🎯 Vấn Đề Cần Giải Quyết

Trong đồ thị tri thức, **quan hệ** (relation) là thứ kết nối các thực thể với nhau. Không có quan hệ, các thực thể chỉ là danh sách rời rạc, vô nghĩa.

**Ví dụ:**
- Có 2 thực thể: `"Hà Nội"` và `"Việt Nam"`
- Nếu không có quan hệ → chỉ biết 2 cái tên, không biết liên quan gì
- Nếu có quan hệ `"Hà Nội" --là_thủ_đô_của--> "Việt Nam"` → có ý nghĩa!

**Vấn đề:** Làm sao máy tính tự động tìm ra quan hệ từ văn bản thô?

#### 💡 SAT Làm Gì? (Không Có Relation Extraction!)

**Quan trọng:** SAT **KHÔNG tự trích xuất quan hệ từ văn bản**. SAT dùng **KG có sẵn** (FB15k-237) với:
- **237 loại quan hệ** đã được định nghĩa sẵn bởi Freebase
- Các quan hệ được lưu trong file `rel2id.txt` và `train.txt`

**Ví dụ file `rel2id.txt` của SAT:**
```
/people/person/profession                    4
/film/film/genre                             8
/location/location/contains                  13
/people/person/nationality                   14
/people/person/place_of_birth                30
...
(tổng cộng 237 loại quan hệ)
```

**Ví dụ file `train.txt` của SAT:**
```
/m/027rn    /location/country/form_of_government    /m/06cx9
/m/0h3y     /location/country/capital               /m/0rtv
```
→ Các triple đã có sẵn, chỉ việc đọc vào!

#### 🔧 Cách Tôi Làm: TỰ VIẾT CODE Trích Xuất Quan Hệ

Vì tôi tự xây KG từ văn bản (không có sẵn như SAT), tôi phải **tự viết code** để trích xuất quan hệ. Tôi dùng **Dependency Parsing** (Phân tích cú pháp phụ thuộc).

**⚠️ Lưu ý quan trọng:** Phần này **KHÔNG lấy từ code SAT**. Đây là code tôi tự viết dựa trên kiến thức NLP.

##### Dependency Parsing là gì?

**Dependency Parsing** = Phân tích cấu trúc ngữ pháp của câu, tìm ra từ nào phụ thuộc vào từ nào.

**Ví dụ với câu:** `"UCF is located in Florida"`

```
       is located (ROOT - động từ chính)
           │
     ┌─────┼─────┐
     │           │
    UCF      in Florida
  (nsubj)      (prep)
  chủ ngữ    giới từ
```

- `"UCF"` là **chủ ngữ** (subject) của động từ `"is located"`
- `"in Florida"` là **cụm giới từ** chỉ địa điểm
- Từ đây suy ra: `UCF` có quan hệ `located_in` với `Florida`

##### Quy Trình Trích Xuất Trong Code

```
Câu: "UCF is located in Florida"
         ↓
   spaCy phân tích dependency
         ↓
   Tìm pattern: Chủ ngữ - Động từ - Tân ngữ/Giới từ
         ↓
   Tạo triple: (UCF, in, Florida)
```

**Code thực tế trong `extract_relations_from_sentence()`:**

```python
def extract_relations_from_sentence(self, sent):
    """Trích xuất quan hệ từ 1 câu dùng dependency parsing"""
    relations = []
    
    for token in sent:
        # Tìm pattern: Chủ ngữ - Động từ - Tân ngữ
        if "subj" in token.dep_:  # token là chủ ngữ
            subj = token.text              # Lấy chủ ngữ: "UCF"
            verb = token.head              # Lấy động từ: "located"
            
            for child in verb.children:
                if "obj" in child.dep_:    # Tìm tân ngữ
                    obj = child.text
                    rel = verb.lemma_      # Lấy dạng gốc động từ
                    
                    # Tạo triple với độ tin cậy 0.8
                    relations.append((subj, rel, obj, 0.8))
        
        # Tìm pattern: Danh từ - Giới từ - Danh từ
        if token.dep_ == "prep":           # token là giới từ (in, at, of,...)
            head = token.head.text         # Từ đứng trước giới từ
            for child in token.children:
                if child.dep_ == "pobj":   # Tân ngữ của giới từ
                    rel = token.text       # Giới từ làm quan hệ
                    
                    # Tạo triple với độ tin cậy 0.6 (thấp hơn)
                    relations.append((head, rel, child.text, 0.6))
    
    return relations
```

##### Bảng Mẫu Quan Hệ Định Sẵn

Code có định nghĩa sẵn một số mẫu để ánh xạ động từ → quan hệ chuẩn:

```python
relation_patterns = {
    "treats": ["treat", "cure", "heal", "remedy"],      # chữa trị
    "causes": ["cause", "lead to", "result in"],        # gây ra
    "prevents": ["prevent", "avoid", "reduce risk"],    # ngăn ngừa
    "part_of": ["part of", "component", "include"],     # là một phần của
    "type_of": ["type of", "kind of", "is a"],          # là một loại
    # ...
}
```

**Ví dụ:** Nếu gặp động từ `"cure"` → ánh xạ thành quan hệ chuẩn `"treats"`

##### Fallback: Quan Hệ Đồng Xuất Hiện (Co-occurrence)

Khi **không tìm được quan hệ rõ ràng** từ dependency parsing, code sẽ **fallback** (dùng phương án dự phòng):

> "Nếu 2 thực thể xuất hiện trong cùng 1 câu → tạo cạnh `co_occurs_with`"

```python
# Fallback: Co-occurrence relations
if add_cooccurrence:
    sent_ents_list = list(sent_entities & entities)
    for i, e1 in enumerate(sent_ents_list):
        for e2 in sent_ents_list[i+1:]:
            # Nếu chưa có cạnh giữa e1 và e2
            if not self.kg.has_edge(e1, e2) and not self.kg.has_edge(e2, e1):
                # Tạo cạnh co_occurs_with với độ tin cậy thấp (0.3)
                self._add_triple(e1, "co_occurs_with", e2, 0.3, chunk_idx)
```

#### 📊 Kết Quả Thực Tế: Vấn Đề Nghiêm Trọng

Phân tích KG đã xây dựng:

| Loại quan hệ | Số cạnh | Tỉ lệ |
|--------------|---------|-------|
| `co_occurs_with` | 8,442 | **99.9%** |
| `as` | 2 | 0.02% |
| `of` | 2 | 0.02% |
| `in` | 1 | 0.01% |
| Các quan hệ khác | 4 | 0.05% |
| **Tổng** | **8,451** | 100% |

**Kết luận đau lòng:** 

- **99.9% quan hệ là `co_occurs_with`** (đồng xuất hiện)
- Dependency parsing **gần như không hoạt động**
- Code phải fallback về co-occurrence cho hầu hết trường hợp

#### ❌ Tại Sao Dependency Parsing Thất Bại?

**1. spaCy model quá yếu:**
- `en_core_web_sm` là model nhỏ nhất, độ chính xác thấp
- Không nhận diện đúng cấu trúc câu phức tạp

**2. Văn bản Wikipedia có cấu trúc phức tạp:**
```
"The University of Central Florida, commonly known as UCF, 
is a public research university with its main campus in 
unincorporated Orange County, Florida."
```
- Câu dài, nhiều mệnh đề
- Nhiều dấu phẩy, từ nối
- spaCy khó parse đúng

**3. Pattern quá đơn giản:**
- Code chỉ tìm `Chủ ngữ - Động từ - Tân ngữ`
- Nhiều quan hệ không theo pattern này

#### 📊 So Sánh: SAT vs Code Của Tôi

| Tiêu chí | SAT gốc | Code của tôi |
|----------|---------|--------------|
| **Nguồn quan hệ** | Có sẵn trong FB15k-237 (237 loại) | Tự trích xuất từ văn bản |
| **Phương pháp** | Đọc từ file `train.txt` | Dependency parsing + co-occurrence |
| **Code relation extraction** | ❌ **KHÔNG CÓ** (không cần) | ✅ **TỰ VIẾT** |
| **Chất lượng quan hệ** | Cao, đa dạng, có ngữ nghĩa rõ ràng | Thấp, 99.9% là co-occurrence |
| **Số loại quan hệ** | 237 loại | 8 loại (hầu hết vô nghĩa) |

#### 💡 Bài Học Rút Ra

1. **SAT không làm relation extraction** - họ dùng KG có sẵn
2. **Dependency parsing không đủ mạnh** để trích xuất quan hệ từ văn bản thực tế
3. **Co-occurrence không mang ngữ nghĩa** - chỉ nói 2 thực thể xuất hiện cùng nhau
4. **Cần phương pháp mạnh hơn:** Dùng LLM để extract relations, hoặc dùng KG có sẵn như SAT

---

### 1.3. Hybrid Retrieval (Tìm Kiếm Kết Hợp)

#### 🎯 Vấn Đề Cần Giải Quyết

Khi tìm kiếm thông tin để trả lời câu hỏi, có 2 cách tiếp cận:

1. **Tìm kiếm ngữ nghĩa (Semantic Search):** Dựa trên ý nghĩa của câu hỏi
2. **Tìm kiếm dựa trên đồ thị (Graph Search):** Dựa trên các thực thể được nhắc đến

**Vấn đề:** Dùng riêng 1 cách có thể bỏ sót thông tin quan trọng.

#### 💡 SAT Làm Gì? (Không Phải Hybrid Retrieval!)

**Quan trọng:** SAT **KHÔNG làm hybrid retrieval** như code của tôi. SAT dùng phương pháp phức tạp hơn nhiều:

**CLIP-style Contrastive Learning:**
- SAT huấn luyện một mô hình để **căn chỉnh (align)** biểu diễn văn bản và biểu diễn đồ thị
- Dùng **InfoNCE loss** (contrastive loss) để học
- Text embedding và Graph embedding được đưa vào **cùng không gian vector**

```python
# Code SAT (trong clip_graph.py) - Contrastive Learning
def forward(self, g, src, rel, dst, src_text, dst_text, device):
    # Encode graph nodes
    s_graph_feats = self.encode_graph(src, g)
    # Encode text 
    s_text_feats = self.encode_text(src_text)
    t_text_feats = self.encode_text(dst_text)
    
    # Normalize features
    s_graph_feats = s_graph_feats / s_graph_feats.norm(dim=-1, keepdim=True)
    s_text_feats = s_text_feats / s_text_feats.norm(dim=-1, keepdim=True)
    
    # Contrastive loss sẽ kéo text và graph embedding gần nhau
    return s_graph_feats, s_text_feats, t_text_feats, text_labels
```

**Đặc điểm của SAT:**
- **Học được** (learnable): Mô hình được huấn luyện trên dữ liệu
- **End-to-end**: Text encoder và Graph encoder được train cùng nhau
- **Contrastive**: Học bằng cách so sánh cặp positive/negative

#### 🔧 Cách Tôi Làm: TỰ VIẾT Hybrid Scoring Đơn Giản

**⚠️ Lưu ý:** Phần này **KHÔNG lấy từ SAT**. Đây là công thức kết hợp đơn giản tôi tự viết.

**Công thức:**
```
final_score = α × semantic_score + (1-α) × graph_score
```

Trong đó:
- `semantic_score`: Điểm từ FAISS (độ tương đồng cosine giữa câu hỏi và chunk)
- `graph_score`: Điểm dựa trên số thực thể trùng khớp giữa câu hỏi và chunk
- `α` (alpha): Trọng số, mặc định = 0.7 (70% semantic, 30% graph)

**Code thực tế:**

```python
def query(self, query: str, top_k: int = 5, alpha: float = 0.7):
    # Bước 1: Tìm kiếm ngữ nghĩa
    sem_results = self._semantic_search(query, top_k=top_k * 2)
    
    # Bước 2: Tính điểm dựa trên đồ thị
    graph_scores = self._graph_search(query)
    
    # Bước 3: Kết hợp điểm
    combined = []
    for idx, sem_score in sem_results:
        gscore = graph_scores[idx]
        final_score = alpha * sem_score + (1 - alpha) * gscore
        combined.append((idx, final_score))
    
    # Sắp xếp và lấy top-k
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]
```

**Cách tính `graph_score`:**
```python
def _graph_search(self, query: str) -> np.ndarray:
    # Trích xuất entities từ câu hỏi bằng NER
    doc = self.nlp(query)
    q_entities = [ent.text for ent in doc.ents]
    
    # Đếm số entities trùng khớp trong mỗi chunk
    scores = []
    for chunk_entities in self.chunk_entities:
        overlap = len(set(q_entities) & chunk_entities)
        scores.append(overlap)
    
    # Chuẩn hóa về [0, 1]
    scores = np.array(scores) / (max(scores) + 1e-12)
    return scores
```

#### 📊 So Sánh: SAT vs Code Của Tôi

| Tiêu chí | SAT gốc | Code của tôi |
|----------|---------|--------------|
| **Phương pháp** | CLIP-style Contrastive Learning | Công thức cộng trọng số đơn giản |
| **Có học (learnable)** | ✅ Có - train neural network | ❌ Không - công thức cố định |
| **Text-Graph alignment** | Học để đưa vào cùng không gian | Chỉ cộng điểm, không align |
| **Độ phức tạp** | Cao (cần train model) | Thấp (chỉ cần công thức) |
| **Hiệu quả** | Cao (nếu train tốt) | Thấp (phụ thuộc NER) |

#### ❌ Tại Sao Hybrid Của Tôi Không Hiệu Quả?

Như đã phân tích ở Phần 3, `graph_score` gần như **luôn bằng 0** vì:
1. NER không trích xuất được entities từ câu hỏi
2. Entities trích xuất được không khớp với KG

**Kết quả test:**
```
α = 1.0 (100% semantic): 6/64 đúng
α = 0.7 (70% semantic): 6/64 đúng  
α = 0.0 (100% graph):   6/64 đúng
→ Thay đổi α không ảnh hưởng gì!
```

#### 💡 Bài Học Rút Ra

1. **SAT dùng contrastive learning**, không phải hybrid scoring đơn giản
2. **Công thức cộng trọng số** là cách tiếp cận naive, không hiệu quả
3. **Cần học alignment** giữa text và graph thay vì chỉ cộng điểm
4. **Graph score vô nghĩa** nếu NER không hoạt động

---

## 🔄 Phần 2: Flow Xử Lý Query Chi Tiết

### 2.1. Sơ Đồ Tổng Quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Question                              │
│          "Where is UCF's main campus located?"                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BƯỚC 1: NER - Trích Xuất Entities từ Question          │
│                                                                      │
│   spaCy NER xử lý câu hỏi → Tìm entities                            │
│   "Where is UCF's main campus located?"                              │
│                    ↓                                                 │
│   Entities tìm được: ["UCF"]                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌────────────────────────────────┐    ┌────────────────────────────────────┐
│   BƯỚC 2A: Semantic Search     │    │    BƯỚC 2B: Graph Search           │
│                                │    │                                     │
│ • Encode question → vector     │    │ • Lấy entities: ["UCF"]             │
│ • FAISS tìm chunks gần nhất    │    │ • Đếm mỗi chunk có bao nhiêu       │
│ • Trả về: [(chunk_idx,         │    │   entities trùng với query         │
│            score), ...]        │    │ • Trả về: [0, 0, 1, 0, 1, ...]     │
│                                │    │   (chunk 2 và 4 có "UCF")           │
└────────────────────────────────┘    └────────────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BƯỚC 3: Kết Hợp Điểm (Hybrid)                    │
│                                                                      │
│   final_score = 0.6 × semantic_score + 0.4 × graph_score            │
│                                                                      │
│   Ví dụ chunk #5:                                                   │
│   - semantic_score = 0.75 (nghĩa gần)                               │
│   - graph_score = 1.0 (có entity "UCF")                             │
│   - final = 0.6 × 0.75 + 0.4 × 1.0 = 0.85                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BƯỚC 4: Lấy KG Facts                              │
│                                                                      │
│   Với mỗi entity trong question, tìm các cạnh liên quan trong KG    │
│   Entity "UCF" có cạnh:                                              │
│   - UCF co_occurs_with Florida                                       │
│   - UCF co_occurs_with Orange County                                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BƯỚC 5: Tạo Context cho LLM                       │
│                                                                      │
│   context = top_chunks + kg_facts                                    │
│                                                                      │
│   "UCF is a public research university with its main campus in      │
│    unincorporated Orange County, Florida..."                         │
│   + "[KG Fact] UCF co_occurs_with Florida"                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BƯỚC 6: LLM Trả Lời                               │
│                                                                      │
│   Gửi context + question cho Kimi LLM                                │
│   → "UCF's main campus is located in unincorporated Orange County,  │
│       Florida."                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Phần 3: Hạn Chế Phát Hiện - Graph KHÔNG Hoạt Động

### 3.1. Vấn Đề Chính: Graph Score Gần Như Bằng 0

Khi test với nhiều câu hỏi, phát hiện:

| Question | Entities NER tìm được | Graph Score |
|----------|----------------------|-------------|
| "What is a government..." | ❌ **KHÔNG TÌM THẤY** | 0 |
| "Where is UCF's..." | ✅ "UCF" | 1.0 |
| "Who formed The Roots..." | ❌ "what year" (sai!) | 0 |
| "What is Mario Puzo..." | ✅ "Mario Puzo" | 0 (không có trong KG) |

**Kết quả test với các giá trị alpha:**
```
α = 1.0 (100% semantic, 0% graph): 6/64 đúng (9.4%)
α = 0.7 (70% semantic, 30% graph): 6/64 đúng (9.4%)
α = 0.0 (0% semantic, 100% graph): 6/64 đúng (9.4%)
→ Thay đổi alpha KHÔNG ảnh hưởng gì vì graph_score = 0!
```

### 3.2. Nguyên Nhân: NER và spaCy Là Gì?

#### NER (Named Entity Recognition) là gì?

**NER** = Named Entity Recognition = **Nhận Diện Thực Thể Có Tên**

Đây là một task trong NLP (Xử lý Ngôn ngữ Tự nhiên) với mục tiêu:
- Đọc một câu văn bản
- Tìm và đánh dấu các "thực thể có tên" như: người, địa điểm, tổ chức, ngày tháng, v.v.

**Ví dụ:**
```
Input:  "Steve Jobs founded Apple in California."
Output: 
  - "Steve Jobs" → PERSON (người)
  - "Apple" → ORG (tổ chức)  
  - "California" → GPE (địa điểm)
```

#### spaCy là gì?

**spaCy** là một thư viện Python mã nguồn mở cho NLP (https://spacy.io/)

- Được phát triển bởi Explosion AI
- Cung cấp các model pre-trained cho nhiều ngôn ngữ
- Tích hợp sẵn nhiều chức năng: NER, POS tagging, Dependency Parsing, v.v.

**Model tôi dùng: `en_core_web_sm`**
- "en" = English (tiếng Anh)
- "core" = model cơ bản
- "web" = train trên dữ liệu web
- "sm" = small (nhỏ, ~12MB)

```python
import spacy
nlp = spacy.load("en_core_web_sm")  # Load model

doc = nlp("UCF is located in Florida")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Output: UCF → ORG, Florida → GPE
```

### 3.3. Tại Sao NER/spaCy Gây Ra Vấn Đề?

#### Vấn đề 1: Model quá nhỏ và yếu

`en_core_web_sm` là model nhỏ nhất, accuracy thấp:
- Chỉ ~86% F1-score cho NER trên benchmark
- Không nhận ra nhiều entities không phổ biến

**Ví dụ thất bại:**
```python
doc = nlp("What is a government?")
print([ent.text for ent in doc.ents])
# Output: [] ← Không tìm thấy gì!

doc = nlp("Who formed The Roots?")  
print([ent.text for ent in doc.ents])
# Output: ['what year'] ← Nhận sai!
```

#### Vấn đề 2: Câu hỏi ngắn, thiếu context

NER hoạt động tốt hơn khi có nhiều context:
```python
# Câu dài (có context) → NER tốt
doc = nlp("The University of Central Florida (UCF) is a public research university.")
# Tìm được: "The University of Central Florida", "UCF"

# Câu hỏi ngắn → NER yếu
doc = nlp("Where is UCF located?")
# Chỉ tìm được: "UCF" (may mắn)
```

#### Vấn đề 3: Entity không match với KG

Ngay cả khi NER tìm được entity, nó có thể không khớp với KG:
```
NER tìm được: "Mario Puzo"
KG chứa: "mario puzo", "Mario Gennaro Puzo"
→ Không match! (do normalize khác nhau)
```

### 3.4. Vấn Đề Với Relations

**Phân tích edges trong KG:**
```
co_occurs_with: 8,442 edges (99.9%!)
as: 2 edges
of: 2 edges
in: 1 edge
...
```

**99.9% relations là `co_occurs_with`** - nghĩa là:
- Dependency parsing KHÔNG hoạt động
- Code fallback về: "Nếu A và B xuất hiện cùng câu → thêm cạnh co_occurs_with"
- Đây là quan hệ VÔ NGHĨA, không mang thông tin gì hữu ích

**Ví dụ:**
```
KG Facts trả về:
- "UCF co_occurs_with Florida" ← Chỉ nói UCF và Florida xuất hiện cùng câu
- "UCF co_occurs_with 68,442 students" ← Vô nghĩa

Thay vì:
- "UCF is_located_in Florida" ← Thông tin hữu ích
- "UCF has_enrollment 68,442" ← Thông tin hữu ích
```

---

## 📊 Phần 4: Kết Luận

### 4.1. Thực Tế Hệ Thống Hoạt Động

| Component | Đóng Góp Thực Sự |
|-----------|------------------|
| **Semantic Search (FAISS)** | ✅ **~100%** - Tìm đúng chunks chứa câu trả lời |
| **Graph Search (Entity Overlap)** | ❌ **~0%** - NER không extract được entities |
| **KG Facts** | ❌ **~0%** - Chỉ có co-occurrence vô nghĩa |
| **LLM (Kimi)** | ✅ **100%** - Đọc context và trả lời |

### 4.2. Tại Sao Vẫn Đạt 67.2% Accuracy?

Mặc dù KG không hoạt động, hệ thống vẫn đạt 67.2% vì:

1. **Dataset nhỏ**: Chỉ 173 chunks → Semantic search dễ tìm đúng
2. **Chunks chứa đầy đủ thông tin**: Mỗi chunk ~700-800 ký tự, chứa nhiều thông tin liên quan
3. **LLM mạnh**: Kimi K2 có khả năng suy luận tốt từ context

**Kết luận: Giai đoạn 1 thực chất là PURE RAG, KG được xây dựng nhưng KHÔNG được sử dụng hiệu quả.**

### 4.3. Cải Tiến Cần Thiết

Để KG thực sự hữu ích, cần:

1. **Nâng cấp NER model:**
   - Dùng `en_core_web_lg` (lớn hơn, chính xác hơn)
   - Hoặc dùng transformer-based NER (BERT, spaCy transformers)

2. **Cải thiện Relation Extraction:**
   - Dùng model chuyên cho relation extraction (OpenIE, REBEL)
   - Hoặc dùng LLM để extract relations

3. **Entity Linking:**
   - Thêm bước match entities từ question với KG
   - Dùng fuzzy matching, alias expansion

4. **Hoặc dùng Pre-built KG:**
   - Dùng KG có sẵn như FB15k-237 (Giai đoạn 2)
   - KG chất lượng cao, có relations đa dạng

---

## 📁 Files Đã Tạo

```
enhanced_sat_data/
├── chunks.json          # 173 đoạn văn bản
├── embeddings.npy       # Vector 384-dim cho mỗi chunk
├── faiss.index          # FAISS index để tìm kiếm nhanh
├── kg.pkl               # NetworkX graph (5088 nodes, 8451 edges)
├── entity2id.pkl        # Dict: entity_name → ID
├── relation2id.pkl      # Dict: relation_name → ID
├── chunk_entities.pkl   # List: mỗi chunk chứa entities nào
└── meta.json            # Metadata
```

---

*Cập nhật: 03/02/2026*
