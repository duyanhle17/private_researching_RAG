# 📋 Next Steps: Integrate SAT vào Enhanced GraphRAG

> **Ngày tạo**: 03/02/2026
> **Mục tiêu**: Sử dụng Graph Transformer và Text-Graph Aligner từ SAT paper

---

## 🎯 Tổng Quan

### Vấn đề hiện tại của Enhanced GraphRAG:
1. ❌ **Graph Transformer disabled** - Segfault với large KG
2. ❌ **Text-Graph Aligner chưa train** - Thiếu labeled data

### Giải pháp từ SAT:
1. ✅ SAT đã có **Graph Transformer** hoạt động được với FB15k-237N (14,542 entities)
2. ✅ SAT có sẵn **id2text.txt** - text descriptions cho mỗi entity
3. ✅ SAT có **CLIP-style training** với contrastive loss

---

## 📁 Data Structure (FB15k-237N)

Data đã được copy vào `sat_data_backup/FB15k-237N/`:

```
sat_data_backup/FB15k-237N/
├── id2text.txt      # 14,542 entities với text descriptions
├── id2title.txt     # 14,542 entity titles (short names)
├── mid2id.txt       # Freebase MID → internal ID mapping
├── rel2id.txt       # 238 relation types với IDs
├── train.txt        # 87,283 training triples
├── valid.txt        # Validation triples
├── test.txt         # Test triples
├── neg_train.txt    # 261,847 negative samples
├── neg_valid.txt    # Negative validation samples
└── neg_test.txt     # Negative test samples
```

### Data Format:

**id2text.txt** (entity descriptions):
```
0	A government is the system or group of people governing...
1	The University of Central Florida (UCF) is a public research university...
2	The Satellite Award for Best Cinematography is one of the annual...
```

**train.txt** (triples):
```
/m/027rn	/location/country/form_of_government	/m/06cx9
/m/07s9rl0	/media_common/netflix_genre/titles	/m/0170z3
```

**rel2id.txt** (238 relations):
```
/film/film/genre	8
/people/person/nationality	14
/location/location/contains	13
```

---

## 🚀 Phase 1: Chạy SAT Training (Verify Setup)

### 1.1 Cài đặt dependencies

```powershell
# Trong terminal với venv activated
pip install torch-scatter torch-geometric tqdm scikit-learn
```

### 1.2 Chạy training với FB15k-237N

```powershell
cd SAT/aligner/model

# Training với config nhỏ để test
python main.py --data_path ../data --data_name FB15k-237N --epoch_num 5 --batch_size 32 --lr 2e-5
```

**Expected output:**
- Log file: `logs/FB15k-237N/aligner_xxx.log`
- Model checkpoint: `checkpoints/FB15k-237N/gt-xxx.pkl`

### 1.3 Kiểm tra kết quả

```python
import torch

# Load trained entity embeddings
entity_emb = torch.load("checkpoints/FB15k-237N/entity_embedding.pt")
print(f"Entity embeddings shape: {entity_emb.shape}")
# Expected: (14541, 128)
```

---

## 🔧 Phase 2: Tạo SAT Adapter

### 2.1 File: `sat_adapter.py`

```python
"""
SAT Adapter - Load và sử dụng SAT models cho Enhanced GraphRAG
"""
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Add SAT to path
sys.path.append("SAT/aligner/model")
from model_gt import CLIP, tokenize
from data_helper import get_mid2id, get_rel2id, get_id2text, construct_graph

class SATAdapter:
    """Adapter để sử dụng SAT's Graph Transformer và Text Encoder"""
    
    def __init__(
        self,
        data_path: str = "sat_data_backup/FB15k-237N",
        checkpoint_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.data_path = data_path
        self.device = device
        
        # Load mappings
        self.id2text = get_id2text(os.path.join(data_path, "id2text.txt"))
        self.mid2id = get_mid2id(os.path.join(data_path, "mid2id.txt"))
        self.rel2id = get_rel2id(os.path.join(data_path, "rel2id.txt"))
        
        self.entity_num = len(self.mid2id)
        self.relation_num = len(self.rel2id)
        
        # Build args for CLIP model
        self.args = self._build_args()
        
        # Initialize model
        self.model = CLIP(self.args).to(device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"✅ Loaded checkpoint from {checkpoint_path}")
        
        self.model.eval()
    
    def _build_args(self):
        """Build args object for CLIP model"""
        class Args:
            pass
        
        args = Args()
        args.entity_num = self.entity_num
        args.relation_num = self.relation_num
        args.context_length = 128
        args.embed_dim = 128
        args.transformer_heads = 8
        args.transformer_layers = 12
        args.transformer_width = 512
        args.vocab_size = 49408
        args.gnn_type = "gt"
        args.gnn_input = 128
        args.gnn_hidden = 128
        args.gnn_output = 128
        args.gt_layers = 3
        args.att_d_model = 128
        args.gt_head = 8
        args.att_norm = True
        args.if_pos = False
        args.node_num = 1
        args.edge_coef = 10
        args.neigh_num = 3
        args.lr = 2e-5
        
        return args
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode text descriptions to embeddings"""
        tokens = tokenize(texts, context_length=self.args.context_length)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            text_feats = self.model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        
        return text_feats.cpu()
    
    def encode_entities(self, entity_ids: list, graph: Data) -> torch.Tensor:
        """Encode entity IDs to graph embeddings"""
        entity_ids = torch.tensor(entity_ids).to(self.device)
        graph = graph.to(self.device)
        
        with torch.no_grad():
            graph_feats = self.model.encode_graph(entity_ids, graph)
            graph_feats = graph_feats / graph_feats.norm(dim=-1, keepdim=True)
        
        return graph_feats.cpu()
    
    def get_entity_text(self, entity_id: int) -> str:
        """Get text description for an entity"""
        return self.id2text.get(entity_id, "")
    
    def compute_similarity(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between text and graph embeddings"""
        return torch.mm(text_emb, graph_emb.t())


# Usage example
if __name__ == "__main__":
    adapter = SATAdapter(
        data_path="sat_data_backup/FB15k-237N",
        checkpoint_path="SAT/aligner/checkpoints/FB15k-237N/gt-xxx_best.pkl"
    )
    
    # Test text encoding
    texts = ["University of Central Florida", "A public research university"]
    text_embs = adapter.encode_text(texts)
    print(f"Text embeddings: {text_embs.shape}")
```

---

## 🔄 Phase 3: Integrate vào Enhanced GraphRAG

### 3.1 Sửa đổi `enhanced_graphrag.py`

```python
# Thêm vào EnhancedGraphRAG class

class EnhancedGraphRAG:
    def __init__(
        self,
        # ... existing params ...
        use_sat_encoder: bool = False,
        sat_checkpoint_path: str = None,
    ):
        # ... existing init ...
        
        # SAT Encoder (optional)
        self.use_sat_encoder = use_sat_encoder
        if use_sat_encoder:
            from sat_adapter import SATAdapter
            self.sat_adapter = SATAdapter(
                checkpoint_path=sat_checkpoint_path
            )
    
    def _encode_query_sat(self, query: str) -> np.ndarray:
        """Encode query using SAT's text encoder"""
        text_emb = self.sat_adapter.encode_text([query])
        return text_emb.numpy()
    
    def hybrid_retrieve_v2(self, query: str, top_k: int = 10, alpha: float = 0.6):
        """
        Hybrid retrieval với SAT encoders
        - alpha: weight cho semantic score
        - (1-alpha): weight cho graph score (từ SAT)
        """
        # Semantic search (existing)
        query_emb = self.embedding_model.encode([query])
        semantic_scores, indices = self.index.search(query_emb, top_k * 2)
        
        if self.use_sat_encoder:
            # SAT-based scoring
            sat_query_emb = self._encode_query_sat(query)
            # ... compute SAT-based relevance scores ...
        
        # Combine scores
        # ... hybrid scoring logic ...
        
        return top_chunks
```

---

## 📊 Phase 4: Training Pipeline cho Custom Data

Nếu muốn train SAT trên data riêng (medical, etc.):

### 4.1 Tạo data format

```python
# data_converter.py

def convert_to_sat_format(
    entities: dict,          # {entity_name: description}
    triples: list,           # [(head, rel, tail), ...]
    output_dir: str
):
    """Convert custom data to SAT format"""
    
    # Create entity2id
    entity2id = {ent: i for i, ent in enumerate(entities.keys())}
    
    # Create id2text.txt
    with open(f"{output_dir}/id2text.txt", "w") as f:
        for ent, desc in entities.items():
            f.write(f"{entity2id[ent]}\t{desc}\n")
    
    # Create mid2id.txt (use entity names as MIDs)
    with open(f"{output_dir}/mid2id.txt", "w") as f:
        for ent, eid in entity2id.items():
            f.write(f"{ent}\t{eid}\n")
    
    # Create rel2id.txt
    relations = list(set(t[1] for t in triples))
    rel2id = {rel: i for i, rel in enumerate(relations)}
    with open(f"{output_dir}/rel2id.txt", "w") as f:
        for rel, rid in rel2id.items():
            f.write(f"{rel}\t{rid}\n")
    
    # Create train.txt
    with open(f"{output_dir}/train.txt", "w") as f:
        for head, rel, tail in triples:
            f.write(f"{head}\t{rel}\t{tail}\n")
    
    print(f"✅ Converted {len(entities)} entities, {len(triples)} triples")
```

### 4.2 Generate entity descriptions với LLM

```python
# generate_descriptions.py

from openai import OpenAI

def generate_entity_descriptions(entities: list, domain: str = "medical"):
    """Use LLM to generate descriptions for entities"""
    client = OpenAI()
    
    descriptions = {}
    for entity in entities:
        prompt = f"""Write a concise 2-3 sentence description of "{entity}" 
        in the context of {domain}. Be factual and informative."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        descriptions[entity] = response.choices[0].message.content
    
    return descriptions
```

---

## ✅ Checklist

### Phase 1: Setup & Verify
- [ ] Cài đặt `torch-scatter`, `torch-geometric`
- [ ] Chạy `main.py` với FB15k-237N
- [ ] Verify checkpoint được tạo

### Phase 2: Integration
- [ ] Tạo `sat_adapter.py`
- [ ] Test encode text/entities
- [ ] Integrate vào EnhancedGraphRAG

### Phase 3: Evaluation
- [ ] So sánh performance với/không SAT
- [ ] Đo memory usage
- [ ] Benchmark speed

### Phase 4: Custom Data (Optional)
- [ ] Convert medical data sang SAT format
- [ ] Generate entity descriptions
- [ ] Fine-tune từ FB15k-237N checkpoint

---

## 🔗 References

- SAT Paper: "Structure Aware Alignment and Tuning for Knowledge Graph Question Answering"
- SAT GitHub: https://github.com/liuyudiy/SAT
- FB15k-237: Benchmark dataset cho Knowledge Graph Completion
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

---

*Tiếp tục ngày mai! 🚀*
