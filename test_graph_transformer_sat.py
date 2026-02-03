# test_graph_transformer_sat.py
"""
Test Graph Transformer với FB15k-237N dataset (KG có sẵn).
Không cần build KG - chỉ load và chạy Graph Transformer.
"""
import os
import torch
import time
from typing import Dict, List, Tuple

from graph_transformer_v2 import GraphTransformerEmbedder

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_mid2id(path: str) -> Dict[str, int]:
    """Load Freebase MID → internal ID"""
    mid2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mid2id[parts[0]] = int(parts[1])
    return mid2id


def load_rel2id(path: str) -> Dict[str, int]:
    """Load relation → ID"""
    rel2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rel2id[parts[0]] = int(parts[1])
    return rel2id


def load_id2text(path: str) -> Dict[int, str]:
    """Load entity ID → text description"""
    id2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                id2text[int(parts[0])] = parts[1]
    return id2text


def load_triples(path: str, mid2id: Dict[str, int], rel2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """Load triples và convert sang IDs"""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h_mid, rel, t_mid = parts
                h_id = mid2id.get(h_mid)
                r_id = rel2id.get(rel)
                t_id = mid2id.get(t_mid)
                
                if h_id is not None and r_id is not None and t_id is not None:
                    triples.append((h_id, r_id, t_id))
    return triples


def build_edge_tensors(triples: List[Tuple[int, int, int]], num_relations: int):
    """Build edge_index và edge_type tensors"""
    src_list, dst_list, rel_list = [], [], []
    
    for h_id, r_id, t_id in triples:
        # Forward edge
        src_list.append(h_id)
        dst_list.append(t_id)
        rel_list.append(r_id)
        
        # Inverse edge
        src_list.append(t_id)
        dst_list.append(h_id)
        rel_list.append(r_id + num_relations)
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(rel_list, dtype=torch.long)
    
    return edge_index, edge_type


# ============================================================================
# Main Test
# ============================================================================

def main():
    data_dir = "sat_data_backup/FB15k-237N"
    
    print("=" * 60)
    print("🧪 Test Graph Transformer với FB15k-237N")
    print("=" * 60)
    
    # === 1. Load Data ===
    print("\n📂 Step 1: Loading data...")
    
    mid2id = load_mid2id(os.path.join(data_dir, "mid2id.txt"))
    print(f"   ✅ {len(mid2id)} entities")
    
    rel2id = load_rel2id(os.path.join(data_dir, "rel2id.txt"))
    print(f"   ✅ {len(rel2id)} relations")
    
    id2text = load_id2text(os.path.join(data_dir, "id2text.txt"))
    print(f"   ✅ {len(id2text)} entity descriptions")
    
    triples = load_triples(os.path.join(data_dir, "train.txt"), mid2id, rel2id)
    print(f"   ✅ {len(triples)} triples")
    
    # === 2. Build Graph Tensors ===
    print("\n🔧 Step 2: Building graph tensors...")
    
    num_entities = len(mid2id)
    num_relations = len(rel2id)
    
    edge_index, edge_type = build_edge_tensors(triples, num_relations)
    print(f"   Edge index: {edge_index.shape}")
    print(f"   Edge type: {edge_type.shape}")
    print(f"   Total edges (forward + inverse): {edge_index.shape[1]}")
    
    # === 3. Run Graph Transformer ===
    print("\n🔮 Step 3: Running Graph Transformer...")
    
    embedder = GraphTransformerEmbedder(
        num_entities=num_entities,
        num_relations=num_relations,
        d_model=128,
        n_layers=2,
        n_heads=4,
        device="cpu"
    )
    
    start = time.time()
    node_embeddings = embedder.compute_embeddings(edge_index, edge_type)
    elapsed = time.time() - start
    
    print(f"   ✅ Done in {elapsed:.2f}s")
    print(f"   Node embeddings: {node_embeddings.shape}")
    print(f"   Memory: {node_embeddings.numel() * 4 / 1024 / 1024:.2f} MB")
    
    # === 4. Test Similarity Search ===
    print("\n🔍 Step 4: Testing similarity search...")
    
    # Test với entity 1: "University of Central Florida"
    test_entity_id = 1
    test_entity_text = id2text.get(test_entity_id, "Unknown")[:80]
    print(f"\n   Query entity [{test_entity_id}]: {test_entity_text}...")
    
    similar_ids, similar_scores = embedder.get_similar_entities(test_entity_id, top_k=5)
    
    print(f"\n   Top 5 similar entities (by Graph Transformer embedding):")
    for rank, (ent_id, score) in enumerate(zip(similar_ids.tolist(), similar_scores.tolist()), 1):
        ent_text = id2text.get(ent_id, "Unknown")[:60]
        print(f"   {rank}. [{ent_id}] (score={score:.4f})")
        print(f"      {ent_text}...")
    
    # === 5. Save Results ===
    print("\n💾 Step 5: Saving embeddings...")
    
    out_dir = "sat_kg_data"
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save(node_embeddings, os.path.join(out_dir, "node_embeddings.pt"))
    torch.save({
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_entities": num_entities,
        "num_relations": num_relations
    }, os.path.join(out_dir, "graph_data.pt"))
    
    print(f"   ✅ Saved to {out_dir}/")
    
    print("\n" + "=" * 60)
    print("✅ Graph Transformer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
