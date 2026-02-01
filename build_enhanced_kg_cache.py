# build_enhanced_kg_cache.py
"""
Build KG cache using EnhancedGraphRAG (với Graph Transformer từ SAT)
Output: enhanced_sat_data/
"""
import os
import json
import pickle
import numpy as np
import faiss
import torch

from enhanced_graphrag import EnhancedGraphRAG

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_long_context(text, max_chars=800):
    """Split long text into ~800-char paragraphs by sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def main():
    # === CONFIG ===
    corpus_path = "SAT/aligner/data/FB15k-237N/id3text.txt"
    out_dir = "enhanced_sat_data"
    os.makedirs(out_dir, exist_ok=True)
    
    # === INIT EnhancedGraphRAG ===
    print("🚀 Initializing EnhancedGraphRAG...")
    rag = EnhancedGraphRAG(
        embedding_model_name="all-MiniLM-L6-v2",
        use_graph_transformer=False,  # Tắt Graph Transformer để tránh segfault với large KG
        graph_transformer_dim=128,
        graph_transformer_layers=3,
        working_dir=out_dir
    )
    
    # === LOAD & CHUNK TEXT ===
    print(f"📄 Loading corpus from: {corpus_path}")
    text = load_txt(corpus_path)
    chunks = split_long_context(text, max_chars=800)
    print(f"   Created {len(chunks)} chunks")
    
    # === BUILD SYSTEM ===
    print("🔧 Building system...")
    
    # 1. Add documents
    rag.add_documents(chunks)
    
    # 2. Build KG (entities, relations, triples)
    print("   Building Knowledge Graph...")
    rag.build_kg(add_cooccurrence=True)
    
    # 3. Build embeddings + FAISS index
    print("   Building embeddings & FAISS index...")
    rag.build_embeddings(normalize=True, batch_size=32)
    
    # 4. Build Graph Transformer embeddings (optional - có thể bỏ qua nếu gặp lỗi)
    print("   Building Graph Transformer embeddings...")
    device = "cpu"  # Dùng CPU để tránh segfault trên MPS
    try:
        rag.build_graph_transformer(device=device)
    except Exception as e:
        print(f"   ⚠️ Graph Transformer failed: {e}")
        print("   Continuing without Graph Transformer embeddings...")
    
    # === SAVE ALL ARTIFACTS ===
    print("💾 Saving cache files...")
    
    # chunks.json
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(rag.chunks, f, ensure_ascii=False, indent=2)
    
    # embeddings.npy
    np.save(os.path.join(out_dir, "embeddings.npy"), rag.embeddings)
    
    # faiss.index
    faiss.write_index(rag.index, os.path.join(out_dir, "faiss.index"))
    
    # kg.pkl (NetworkX graph từ kg_builder)
    with open(os.path.join(out_dir, "kg.pkl"), "wb") as f:
        pickle.dump(rag.kg_builder.kg, f)
    
    # chunk_entities.pkl
    with open(os.path.join(out_dir, "chunk_entities.pkl"), "wb") as f:
        pickle.dump(rag.chunk_entities, f)
    
    # entity2id.pkl & relation2id.pkl (SAT-style mappings)
    with open(os.path.join(out_dir, "entity2id.pkl"), "wb") as f:
        pickle.dump(rag.kg_builder.entity2id, f)
    
    with open(os.path.join(out_dir, "relation2id.pkl"), "wb") as f:
        pickle.dump(rag.kg_builder.relation2id, f)
    
    # node_embeddings.pt (Graph Transformer output)
    if rag.node_embeddings is not None:
        torch.save(rag.node_embeddings, os.path.join(out_dir, "node_embeddings.pt"))
    
    # triples list
    triples_data = [
        {"head": t.head, "relation": t.relation, "tail": t.tail, "confidence": t.confidence}
        for t in rag.kg_builder.triples
    ]
    with open(os.path.join(out_dir, "triples.json"), "w", encoding="utf-8") as f:
        json.dump(triples_data, f, ensure_ascii=False, indent=2)
    
    # meta.json
    meta = {
        "corpus_path": corpus_path,
        "num_chunks": len(rag.chunks),
        "num_entities": len(rag.kg_builder.entity2id),
        "num_relations": len(rag.kg_builder.relation2id),
        "num_triples": len(rag.kg_builder.triples),
        "kg_nodes": int(rag.kg_builder.kg.number_of_nodes()),
        "kg_edges": int(rag.kg_builder.kg.number_of_edges()),
        "graph_transformer_dim": 128,
        "graph_transformer_layers": 3,
        "device": device
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Saved all cache to enhanced_sat_data/")
    print("=" * 50)
    print(f"📊 Statistics:")
    print(f"   Chunks: {meta['num_chunks']}")
    print(f"   Entities: {meta['num_entities']}")
    print(f"   Relations: {meta['num_relations']}")
    print(f"   Triples: {meta['num_triples']}")
    print(f"   KG Nodes: {meta['kg_nodes']}")
    print(f"   KG Edges: {meta['kg_edges']}")
    print("=" * 50)

if __name__ == "__main__":
    main()
