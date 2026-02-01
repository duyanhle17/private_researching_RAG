# build_kg_cache.py
import os
import json
import pickle
import numpy as np
import faiss

from simple_graphrag import SimpleGraphRAG, split_long_context

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    corpus_path = "SAT/aligner/data/FB15k-237N/id3text.txt"
    out_dir = "sat_data"
    os.makedirs(out_dir, exist_ok=True)

    rag = SimpleGraphRAG(embedding_model_name="all-MiniLM-L6-v2", chunk_size=1200, chunk_overlap=100)

    text = load_txt(corpus_path)
    rag.dataset = split_long_context(text, max_chars=800)

    rag.build_embeddings_and_index()
    rag.build_kg(use_dependency=True)

    # save all needed artifacts (not only kg.pkl)
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(rag.dataset, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(out_dir, "embeddings.npy"), rag.embeddings)
    faiss.write_index(rag.index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "kg.pkl"), "wb") as f:
        pickle.dump(rag.kg, f)

    with open(os.path.join(out_dir, "chunk_entities.pkl"), "wb") as f:
        pickle.dump(rag.chunk_entities, f)

    meta = {
        "corpus_path": corpus_path,
        "num_chunks": len(rag.dataset),
        "kg_nodes": int(rag.kg.number_of_nodes()),
        "kg_edges": int(rag.kg.number_of_edges()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Saved all cache to sat_data/")
    print(meta)

if __name__ == "__main__":
    main()
