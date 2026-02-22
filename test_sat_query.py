# Quick test: verify SATGraphRAG loads and queries correctly
from run_sat_baseline import SATGraphRAG

rag = SATGraphRAG("SAT/aligner/data/FB15k-237N", "sat_fb15k_cache")

print(f"\n=== SAT GraphRAG Stats ===")
print(f"Chunks (entity texts): {len(rag.chunks)}")
print(f"Entities with KG neighbors: {len(rag.neighbors)}")
print(f"FAISS index vectors: {rag.index.ntotal}")

# Test query
q = "What is a government, according to the given definition?"
r = rag.query(q, top_k=5, alpha=0.7)

print(f"\n=== Query: {q} ===")
print(f"KG facts: {r['kg_facts']}")
print(f"\nTop scores:")
for s in r["scores"]:
    print(f"  chunk[{s['chunk_idx']}] combined={s['combined']:.4f}  sem={s['semantic']:.4f}  graph={s['graph']:.4f}")

print(f"\nContext (first 500 chars):")
print(r["context"][:500])
print("...")
