# run_enhanced_baseline.py
"""
Run QA evaluation using EnhancedGraphRAG cache (với Graph Transformer từ SAT)
Load từ: enhanced_sat_data/
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import json
import time
import pickle
import numpy as np
import faiss
import torch
import httpx
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

from enhanced_graphrag import EnhancedGraphRAG

# === CONFIG ===
CACHE_DIR = "enhanced_sat_data"
QA_PATH = "qa_eval.json"
OUT_PATH = "enhanced_results.json"

TOP_K = 6
ALPHA = 0.7  # Có thể điều chỉnh (0.7 semantic + 0.3 graph)
SLEEP = 1.5
MAX_RETRY = 3

def get_client():
    api_key = os.getenv("NVAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing NVAPI_KEY. Set it in terminal first.")
    logging.getLogger("openai").setLevel(logging.WARNING)

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        max_retries=0,
        http_client=httpx.Client(timeout=httpx.Timeout(90.0, connect=15.0))
    )

def load_enhanced_cache() -> EnhancedGraphRAG:
    """Load EnhancedGraphRAG từ cache files"""
    print(f"📂 Loading cache from: {CACHE_DIR}/")
    
    rag = EnhancedGraphRAG(
        embedding_model_name="all-MiniLM-L6-v2",
        use_graph_transformer=True,
        working_dir=CACHE_DIR
    )
    
    # Load chunks
    with open(os.path.join(CACHE_DIR, "chunks.json"), encoding="utf-8") as f:
        rag.chunks = json.load(f)
    
    # Load embeddings
    rag.embeddings = np.load(os.path.join(CACHE_DIR, "embeddings.npy"))
    
    # Load FAISS index
    rag.index = faiss.read_index(os.path.join(CACHE_DIR, "faiss.index"))
    
    # Load KG (NetworkX)
    with open(os.path.join(CACHE_DIR, "kg.pkl"), "rb") as f:
        rag.kg_builder.kg = pickle.load(f)
    
    # Load chunk_entities
    with open(os.path.join(CACHE_DIR, "chunk_entities.pkl"), "rb") as f:
        rag.chunk_entities = pickle.load(f)
    
    # Load entity2id & relation2id
    with open(os.path.join(CACHE_DIR, "entity2id.pkl"), "rb") as f:
        rag.kg_builder.entity2id = pickle.load(f)
        rag.kg_builder.id2entity = {v: k for k, v in rag.kg_builder.entity2id.items()}
    
    with open(os.path.join(CACHE_DIR, "relation2id.pkl"), "rb") as f:
        rag.kg_builder.relation2id = pickle.load(f)
        rag.kg_builder.id2relation = {v: k for k, v in rag.kg_builder.relation2id.items()}
    
    # Load node_embeddings (optional)
    node_emb_path = os.path.join(CACHE_DIR, "node_embeddings.pt")
    if os.path.exists(node_emb_path):
        rag.node_embeddings = torch.load(node_emb_path)
        print(f"   Loaded node embeddings: {rag.node_embeddings.shape}")
    
    return rag

def kimi_answer(question: str, context: str, client, max_retry=MAX_RETRY) -> str:
    prompt = f"""
You are a precise QA assistant.

Answer using ONLY the provided context.
- If the answer is a number/date/location/name, copy it EXACTLY as in the context.
- Do not guess or use outside knowledge.
- If the context does not contain the answer, reply exactly:
not stated in the text

Context:
{context}

Question:
{question}

Answer (1 sentence max):
""".strip()

    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64
            )
            return resp.choices[0].message.content.strip()

        except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as e:
            wait = 2.0 * (attempt + 1)
            print(f"⚠️ API transient: {type(e).__name__} retry {attempt+1}/{max_retry} (sleep {wait:.1f}s)")
            time.sleep(wait)

        except Exception as e:
            print(f"❌ API fatal: {type(e).__name__}: {e}")
            return "ERROR: api_fatal"

    return "ERROR: api_failed"

def main():
    client = get_client()
    rag = load_enhanced_cache()

    print("\n✅ Enhanced cache loaded")
    print(f"   Chunks: {len(rag.chunks)}")
    print(f"   Entities: {len(rag.kg_builder.entity2id)}")
    print(f"   Relations: {len(rag.kg_builder.relation2id)}")
    print(f"   KG: {rag.kg_builder.kg.number_of_nodes()} nodes | {rag.kg_builder.kg.number_of_edges()} edges")
    
    # Load meta
    meta_path = os.path.join(CACHE_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"   Triples: {meta.get('num_triples', 'N/A')}")

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    outputs = []
    for i, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("answer") or item.get("groundtruth") or ""

        print(f"\n[{i}/{len(data)}] {q}")
        
        # Query using EnhancedGraphRAG
        r = rag.query(q, top_k=TOP_K, alpha=ALPHA, include_kg_facts=True)
        ctx = r["context"]
        kg_facts = r.get("kg_facts", [])

        ans = kimi_answer(q, ctx, client)

        outputs.append({
            "question": q,
            "ans": ans,
            "groundtruth": gt,
            "kg_facts": kg_facts,  # Thêm KG facts để debug
            "retrieval_scores": r.get("retrieval_scores", [])
        })

        time.sleep(SLEEP)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {OUT_PATH}")
    
    # Quick accuracy summary
    correct = sum(1 for o in outputs if o["groundtruth"].lower() in o["ans"].lower())
    print(f"📊 Quick check: {correct}/{len(outputs)} answers contain groundtruth")

if __name__ == "__main__":
    main()
