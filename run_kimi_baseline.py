import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import json
import time
import pickle
import numpy as np
import faiss
import httpx
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

from simple_graphrag import SimpleGraphRAG

CACHE_DIR = "sat_data"
QA_PATH = "qa_eval.json"
OUT_PATH = "baseline_results.json"

TOP_K = 6
ALPHA = 0.8
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
        max_retries=0,  # IMPORTANT: disable hidden long retries
        http_client=httpx.Client(timeout=httpx.Timeout(90.0, connect=15.0))
    )

def load_cache() -> SimpleGraphRAG:
    rag = SimpleGraphRAG(embedding_model_name="all-MiniLM-L6-v2")

    with open(os.path.join(CACHE_DIR, "chunks.json"), encoding="utf-8") as f:
        rag.dataset = json.load(f)

    rag.embeddings = np.load(os.path.join(CACHE_DIR, "embeddings.npy"))
    rag.index = faiss.read_index(os.path.join(CACHE_DIR, "faiss.index"))

    with open(os.path.join(CACHE_DIR, "kg.pkl"), "rb") as f:
        rag.kg = pickle.load(f)

    with open(os.path.join(CACHE_DIR, "chunk_entities.pkl"), "rb") as f:
        rag.chunk_entities = pickle.load(f)

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
    rag = load_cache()

    print("✅ Cache loaded")
    print("Chunks:", len(rag.dataset))
    print("KG:", rag.kg.number_of_nodes(), "nodes |", rag.kg.number_of_edges(), "edges")

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    outputs = []
    for i, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("answer") or item.get("groundtruth") or ""

        print(f"\n[{i}/{len(data)}] {q}")
        r = rag.graphrag_query(q, top_k=TOP_K, alpha=ALPHA)
        ctx = r["context"]

        ans = kimi_answer(q, ctx, client)

        outputs.append({
            "question": q,
            "ans": ans,
            "groundtruth": gt
        })

        time.sleep(SLEEP)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
