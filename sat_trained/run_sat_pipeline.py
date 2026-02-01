# # =========================
# # MODE 2: GraphRAG + SAT LoRA (Qwen) + Kimi Final
# # macOS-safe: avoid segfault by setting env FIRST and lazy-import heavy libs
# # =========================

# import os

# # MUST be set BEFORE importing numpy/faiss/spacy/torch/transformers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# import json
# import time
# import pickle

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# import httpx
# from openai import OpenAI

# # =========================
# # CONFIG
# # =========================
# CACHE_DIR = "sat_data"
# KG_PATH = os.path.join(CACHE_DIR, "kg.pkl")
# FAISS_PATH = os.path.join(CACHE_DIR, "faiss.index")
# EMB_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
# CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
# CHUNK_ENTS_PKL = os.path.join(CACHE_DIR, "chunk_entities.pkl")

# QA_PATH = "qa_eval.json"
# OUT_PATH = "sat_rag_kimi_results.json"

# TOP_K = 5
# ALPHA = 0.8

# QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# SAT_LORA_PATH = "sat_lora_model"

# SAT_MAX_NEW_TOKENS = 120
# SAT_TEMPERATURE = 0.2
# SAT_TOP_P = 0.9

# SLEEP_BETWEEN_CALLS = 1.2
# KIMI_MODEL = "moonshotai/kimi-k2-instruct-0905"

# # Use env var only (do NOT hardcode keys in code)
# NVAPI_KEY = os.getenv("NVAPI_KEY")
# if not NVAPI_KEY:
#     raise RuntimeError(
#         "Missing NVAPI_KEY. Set it in your shell first, e.g.\n"
#         "export NVAPI_KEY='nvapi-...'\n"
#     )

# # =========================
# # KIMI CLIENT (NVIDIA NIM)
# # =========================
# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=NVAPI_KEY,
#     http_client=httpx.Client(timeout=60.0),
# )

# # =========================
# # CACHED GRAPHRAG LOADER (lazy-import heavy libs)
# # =========================
# class CachedGraphRAG:
#     def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
#         # Lazy imports to avoid native-lib conflicts during SAT model loading
#         import numpy as np
#         import faiss
#         import spacy
#         import networkx as nx
#         from sentence_transformers import SentenceTransformer

#         self.np = np
#         self.faiss = faiss
#         self.nx = nx

#         self.embedding_model = SentenceTransformer(embedding_model_name)
#         self.nlp = spacy.load("en_core_web_sm")
#         self.nlp.max_length = 10_000_000

#         # load chunks
#         with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
#             self.dataset = json.load(f)  # list[str]

#         # load chunk_entities
#         with open(CHUNK_ENTS_PKL, "rb") as f:
#             ce = pickle.load(f)
#         self.chunk_entities = [set(x) for x in ce]

#         # load KG
#         with open(KG_PATH, "rb") as f:
#             self.kg: nx.DiGraph = pickle.load(f)

#         # load embeddings + faiss index
#         self.embeddings = self.np.load(EMB_PATH).astype("float32")
#         self.index = self.faiss.read_index(FAISS_PATH)

#         # sanity
#         if len(self.dataset) != self.embeddings.shape[0]:
#             raise ValueError(
#                 f"❌ Mismatch: chunks={len(self.dataset)} vs embeddings={self.embeddings.shape[0]}.\n"
#                 f"Bạn đang dùng chunks khác với embeddings/index cũ."
#             )

#     def _semantic_search(self, query: str, top_k: int = 5):
#         q_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
#         D, I = self.index.search(q_emb, top_k)
#         return [(int(i), float(D[0][idx])) for idx, i in enumerate(I[0])]

#     def _graph_search_scores(self, query: str):
#         q_doc = self.nlp(query)
#         q_entities = set(ent.text.strip() for ent in q_doc.ents if ent.text.strip())
#         if not q_entities:
#             return self.np.zeros(len(self.dataset), dtype=float)

#         scores = self.np.array(
#             [len(self.chunk_entities[i].intersection(q_entities)) for i in range(len(self.dataset))],
#             dtype=float
#         )
#         if scores.max() > 0:
#             scores = scores / (scores.max() + 1e-12)
#         return scores

#     def graphrag_query(self, query: str, top_k: int = 5, alpha: float = 0.8):
#         sem = self._semantic_search(query, top_k=top_k * 2)
#         graph_scores = self._graph_search_scores(query)

#         combined = []
#         for i, sem_score in sem:
#             gscore = graph_scores[i] if i < len(graph_scores) else 0.0
#             score = alpha * sem_score + (1.0 - alpha) * gscore
#             combined.append((i, score))

#         combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
#         combined_idxs = [i for i, _ in combined_sorted]
#         combined_chunks = [self.dataset[i] for i in combined_idxs]

#         # KG neighbor facts (optional boost)
#         kg_facts = []
#         q_doc = self.nlp(query)
#         q_entities = set(ent.text.strip() for ent in q_doc.ents if ent.text.strip())

#         for qe in q_entities:
#             if self.kg is not None and qe in self.kg:
#                 # outgoing
#                 for nb in list(self.kg.successors(qe))[:3]:
#                     edge = self.kg.get_edge_data(qe, nb) or {}
#                     rel = edge.get("relation", "related_to")
#                     kg_facts.append(f"{qe} {rel} {nb}.")
#                 # incoming
#                 for nb in list(self.kg.predecessors(qe))[:3]:
#                     edge = self.kg.get_edge_data(nb, qe) or {}
#                     rel = edge.get("relation", "related_to")
#                     kg_facts.append(f"{nb} {rel} {qe}.")

#         context = " ".join(combined_chunks + kg_facts)
#         return {"query": query, "context": context}


# # =========================
# # LOAD QWEN + SAT LORA (CPU)
# # =========================
# print("🔹 Loading Qwen + SAT LoRA (CPU)")

# print("A) tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(
#     QWEN_MODEL_NAME,
#     trust_remote_code=True,
#     use_fast=True,
# )
# print("A) tokenizer OK")

# print("B) base model...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     QWEN_MODEL_NAME,
#     trust_remote_code=True,
#     device_map=None,
#     dtype=torch.float32,
#     low_cpu_mem_usage=False,
# )
# print("B) base model OK")

# print("C) lora...")
# sat_model = PeftModel.from_pretrained(base_model, SAT_LORA_PATH)
# sat_model.eval()
# print("✅ SAT model ready")


# def sat_reason(question: str, context: str) -> str:
#     prompt = f"""You are a SAT-style QA reasoner.

# RULES:
# - Use ONLY the provided CONTEXT.
# - Do NOT guess.
# - If the answer is not in the context, output exactly: not stated in the text
# - If the answer is a number/name/location, copy EXACTLY.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# SAT_ANSWER (1-2 short sentences):
# """.strip()

#     inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

#     with torch.inference_mode():
#         out = sat_model.generate(
#             **inputs,
#             max_new_tokens=SAT_MAX_NEW_TOKENS,
#             do_sample=True,
#             temperature=SAT_TEMPERATURE,
#             top_p=SAT_TOP_P,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     text = tokenizer.decode(out[0], skip_special_tokens=True)
#     if "SAT_ANSWER" in text:
#         text = text.split("SAT_ANSWER", 1)[-1].replace(":", "").strip()
#     return text.strip()


# def kimi_final(question: str, context: str, sat_draft: str) -> str:
#     prompt = f"""
# You are a high-accuracy QA assistant.

# You MUST answer using ONLY the CONTEXT below.
# You may use SAT_DRAFT only as reasoning guidance, but do not add facts not in CONTEXT.
# If the answer is not in CONTEXT, reply exactly:
# not stated in the text

# CONTEXT:
# {context}

# SAT_DRAFT:
# {sat_draft}

# QUESTION:
# {question}

# FINAL ANSWER (1 sentence, concise):
# """.strip()

#     resp = client.chat.completions.create(
#         model=KIMI_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=96
#     )
#     return resp.choices[0].message.content.strip()


# def main():
#     print("🔹 Loading cached GraphRAG artifacts...")
#     rag = CachedGraphRAG()
#     print(f"✅ Cache loaded: chunks={len(rag.dataset)} | KG nodes={rag.kg.number_of_nodes()} edges={rag.kg.number_of_edges()}")

#     with open(QA_PATH, "r", encoding="utf-8") as f:
#         qa = json.load(f)
#     print(f"🔹 Loaded {len(qa)} questions")

#     outputs = []
#     for i, item in enumerate(qa, 1):
#         q = item["question"]
#         gt = item.get("answer", "")

#         print(f"\n[{i}/{len(qa)}] {q}")

#         # 1) retrieve context
#         r = rag.graphrag_query(q, top_k=TOP_K, alpha=ALPHA)
#         context = r["context"]

#         # 2) SAT draft
#         sat_draft = sat_reason(q, context)
#         print("SAT_DRAFT:", sat_draft)

#         # 3) Kimi final
#         try:
#             ans = kimi_final(q, context, sat_draft)
#         except Exception as e:
#             ans = f"ERROR: {type(e).__name__}"
#         print("KIMI:", ans)

#         outputs.append({
#             "question": q,
#             "context": context,
#             "sat_draft": sat_draft,
#             "ans": ans,
#             "groundtruth": gt
#         })

#         time.sleep(SLEEP_BETWEEN_CALLS)

#     with open(OUT_PATH, "w", encoding="utf-8") as f:
#         json.dump(outputs, f, indent=2, ensure_ascii=False)

#     print(f"\n✅ Saved {OUT_PATH}")


# if __name__ == "__main__":
#     main()


# =========================
# MODE 2 (DEBUG): GraphRAG + SAT LoRA (Qwen) + Kimi Final
# macOS-safe: set env FIRST + lazy-import heavy libs
# Adds step-by-step logs + timing + API error classification
# =========================

import os

# MUST be set BEFORE importing numpy/faiss/spacy/torch/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import time
import pickle
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import httpx
from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)

# =========================
# CONFIG
# =========================
CACHE_DIR = "sat_data"
KG_PATH = os.path.join(CACHE_DIR, "kg.pkl")
FAISS_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMB_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
CHUNK_ENTS_PKL = os.path.join(CACHE_DIR, "chunk_entities.pkl")

QA_PATH = "qa_eval.json"
OUT_PATH = "sat_rag_kimi_results.json"

TOP_K = 5
ALPHA = 0.8

QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAT_LORA_PATH = "sat_lora_model"

# If SAT draft is too slow on CPU, reduce these:
SAT_MAX_NEW_TOKENS = 32
SAT_TEMPERATURE = 0.0
SAT_TOP_P = 1.0
SAT_DO_SAMPLE = False  # set False for faster, deterministic draft

SLEEP_BETWEEN_CALLS = 2.5
KIMI_MODEL = "moonshotai/kimi-k2-instruct-0905"

# =========================
# LOGGING
# =========================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# API KEY
# =========================
NVAPI_KEY = os.getenv("NVAPI_KEY")
if not NVAPI_KEY:
    raise RuntimeError(
        "Missing NVAPI_KEY.\n"
        "Set it in your shell first, e.g.\n"
        "  export NVAPI_KEY='nvapi-...'\n"
    )

# =========================
# KIMI CLIENT (NVIDIA NIM)
# =========================
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVAPI_KEY,
    http_client=httpx.Client(timeout=60.0),
)

# =========================
# CACHED GRAPHRAG LOADER (lazy-import heavy libs)
# =========================
class CachedGraphRAG:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        # Lazy imports to avoid native-lib conflicts during SAT model loading
        import numpy as np
        import faiss
        import spacy
        import networkx as nx
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.faiss = faiss
        self.nx = nx

        log("  [RAG] Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        log("  [RAG] Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 10_000_000

        log("  [RAG] Loading chunks.json ...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)  # list[str]

        log("  [RAG] Loading chunk_entities.pkl ...")
        with open(CHUNK_ENTS_PKL, "rb") as f:
            ce = pickle.load(f)
        self.chunk_entities = [set(x) for x in ce]

        log("  [RAG] Loading kg.pkl ...")
        with open(KG_PATH, "rb") as f:
            self.kg: nx.DiGraph = pickle.load(f)

        log("  [RAG] Loading embeddings.npy + faiss.index ...")
        self.embeddings = self.np.load(EMB_PATH).astype("float32")
        self.index = self.faiss.read_index(FAISS_PATH)

        if len(self.dataset) != self.embeddings.shape[0]:
            raise ValueError(
                f"❌ Mismatch: chunks={len(self.dataset)} vs embeddings={self.embeddings.shape[0]}.\n"
                f"Bạn đang dùng chunks khác với embeddings/index cũ."
            )

        log(f"  [RAG] Ready: chunks={len(self.dataset)} | KG nodes={self.kg.number_of_nodes()} edges={self.kg.number_of_edges()}")

    def _semantic_search(self, query: str, top_k: int = 5):
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        return [(int(i), float(D[0][idx])) for idx, i in enumerate(I[0])]

    def _graph_search_scores(self, query: str):
        q_doc = self.nlp(query)
        q_entities = set(ent.text.strip() for ent in q_doc.ents if ent.text.strip())
        if not q_entities:
            return self.np.zeros(len(self.dataset), dtype=float)

        scores = self.np.array(
            [len(self.chunk_entities[i].intersection(q_entities)) for i in range(len(self.dataset))],
            dtype=float
        )
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-12)
        return scores

    def graphrag_query(self, query: str, top_k: int = 5, alpha: float = 0.8):
        sem = self._semantic_search(query, top_k=top_k * 2)
        graph_scores = self._graph_search_scores(query)

        combined = []
        for i, sem_score in sem:
            gscore = graph_scores[i] if i < len(graph_scores) else 0.0
            score = alpha * sem_score + (1.0 - alpha) * gscore
            combined.append((i, score))

        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
        combined_idxs = [i for i, _ in combined_sorted]
        combined_chunks = [self.dataset[i] for i in combined_idxs]

        kg_facts = []
        q_doc = self.nlp(query)
        q_entities = set(ent.text.strip() for ent in q_doc.ents if ent.text.strip())

        for qe in q_entities:
            if self.kg is not None and qe in self.kg:
                for nb in list(self.kg.successors(qe))[:3]:
                    edge = self.kg.get_edge_data(qe, nb) or {}
                    rel = edge.get("relation", "related_to")
                    kg_facts.append(f"{qe} {rel} {nb}.")
                for nb in list(self.kg.predecessors(qe))[:3]:
                    edge = self.kg.get_edge_data(nb, qe) or {}
                    rel = edge.get("relation", "related_to")
                    kg_facts.append(f"{nb} {rel} {qe}.")

        context = " ".join(combined_chunks + kg_facts)
        return {"query": query, "context": context}

# =========================
# LOAD QWEN + SAT LORA (CPU)
# =========================
log("🔹 Loading Qwen + SAT LoRA (CPU)")

log("A) tokenizer: start")
tokenizer = AutoTokenizer.from_pretrained(
    QWEN_MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)
log("A) tokenizer: OK")

log("B) base model: start")
base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    trust_remote_code=True,
    device_map=None,
    dtype=torch.float32,
    low_cpu_mem_usage=False,
)
log("B) base model: OK")

log("C) LoRA: start")
sat_model = PeftModel.from_pretrained(base_model, SAT_LORA_PATH)
sat_model.eval()
log("✅ SAT model ready")

# =========================
# SAT draft
# =========================
def sat_reason(question: str, context: str) -> str:
    prompt = f"""You are a SAT-style QA reasoner.

RULES:
- Use ONLY the provided CONTEXT.
- Do NOT guess.
- If the answer is not in the context, output exactly: not stated in the text
- If the answer is a number/name/location, copy EXACTLY.

CONTEXT:
{context}

QUESTION:
{question}

SAT_ANSWER (1-2 short sentences):
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.inference_mode():
        out = sat_model.generate(
            # **inputs,
            # max_new_tokens=SAT_MAX_NEW_TOKENS,
            # do_sample=SAT_DO_SAMPLE,
            # temperature=SAT_TEMPERATURE,
            # top_p=SAT_TOP_P,
            # pad_token_id=tokenizer.eos_token_id
             **inputs,
            max_new_tokens=SAT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "SAT_ANSWER" in text:
        text = text.split("SAT_ANSWER", 1)[-1].replace(":", "").strip()
    return text.strip()

# =========================
# KIMI final with detailed error handling
# =========================
def kimi_final(question: str, context: str, sat_draft: str, max_retry: int = 3) -> str:
    prompt = f"""
You are a high-accuracy QA assistant.

You MUST answer using ONLY the CONTEXT below.
You may use SAT_DRAFT only as reasoning guidance, but do not add facts not in CONTEXT.
If the answer is not in CONTEXT, reply exactly:
not stated in the text

CONTEXT:
{context}

SAT_DRAFT:
{sat_draft}

QUESTION:
{question}

FINAL ANSWER (1 sentence, concise):
""".strip()

    for attempt in range(1, max_retry + 1):
        try:
            log(f"  -> Kimi API call (attempt {attempt}/{max_retry})")
            resp = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=96
            )
            return resp.choices[0].message.content.strip()

        except AuthenticationError as e:
            return f"ERROR: AuthError (NVAPI_KEY invalid/expired). {str(e)[:160]}"

        except RateLimitError as e:
            wait_s = 3 * attempt
            log(f"  !! RateLimitError (429). sleep {wait_s}s")
            time.sleep(wait_s)

        except (APITimeoutError, APIConnectionError) as e:
            wait_s = 2 * attempt
            log(f"  !! Timeout/Connection error: {type(e).__name__}. sleep {wait_s}s")
            time.sleep(wait_s)

        except BadRequestError as e:
            return f"ERROR: BadRequest (prompt too long / invalid). {str(e)[:200]}"

        except Exception as e:
            wait_s = 2 * attempt
            log(f"  !! Unknown API error: {type(e).__name__}. sleep {wait_s}s")
            time.sleep(wait_s)

    return "ERROR: kimi_failed_after_retries"

# =========================
# MAIN
# =========================
def main():
    log("🔹 Loading cached GraphRAG artifacts...")
    rag = CachedGraphRAG()

    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa = json.load(f)
    log(f"🔹 Loaded {len(qa)} questions")

    outputs = []
    for i, item in enumerate(qa, 1):
        q = item["question"]
        gt = item.get("answer", "")

        log(f"\n[{i}/{len(qa)}] Q: {q}")

        # 1) retrieve context
        t0 = time.time()
        log("  -> RETRIEVE: start")
        r = rag.graphrag_query(q, top_k=TOP_K, alpha=ALPHA)
        context = r["context"]
        log(f"  -> RETRIEVE: done in {time.time()-t0:.2f}s | ctx_chars={len(context)}")

        # 2) SAT draft
        t1 = time.time()
        log("  -> SAT: start (local generate)")
        sat_draft = sat_reason(q, context)
        log(f"  -> SAT: done in {time.time()-t1:.2f}s")
        log(f"SAT_DRAFT: {sat_draft}")

        # 3) Kimi final
        t2 = time.time()
        ans = kimi_final(q, context, sat_draft, max_retry=3)
        log(f"  -> KIMI: done in {time.time()-t2:.2f}s")
        log(f"KIMI: {ans}")

        outputs.append({
            "question": q,
            "context": context,
            "sat_draft": sat_draft,
            "ans": ans,
            "groundtruth": gt
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    log(f"\n✅ Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
