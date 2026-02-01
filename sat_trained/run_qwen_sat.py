# # import os
# # import json
# # import torch
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from peft import PeftModel
# # from openai import OpenAI

# # # =========================================================
# # # CONFIG
# # # =========================================================
# # QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# # SAT_LORA_PATH = "sat_lora_model"
# # DEVICE = "cpu"

# # # =========================================================
# # # LOAD QWEN + SAT (LOAD ONLY – NO GENERATE)
# # # =========================================================
# # print("🔹 Loading Qwen + SAT LoRA (LOAD ONLY, CPU safe)")

# # tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

# # base_model = AutoModelForCausalLM.from_pretrained(
# #     QWEN_MODEL_NAME,
# #     torch_dtype=torch.float32,
# #     device_map=None
# # )

# # # 🔥 SAT MODEL THẬT – ĐÃ TRAIN
# # model = PeftModel.from_pretrained(base_model, SAT_LORA_PATH)
# # model.eval()

# # print("✅ Qwen + SAT loaded successfully (no generation mode)")

# # # =========================================================
# # # SAT SIGNAL (REASONING BIAS)
# # # =========================================================
# # sat_signal = (
# #     "The model has been fine-tuned with SAT-style logical constraints. "
# #     "Answers should be concise, logically valid, and directly grounded in the context."
# # )

# # # =========================================================
# # # LOAD KIMI K2 (NVIDIA NIM)
# # # =========================================================
# # os.environ["NVAPI_KEY"] = os.getenv("NVAPI_KEY") or "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"

# # client = OpenAI(
# #     base_url="https://integrate.api.nvidia.com/v1",
# #     api_key=os.getenv("NVAPI_KEY")
# # )





# # # =========================================================
# # # KIMI FINAL ANSWER
# # # =========================================================
# # def kimi_sat(question: str, context: str, sat_signal: str) -> str:
# #     prompt = f"""
# # You are a high-accuracy QA assistant.

# # The system has been conditioned with SAT-style logical reasoning constraints.

# # SAT bias:
# # {sat_signal}

# # Answer using ONLY the provided context.

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """

# #     resp = client.chat.completions.create(
# #         model="moonshotai/kimi-k2-instruct-0905",
# #         messages=[{"role": "user", "content": prompt}],
# #         temperature=0.2,
# #         max_tokens=128
# #     )

# #     return resp.choices[0].message.content.strip()

# # # =========================================================
# # # LOAD DATA
# # # =========================================================
# # with open("qa_eval.json", encoding="utf-8") as f:
# #     data = json.load(f)

# # print(f"🔹 Loaded {len(data)} questions")

# # # =========================================================
# # # RUN INFERENCE (10 QUESTIONS – STABLE)
# # # =========================================================
# # # for i, item in enumerate(data[:10]):
# # #     print("\n" + "=" * 80)
# # #     print(f"Q{i+1}: {item['question']}")
# # #     print("-" * 80)

# # #     answer = kimi_answer(
# # #         item["question"],
# # #         item["context"],
# # #         sat_signal
# # #     )

# # #     print("✨ Kimi Final Answer:")
# # #     print(answer)

# # # print("\n🎉 DONE – SAT-loaded + Kimi inference finished successfully")


# # baseline_outputs = []
# # sat_outputs = []

# # for item in data:
# #     q = item["question"]
# #     gt = item["answer"]

# #     # BASELINE
# #     ans_base = kimi_baseline(q)
# #     baseline_outputs.append({
# #         "question": q,
# #         "ans": ans_base,
# #         "groundtruth": gt
# #     })

# #     # SAT
# #     ans_sat = kimi_sat(q, sat_signal)
# #     sat_outputs.append({
# #         "question": q,
# #         "ans": ans_sat,
# #         "groundtruth": gt
# #     })

# # # SAVE FILES
# # with open("baseline_results.json", "w", encoding="utf-8") as f:
# #     json.dump(baseline_outputs, f, indent=2, ensure_ascii=False)

# # with open("sat_results.json", "w", encoding="utf-8") as f:
# #     json.dump(sat_outputs, f, indent=2, ensure_ascii=False)

# # print("✅ Saved baseline_results.json & sat_results.json")


# import os
# import json
# import time
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from openai import OpenAI, InternalServerError
# from openai import OpenAI
# import httpx

# # from simple_graphrag import SimpleGraphRAG, split_long_context, load_txt

# # rag = SimpleGraphRAG(embedding_model_name="all-MiniLM-L6-v2", chunk_size=1200, chunk_overlap=100)

# # corpus_path = "SAT/aligner/data/FB15k-237N/id3text.txt"
# # text = load_txt(corpus_path)

# # rag.dataset = split_long_context(text, max_chars=800)
# # rag.build_embeddings_and_index()
# # rag.build_kg(use_dependency=True)





# # =========================================================
# # CONFIG
# # =========================================================
# QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# SAT_LORA_PATH = "sat_lora_model"
# DEVICE = "cpu"

# SLEEP_BETWEEN_CALLS = 2.0
# MAX_RETRY = 3

# # =========================================================
# # LOAD QWEN + SAT (LOAD ONLY – NO GENERATE)
# # =========================================================
# print("🔹 Loading Qwen + SAT LoRA (LOAD ONLY, CPU safe)")

# tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

# base_model = AutoModelForCausalLM.from_pretrained(
#     QWEN_MODEL_NAME,
#     torch_dtype=torch.float32,
#     device_map=None
# )

# model = PeftModel.from_pretrained(base_model, SAT_LORA_PATH)
# model.eval()

# print("✅ Qwen + SAT loaded successfully (no generation mode)")

# # =========================================================
# # SAT SIGNAL (REASONING BIAS)
# # =========================================================
# sat_signal = (
#     "The model has been fine-tuned with SAT-style logical constraints. "
#     "Answers should be concise, logically valid, and directly grounded in the context."
# )

# # =========================================================
# # LOAD KIMI K2 (NVIDIA NIM)
# # =========================================================
# os.environ["NVAPI_KEY"] = os.getenv("NVAPI_KEY") or "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"

# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("NVAPI_KEY")
# )

# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("NVAPI_KEY"),
#     http_client=httpx.Client(timeout=30.0)  # 👈 30s HARD TIMEOUT
# )

# # =========================================================
# # KIMI BASELINE (NO CONTEXT)
# # =========================================================
# def kimi_baseline_rag(question: str, rag, top_k: int = 6, alpha: float = 0.8, max_retry: int = 3) -> str:
#     """
#     Baseline = GraphRAG retrieval + Kimi answer (NO SAT bias).
#     """
#     # 1) Lấy context từ KG + FAISS
#     result = rag.graphrag_query(question, top_k=top_k, alpha=alpha)
#     context = result["context"]

#     prompt = f"""
# You are a precise QA assistant.

# Answer using ONLY the provided context.
# - If the answer is a number/date/location/name, copy it EXACTLY as in the context.
# - Do not guess or use outside knowledge.
# - If the context does not contain the answer, reply exactly:
# not stated in the text

# Context:
# {context}

# Question:
# {question}

# Answer (1 sentence max):
# """.strip()

#     import time
#     for attempt in range(max_retry):
#         try:
#             resp = client.chat.completions.create(
#                 model="moonshotai/kimi-k2-instruct-0905",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,   # giảm hallucination cho fact
#                 max_tokens=80
#             )
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"⚠️ Baseline error: {type(e).__name__} – retry {attempt+1}")
#             time.sleep(2)

#     return "ERROR: baseline_timeout"


# # =========================================================
# # KIMI + SAT (CONTEXT OPTIONAL)
# # =========================================================
# def kimi_sat(question: str, context: str, sat_signal: str, max_retry=MAX_RETRY) -> str:
#     prompt = f"""
# You are a reasoning-based QA assistant.

# Using include signal


# Answer style rules:
# - Be concise.
# - Do NOT paraphrase numbers or names incorrectly.
# - Do NOT add new information.

# {sat_signal}

# Question:
# {question}

# Answer:
# """
#     for attempt in range(max_retry):
#         try:
#             resp = client.chat.completions.create(
#                 model="moonshotai/kimi-k2-instruct-0905",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 max_tokens=128
#             )
#             return resp.choices[0].message.content.strip()

#         except InternalServerError:
#             print(f"⚠️ SAT retry {attempt+1}/{max_retry} (504)")
#             time.sleep(3)

#     return "ERROR: sat_failed"

# # =========================================================
# # LOAD DATA
# # qa_eval.json FORMAT:
# # [
# #   { "question": "...", "answer": "...", "context": "..." (optional) }
# # ]
# # =========================================================
# with open("qa_eval.json", encoding="utf-8") as f:
#     data = json.load(f)

# print(f"🔹 Loaded {len(data)} questions")

# # =========================================================
# # RUN INFERENCE
# # =========================================================
# baseline_outputs = []
# sat_outputs = []

# for i, item in enumerate(data, 1):
#     q = item["question"]
#     gt = item["answer"]
#     context = item.get("context", "")  # có hoặc không đều OK

#     print(f"\n[{i}/{len(data)}] Processing question...")

#     # # BASELINE
#     # ans_base = kimi_baseline_rag(q,rag)
#     # baseline_outputs.append({
#     #     "question": q,
#     #     "ans": ans_base,
#     #     "groundtruth": gt
#     # })
#     # time.sleep(SLEEP_BETWEEN_CALLS)

#     # SAT
#     ans_sat = kimi_sat(q, context, sat_signal)
#     sat_outputs.append({
#         "question": q,
#         "ans": ans_sat,
#         "groundtruth": gt
#     })
#     time.sleep(SLEEP_BETWEEN_CALLS)


# # SAVE RESULTS

# # with open("baseline_results.json", "w", encoding="utf-8") as f:
# #     json.dump(baseline_outputs, f, indent=2, ensure_ascii=False)

# with open("sat_results.json", "w", encoding="utf-8") as f:
#     json.dump(sat_outputs, f, indent=2, ensure_ascii=False)

# print("\n✅ DONE")
# print("📁 Saved baseline_results.json")
# print("📁 Saved sat_results.json")


import os

# set early to reduce native thread issues on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import time
from datetime import datetime

import httpx
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
CACHE_DIR = "sat_data"
FAISS_PATH = os.path.join(CACHE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")

QA_PATH = "qa_eval.json"
OUT_PATH = "sat_rag_kimi_results.json"

# retrieval
TOP_K = 5
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# SAT LoRA
QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAT_LORA_PATH = "sat_lora_model"
SAT_MAX_NEW_TOKENS = 32          # keep short (fast)
SAT_DO_SAMPLE = False            # deterministic (fast)

# Kimi
KIMI_MODEL = "moonshotai/kimi-k2-instruct-0905"
SLEEP_BETWEEN_CALLS = 1.2

# truncate to reduce API timeouts
MAX_CTX_CHARS = 8000

# =========================
# LOG
# =========================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def truncate(text: str, max_chars: int = MAX_CTX_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

# =========================
# KIMI CLIENT
# =========================
NVAPI_KEY = os.getenv("NVAPI_KEY")
if not NVAPI_KEY:
    raise RuntimeError("Missing NVAPI_KEY. export NVAPI_KEY='nvapi-...'")

timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVAPI_KEY,
    http_client=httpx.Client(timeout=timeout),
)

# =========================
# RETRIEVER (FAISS + SentenceTransformer)
# =========================
class FaissRetriever:
    def __init__(self, chunks_path: str, faiss_path: str, emb_model: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        log("🔹 Loading chunks...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)  # list[str]

        log("🔹 Loading FAISS index...")
        self.index = faiss.read_index(faiss_path)

        log("🔹 Loading embedding model...")
        self.embedder = SentenceTransformer(emb_model)

        log(f"✅ Retriever ready: chunks={len(self.chunks)}")

    def retrieve(self, query: str, top_k: int = 5) -> str:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        picks = [self.chunks[int(i)] for i in I[0] if int(i) >= 0]
        return " ".join(picks)

# =========================
# SAT MODEL (Qwen + LoRA)
# =========================
log("🔹 Loading Qwen + SAT LoRA (CPU)")

log("A) tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True, use_fast=True)
log("A) tokenizer OK")

log("B) base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    trust_remote_code=True,
    device_map=None,
    dtype=torch.float32,
    low_cpu_mem_usage=False,
)
log("B) base model OK")

log("C) LoRA...")
sat_model = PeftModel.from_pretrained(base_model, SAT_LORA_PATH)
sat_model.eval()
log("✅ SAT model ready")

def sat_draft(question: str, context: str) -> str:
    context = truncate(context)

    prompt = f"""You are a SAT-style QA reasoner.

RULES:
- Use ONLY the provided CONTEXT.
- Do NOT guess.
- If the answer is not in the context, output exactly: not stated in the text

CONTEXT:
{context}

QUESTION:
{question}

SAT_DRAFT (one short sentence):
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.inference_mode():
        out = sat_model.generate(
            **inputs,
            max_new_tokens=SAT_MAX_NEW_TOKENS,
            do_sample=SAT_DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "SAT_DRAFT" in text:
        text = text.split("SAT_DRAFT", 1)[-1].replace(":", "").strip()
    return text.strip()

def kimi_final(question: str, context: str, sat_hint: str, max_retry: int = 4) -> str:
    context = truncate(context)

    prompt = f"""
You are a high-accuracy QA assistant.

You MUST answer using ONLY the CONTEXT below.
You may use SAT_DRAFT only as guidance, but do not add facts not in CONTEXT.
If the answer is not in CONTEXT, reply exactly:
not stated in the text

CONTEXT:
{context}

SAT_DRAFT:
{sat_hint}

QUESTION:
{question}

FINAL ANSWER (1 sentence):
""".strip()

    for attempt in range(1, max_retry + 1):
        try:
            log(f"  -> Kimi API call (attempt {attempt}/{max_retry})")
            resp = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=96,
            )
            return resp.choices[0].message.content.strip()

        except AuthenticationError:
            return "ERROR: AuthError (check NVAPI_KEY)"
        except RateLimitError:
            wait_s = 3 * attempt
            log(f"  !! 429 RateLimit -> sleep {wait_s}s")
            time.sleep(wait_s)
        except (APITimeoutError, APIConnectionError):
            wait_s = 5 * attempt
            log(f"  !! Timeout/Connection -> sleep {wait_s}s")
            time.sleep(wait_s)
        except BadRequestError as e:
            return f"ERROR: BadRequest (prompt too long/invalid): {str(e)[:140]}"
        except Exception as e:
            wait_s = 2 * attempt
            log(f"  !! Unknown error {type(e).__name__} -> sleep {wait_s}s")
            time.sleep(wait_s)

    return "ERROR: kimi_failed_after_retries"

# =========================
# MAIN
# =========================
def main():
    retriever = FaissRetriever(CHUNKS_PATH, FAISS_PATH, EMB_MODEL)

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
        log("  -> RETRIEVE start")
        context = retriever.retrieve(q, top_k=TOP_K)
        log(f"  -> RETRIEVE done in {time.time()-t0:.2f}s | ctx_chars={len(context)}")

        # 2) SAT draft
        t1 = time.time()
        log("  -> SAT_DRAFT start")
        draft = sat_draft(q, context)
        log(f"  -> SAT_DRAFT done in {time.time()-t1:.2f}s | draft={draft}")

        # 3) Kimi final
        t2 = time.time()
        log("  -> KIMI start")
        ans = kimi_final(q, context, draft)
        log(f"  -> KIMI done in {time.time()-t2:.2f}s | ans={ans}")

        outputs.append({
            "question": q,
            "context": context,
            "sat_draft": draft,
            "ans": ans,
            "groundtruth": gt
        })

        # checkpoint each question
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_BETWEEN_CALLS)

    log(f"\n✅ Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
