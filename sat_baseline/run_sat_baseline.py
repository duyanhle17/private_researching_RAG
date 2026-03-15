# run_sat_baseline.py
"""
Run QA evaluation using SAT/aligner/data/FB15k-237N directly.

Pipeline:
  1. Load entity texts from id2text.txt as "chunks"
  2. Build FAISS index for semantic search (cached after first run)  
  3. Build KG adjacency from train/valid/test triplets
  4. For each question:  semantic + graph search → context → Kimi K2 LLM → answer
  5. Save results to sat_baseline_results.json
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time
import logging
import numpy as np
import faiss
import httpx
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

# ============================================================
# CONFIG
# ============================================================
SAT_DATA_DIR = "SAT/aligner/data/FB15k-237N"
CACHE_DIR = "sat_fb15k_cache"       # embeddings + index cached here
QA_PATH = "qa_eval.json"
OUT_PATH = "sat_baseline_results.json"

TOP_K = 10          # Number of chunks to retrieve
ALPHA = 0.7         # 0.7 semantic + 0.3 graph
SLEEP = 1.5         # Delay between API calls (rate limit)
MAX_RETRY = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# HELPER: Load SAT data files
# ============================================================
def load_id2text(path):
    """id2text.txt → {int_id: text_description}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def load_id2title(path):
    """id2title.txt → {int_id: title}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def load_mid2id(path):
    """mid2id.txt → {freebase_mid: int_id}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mid, id_str = line.strip().split("\t")
            mapping[mid] = int(id_str)
    return mapping


def load_rel2id(path):
    """rel2id.txt → {relation_path: int_id}"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rel, id_str = line.strip().split("\t")
            mapping[rel] = int(id_str)
    return mapping


def load_triplets(path, mid2id):
    """Load a triplet file → list of (src_id, rel_string, dst_id)"""
    triplets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                src_mid, rel, dst_mid = parts
                if src_mid in mid2id and dst_mid in mid2id:
                    triplets.append((mid2id[src_mid], rel, mid2id[dst_mid]))
    return triplets


def make_rel_readable(rel_path):
    """Convert '/people/person/profession' → 'profession'"""
    if "/" in rel_path:
        parts = rel_path.rstrip("/").split("/")
        return parts[-1].replace("_", " ")
    return rel_path.replace("_", " ")


# ============================================================
# LLM CLIENT
# ============================================================
def get_client():
    api_key = os.getenv("NVAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NVAPI_KEY. Set it first:\n"
            "  PowerShell: $env:NVAPI_KEY = 'nvapi-xxxxx'\n"
            "  CMD:        set NVAPI_KEY=nvapi-xxxxx"
        )
    logging.getLogger("openai").setLevel(logging.WARNING)
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        max_retries=0,
        http_client=httpx.Client(timeout=httpx.Timeout(90.0, connect=15.0)),
    )


def kimi_answer(question, context, client, max_retry=MAX_RETRY):
    prompt = f"""\
You are a precise QA assistant with strong reasoning abilities.

INSTRUCTIONS:
1. Read the context carefully and answer based on it.
2. For "WHAT/WHO/WHERE/WHEN" questions: Extract the answer directly from context.
3. For "WHY/HOW" questions: Use reasoning to infer the answer from context clues.
4. If you can reasonably infer an answer from the context, provide it.
5. ONLY say "not stated in the text" if there is absolutely NO relevant information.

Context:
{context}

Question:
{question}

Answer concisely (1-2 sentences):"""

    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()

        except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as e:
            wait = 2.0 * (attempt + 1)
            print(f"  ⚠️ API retry {attempt+1}/{max_retry}: {type(e).__name__} (sleep {wait:.1f}s)")
            time.sleep(wait)

        except Exception as e:
            print(f"  ❌ API fatal: {type(e).__name__}: {e}")
            return "ERROR: api_fatal"

    return "ERROR: api_failed"


# ============================================================
# SAT GraphRAG — loads directly from FB15k-237N
# ============================================================
class SATGraphRAG:
    """
    GraphRAG system built directly from SAT/aligner/data/FB15k-237N.

    - id2text.txt  → chunks (entity descriptions = retrieval corpus)
    - id2title.txt → entity names (for KG fact readability)
    - train/valid/test.txt → KG triplets (entity ↔ entity via relation)
    - FAISS index  → fast semantic search over chunks
    """

    def __init__(self, data_dir, cache_dir, embed_model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # ------ 1. Load SAT data files ------
        logger.info("Loading SAT data files from %s ...", data_dir)

        self.id2text = load_id2text(os.path.join(data_dir, "id2text.txt"))
        self.id2title = load_id2title(os.path.join(data_dir, "id2title.txt"))
        self.mid2id = load_mid2id(os.path.join(data_dir, "mid2id.txt"))
        self.rel2id = load_rel2id(os.path.join(data_dir, "rel2id.txt"))

        # Sorted entity IDs → ordered chunk list
        self.entity_ids = sorted(self.id2text.keys())
        self.chunks = [self.id2text[eid] for eid in self.entity_ids]
        self.eid_to_chunk_idx = {eid: idx for idx, eid in enumerate(self.entity_ids)}

        logger.info("  %d entity texts (chunks), %d relations", len(self.chunks), len(self.rel2id))

        # ------ 2. Build KG adjacency from triplets ------
        self.neighbors = defaultdict(set)  # eid → set of (rel_str, neighbor_eid)
        total_triplets = 0

        for split in ("train.txt", "valid.txt", "test.txt"):
            path = os.path.join(data_dir, split)
            if os.path.exists(path):
                trips = load_triplets(path, self.mid2id)
                total_triplets += len(trips)
                for src, rel, dst in trips:
                    self.neighbors[src].add((rel, dst))
                    self.neighbors[dst].add((rel, src))   # undirected for search

        logger.info("  %d triplets loaded → KG adjacency built", total_triplets)

        # ------ 3. Build / load FAISS index ------
        logger.info("Loading embedding model: %s", embed_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        self._build_or_load_index()

    # ---- Index management ----
    def _build_or_load_index(self):
        emb_path = os.path.join(self.cache_dir, "embeddings.npy")
        idx_path = os.path.join(self.cache_dir, "faiss.index")

        if os.path.exists(emb_path) and os.path.exists(idx_path):
            logger.info("Loading cached embeddings + FAISS index from %s/", self.cache_dir)
            self.embeddings = np.load(emb_path)
            self.index = faiss.read_index(idx_path)
            logger.info("  Index loaded: %d vectors, dim=%d", self.index.ntotal, self.embeddings.shape[1])
        else:
            logger.info("Computing embeddings for %d chunks (first run — will be cached)...", len(self.chunks))
            batch_size = 64
            emb_list = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                emb = self.embed_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                emb_list.append(emb)
                done = min(i + batch_size, len(self.chunks))
                if (i // batch_size) % 20 == 0:
                    logger.info("  Embedded %d / %d", done, len(self.chunks))

            self.embeddings = np.vstack(emb_list).astype("float32")

            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)

            np.save(emb_path, self.embeddings)
            faiss.write_index(self.index, idx_path)
            logger.info("  Saved cache → %s/  (%d vectors, dim=%d)", self.cache_dir, self.index.ntotal, d)

    # ---- Search methods ----
    def _semantic_search(self, query, top_k=30):
        """Return [(chunk_idx, score), ...] via FAISS inner-product search."""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        return [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]

    def _graph_search_scores(self, query):
        """Score every chunk by entity-name overlap with query + KG neighbor boost."""
        query_lower = query.lower()
        matched_eids = set()

        for eid in self.entity_ids:
            title = self.id2title.get(eid, "")
            if title and len(title) > 2 and title.lower() in query_lower:
                matched_eids.add(eid)

        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not matched_eids:
            return scores

        for eid in matched_eids:
            # Direct match → high score
            if eid in self.eid_to_chunk_idx:
                scores[self.eid_to_chunk_idx[eid]] += 1.0

            # 1-hop neighbors → moderate boost
            for _rel, nb in self.neighbors.get(eid, set()):
                if nb in self.eid_to_chunk_idx:
                    scores[self.eid_to_chunk_idx[nb]] += 0.3

        mx = scores.max()
        if mx > 0:
            scores /= mx
        return scores

    def _get_kg_facts(self, query, max_facts=5):
        """Extract readable KG facts for entities mentioned in the query."""
        query_lower = query.lower()
        facts = []

        for eid in self.entity_ids:
            title = self.id2title.get(eid, "")
            if not title or len(title) <= 2 or title.lower() not in query_lower:
                continue

            for rel, nb in list(self.neighbors.get(eid, set()))[:max_facts]:
                nb_title = self.id2title.get(nb, f"entity_{nb}")
                rel_name = make_rel_readable(rel)
                facts.append(f"{title} → {rel_name} → {nb_title}")

            if len(facts) >= max_facts:
                break

        return facts[:max_facts]

    # ---- Main query ----
    def query(self, question, top_k=10, alpha=0.7):
        """
        Retrieve context for a question.

        Args:
            question: The question string
            top_k:    Number of chunks to retrieve
            alpha:    Weight for semantic score (1-alpha for graph score)

        Returns:
            dict with keys: query, context, chunks, kg_facts, scores
        """
        # Semantic search (get more candidates than top_k)
        sem_results = self._semantic_search(question, top_k=top_k * 3)

        # Graph-based scores
        graph_scores = self._graph_search_scores(question)

        # Combine: alpha * semantic + (1-alpha) * graph
        combined = []
        for idx, sem_score in sem_results:
            gscore = graph_scores[idx] if idx < len(graph_scores) else 0.0
            final = alpha * sem_score + (1 - alpha) * gscore
            combined.append((idx, final, sem_score, gscore))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_results = combined[:top_k]

        # Retrieve chunks
        retrieved_chunks = [self.chunks[idx] for idx, _, _, _ in top_results]

        # KG facts
        kg_facts = self._get_kg_facts(question)

        # Build context string
        parts = list(retrieved_chunks)
        if kg_facts:
            parts.append("[Knowledge Graph Facts]\n" + "\n".join(kg_facts))

        context = "\n\n".join(parts)

        return {
            "query": question,
            "context": context,
            "chunks": retrieved_chunks,
            "kg_facts": kg_facts,
            "scores": [
                {
                    "chunk_idx": idx,
                    "combined": round(comb, 4),
                    "semantic": round(sem, 4),
                    "graph": round(gs, 4),
                }
                for idx, comb, sem, gs in top_results
            ],
        }


# ============================================================
# MAIN
# ============================================================
def main():
    client = get_client()

    # Build / load SAT GraphRAG
    rag = SATGraphRAG(
        data_dir=SAT_DATA_DIR,
        cache_dir=CACHE_DIR,
        embed_model_name=EMBED_MODEL,
    )

    print(f"\n{'='*60}")
    print(f"✅ SAT GraphRAG ready")
    print(f"   Chunks (entity texts): {len(rag.chunks)}")
    print(f"   KG relations:          {len(rag.rel2id)}")
    print(f"   KG entities with neighbors: {len(rag.neighbors)}")
    print(f"   FAISS index vectors:   {rag.index.ntotal}")
    print(f"{'='*60}")

    # Load questions
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Questions to evaluate:  {len(data)}\n")

    outputs = []
    for i, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("answer") or item.get("groundtruth") or ""

        print(f"[{i}/{len(data)}] {q}")

        r = rag.query(q, top_k=TOP_K, alpha=ALPHA)
        ctx = r["context"]

        if r["kg_facts"]:
            print(f"  📊 KG facts: {len(r['kg_facts'])}")
        print(f"  📝 Context: {len(ctx)} chars, {len(r['chunks'])} chunks")

        ans = kimi_answer(q, ctx, client)
        print(f"  💬 {ans[:120]}")

        outputs.append({
            "question": q,
            "answer": ans,
            "groundtruth": gt,
        })

        time.sleep(SLEEP)

    # Save results
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved → {OUT_PATH}")

    # Quick accuracy summary
    correct = sum(
        1 for o in outputs
        if o["groundtruth"].lower() in o["answer"].lower()
    )
    print(f"📊 Quick substring match: {correct}/{len(outputs)} answers contain groundtruth")


if __name__ == "__main__":
    main()
