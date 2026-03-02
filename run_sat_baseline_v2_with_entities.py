# run_sat_baseline_v2_with_entities.py
"""
SAT GraphRAG v2 — Entity-First Retrieval

Thay đổi chính so với v1:
  - v1: Semantic search toàn bộ corpus → boost bằng graph score
  - v2: Trích xuất entity từ câu hỏi TRƯỚC → lấy chunk của entity + neighbors
         → bổ sung thêm semantic search → gộp context

Pipeline:
  1. Load entity texts (id2text.txt) + titles (id2title.txt) từ SAT
  2. Build FAISS index + KG adjacency (giống v1)
  3. For each question:
     a) Entity Extraction: tìm entity nào xuất hiện trong câu hỏi
     b) Entity Chunks: lấy trực tiếp chunk (mô tả) của entity đó
     c) Neighbor Chunks: lấy chunk của các entity lân cận trong KG (1-hop)
     d) Semantic Supplement: bổ sung thêm chunks liên quan bằng FAISS
     e) Deduplicate + build context → LLM → answer
  4. Save results
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time
import logging
import numpy as np
import faiss
import httpx
from collections import defaultdict, OrderedDict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

# ============================================================
# CONFIG
# ============================================================
SAT_DATA_DIR = "SAT/aligner/data/FB15k-237N"
CACHE_DIR = "sat_fb15k_cache"
QA_PATH = "qa_eval.json"
OUT_PATH = "sat_baseline_v2_entities_results.json"

TOP_K = 10              # Tổng số chunks tối đa trong context
MAX_NEIGHBOR_CHUNKS = 5 # Số chunk neighbor tối đa lấy từ KG
SEMANTIC_SUPPLEMENT = 3 # Số chunk bổ sung từ semantic search
SLEEP = 1.5
MAX_RETRY = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# HELPER: Load SAT data files (giống v1)
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
# LLM CLIENT (giống v1)
# ============================================================
def get_client():
    api_key = os.getenv("NVAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NVAPI_KEY. Set it first:\n"
            "  export NVAPI_KEY='nvapi-xxxxx'"
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
# SAT GraphRAG v2 — Entity-First Retrieval
# ============================================================
class SATGraphRAG_v2:
    """
    GraphRAG v2: Entity-First Retrieval

    Khác v1 ở chỗ: thay vì search toàn bộ corpus rồi rerank,
    v2 trích xuất entity từ câu hỏi TRƯỚC, sau đó:
      1. Lấy trực tiếp chunk (mô tả) của entity đó       → "entity chunks"
      2. Lấy chunk của các neighbor trong KG (1-hop)       → "neighbor chunks"
      3. Bổ sung thêm bằng semantic search (fallback)      → "semantic chunks"
      4. Gộp tất cả + KG facts → context
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

        # Reverse mapping: title_lower → eid (for entity extraction)
        # Sắp xếp theo chiều dài title giảm dần để ưu tiên match dài nhất
        self.title2eid = {}
        for eid in self.entity_ids:
            title = self.id2title.get(eid, "")
            if title and len(title) > 2:
                self.title2eid[title.lower()] = eid

        # Sorted titles by length (longest first) — greedy matching
        self.sorted_titles = sorted(self.title2eid.keys(), key=len, reverse=True)

        logger.info("  %d entity texts (chunks), %d searchable titles, %d relations",
                     len(self.chunks), len(self.title2eid), len(self.rel2id))

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
                    self.neighbors[dst].add((rel, src))

        logger.info("  %d triplets loaded → KG adjacency built", total_triplets)

        # ------ 3. Build / load FAISS index (for fallback semantic search) ------
        logger.info("Loading embedding model: %s", embed_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        self._build_or_load_index()

    def _build_or_load_index(self):
        emb_path = os.path.join(self.cache_dir, "embeddings.npy")
        idx_path = os.path.join(self.cache_dir, "faiss.index")

        if os.path.exists(emb_path) and os.path.exists(idx_path):
            logger.info("Loading cached embeddings + FAISS index from %s/", self.cache_dir)
            self.embeddings = np.load(emb_path)
            self.index = faiss.read_index(idx_path)
            logger.info("  Index loaded: %d vectors, dim=%d", self.index.ntotal, self.embeddings.shape[1])
        else:
            logger.info("Computing embeddings for %d chunks ...", len(self.chunks))
            batch_size = 64
            emb_list = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                emb = self.embed_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                emb_list.append(emb)
                if (i // batch_size) % 20 == 0:
                    logger.info("  Embedded %d / %d", min(i + batch_size, len(self.chunks)), len(self.chunks))

            self.embeddings = np.vstack(emb_list).astype("float32")
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)

            np.save(emb_path, self.embeddings)
            faiss.write_index(self.index, idx_path)
            logger.info("  Saved cache → %s/ (%d vectors, dim=%d)", self.cache_dir, self.index.ntotal, d)

    # ================================================================
    # BƯỚC 1: Entity Extraction — tìm entity nào xuất hiện trong câu hỏi
    # ================================================================
    def extract_entities(self, question):
        """
        Trích xuất entity từ câu hỏi bằng cách so khớp tên entity (greedy longest match).

        Returns:
            list of (eid, title) — các entity được nhận diện trong câu hỏi
        """
        question_lower = question.lower()
        matched = []
        used_spans = []  # Để tránh overlap (ví dụ "University of Essex" vs "Essex")

        for title_lower in self.sorted_titles:
            pos = question_lower.find(title_lower)
            if pos == -1:
                continue

            # Kiểm tra xem span này có bị overlap với entity đã match chưa
            end = pos + len(title_lower)
            overlap = False
            for s, e in used_spans:
                if pos < e and end > s:  # có overlap
                    overlap = True
                    break

            if not overlap:
                eid = self.title2eid[title_lower]
                original_title = self.id2title.get(eid, title_lower)
                matched.append((eid, original_title))
                used_spans.append((pos, end))

        return matched

    # ================================================================
    # BƯỚC 2: Lấy chunks của entity trực tiếp
    # ================================================================
    def _get_entity_chunks(self, entity_ids):
        """
        Trả về danh sách chunks (mô tả) của các entity đã xác định.

        Returns:
            list of (chunk_idx, chunk_text, source="entity")
        """
        results = []
        for eid in entity_ids:
            if eid in self.eid_to_chunk_idx:
                idx = self.eid_to_chunk_idx[eid]
                results.append((idx, self.chunks[idx], "entity"))
        return results

    # ================================================================
    # BƯỚC 3: Lấy chunks của neighbors (1-hop) trong KG
    # ================================================================
    def _get_neighbor_chunks(self, entity_ids, max_per_entity=5):
        """
        Lấy chunk (mô tả) của các entity láng giềng trong KG.

        Ưu tiên: sắp xếp neighbor theo số quan hệ (nhiều quan hệ = quan trọng hơn).

        Returns:
            list of (chunk_idx, chunk_text, source="neighbor")
        """
        # Đếm tần suất xuất hiện của mỗi neighbor qua tất cả entity
        neighbor_count = defaultdict(int)
        seen_entity_ids = set(entity_ids)  # Tránh trùng với entity chính

        for eid in entity_ids:
            for _rel, nb in self.neighbors.get(eid, set()):
                if nb not in seen_entity_ids:
                    neighbor_count[nb] += 1

        # Sắp xếp theo tần suất giảm dần
        sorted_neighbors = sorted(neighbor_count.items(), key=lambda x: x[1], reverse=True)

        results = []
        for nb_eid, _count in sorted_neighbors[:max_per_entity]:
            if nb_eid in self.eid_to_chunk_idx:
                idx = self.eid_to_chunk_idx[nb_eid]
                results.append((idx, self.chunks[idx], "neighbor"))

        return results

    # ================================================================
    # BƯỚC 4: Semantic search bổ sung (fallback / supplement)
    # ================================================================
    def _semantic_search(self, query, top_k=10, exclude_indices=None):
        """
        FAISS semantic search, loại bỏ các chunk đã có (avoid duplication).

        Returns:
            list of (chunk_idx, chunk_text, source="semantic")
        """
        exclude = set(exclude_indices or [])
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        # Lấy nhiều hơn top_k để bù cho các chunk bị loại
        scores, indices = self.index.search(q_emb, top_k * 3)

        results = []
        for idx, sc in zip(indices[0], scores[0]):
            if int(idx) < 0 or int(idx) in exclude:
                continue
            results.append((int(idx), self.chunks[int(idx)], "semantic"))
            if len(results) >= top_k:
                break

        return results

    # ================================================================
    # BƯỚC 5: Lấy KG facts cho các entity đã xác định
    # ================================================================
    def _get_kg_facts(self, entity_ids, max_facts=8):
        """
        Trích xuất các fact từ KG dưới dạng readable cho các entity đã xác định.
        """
        facts = []
        for eid in entity_ids:
            title = self.id2title.get(eid, f"entity_{eid}")
            for rel, nb in list(self.neighbors.get(eid, set())):
                nb_title = self.id2title.get(nb, f"entity_{nb}")
                rel_name = make_rel_readable(rel)
                facts.append(f"{title} → {rel_name} → {nb_title}")
                if len(facts) >= max_facts:
                    return facts
        return facts

    # ================================================================
    # MAIN QUERY — Entity-First Pipeline
    # ================================================================
    def query(self, question, top_k=TOP_K, max_neighbor=MAX_NEIGHBOR_CHUNKS,
              semantic_supplement=SEMANTIC_SUPPLEMENT):
        """
        Entity-First Retrieval Pipeline:

        1. Extract entities từ câu hỏi
        2. Lấy chunk của entity trực tiếp
        3. Lấy chunk của neighbors (1-hop)
        4. Bổ sung semantic search (loại trùng)
        5. Gộp context + KG facts

        Returns:
            dict with keys: query, context, matched_entities, entity_chunks,
                            neighbor_chunks, semantic_chunks, kg_facts
        """
        # ---- Bước 1: Entity Extraction ----
        matched = self.extract_entities(question)
        matched_eids = [eid for eid, _title in matched]
        matched_names = [title for _eid, title in matched]

        # ---- Bước 2: Entity Chunks (trực tiếp) ----
        entity_chunks = self._get_entity_chunks(matched_eids)

        # ---- Bước 3: Neighbor Chunks (1-hop KG) ----
        neighbor_chunks = self._get_neighbor_chunks(matched_eids, max_per_entity=max_neighbor)

        # ---- Tổng hợp chunk indices đã có (để loại trùng khi semantic search) ----
        used_indices = set()
        for idx, _text, _src in entity_chunks:
            used_indices.add(idx)
        for idx, _text, _src in neighbor_chunks:
            used_indices.add(idx)

        # ---- Bước 4: Semantic Search bổ sung ----
        # Nếu không tìm được entity nào → fallback hoàn toàn sang semantic
        if not matched_eids:
            sem_top_k = top_k  # Dùng toàn bộ quota cho semantic
        else:
            sem_top_k = semantic_supplement  # Chỉ bổ sung thêm vài chunk

        semantic_chunks = self._semantic_search(question, top_k=sem_top_k, exclude_indices=used_indices)

        # ---- Bước 5: Gộp context ----
        all_chunks = []  # OrderedDict-style dedup
        seen = set()

        # Ưu tiên: entity chunks > neighbor chunks > semantic chunks
        for idx, text, src in entity_chunks + neighbor_chunks + semantic_chunks:
            if idx not in seen and len(all_chunks) < top_k:
                all_chunks.append((idx, text, src))
                seen.add(idx)

        # KG facts
        kg_facts = self._get_kg_facts(matched_eids)

        # Build context string
        context_parts = []

        # Đánh dấu nguồn gốc chunk trong context
        if entity_chunks:
            entity_texts = [text for _, text, _ in all_chunks if _ == "entity" or (_, text, _) in entity_chunks]
            # Thêm tất cả chunks theo thứ tự ưu tiên
        for idx, text, src in all_chunks:
            label = {"entity": "[Entity Description]",
                     "neighbor": "[Related Entity]",
                     "semantic": "[Relevant Context]"}.get(src, "")
            context_parts.append(f"{label}\n{text}" if label else text)

        if kg_facts:
            context_parts.append("[Knowledge Graph Facts]\n" + "\n".join(kg_facts))

        context = "\n\n".join(context_parts)

        return {
            "query": question,
            "context": context,
            "matched_entities": matched_names,
            "entity_chunks": [(idx, src) for idx, _, src in entity_chunks],
            "neighbor_chunks": [(idx, src) for idx, _, src in neighbor_chunks],
            "semantic_chunks": [(idx, src) for idx, _, src in semantic_chunks],
            "kg_facts": kg_facts,
            "total_chunks": len(all_chunks),
        }


# ============================================================
# MAIN
# ============================================================
def main():
    client = get_client()

    rag = SATGraphRAG_v2(
        data_dir=SAT_DATA_DIR,
        cache_dir=CACHE_DIR,
        embed_model_name=EMBED_MODEL,
    )

    print(f"\n{'='*60}")
    print(f"✅ SAT GraphRAG v2 (Entity-First) ready")
    print(f"   Chunks (entity texts):       {len(rag.chunks)}")
    print(f"   Searchable entity titles:    {len(rag.title2eid)}")
    print(f"   KG relations:                {len(rag.rel2id)}")
    print(f"   KG entities with neighbors:  {len(rag.neighbors)}")
    print(f"   FAISS index vectors:         {rag.index.ntotal}")
    print(f"{'='*60}")

    # Load questions
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Questions to evaluate:        {len(data)}\n")

    outputs = []
    stats = {"entity_found": 0, "fallback_semantic": 0}

    for i, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("answer") or item.get("groundtruth") or ""

        print(f"[{i}/{len(data)}] {q}")

        r = rag.query(q, top_k=TOP_K, max_neighbor=MAX_NEIGHBOR_CHUNKS,
                       semantic_supplement=SEMANTIC_SUPPLEMENT)
        ctx = r["context"]

        # Log retrieval strategy
        if r["matched_entities"]:
            stats["entity_found"] += 1
            print(f"  🎯 Entities found: {r['matched_entities']}")
            print(f"  📦 Chunks: {len(r['entity_chunks'])} entity + "
                  f"{len(r['neighbor_chunks'])} neighbor + "
                  f"{len(r['semantic_chunks'])} semantic")
        else:
            stats["fallback_semantic"] += 1
            print(f"  🔍 No entity matched → full semantic fallback")
            print(f"  📦 Chunks: {len(r['semantic_chunks'])} semantic")

        if r["kg_facts"]:
            print(f"  📊 KG facts: {len(r['kg_facts'])}")

        print(f"  📝 Context: {len(ctx)} chars, {r['total_chunks']} chunks total")

        ans = kimi_answer(q, ctx, client)
        print(f"  💬 {ans[:120]}")

        outputs.append({
            "question": q,
            "answer": ans,
            "groundtruth": gt,
            "matched_entities": r["matched_entities"],
            "retrieval": {
                "entity_chunks": len(r["entity_chunks"]),
                "neighbor_chunks": len(r["neighbor_chunks"]),
                "semantic_chunks": len(r["semantic_chunks"]),
                "kg_facts": len(r["kg_facts"]),
            }
        })

        time.sleep(SLEEP)

    # Save results
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved → {OUT_PATH}")

    # Summary
    total = len(outputs)
    print(f"\n{'='*60}")
    print(f"📊 Retrieval Strategy Summary:")
    print(f"   Entity found:        {stats['entity_found']}/{total} "
          f"({100*stats['entity_found']/total:.1f}%)")
    print(f"   Semantic fallback:   {stats['fallback_semantic']}/{total} "
          f"({100*stats['fallback_semantic']/total:.1f}%)")

    correct = sum(
        1 for o in outputs
        if o["groundtruth"].lower() in o["answer"].lower()
    )
    print(f"   Substring match:     {correct}/{total} "
          f"({100*correct/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
