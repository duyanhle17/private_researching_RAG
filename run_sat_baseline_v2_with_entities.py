# run_sat_baseline_v2_with_entities.py
"""
SAT GraphRAG v2 — Entity-First Retrieval + BM25 Hybrid

Pipeline retrieval:
  1. Load entity texts (id2text.txt) + titles (id2title.txt) từ SAT data
     - Mỗi entity = 1 "chunk" (Wikipedia description)
     - VD: entity "University of Essex" (id=3) → chunk = đoạn mô tả Wikipedia của trường đó
     - Các triplets (train/valid/test.txt) build KG: (src_entity, relation, dst_entity)
  2. Build FAISS index (semantic) + BM25 index + KG adjacency
  3. For each question:
     a) Entity Extraction: string match tên entity trong câu hỏi
     b) Entity Chunks: lấy Wikipedia description của entity tìm được (1 chunk/entity)
     c) Neighbor Chunks: lấy description của entities 1-hop trong KG
     d) BM25 Search: keyword search trên toàn bộ corpus → lấy top-k
     e) Fallback: nếu stage a-d vẫn thiếu → bổ sung 5 chunks từ FAISS semantic
     f) Deduplicate + rank + build context → LLM → answer
  4. Save results

Lưu ý data structure:
  - id2text.txt:  int_id  TAB  wikipedia_description
  - id2title.txt: int_id  TAB  entity_title
  - mid2id.txt:   freebase_mid  TAB  int_id
  - train.txt:    src_mid  TAB  relation_path  TAB  dst_mid
  - → sau khi map qua mid2id → (src_int_id, relation_str, dst_int_id)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time
import logging
import math
import re
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
CACHE_DIR = "sat_fb15k_cache"
QA_PATH = "qa_eval.json"
OUT_PATH = "sat_baseline_v2_entities_results.json"

TOP_K = 15              # Tổng số chunks tối đa trong context
BM25_TOP_K = 4          # Số chunk từ BM25 keyword search
SEMANTIC_K = 5          # Số chunk luôn lấy từ FAISS semantic search
MAX_KG_FACTS = 8        # Số KG facts tối đa
SLEEP = 1.5
MAX_RETRY = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
# EMBED_MODEL = "BAAI/bge-m3"
# CACHE_DIR = f"sat_fb15k_cache_{EMBED_MODEL.replace('/', '_')}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# HELPER: Load SAT data files
# ============================================================
def load_id2text(path):
    """id2text.txt → {int_id: text_description}
    
    Mỗi dòng: int_id TAB wikipedia_description
    → đây là "chunk" của entity, tức là đoạn mô tả từ Wikipedia
    """
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
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_triplets(path, mid2id):
    """Load a triplet file → list of (src_id, rel_string, dst_id)
    
    train.txt format: src_mid TAB relation_path TAB dst_mid
    → sau khi map qua mid2id: (src_int_id, relation_string, dst_int_id)
    """
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
# BM25 IMPLEMENTATION (không cần external lib)
# ============================================================
class BM25:
    """
    BM25 (Okapi BM25) implementation đơn giản không cần thư viện ngoài.
    
    Được build trên corpus chunks (Wikipedia descriptions của entities).
    Mỗi chunk là mô tả văn bản của một entity trong KG.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: list of document strings (chunks)
            k1: term frequency saturation parameter
            b: length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # Tokenize: lowercase + split by non-word chars
        self.tokenized = [self._tokenize(doc) for doc in corpus]

        # Document lengths
        doc_lens = [len(t) for t in self.tokenized]
        self.avgdl = sum(doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

        # Build inverted index: token → {doc_idx: freq}
        self.idf: dict[str, float] = {}
        self.tf: list[dict[str, int]] = []

        df: dict[str, int] = defaultdict(int)  # document frequency
        for tok_doc in self.tokenized:
            tf_doc: dict[str, int] = defaultdict(int)
            seen = set()
            for tok in tok_doc:
                tf_doc[tok] += 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)
            self.tf.append(dict(tf_doc))

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)  (Robertson IDF)
        N = self.corpus_size
        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase + split on non-alphanumeric (loại stopwords đơn giản)."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        # Loại bỏ stopwords thông dụng để tăng precision
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "it", "its", "of", "in", "on", "at",
            "to", "for", "and", "or", "but", "with", "by", "from", "as",
            "this", "that", "which", "who", "not", "also", "than", "more",
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for all documents given a query.
        
        Returns:
            np.ndarray of shape (corpus_size,) with BM25 scores
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for tok in query_tokens:
            if tok not in self.idf:
                continue
            idf_val = self.idf[tok]
            for doc_idx, tf_doc in enumerate(self.tf):
                tf_val = tf_doc.get(tok, 0)
                if tf_val == 0:
                    continue
                dl = len(self.tokenized[doc_idx])
                denom = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf_val * (tf_val * (self.k1 + 1)) / denom

        return scores

    def get_top_k(self, query: str, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """
        Returns top-k (doc_idx, score) pairs sorted by score desc.
        
        Args:
            query: search query
            k: number of results
            exclude: set of doc indices to exclude
        """
        scores = self.get_scores(query)
        if exclude:
            for idx in exclude:
                if 0 <= idx < self.corpus_size:
                    scores[idx] = -1.0

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append((int(idx), float(scores[idx])))
            if len(results) >= k:
                break
        return results


# ============================================================
# LLM CLIENT
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
5. ONLY say "not stated in the text" if there is absolutely NO relevant information. Although, if the context nearly close to the question, you can say the relevant context into the answer with your solution.

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
# SAT GraphRAG v2 — Entity-First + BM25 Hybrid Retrieval
# ============================================================
class SATGraphRAG_v2:
    """
    GraphRAG v2: Entity-First + BM25 Hybrid Retrieval

    Hiểu về data:
    - Mỗi entity trong FB15k-237N có:
      * id (int): khóa nội bộ
      * title: tên entity (VD: "University of Essex")
      * text: mô tả Wikipedia → đây là "CHUNK"
    - Các triplets (train/valid/test) xây dựng KG:
      (entity_A, relation, entity_B) → adjacency graph

    Retrieval pipeline:
    1. Entity Extraction: string match tên entity trong câu hỏi
       → lấy "chunk" (Wikipedia description) của entity đó
    2. Neighbor Chunks: dùng KG adjacency để lấy chunk của entities
       có quan hệ trực tiếp (1-hop) với entity tìm được
    3. BM25 Search: keyword matching trên toàn bộ corpus
       → tìm chunks có từ khóa trùng với query
    4. Fallback FAISS: nếu tổng chunks < threshold
       → bổ sung additional 5 chunks từ semantic search
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
        # Mỗi chunk = Wikipedia description của một entity
        self.entity_ids = sorted(self.id2text.keys())
        self.chunks = [self.id2text[eid] for eid in self.entity_ids]
        self.eid_to_chunk_idx = {eid: idx for idx, eid in enumerate(self.entity_ids)}

        # Reverse mapping: title_lower → eid (for entity extraction)
        # Sắp xếp theo chiều dài title giảm dần để ưu tiên match dài nhất (greedy)
        self.title2eid: dict[str, int] = {}
        for eid in self.entity_ids:
            title = self.id2title.get(eid, "")
            if title and len(title) > 2:
                self.title2eid[title.lower()] = eid

        # Sorted titles by length (longest first) — greedy matching
        self.sorted_titles = sorted(self.title2eid.keys(), key=len, reverse=True)

        logger.info("  %d entity texts (chunks), %d searchable titles, %d relations",
                     len(self.chunks), len(self.title2eid), len(self.rel2id))

        # ------ 2. Build KG adjacency from triplets ------
        # neighbors[eid] = set of (relation_string, neighbor_eid)
        self.neighbors: dict[int, set] = defaultdict(set)
        total_triplets = 0

        for split in ("train.txt", "valid.txt", "test.txt"):
            path = os.path.join(data_dir, split)
            if os.path.exists(path):
                trips = load_triplets(path, self.mid2id)
                total_triplets += len(trips)
                for src, rel, dst in trips:
                    self.neighbors[src].add((rel, dst))
                    self.neighbors[dst].add((rel, src))  # undirected

        logger.info("  %d triplets loaded → KG adjacency built", total_triplets)

        # ------ 3. Build BM25 index ------
        logger.info("Building BM25 index over %d chunks ...", len(self.chunks))
        self.bm25 = BM25(self.chunks)
        logger.info("  BM25 index built.")

        # ------ 4. Build / load FAISS index (for fallback semantic search) ------
        logger.info("Loading embedding model: %s", embed_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        self._build_or_load_index()

    def _build_or_load_index(self):
        """Build or load cached FAISS index for semantic fallback."""
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
    # BƯỚC 1: Entity Extraction — string match tên entity trong câu hỏi
    # ================================================================
    def extract_entities(self, question: str) -> list[tuple[int, str]]:
        """
        Trích xuất entity từ câu hỏi bằng greedy longest-match string matching.

        Cách hoạt động:
        - So sánh từng title entity với text câu hỏi (lowercase)
        - Ưu tiên match dài nhất (longest-first) để tránh "Essex" thay vì "University of Essex"
        - Tránh overlap giữa các match

        Returns:
            list of (eid, title) — các entity được nhận diện trong câu hỏi
        """
        question_lower = question.lower()
        matched = []
        used_spans: list[tuple[int, int]] = []

        for title_lower in self.sorted_titles:
            pos = question_lower.find(title_lower)
            if pos == -1:
                continue

            end = pos + len(title_lower)
            overlap = any(pos < e and end > s for s, e in used_spans)

            if not overlap:
                eid = self.title2eid[title_lower]
                original_title = self.id2title.get(eid, title_lower)
                matched.append((eid, original_title))
                used_spans.append((pos, end))

        return matched

    # ================================================================
    # BƯỚC 1b: Phrase Search — tìm chunk chứa cụm từ chính trong câu hỏi
    # ================================================================
    def _phrase_search(self, question: str,
                       exclude_indices: set[int] | None = None) -> list[tuple[int, str, str]]:
        """
        Tìm chunks chứa cụm từ noun phrase từ câu hỏi bằng substring search.

        Giải quyết vấn đề:
        - "Planet Terror" không phải entity title riêng → không được extract
        - Nhưng mô tả của Grindhouse (id=9) CÓ CHỨA cụm "Planet Terror"
        - → Phrase search trực tiếp trên nội dung chunk tìm được đúng chunk

        Cách hoạt động:
        1. Trích các cụm từ danh từ (NP) từ câu hỏi (2-4 words bắt đầu bằng hoa)
        2. Tìm trong tất cả chunks xem có chứa cụm đó không
        3. Trả về chunks khớp, ưu tiên cụm dài hơn

        Returns:
            list of (chunk_idx, chunk_text, source="phrase")
        """
        exclude = set(exclude_indices or [])
        results: list[tuple[int, str, str]] = []
        seen: set[int] = set()

        # Trích noun phrase: chuỗi 2-4 từ viết hoa liền nhau (tên riêng)
        # VD: "Planet Terror", "Death Proof", "White Hart Lane"
        phrase_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        )
        phrases = phrase_pattern.findall(question)

        # Sắp xếp dài nhất trước để ưu tiên cụm chính xác hơn
        phrases = sorted(set(phrases), key=len, reverse=True)

        for phrase in phrases:
            if len(phrase) < 5:  # Bỏ qua cụm quá ngắn
                continue
            phrase_lower = phrase.lower()
            for idx, chunk_text in enumerate(self.chunks):
                if idx in exclude or idx in seen:
                    continue
                if phrase_lower in chunk_text.lower():
                    results.append((idx, chunk_text, "phrase"))
                    seen.add(idx)
                    break  # Mỗi phrase chỉ lấy chunk đầu tiên match

        return results

    # ================================================================
    # BƯỚC 2: Lấy chunks của entity trực tiếp (Wikipedia description)
    # ================================================================
    def _get_entity_chunks(self, entity_ids: list[int]) -> list[tuple[int, str, str]]:
        """
        Trả về danh sách chunks (Wikipedia descriptions) của các entity đã xác định.

        Mỗi entity → 1 chunk = Wikipedia description từ id2text.txt

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
    def _get_neighbor_chunks(self, entity_ids: list[int], max_per_entity: int = 5) -> list[tuple[int, str, str]]:
        """
        Lấy chunk (Wikipedia description) của entities lân cận trong KG (1-hop).

        Ví dụ: entity "University of Essex" có neighbor "Essex" (via location/contains)
        → lấy Wikipedia description của "Essex" làm additional context

        Ưu tiên neighbors xuất hiện nhiều lần (shared across multiple query entities).

        Returns:
            list of (chunk_idx, chunk_text, source="neighbor")
        """
        neighbor_count: dict[int, int] = defaultdict(int)
        seen_entity_ids = set(entity_ids)

        for eid in entity_ids:
            for _rel, nb in self.neighbors.get(eid, set()):
                if nb not in seen_entity_ids:
                    neighbor_count[nb] += 1

        sorted_neighbors = sorted(neighbor_count.items(), key=lambda x: x[1], reverse=True)

        results = []
        for nb_eid, _count in sorted_neighbors[:max_per_entity]:
            if nb_eid in self.eid_to_chunk_idx:
                idx = self.eid_to_chunk_idx[nb_eid]
                results.append((idx, self.chunks[idx], "neighbor"))

        return results

    # ================================================================
    # BƯỚC 4: BM25 keyword search
    # ================================================================
    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K,
                     exclude_indices: set[int] | None = None) -> list[tuple[int, str, str]]:
        """
        BM25 keyword search trên toàn bộ corpus chunks.

        BM25 tốt cho:
        - Câu hỏi có từ khóa cụ thể (tên riêng, thuật ngữ)
        - Trường hợp entity extraction không match được nhưng từ khóa xuất hiện trong chunks
        
        VD: query "Who coined the term telepathy" → BM25 sẽ tìm chunk chứa "telepathy", "coined", "term"

        Returns:
            list of (chunk_idx, chunk_text, source="bm25")
        """
        exclude = exclude_indices or set()
        top_results = self.bm25.get_top_k(query, top_k, exclude=exclude)

        results = []
        for idx, score in top_results:
            results.append((idx, self.chunks[idx], "bm25"))
        return results

    # ================================================================
    # BƯỚC 5: FAISS semantic fallback
    # ================================================================
    def _semantic_fallback(self, query: str, top_k: int = SEMANTIC_K,
                           exclude_indices: set[int] | None = None) -> list[tuple[int, str, str]]:
        """
        FAISS semantic search dùng làm fallback khi entity + BM25 không đủ chunks.

        Dùng embedding similarity để tìm chunks ngữ nghĩa gần với query,
        bất kể có từ khóa trùng hay không.

        Returns:
            list of (chunk_idx, chunk_text, source="semantic_fallback")
        """
        exclude = set(exclude_indices or [])
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k * 3)

        results = []
        for idx, sc in zip(indices[0], scores[0]):
            if int(idx) < 0 or int(idx) in exclude:
                continue
            results.append((int(idx), self.chunks[int(idx)], "semantic_fallback"))
            if len(results) >= top_k:
                break

        return results

    # ================================================================
    # BƯỚC 6: Lấy KG facts (human-readable)
    # ================================================================
    def _get_kg_facts(self, entity_ids: list[int], max_facts: int = MAX_KG_FACTS) -> list[str]:
        """
        Trích xuất KG facts dưới dạng readable: "EntityA → relation → EntityB"

        Ví dụ: "University of Essex → location → Essex"
                "Thomas Hobbes → profession → philosopher"
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
    # MAIN QUERY — Entity + BM25 + Semantic Pipeline
    # ================================================================
    def query(self, question: str,
              top_k: int = TOP_K,
              bm25_top_k: int = BM25_TOP_K,
              semantic_k: int = SEMANTIC_K) -> dict:
        """
        Streamlined Hybrid Retrieval Pipeline:

        Stage 1 — Entity chunks (direct match, no neighbors):
          a) Entity Extraction: tìm entity trong câu hỏi
          b) Entity Chunks: lấy Wikipedia description của entity đó

        Stage 2 — BM25 keyword search (3-4 chunks):
          c) BM25: tìm thêm chunks bằng keyword matching

        Stage 3 — Semantic search (4-5 chunks, luôn chạy):
          d) FAISS: bổ sung chunks bằng semantic similarity

        Stage 4 — Aggregate:
          e) Deduplicate + rank (entity > bm25 > semantic)
          f) Thêm KG facts
          g) Build context string → LLM
        """
        # ---- Stage 1: Entity Extraction + Entity Chunks ----
        matched = self.extract_entities(question)
        matched_eids = [eid for eid, _title in matched]
        matched_names = [title for _eid, title in matched]

        entity_chunks = self._get_entity_chunks(matched_eids)

        # ---- Collect used indices để tránh trùng ----
        used_indices: set[int] = set()
        for idx, _text, _src in entity_chunks:
            used_indices.add(idx)

        # ---- Stage 1b: Phrase Search (noun phrase trong câu hỏi → tìm trong chunk) ----
        # Xử lý trường hợp tên phim/entity không có trong id2title nhưng có trong id2text
        # VD: "Planet Terror" không phải entity title nhưng xuất hiện trong chunk Grindhouse
        phrase_chunks = self._phrase_search(question, exclude_indices=used_indices)
        for idx, _text, _src in phrase_chunks:
            used_indices.add(idx)

        # ---- Stage 2: BM25 keyword search (3-4 chunks) ----
        bm25_chunks = self._bm25_search(question, top_k=bm25_top_k, exclude_indices=used_indices)
        for idx, _text, _src in bm25_chunks:
            used_indices.add(idx)

        # ---- Stage 3: Semantic search (4-5 chunks, LUÔN CHẠY) ----
        semantic_chunks = self._semantic_fallback(
            question, top_k=semantic_k, exclude_indices=used_indices
        )

        # ---- Stage 4a: Gộp & deduplicate ----
        # Thứ tự ưu tiên: entity > phrase > bm25 > semantic
        all_chunks: list[tuple[int, str, str]] = []
        seen_final: set[int] = set()

        for idx, text, src in (entity_chunks + phrase_chunks + bm25_chunks + semantic_chunks):
            if idx not in seen_final and len(all_chunks) < top_k:
                all_chunks.append((idx, text, src))
                seen_final.add(idx)

        # ---- Stage 4b: KG facts ----
        kg_facts = self._get_kg_facts(matched_eids)

        # ---- Stage 4c: Build context string ----
        source_labels = {
            "entity": "[Entity Description]",
            "phrase": "[Phrase Match]",
            "bm25": "[Keyword Match]",
            "semantic_fallback": "[Semantic Context]",
        }

        context_parts = []
        for idx, text, src in all_chunks:
            label = source_labels.get(src, "")
            context_parts.append(f"{label}\n{text}" if label else text)

        if kg_facts:
            context_parts.append("[Knowledge Graph Facts]\n" + "\n".join(kg_facts))

        context = "\n\n".join(context_parts)

        return {
            "query": question,
            "context": context,
            "matched_entities": matched_names,
            "entity_chunks": [(idx, src) for idx, _, src in entity_chunks],
            "phrase_chunks": [(idx, src) for idx, _, src in phrase_chunks],
            "bm25_chunks": [(idx, src) for idx, _, src in bm25_chunks],
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

    print(f"\n{'='*65}")
    print(f"✅ SAT GraphRAG v2 (Entity-First + BM25 + FAISS Fallback) ready")
    print(f"   Chunks (entity Wikipedia texts): {len(rag.chunks)}")
    print(f"   Searchable entity titles:        {len(rag.title2eid)}")
    print(f"   KG relations types:              {len(rag.rel2id)}")
    print(f"   KG entities with neighbors:      {len(rag.neighbors)}")
    print(f"   FAISS index vectors:             {rag.index.ntotal}")
    print(f"   BM25 corpus size:                {rag.bm25.corpus_size}")
    print(f"{'='*65}")
    print(f"   Retrieval config:")
    print(f"     Entity chunks:   direct match (no neighbors)")
    print(f"     BM25 chunks:     top-{BM25_TOP_K}")
    print(f"     Semantic chunks: {SEMANTIC_K} (FAISS, always-on)")
    print(f"     Context max:     {TOP_K} chunks total")
    print(f"{'='*65}")

    # Load questions
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Questions to evaluate: {len(data)}\n")

    outputs = []
    stats = {
        "entity_found": 0,
        "bm25_only": 0,
        "full_fallback": 0,
    }

    for i, item in enumerate(data, 1):
        q = item["question"]
        gt = item.get("answer") or item.get("groundtruth") or ""

        print(f"[{i}/{len(data)}] {q}")

        r = rag.query(
            q,
            top_k=TOP_K,
            bm25_top_k=BM25_TOP_K,
            semantic_k=SEMANTIC_K,
        )
        ctx = r["context"]

        # Log retrieval strategy
        has_entity = bool(r["matched_entities"])

        if has_entity:
            stats["entity_found"] += 1
            print(f"  🎯 Entities: {r['matched_entities']}")
            print(f"  📦 entity={len(r['entity_chunks'])} "
                  f"+ bm25={len(r['bm25_chunks'])} "
                  f"+ semantic={len(r['semantic_chunks'])}")
        elif r["bm25_chunks"]:
            stats["bm25_only"] += 1
            print(f"  🔍 No entity match → BM25 + semantic")
            print(f"  📦 bm25={len(r['bm25_chunks'])} + semantic={len(r['semantic_chunks'])}")
        else:
            stats["full_fallback"] += 1
            print(f"  🔄 Semantic only")
            print(f"  📦 semantic={len(r['semantic_chunks'])}")

        if r["kg_facts"]:
            print(f"  📊 KG facts: {len(r['kg_facts'])}")

        print(f"  📝 Context: {len(ctx)} chars, {r['total_chunks']} chunks")

        ans = kimi_answer(q, ctx, client)
        print(f"  💬 {ans[:120]}")

        outputs.append({
            "question": q,
            "answer": ans,
            "groundtruth": gt,
            "matched_entities": r["matched_entities"],
            "retrieval": {
                "entity_chunks": len(r["entity_chunks"]),
                "bm25_chunks": len(r["bm25_chunks"]),
                "semantic_chunks": len(r["semantic_chunks"]),
                "kg_facts": len(r["kg_facts"]),
                "total_chunks": r["total_chunks"],
            }
        })

        time.sleep(SLEEP)

    # Save results
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved → {OUT_PATH}")

    # Summary
    total = len(outputs)
    print(f"\n{'='*65}")
    print(f"📊 Retrieval Strategy Summary:")
    print(f"   Entity found:     {stats['entity_found']}/{total} "
          f"({100*stats['entity_found']/total:.1f}%)")
    print(f"   BM25 only:        {stats['bm25_only']}/{total} "
          f"({100*stats['bm25_only']/total:.1f}%)")
    print(f"   Full fallback:    {stats['full_fallback']}/{total} "
          f"({100*stats['full_fallback']/total:.1f}%)")

    correct = sum(
        1 for o in outputs
        if o["groundtruth"].lower() in o["answer"].lower()
    )
    print(f"   Substring match:  {correct}/{total} "
          f"({100*correct/total:.1f}%)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
