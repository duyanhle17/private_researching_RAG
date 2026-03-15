# simple_graphrag.py
import os
import json
import logging
from typing import List, Set, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
import networkx as nx
from rouge import Rouge

from openai import OpenAI

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleGraphRAG")

# ---------- OpenAI-compatible client (NVIDIA NIM) ----------
def get_kimi_client() -> OpenAI:
    api_key = os.getenv("NVAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing NVAPI_KEY. Set it via: export NVAPI_KEY=... (mac/linux) or $env:NVAPI_KEY=... (powershell)")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

# ---------- Utility LLM ----------
def llm_generate(
    prompt: str,
    model: str = "moonshotai/kimi-k2-instruct-0905",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call Kimi-K2 via NVIDIA NIM (OpenAI-compatible).
    """
    try:
        client = get_kimi_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""

# Optional / extra libs (used if installed)
try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from node2vec import Node2Vec
except Exception:
    Node2Vec = None

try:
    from networkx.algorithms import community as nx_community
except Exception:
    nx_community = None

# ---------------- Helpers ----------------
def char_splitter(text: str, chunk_size: int = 1200, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

import re
def split_long_context(text, max_chars=800):
    """
    Split long text into ~800-char paragraphs by sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ---------------- Configuration ----------------
@dataclass
class GraphRAGConfig:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    enable_local: bool = True
    enable_naive_rag: bool = False

    tokenizer_type: str = "tiktoken"
    tiktoken_model_name: str = "cl100k_base"

    huggingface_model_name: str = "bert-base-uncased"
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    chunk_char_size: int = 1200
    chunk_char_overlap: int = 100
    chunk_func: Optional[Callable] = None

    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    graph_cluster_algorithm: str = "greedy"
    max_graph_cluster_size: int = 8
    graph_cluster_seed: int = 0xDEADBEEF

    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 128,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 3,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2


# ---------------- Token-based splitter ----------------
class TokenSplitter:
    def __init__(self, cfg: GraphRAGConfig):
        self.cfg = cfg
        if tiktoken is None:
            raise RuntimeError("tiktoken not installed; pip install tiktoken")

        try:
            self.enc = tiktoken.encoding_for_model(cfg.tiktoken_model_name)
        except KeyError:
            logger.warning(f"tiktoken cannot map '{cfg.tiktoken_model_name}', fallback cl100k_base")
            self.enc = tiktoken.get_encoding("cl100k_base")

        self.chunk_size = cfg.chunk_token_size
        self.overlap = cfg.chunk_overlap_token_size

    def chunk(self, text: str) -> List[str]:
        tokens = self.enc.encode(text)
        chunks: List[str] = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(tokens), step):
            sub = tokens[i:i + self.chunk_size]
            if not sub:
                continue
            chunks.append(self.enc.decode(sub))
        return chunks


# ---------------- Main SimpleGraphRAG ----------------
class SimpleGraphRAG:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        faiss_index_factory: Optional[str] = None,
        cfg: Optional[GraphRAGConfig] = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_factory = faiss_index_factory

        self.cfg = cfg or GraphRAGConfig()
        os.makedirs(self.cfg.working_dir, exist_ok=True)

        self.dataset: List[str] = []
        self.chunk_entities: List[Set[str]] = []
        self.nlp = None
        self.embedding_model = None
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.kg: Optional[nx.DiGraph] = None
        self.node_embeddings: Optional[Dict[str, np.ndarray]] = None

        self._init_models()

    def _init_models(self):
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.max_length = 10_000_000
        except OSError:
            raise RuntimeError("Missing spaCy model. Run: python -m spacy download en_core_web_sm")

    def load_json_and_concat(self, json_path: str, text_key: str = "context") -> str:
        logger.info(f"Loading JSON from {json_path} ...")
        try:
            df = pd.read_json(json_path)
            if text_key not in df.columns:
                raise ValueError(f"Key '{text_key}' not found. Columns: {df.columns.tolist()}")
            full_text = " ".join(df[text_key].astype(str).tolist())
        except Exception as e:
            logger.warning(f"Read JSON failed: {e}. Using fallback text.")
            full_text = (
                "Basal cell carcinoma (BCC) is the most common type of skin cancer. "
                "It is frequently treated with surgery. Fair skin increases the risk."
            )
        return full_text

    def chunk_text(self, text: str, method: str = "auto"):
        logger.info("Chunking text ...")
        if self.cfg.chunk_func is not None:
            chunks = self.cfg.chunk_func(text)
            self.dataset = [c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in chunks]
            logger.info(f"Created {len(self.dataset)} chunks (custom)")
            return

        mode = method
        if method == "auto":
            mode = "token" if self.cfg.tokenizer_type == "tiktoken" and tiktoken is not None else "char"

        if mode == "token":
            if tiktoken is None:
                self.dataset = char_splitter(text, self.cfg.chunk_char_size, self.cfg.chunk_char_overlap)
            else:
                splitter = TokenSplitter(self.cfg)
                self.dataset = splitter.chunk(text)
        else:
            self.dataset = char_splitter(text, self.cfg.chunk_char_size, self.cfg.chunk_char_overlap)

        logger.info(f"Created {len(self.dataset)} chunks (mode={mode})")

    def build_embeddings_and_index(self, normalize: bool = True, batch_size: Optional[int] = None):
        if not self.dataset:
            raise RuntimeError("Dataset empty. Set rag.dataset or run chunk_text() first.")

        batch_size = batch_size or self.cfg.embedding_batch_num
        logger.info("Computing embeddings ...")
        emb_list = []
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i + batch_size]
            emb_batch = self.embedding_model.encode(batch, normalize_embeddings=normalize)
            emb_list.append(emb_batch)

        emb = np.vstack(emb_list).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        emb = emb / norms
        self.embeddings = emb

        d = emb.shape[1]
        logger.info(f"Creating FAISS index (dimension={d})")
        if self.faiss_index_factory:
            try:
                self.index = faiss.index_factory(d, self.faiss_index_factory)
            except Exception as e:
                logger.warning(f"faiss.index_factory failed: {e}, fallback IndexFlatIP")
                self.index = faiss.IndexFlatIP(d)
        else:
            self.index = faiss.IndexFlatIP(d)

        self.index.add(emb)
        logger.info("FAISS index built")

    def build_kg(self, use_dependency: bool = True):
        logger.info("Building Knowledge Graph (chunk-based NER)...")
        self.kg = nx.DiGraph()
        entity_nodes = set()
        self.chunk_entities = []

        for chunk in self.dataset:
            cdoc = self.nlp(chunk)

            ents = set()
            for ent in cdoc.ents:
                ents.add(ent.text.strip())

            for np_ in cdoc.noun_chunks:
                txt = np_.text.strip().lower()
                if len(txt.split()) >= 2:
                    ents.add(txt)

            ent_set = set(list(ents))
            self.chunk_entities.append(ent_set)

            for e in ent_set:
                if e not in self.kg:
                    label = None
                    for ent in cdoc.ents:
                        if ent.text.strip() == e:
                            label = ent.label_
                            break
                    self.kg.add_node(e, label=label or "UNKNOWN")
                entity_nodes.add(e)

            for sent in cdoc.sents:
                sent_ents = [e.text.strip() for e in sent.ents]

                if use_dependency:
                    for token in sent:
                        if "subj" in token.dep_:
                            subj = token.text
                            verb = token.head.lemma_
                            for child in token.head.children:
                                if "obj" in child.dep_:
                                    obj = child.text
                                    if subj in entity_nodes and obj in entity_nodes:
                                        self.kg.add_edge(subj, obj, relation=verb)

                for i in range(len(sent_ents)):
                    for j in range(i + 1, len(sent_ents)):
                        a, b = sent_ents[i], sent_ents[j]
                        if a != b and not self.kg.has_edge(a, b):
                            self.kg.add_edge(a, b, relation="co_occurs_with")

        logger.info(f"KG built with {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")

    def _semantic_search(self, query: str, top_k: int = 5):
        q_emb = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        return [(int(i), float(D[0][idx])) for idx, i in enumerate(I[0])]


    def _graph_search_scores(self, query: str):
        q_doc = self.nlp(query)
        q_entities = set([ent.text.strip() for ent in q_doc.ents if ent.text.strip()])
        if len(q_entities) == 0:
            return np.zeros(len(self.dataset), dtype=float)
        scores = np.array([len(self.chunk_entities[i].intersection(q_entities)) for i in range(len(self.dataset))], dtype=float)
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

        top_graph_idx = np.argsort(graph_scores)[-top_k:][::-1]
        graph_chunks = [self.dataset[i] for i in top_graph_idx if graph_scores[i] > 0 and i not in combined_idxs]

        kg_facts = []
        q_doc = self.nlp(query)
        q_entities = set([ent.text.strip() for ent in q_doc.ents if ent.text.strip()])
        for qe in q_entities:
            if self.kg is not None and qe in self.kg:
                for nb in list(self.kg.successors(qe))[:3]:
                    rel = self.kg.get_edge_data(qe, nb).get("relation", "related_to")
                    kg_facts.append(f"{qe} {rel} {nb}.")
                for nb in list(self.kg.predecessors(qe))[:3]:
                    rel = self.kg.get_edge_data(nb, qe).get("relation", "related_to")
                    kg_facts.append(f"{nb} {rel} {qe}.")

        combined_context = " ".join(combined_chunks + graph_chunks + kg_facts)
        return {"query": query, "context": combined_context, "sem_idxs": combined_idxs}

    def answer_question_llm(self, query: str, llm_model: str = "moonshotai/kimi-k2-instruct-0905", top_k: int = 5, alpha: float = 0.8):
        result = self.graphrag_query(query, top_k=top_k, alpha=alpha)
        context_text = result["context"]
        prompt = f"""
You are a high-accuracy QA assistant.

You MUST answer using ONLY the provided context.
If no relevant information exists, reply EXACTLY:
no relevant information found

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
""".strip()
        answer = llm_generate(prompt, model=llm_model, temperature=0.0, max_tokens=128)
        return answer, result

    def evaluate_answer(self, pred: str, truth: str) -> float:
        scorer = Rouge()
        scores = scorer.get_scores(pred, truth)
        return scores[0]["rouge-l"]["f"]
