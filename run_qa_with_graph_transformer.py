# run_qa_with_graph_transformer.py
"""
Test QA với Graph Transformer embeddings trên FB15k-237N KG.

Flow:
1. Load KG data + Graph Transformer embeddings
2. Load câu hỏi từ qa_eval.json
3. Với mỗi câu hỏi:
   - Semantic search (text embeddings)
   - Graph-enhanced search (node embeddings)
   - Combine scores
   - Generate answer với LLM
4. Evaluate results
"""
import os
import json
import pickle
import numpy as np
import torch
import faiss
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import torch.nn.functional as F

# ============================================================================
# Config
# ============================================================================
DATA_DIR = "sat_data_backup/FB15k-237N"
CACHE_DIR = "sat_kg_data"
QA_FILE = "qa_eval.json"
OUTPUT_FILE = "qa_results_graph_transformer.json"

TOP_K = 5           # Số chunks để retrieve
ALPHA = 1.0         # Weight cho semantic (1-alpha cho graph) - 1.0 = semantic only

# ============================================================================
# Data Loading
# ============================================================================

def load_id2text(path: str) -> Dict[int, str]:
    """Load entity ID → text description"""
    id2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                id2text[int(parts[0])] = parts[1]
    return id2text


def get_kimi_client() -> OpenAI:
    """Get Kimi client via NVIDIA NIM"""
    api_key = os.getenv("NVAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing NVAPI_KEY")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )


def llm_generate(prompt: str, max_tokens: int = 256) -> str:
    """Generate answer with LLM"""
    try:
        client = get_kimi_client()
        resp = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"


# ============================================================================
# Main QA System
# ============================================================================

class GraphTransformerQA:
    def __init__(self):
        print("🚀 Initializing GraphTransformerQA...")
        
        # Load entity descriptions (these are our "chunks")
        self.id2text = load_id2text(os.path.join(DATA_DIR, "id2text.txt"))
        self.chunks = [self.id2text[i] for i in range(len(self.id2text))]
        print(f"   ✅ Loaded {len(self.chunks)} entity descriptions")
        
        # Load or build text embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_path = os.path.join(CACHE_DIR, "text_embeddings.npy")
        
        if os.path.exists(emb_path):
            self.text_embeddings = np.load(emb_path)
            print(f"   ✅ Loaded text embeddings from cache")
        else:
            print("   Building text embeddings...")
            self.text_embeddings = self._build_text_embeddings()
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(emb_path, self.text_embeddings)
            print(f"   ✅ Saved text embeddings to {emb_path}")
        
        # Build FAISS index for text search
        self.text_index = faiss.IndexFlatIP(self.text_embeddings.shape[1])
        self.text_index.add(self.text_embeddings)
        print(f"   ✅ FAISS index built ({self.text_embeddings.shape})")
        
        # Load Graph Transformer embeddings
        node_emb_path = os.path.join(CACHE_DIR, "node_embeddings.pt")
        if os.path.exists(node_emb_path):
            self.node_embeddings = torch.load(node_emb_path)
            # Normalize for cosine similarity
            self.node_embeddings = F.normalize(self.node_embeddings, dim=-1)
            print(f"   ✅ Loaded Graph Transformer embeddings ({self.node_embeddings.shape})")
        else:
            print("   ⚠️ No Graph Transformer embeddings found, run test_graph_transformer_sat.py first")
            self.node_embeddings = None
    
    def _build_text_embeddings(self) -> np.ndarray:
        """Build text embeddings for all entity descriptions"""
        batch_size = 64
        emb_list = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            batch_emb = self.embedding_model.encode(batch, normalize_embeddings=True)
            emb_list.append(batch_emb)
            if (i // batch_size) % 50 == 0:
                print(f"      Processed {i}/{len(self.chunks)}...")
        return np.vstack(emb_list).astype("float32")
    
    def _semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Semantic search using text embeddings"""
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.text_index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(I[0], D[0])]
    
    def _graph_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Graph-enhanced search using node embeddings"""
        if self.node_embeddings is None:
            return []
        
        # Encode query
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        q_emb = torch.tensor(q_emb)
        
        # Project query to node embedding space (simple approach: use text embedding similarity 
        # to find anchor nodes, then use their graph embeddings)
        
        # Step 1: Find top semantic matches
        sem_results = self._semantic_search(query, top_k=20)
        
        # Step 2: Get graph embeddings for these entities
        entity_ids = [idx for idx, _ in sem_results]
        graph_embs = self.node_embeddings[entity_ids]  # (20, 128)
        
        # Step 3: Find entities most similar in graph space to these anchors
        # Use mean of anchor embeddings as query
        anchor_emb = graph_embs.mean(dim=0, keepdim=True)  # (1, 128)
        
        # Compute similarity to all nodes
        similarities = torch.mm(anchor_emb, self.node_embeddings.t()).squeeze(0)  # (N,)
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        return [(int(idx), float(score)) for idx, score in zip(top_indices.tolist(), top_scores.tolist())]
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[Tuple[int, str, float]]:
        """
        Hybrid retrieval combining semantic and graph scores.
        
        Args:
            alpha: weight for semantic score (1-alpha for graph score)
        """
        # Semantic search
        sem_results = self._semantic_search(query, top_k=top_k * 2)
        sem_dict = {idx: score for idx, score in sem_results}
        
        # Graph search
        graph_results = self._graph_search(query, top_k=top_k * 2)
        graph_dict = {idx: score for idx, score in graph_results}
        
        # Combine scores
        all_indices = set(sem_dict.keys()) | set(graph_dict.keys())
        combined = []
        
        for idx in all_indices:
            sem_score = sem_dict.get(idx, 0.0)
            graph_score = graph_dict.get(idx, 0.0)
            final_score = alpha * sem_score + (1 - alpha) * graph_score
            combined.append((idx, self.chunks[idx], final_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[2], reverse=True)
        
        return combined[:top_k]
    
    def answer(self, question: str, top_k: int = 5, alpha: float = 0.6) -> Dict:
        """Answer a question using retrieved context"""
        # Retrieve relevant chunks
        results = self.retrieve(question, top_k=top_k, alpha=alpha)
        
        # Build context
        context_parts = [text for _, text, _ in results]
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""You are a precise QA assistant with strong reasoning abilities.

INSTRUCTIONS:
1. Read the context carefully and answer based on it.
2. For "WHAT/WHO/WHERE/WHEN" questions: Extract the answer directly from context.
3. For "WHY/HOW" questions: Use reasoning to infer the answer from context clues.
   - Look for cause-effect relationships, purposes, or explanations implied in the text.
   - Even if not explicitly stated, derive logical conclusions from available information.
4. If you can reasonably infer an answer from the context, provide it.
5. ONLY say "Not stated in the text" if there is absolutely NO relevant information.

Context:
{context}

Question: {question}

Think step by step, then provide a concise answer (1-2 sentences):"""
        
        answer = llm_generate(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved": [(idx, score) for idx, _, score in results]
        }


def main():
    # Initialize QA system
    qa = GraphTransformerQA()
    
    # Load questions
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"\n📝 Loaded {len(questions)} questions from {QA_FILE}")
    
    # Run QA
    print(f"\n🔄 Running QA with Graph Transformer (alpha={ALPHA})...\n")
    
    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        groundtruth = q["answer"]
        
        result = qa.answer(question, top_k=TOP_K, alpha=ALPHA)
        result["groundtruth"] = groundtruth
        results.append(result)
        
        print(f"[{i+1}/{len(questions)}] {question[:50]}...")
        print(f"   Answer: {result['answer'][:60]}...")
        print()
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Quick evaluation
    print("\n" + "=" * 60)
    print("📊 Quick Evaluation")
    print("=" * 60)
    
    correct = 0
    not_found = 0
    
    for r in results:
        ans = r["answer"].lower()
        gt = r["groundtruth"].lower()
        
        if "not stated" in ans:
            not_found += 1
        elif gt in ans or any(word in ans for word in gt.split()[:3]):
            correct += 1
    
    print(f"   Total questions: {len(results)}")
    print(f"   Correct (fuzzy): {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"   Not stated: {not_found}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
