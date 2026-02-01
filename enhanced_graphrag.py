# enhanced_graphrag.py
"""
Enhanced GraphRAG with Structure-Aware components inspired by SAT paper
(Structure Aware Alignment and Tuning)

Key improvements:
1. Learnable entity/relation embeddings
2. Graph Transformer for structure-aware node representations  
3. Text-Graph alignment using contrastive learning
4. Better knowledge graph construction with explicit relations
"""

import os
import json
import logging
from typing import List, Set, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import faiss
from sentence_transformers import SentenceTransformer
import spacy
import networkx as nx

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedGraphRAG")

# ============================================================================
# PART 1: Graph Transformer Components (inspired by SAT)
# ============================================================================

def positional_encoding(seq_len: int, d_model: int, normalize: bool = True) -> torch.Tensor:
    """Sinusoidal positional encoding for nodes"""
    import math
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10 + 1e-8)
    return pe


class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer with multi-head attention on graph edges.
    Inspired by SAT's GTLayer.
    """
    def __init__(self, d_model: int, n_heads: int, use_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, node_embeds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeds: (num_nodes, d_model)
            edge_index: (2, num_edges) - [source_nodes, target_nodes]
        Returns:
            Updated node embeddings
        """
        num_nodes = node_embeds.shape[0]
        device = node_embeds.device
        
        rows, cols = edge_index[0], edge_index[1]
        
        # Get embeddings for source and target of each edge
        src_embeds = node_embeds[rows]  # (num_edges, d_model)
        tgt_embeds = node_embeds[cols]  # (num_edges, d_model)
        
        # Project to Q, K, V
        q = self.q_proj(src_embeds).view(-1, self.n_heads, self.head_dim)
        k = self.k_proj(tgt_embeds).view(-1, self.n_heads, self.head_dim)
        v = self.v_proj(tgt_embeds).view(-1, self.n_heads, self.head_dim)
        
        # Compute attention scores
        att = torch.einsum("ehd,ehd->eh", q, k) / (self.head_dim ** 0.5)
        att = torch.clamp(att, -10.0, 10.0)
        exp_att = torch.exp(att)
        
        # Normalize attention by summing over all edges from each source node
        att_sum = torch.zeros(num_nodes, self.n_heads, device=device)
        att_sum.index_add_(0, rows, exp_att)
        att_norm = exp_att / (att_sum[rows] + 1e-8)
        
        # Aggregate values
        weighted_v = torch.einsum("eh,ehd->ehd", att_norm, v)
        weighted_v = weighted_v.view(-1, self.d_model)
        
        # Scatter add to accumulate messages at each node
        out = torch.zeros(num_nodes, self.d_model, device=device)
        out.index_add_(0, rows, weighted_v)
        
        # Residual connection
        out = out + node_embeds
        
        if self.use_norm:
            out = self.norm(out)
        
        return out


class GraphTransformer(nn.Module):
    """
    Full Graph Transformer encoder for learning structure-aware node embeddings.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        use_pos_encoding: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.entity_embedding = nn.Embedding(num_entities, input_dim)
        self.relation_embedding = nn.Embedding(num_relations * 2, input_dim)  # *2 for inverse relations
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_enc = nn.Parameter(
                positional_encoding(num_entities, hidden_dim),
                requires_grad=True
            )
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads, use_norm=True)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.entity_embedding.weight, std=0.02)
        nn.init.normal_(self.relation_embedding.weight, std=0.02)
    
    def forward(self, entity_ids: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_ids: (num_entities,) - entity indices
            edge_index: (2, num_edges) - graph structure
        Returns:
            node_embeds: (num_entities, output_dim)
        """
        x = self.entity_embedding(entity_ids)
        x = self.input_proj(x)
        
        if self.use_pos_encoding:
            x = self.dropout(x + self.pos_enc[:x.shape[0]])
        else:
            x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        
        return self.output_proj(x)


# ============================================================================
# PART 2: Text-Graph Alignment Module (CLIP-style, inspired by SAT)
# ============================================================================

class TextEncoder(nn.Module):
    """
    Simple text encoder using sentence transformers + projection
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        output_dim: int = 128,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get the embedding dimension from the model
        self.backbone_dim = self.encoder.get_sentence_embedding_dimension()
        self.projection = nn.Linear(self.backbone_dim, output_dim)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: List of text strings
        Returns:
            embeddings: (batch_size, output_dim)
        """
        with torch.no_grad():
            embeds = self.encoder.encode(texts, convert_to_tensor=True)
        return self.projection(embeds)


class TextGraphAligner(nn.Module):
    """
    Aligns text and graph embeddings using contrastive learning (CLIP-style).
    This enables better retrieval by learning joint representations.
    """
    def __init__(
        self,
        graph_encoder: GraphTransformer,
        text_encoder: TextEncoder,
        temperature: float = 0.07
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1 / temperature)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def encode_graph_nodes(self, entity_ids: torch.Tensor, edge_index: torch.Tensor, 
                           node_indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific nodes"""
        all_embeds = self.graph_encoder(entity_ids, edge_index)
        return all_embeds[node_indices]
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self.text_encoder(texts)
    
    def contrastive_loss(
        self,
        graph_feats: torch.Tensor,
        text_feats: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss for aligning graph and text representations.
        """
        # Normalize features
        graph_feats = F.normalize(graph_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        
        # Compute similarity matrix
        logits = self.temperature * graph_feats @ text_feats.t()
        
        # Cross-entropy loss (symmetric)
        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.t(), labels)
        
        return (loss_g2t + loss_t2g) / 2
    
    def forward(
        self,
        entity_ids: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both graph and text embeddings.
        """
        graph_feats = self.encode_graph_nodes(entity_ids, edge_index, node_indices)
        text_feats = self.encode_texts(texts)
        return graph_feats, text_feats


# ============================================================================
# PART 3: Enhanced Knowledge Graph Builder
# ============================================================================

@dataclass
class KGTriple:
    """A knowledge graph triple"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0
    source_chunk_idx: int = -1


class EnhancedKGBuilder:
    """
    Enhanced KG builder with:
    1. Better entity extraction
    2. Explicit relation extraction (not just co-occurrence)
    3. Triple scoring/confidence
    4. Entity/Relation ID mapping (like SAT)
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
            self.nlp.max_length = 10_000_000
        except OSError:
            raise RuntimeError(f"Missing spaCy model. Run: python -m spacy download {spacy_model}")
        
        # Entity and relation mappings (SAT-style)
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        
        # Knowledge graph storage
        self.triples: List[KGTriple] = []
        self.kg = nx.DiGraph()
        
        # Pre-defined relation patterns for extraction
        self.relation_patterns = {
            "treats": ["treat", "cure", "heal", "remedy"],
            "causes": ["cause", "lead to", "result in", "trigger"],
            "prevents": ["prevent", "avoid", "reduce risk"],
            "symptoms_of": ["symptom", "sign", "indicate"],
            "part_of": ["part of", "component", "include"],
            "affects": ["affect", "impact", "influence"],
            "associated_with": ["associate", "relate", "connect", "link"],
            "type_of": ["type of", "kind of", "form of", "is a"],
        }
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text"""
        return text.strip().lower()
    
    def _get_or_create_entity_id(self, entity: str) -> int:
        """Get entity ID, creating new one if needed"""
        entity = self._normalize_entity(entity)
        if entity not in self.entity2id:
            idx = len(self.entity2id)
            self.entity2id[entity] = idx
            self.id2entity[idx] = entity
        return self.entity2id[entity]
    
    def _get_or_create_relation_id(self, relation: str) -> int:
        """Get relation ID, creating new one if needed"""
        relation = relation.lower().strip()
        if relation not in self.relation2id:
            idx = len(self.relation2id)
            self.relation2id[relation] = idx
            self.id2relation[idx] = relation
        return self.relation2id[relation]
    
    def extract_entities(self, doc) -> Set[str]:
        """Extract entities from spaCy doc"""
        entities = set()
        
        # Named entities
        for ent in doc.ents:
            entities.add(self._normalize_entity(ent.text))
        
        # Noun chunks (filter short ones)
        for chunk in doc.noun_chunks:
            text = self._normalize_entity(chunk.text)
            if len(text.split()) >= 2:
                entities.add(text)
        
        return entities
    
    def extract_relations_from_sentence(self, sent) -> List[Tuple[str, str, str, float]]:
        """
        Extract (head, relation, tail, confidence) from a sentence using dependency parsing.
        """
        relations = []
        
        for token in sent:
            # Look for subject-verb-object patterns
            if "subj" in token.dep_:
                subj = token.text
                verb = token.head
                
                for child in verb.children:
                    if "obj" in child.dep_:
                        obj = child.text
                        
                        # Get the full verb phrase as relation
                        rel = verb.lemma_
                        
                        # Map to canonical relation if possible
                        canonical_rel = self._map_to_canonical_relation(rel)
                        
                        relations.append((
                            self._normalize_entity(subj),
                            canonical_rel,
                            self._normalize_entity(obj),
                            0.8  # confidence for dep-parse relations
                        ))
            
            # Look for prepositional phrases
            if token.dep_ == "prep":
                head = token.head.text
                for child in token.children:
                    if child.dep_ == "pobj":
                        rel = f"{token.text}"
                        relations.append((
                            self._normalize_entity(head),
                            rel,
                            self._normalize_entity(child.text),
                            0.6
                        ))
        
        return relations
    
    def _map_to_canonical_relation(self, verb: str) -> str:
        """Map verb to canonical relation type"""
        verb_lower = verb.lower()
        for canonical, patterns in self.relation_patterns.items():
            if any(p in verb_lower for p in patterns):
                return canonical
        return verb_lower
    
    def build_kg_from_chunks(
        self,
        chunks: List[str],
        add_cooccurrence: bool = True,
        cooccurrence_relation: str = "co_occurs_with"
    ):
        """
        Build knowledge graph from text chunks.
        """
        logger.info(f"Building KG from {len(chunks)} chunks...")
        
        chunk_entities = []
        
        for chunk_idx, chunk in enumerate(chunks):
            doc = self.nlp(chunk)
            
            # Extract entities
            entities = self.extract_entities(doc)
            chunk_entities.append(entities)
            
            # Add entities to KG
            for ent in entities:
                ent_id = self._get_or_create_entity_id(ent)
                if ent not in self.kg:
                    self.kg.add_node(ent, entity_id=ent_id)
            
            # Extract relations from each sentence
            for sent in doc.sents:
                sent_entities = set()
                for ent in sent.ents:
                    sent_entities.add(self._normalize_entity(ent.text))
                
                # Dependency-based relations
                relations = self.extract_relations_from_sentence(sent)
                for head, rel, tail, conf in relations:
                    if head in entities and tail in entities and head != tail:
                        self._add_triple(head, rel, tail, conf, chunk_idx)
                
                # Co-occurrence relations (fallback)
                if add_cooccurrence:
                    sent_ents_list = list(sent_entities & entities)
                    for i, e1 in enumerate(sent_ents_list):
                        for e2 in sent_ents_list[i+1:]:
                            if not self.kg.has_edge(e1, e2) and not self.kg.has_edge(e2, e1):
                                self._add_triple(e1, cooccurrence_relation, e2, 0.3, chunk_idx)
        
        logger.info(f"KG built: {len(self.entity2id)} entities, {len(self.relation2id)} relations, {len(self.triples)} triples")
        
        return chunk_entities
    
    def _add_triple(self, head: str, relation: str, tail: str, confidence: float, chunk_idx: int):
        """Add a triple to the KG"""
        rel_id = self._get_or_create_relation_id(relation)
        
        triple = KGTriple(head, relation, tail, confidence, chunk_idx)
        self.triples.append(triple)
        
        self.kg.add_edge(head, tail, relation=relation, confidence=confidence, relation_id=rel_id)
    
    def get_pytorch_geometric_data(self) -> Dict:
        """
        Convert KG to PyTorch Geometric format (like SAT's construct_graph).
        """
        if not self.triples:
            return None
        
        src_list = []
        dst_list = []
        rel_list = []
        
        for triple in self.triples:
            h_id = self.entity2id.get(triple.head)
            t_id = self.entity2id.get(triple.tail)
            r_id = self.relation2id.get(triple.relation)
            
            if h_id is not None and t_id is not None and r_id is not None:
                # Forward edge
                src_list.append(h_id)
                dst_list.append(t_id)
                rel_list.append(r_id)
                
                # Inverse edge (like SAT)
                src_list.append(t_id)
                dst_list.append(h_id)
                rel_list.append(r_id + len(self.relation2id))
        
        src = torch.LongTensor(src_list)
        dst = torch.LongTensor(dst_list)
        edge_index = torch.stack([src, dst], dim=0)
        edge_type = torch.LongTensor(rel_list)
        entity_ids = torch.arange(len(self.entity2id))
        
        return {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "entity_ids": entity_ids,
            "num_entities": len(self.entity2id),
            "num_relations": len(self.relation2id)
        }
    
    def get_entity_texts(self) -> Dict[int, str]:
        """Get id2text mapping (like SAT's get_id2text)"""
        return self.id2entity.copy()
    
    def save_kg_data(self, output_dir: str):
        """Save KG data in SAT-compatible format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save entity mapping (mid2id.txt style)
        with open(os.path.join(output_dir, "entity2id.txt"), "w") as f:
            for ent, idx in self.entity2id.items():
                f.write(f"{ent}\t{idx}\n")
        
        # Save relation mapping (rel2id.txt style)
        with open(os.path.join(output_dir, "relation2id.txt"), "w") as f:
            for rel, idx in self.relation2id.items():
                f.write(f"{rel}\t{idx}\n")
        
        # Save id2text mapping
        with open(os.path.join(output_dir, "id2text.txt"), "w") as f:
            for idx, ent in self.id2entity.items():
                f.write(f"{idx}\t{ent}\n")
        
        # Save triples
        with open(os.path.join(output_dir, "triples.txt"), "w") as f:
            for t in self.triples:
                h_id = self.entity2id[t.head]
                r_id = self.relation2id[t.relation]
                t_id = self.entity2id[t.tail]
                f.write(f"{h_id}\t{r_id}\t{t_id}\t{t.confidence}\n")
        
        logger.info(f"KG data saved to {output_dir}")


# ============================================================================
# PART 4: Enhanced GraphRAG System
# ============================================================================

class EnhancedGraphRAG:
    """
    Enhanced GraphRAG with structure-aware components from SAT.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_graph_transformer: bool = True,
        graph_transformer_dim: int = 128,
        graph_transformer_layers: int = 3,
        working_dir: Optional[str] = None
    ):
        self.embedding_model_name = embedding_model_name
        self.use_graph_transformer = use_graph_transformer
        self.graph_transformer_dim = graph_transformer_dim
        self.graph_transformer_layers = graph_transformer_layers
        
        self.working_dir = working_dir or f"./enhanced_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Components
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.kg_builder = EnhancedKGBuilder()
        
        # Data storage
        self.chunks: List[str] = []
        self.chunk_entities: List[Set[str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        
        # Graph Transformer (initialized after KG is built)
        self.graph_transformer: Optional[GraphTransformer] = None
        self.node_embeddings: Optional[torch.Tensor] = None
        
        # Text-Graph Aligner (optional)
        self.aligner: Optional[TextGraphAligner] = None
    
    def add_documents(self, chunks: List[str]):
        """Add document chunks to the system"""
        self.chunks = chunks
        logger.info(f"Added {len(chunks)} chunks")
    
    def build_kg(self, add_cooccurrence: bool = True):
        """Build knowledge graph from chunks"""
        self.chunk_entities = self.kg_builder.build_kg_from_chunks(
            self.chunks, 
            add_cooccurrence=add_cooccurrence
        )
        
        # Save KG data
        self.kg_builder.save_kg_data(os.path.join(self.working_dir, "kg_data"))
    
    def build_embeddings(self, normalize: bool = True, batch_size: int = 32):
        """Build chunk embeddings and FAISS index"""
        logger.info("Computing chunk embeddings...")
        
        emb_list = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_emb = self.embedding_model.encode(batch, normalize_embeddings=normalize)
            emb_list.append(batch_emb)
        
        self.embeddings = np.vstack(emb_list).astype("float32")
        
        # Build FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)
        
        logger.info(f"FAISS index built with {self.embeddings.shape[0]} vectors, dim={d}")
    
    def build_graph_transformer(self, device: str = "cpu"):
        """Initialize and compute Graph Transformer embeddings"""
        if not self.use_graph_transformer:
            logger.info("Graph Transformer disabled, skipping")
            return
        
        kg_data = self.kg_builder.get_pytorch_geometric_data()
        if kg_data is None:
            logger.warning("No KG data available, skipping Graph Transformer")
            return
        
        logger.info("Building Graph Transformer embeddings...")
        
        self.graph_transformer = GraphTransformer(
            num_entities=kg_data["num_entities"],
            num_relations=kg_data["num_relations"],
            input_dim=self.graph_transformer_dim,
            hidden_dim=self.graph_transformer_dim,
            output_dim=self.graph_transformer_dim,
            n_layers=self.graph_transformer_layers,
            n_heads=8
        ).to(device)
        
        # Compute node embeddings (inference mode)
        self.graph_transformer.eval()
        with torch.no_grad():
            self.node_embeddings = self.graph_transformer(
                kg_data["entity_ids"].to(device),
                kg_data["edge_index"].to(device)
            ).cpu()
        
        # Save embeddings
        torch.save(self.node_embeddings, os.path.join(self.working_dir, "node_embeddings.pt"))
        logger.info(f"Graph Transformer embeddings computed: {self.node_embeddings.shape}")
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Semantic search using FAISS"""
        q_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        return [(int(i), float(D[0][idx])) for idx, i in enumerate(I[0])]
    
    def _graph_search(self, query: str) -> np.ndarray:
        """Graph-based scoring using entity overlap"""
        try:
            nlp = self.kg_builder.nlp
            doc = nlp(query)
            q_entities = set(self.kg_builder._normalize_entity(ent.text) for ent in doc.ents)
        except:
            return np.zeros(len(self.chunks))
        
        if not q_entities:
            return np.zeros(len(self.chunks))
        
        scores = np.array([
            len(self.chunk_entities[i].intersection(q_entities)) 
            for i in range(len(self.chunks))
        ], dtype=float)
        
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-12)
        
        return scores
    
    def _get_kg_facts(self, query: str, max_facts: int = 10) -> List[str]:
        """Get relevant KG facts for query entities"""
        try:
            nlp = self.kg_builder.nlp
            doc = nlp(query)
            q_entities = set(self.kg_builder._normalize_entity(ent.text) for ent in doc.ents)
        except:
            return []
        
        facts = []
        kg = self.kg_builder.kg
        
        for qe in q_entities:
            if qe in kg:
                # Outgoing edges
                for nb in list(kg.successors(qe))[:3]:
                    rel = kg.get_edge_data(qe, nb).get("relation", "related_to")
                    facts.append(f"{qe} {rel} {nb}")
                
                # Incoming edges
                for nb in list(kg.predecessors(qe))[:3]:
                    rel = kg.get_edge_data(nb, qe).get("relation", "related_to")
                    facts.append(f"{nb} {rel} {qe}")
        
        return facts[:max_facts]
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        include_kg_facts: bool = True
    ) -> Dict:
        """
        Query the enhanced GraphRAG system.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            alpha: Weight for semantic search (1-alpha for graph search)
            include_kg_facts: Whether to include KG facts in context
        
        Returns:
            Dict with query results and context
        """
        # Semantic search
        sem_results = self._semantic_search(query, top_k=top_k * 2)
        
        # Graph-based scores
        graph_scores = self._graph_search(query)
        
        # Combine scores
        combined = []
        for idx, sem_score in sem_results:
            gscore = graph_scores[idx] if idx < len(graph_scores) else 0.0
            final_score = alpha * sem_score + (1 - alpha) * gscore
            combined.append((idx, final_score, sem_score, gscore))
        
        # Sort and select top-k
        combined.sort(key=lambda x: x[1], reverse=True)
        top_results = combined[:top_k]
        
        # Get chunks
        retrieved_chunks = [self.chunks[idx] for idx, _, _, _ in top_results]
        
        # Get KG facts
        kg_facts = self._get_kg_facts(query) if include_kg_facts else []
        
        # Build context
        context_parts = retrieved_chunks + [f"[KG Fact] {f}" for f in kg_facts]
        context = "\n\n".join(context_parts)
        
        return {
            "query": query,
            "context": context,
            "chunks": retrieved_chunks,
            "kg_facts": kg_facts,
            "retrieval_scores": [
                {"idx": idx, "combined": comb, "semantic": sem, "graph": gscore}
                for idx, comb, sem, gscore in top_results
            ]
        }
    
    def save(self):
        """Save all data to working directory"""
        # Save chunks
        with open(os.path.join(self.working_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(os.path.join(self.working_dir, "embeddings.npy"), self.embeddings)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.working_dir, "faiss.index"))
        
        logger.info(f"System saved to {self.working_dir}")
    
    def load(self, working_dir: str):
        """Load saved data"""
        self.working_dir = working_dir
        
        # Load chunks
        with open(os.path.join(working_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        # Load embeddings
        emb_path = os.path.join(working_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
        
        # Load FAISS index
        idx_path = os.path.join(working_dir, "faiss.index")
        if os.path.exists(idx_path):
            self.index = faiss.read_index(idx_path)
        
        # Load node embeddings
        node_emb_path = os.path.join(working_dir, "node_embeddings.pt")
        if os.path.exists(node_emb_path):
            self.node_embeddings = torch.load(node_emb_path)
        
        logger.info(f"System loaded from {working_dir}")


# ============================================================================
# PART 5: Training Module for Text-Graph Alignment (optional)
# ============================================================================

class AlignmentTrainer:
    """
    Train the Text-Graph Aligner using contrastive learning.
    Based on SAT's training approach.
    """
    
    def __init__(
        self,
        aligner: TextGraphAligner,
        kg_builder: EnhancedKGBuilder,
        device: str = "cpu",
        lr: float = 2e-5
    ):
        self.aligner = aligner.to(device)
        self.kg_builder = kg_builder
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            [p for p in aligner.parameters() if p.requires_grad],
            lr=lr
        )
    
    def train_step(
        self,
        entity_ids: torch.Tensor,
        edge_index: torch.Tensor,
        batch_entity_ids: torch.Tensor,
        batch_texts: List[str]
    ) -> float:
        """Single training step"""
        self.aligner.train()
        self.optimizer.zero_grad()
        
        entity_ids = entity_ids.to(self.device)
        edge_index = edge_index.to(self.device)
        batch_entity_ids = batch_entity_ids.to(self.device)
        
        graph_feats, text_feats = self.aligner(
            entity_ids, edge_index, batch_entity_ids, batch_texts
        )
        
        labels = torch.arange(len(batch_texts)).to(self.device)
        loss = self.aligner.contrastive_loss(graph_feats, text_feats, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Build enhanced GraphRAG
    rag = EnhancedGraphRAG(
        embedding_model_name="all-MiniLM-L6-v2",
        use_graph_transformer=True,
        graph_transformer_dim=128,
        graph_transformer_layers=2
    )
    
    # Sample medical text chunks
    sample_chunks = [
        "Basal cell carcinoma (BCC) is the most common type of skin cancer. It typically appears on sun-exposed areas of the skin.",
        "Treatment options for BCC include surgical excision, Mohs surgery, and radiation therapy. Early detection improves outcomes.",
        "Fair skin and excessive sun exposure are major risk factors for developing skin cancer, including BCC and melanoma.",
        "Melanoma is a more aggressive form of skin cancer that can metastasize to other organs if not caught early.",
        "Sunscreen use and protective clothing can help prevent skin cancer by reducing UV radiation exposure."
    ]
    
    # Build the system
    rag.add_documents(sample_chunks)
    rag.build_kg()
    rag.build_embeddings()
    rag.build_graph_transformer()
    
    # Query
    result = rag.query("What are the treatments for skin cancer?", top_k=3)
    
    print("=" * 50)
    print("Query:", result["query"])
    print("=" * 50)
    print("Context:")
    print(result["context"])
    print("=" * 50)
    print("KG Facts:", result["kg_facts"])
    print("=" * 50)
    print("Retrieval Scores:")
    for score in result["retrieval_scores"]:
        print(f"  Chunk {score['idx']}: combined={score['combined']:.3f}, semantic={score['semantic']:.3f}, graph={score['graph']:.3f}")
