"""
Hybrid search model combining BM25 and dense retrieval.

This module implements the `HybridSearchIndex` class, which combines sparse keyword search (BM25)
with dense semantic search (SentenceTransformers + Annoy) to retrieve relevant code chunks.
It also supports re-ranking using a CrossEncoder.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
from ..core.utils import tokenize, get_torch_device
import logging
from repgen_logging import get_logger, trace

# Suppress logs from transformers and sentence_transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

logger = get_logger(__name__, component="retrieval.hybrid_search")

class HybridSearchIndex:
    """
    Implements hybrid search using BM25 and Approximate Nearest Neighbors (Annoy).    
    """
    def __init__(
        self,
        embedding_model: str,
        reranker_model: str,
        device: Optional[str] = None,
        config: Optional[Any] = None
    ):
        self.device = get_torch_device(device or "cuda")
        self.encoder = SentenceTransformer(embedding_model, device=self.device)
        self.cross_encoder = CrossEncoder(reranker_model, device=self.device)
        self.max_seq_length = 512
        self.bm25 = None
        self.embeddings = None
        self.code_chunks = None
        self.annoy_index = None
        self.config = config
        trace(logger, "Hybrid search initialized", stage="retrieval", action="init_hybrid_search", status="ok", details={"device": self.device, "embedding_model": embedding_model, "reranker_model": reranker_model})

    def build_index(self, code_chunks: List[dict], corpus: List[List[str]]) -> None:
        """
        Build the hybrid search index from code chunks.

        Computes BM25 frequencies and generates dense embeddings for all chunks.
        Builds the Annoy index for fast nearest neighbor search.

        Args:
            code_chunks: List of code snippet dictionaries.
            corpus: Tokenized corpus for BM25.
        """
        self.code_chunks = code_chunks
        self.bm25 = BM25Okapi(corpus)
        
        texts = [chunk['page_content'] for chunk in code_chunks]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        self.embeddings = normalize(self.embeddings.cpu().numpy())
        
        self._build_annoy_index()
        trace(logger, "Hybrid index built", stage="retrieval", action="build_index", status="ok", details={"chunks": len(code_chunks), "embedding_dim": self.embeddings.shape[1] if len(self.embeddings) else 0})

    def _build_annoy_index(self) -> None:
        """
        Build Annoy index for approximate nearest neighbor search.
        
        Uses Angular distance metric.
        """
        dim = self.embeddings.shape[1]
        self.annoy_index = AnnoyIndex(dim, 'angular')
        for i, vec in enumerate(self.embeddings):
            self.annoy_index.add_item(i, vec)
        self.annoy_index.build(n_trees=50)

    def save_index(self, index_dir: Path) -> None:
        """
        Save the index components to disk.

        Saves embeddings (npy), code chunks (json), and the Annoy index (ann).

        Args:
            index_dir: Directory path to save the index.
        """
        index_dir.mkdir(exist_ok=True)
        
        with open(index_dir / "documents.json", 'w') as f:
            json.dump(self.code_chunks, f)
            
        np.save(index_dir / "embeddings.npy", self.embeddings)
        self.annoy_index.save(str(index_dir / "annoy_index.ann"))

    def load_index(self, index_dir: Path) -> None:
        """
        Load the index components from disk.

        Args:
            index_dir: Directory path containing the saved index files.
        """
        with open(index_dir / "documents.json", 'r') as f:
            self.code_chunks = json.load(f)
            
        self.embeddings = np.load(index_dir / "embeddings.npy")
        
        corpus = [tokenize(doc['page_content']) for doc in self.code_chunks]
        self.bm25 = BM25Okapi(corpus)
        
        self.annoy_index = AnnoyIndex(self.embeddings.shape[1], 'angular')
        self.annoy_index.load(str(index_dir / "annoy_index.ann"))

    def semantic_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform semantic search using the Annoy index.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (indices, scores).
        """
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding.flatten(), top_k, include_distances=True
        )
        return np.array(indices), 1 - np.array(distances)

    def search(
        self,
        query: str,
        top_k: int = 200,
        alpha: float = 0.55,
        rerank_top_k: int = 20,
        ann_top_k: int = 200
    ) -> List[dict]:
        """
        Perform hybrid search query.

        Combines BM25 and Semantic search scores using the formula:
        score = (1 - alpha) * bm25_score + alpha * semantic_score

        Args:
            query: Search query string.
            top_k: Number of results to consider from each method (not fully used in logic but kept for interface).
            alpha: Weight for semantic search (0.0 to 1.0).
            rerank_top_k: Number of results to re-rank using CrossEncoder.
            ann_top_k: Number of results to retrieve from Annoy.

        Returns:
            List of unique code chunks sorted by relevance.
        """
        if not query or not isinstance(query, str):
            return []

        if self.config:
            alpha = self.config.ALPHA
            rerank_top_k = self.config.RERANK_TOP_K
        
        # Ablation: No BM25
        if self.config and self.config.AB_NO_BM25:
            alpha = 1.0  # Fully semantic
        
        # Ablation: No ANN
        if self.config and self.config.AB_NO_ANN:
            alpha = 0.0  # Fully sparse (BM25)

        query = query[:100000]  # Truncate very long queries
        
        # BM25 search
        query_tokens = tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)

        # Semantic search
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        query_embedding = normalize(query_embedding.cpu().numpy().reshape(1, -1))
        ann_indices, ann_scores = self.semantic_search(query_embedding, ann_top_k)
        ann_indices = np.array(ann_indices, dtype=int)

        if len(ann_indices) == 0:
            trace(logger, "Semantic search returned no candidates", level=logging.WARNING, stage="retrieval", action="semantic_search", status="empty")
            return []

        # Combine scores
        combined_scores = (1 - alpha) * bm25_scores[ann_indices] + alpha * ann_scores
        combined_indices_sorted = ann_indices[np.argsort(combined_scores)[::-1]]
        top_combined_indices = combined_indices_sorted[:rerank_top_k]

        # Prepare for cross-encoder
        top_chunks = [self.code_chunks[i] for i in top_combined_indices]
        
        if self.config and self.config.AB_NO_RERANKER:
            trace(logger, "Hybrid search completed without reranker", stage="retrieval", action="hybrid_search", status="ok", details={"ann_candidates": len(ann_indices), "returned": len(top_chunks)})
            return top_chunks
        
        rerank_pairs = [
            [query, chunk['page_content'][:100000]]
            for chunk in top_chunks
        ]
        rerank_scores = np.asarray(
            self.cross_encoder.predict(
                rerank_pairs,
                batch_size=16,
                show_progress_bar=False
            )
        ).reshape(-1)

        # Sort by cross-encoder scores
        reranked_indices = np.argsort(rerank_scores)[::-1]
        results = [top_chunks[i] for i in reranked_indices]
        trace(logger, "Hybrid search completed", stage="retrieval", action="hybrid_search", status="ok", details={"ann_candidates": len(ann_indices), "reranked": len(results), "alpha": alpha})
        return results
