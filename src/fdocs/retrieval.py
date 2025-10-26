"""Hybrid retrieval orchestration with fusion."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .extract import StructureExtractor


def reciprocal_rank_fusion(
    results_list: List[List[str]],
    scores_list: List[List[float]],
    k: int = 60
) -> Tuple[List[str], List[float]]:
    """Combine multiple ranking lists using Reciprocal Rank Fusion.
    
    Args:
        results_list: List of ranked result lists
        scores_list: List of score lists (not used in RRF, kept for API compatibility)
        k: RRF constant
        
    Returns:
        (fused_ids, fused_scores)
    """
    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}
    
    for results in results_list:
        for rank, doc_id in enumerate(results, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_ids = [item[0] for item in sorted_items]
    fused_scores = [item[1] for item in sorted_items]
    
    return fused_ids, fused_scores


def weighted_score_fusion(
    results_list: List[List[str]],
    scores_list: List[List[float]],
    weights: List[float]
) -> Tuple[List[str], List[float]]:
    """Combine multiple ranking lists using weighted score fusion.
    
    Args:
        results_list: List of ranked result lists
        scores_list: List of score lists
        weights: Weight for each list
        
    Returns:
        (fused_ids, fused_scores)
    """
    # Normalize scores to [0, 1] for each list
    normalized_scores_list = []
    for scores in scores_list:
        if not scores or max(scores) == min(scores):
            normalized_scores_list.append(scores)
        else:
            min_score = min(scores)
            max_score = max(scores)
            normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            normalized_scores_list.append(normalized)
    
    # Compute weighted scores
    weighted_scores: Dict[str, float] = {}
    
    for results, norm_scores, weight in zip(results_list, normalized_scores_list, weights):
        for doc_id, score in zip(results, norm_scores):
            if doc_id not in weighted_scores:
                weighted_scores[doc_id] = 0.0
            weighted_scores[doc_id] += weight * score
    
    # Sort by score
    sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_ids = [item[0] for item in sorted_items]
    fused_scores = [item[1] for item in sorted_items]
    
    return fused_ids, fused_scores


class HybridRetriever:
    """Orchestrate hybrid retrieval with dense, sparse, and reranking."""
    
    def __init__(
        self,
        faiss_index,
        bm25_index,
        reranker,
        chunks_df: pd.DataFrame,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rerank_top_n: int = 200,
        extractor: Optional[StructureExtractor] = None
    ):
        """Initialize hybrid retriever.
        
        Args:
            faiss_index: FAISS index
            bm25_index: BM25 index
            reranker: Reranker
            chunks_df: DataFrame with chunk metadata
            fusion_method: Fusion method (rrf, weighted)
            rrf_k: RRF k parameter
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            rerank_top_n: Number of candidates to rerank
        """
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.chunks_df = chunks_df
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rerank_top_n = rerank_top_n
        self.extractor = extractor
    
    def retrieve(
        self,
        query: str,
        query_embedding,
        mode: str = "hybrid",
        k: int = 10,
        rerank: bool = True
    ) -> pd.DataFrame:
        """Retrieve relevant chunks.
        
        Args:
            query: Query text
            query_embedding: Query embedding (for dense retrieval)
            mode: Retrieval mode (dense, sparse, hybrid)
            k: Number of final results
            rerank: Whether to apply reranking
            
        Returns:
            DataFrame with results
        """
        # Retrieve candidates based on mode
        if mode == "dense":
            candidate_ids, candidate_scores = self._retrieve_dense(query_embedding, k=self.rerank_top_n if rerank else k)
        elif mode == "sparse":
            candidate_ids, candidate_scores = self._retrieve_sparse(query, k=self.rerank_top_n if rerank else k)
        elif mode == "hybrid":
            candidate_ids, candidate_scores = self._retrieve_hybrid(query, query_embedding, k=self.rerank_top_n if rerank else k)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if not candidate_ids:
            return pd.DataFrame()
        
        # Rerank if enabled
        if rerank and self.reranker is not None:
            candidate_ids, candidate_scores = self._rerank_candidates(
                query, candidate_ids, candidate_scores, top_n=k
            )
        else:
            # Limit to k
            candidate_ids = candidate_ids[:k]
            candidate_scores = candidate_scores[:k]
        
        # Fetch chunk data
        results_df = self._fetch_chunks(candidate_ids, candidate_scores)
        
        return results_df
    
    def _retrieve_dense(self, query_embedding, k: int) -> Tuple[List[str], List[float]]:
        """Dense retrieval via FAISS."""
        return self.faiss_index.search(query_embedding, k=k)
    
    def _retrieve_sparse(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        """Sparse retrieval via BM25."""
        return self.bm25_index.search(query, k=k)
    
    def _retrieve_hybrid(self, query: str, query_embedding, k: int) -> Tuple[List[str], List[float]]:
        """Hybrid retrieval with fusion."""
        # Get results from both
        dense_ids, dense_scores = self._retrieve_dense(query_embedding, k=k)
        sparse_ids, sparse_scores = self._retrieve_sparse(query, k=k)
        
        # Fuse results
        if self.fusion_method == "rrf":
            fused_ids, fused_scores = reciprocal_rank_fusion(
                [dense_ids, sparse_ids],
                [dense_scores, sparse_scores],
                k=self.rrf_k
            )
        elif self.fusion_method == "weighted":
            fused_ids, fused_scores = weighted_score_fusion(
                [dense_ids, sparse_ids],
                [dense_scores, sparse_scores],
                weights=[self.dense_weight, self.sparse_weight]
            )
        else:
            # Default to dense
            fused_ids, fused_scores = dense_ids, dense_scores
        
        return fused_ids, fused_scores
    
    def _rerank_candidates(
        self,
        query: str,
        candidate_ids: List[str],
        candidate_scores: List[float],
        top_n: int
    ) -> Tuple[List[str], List[float]]:
        """Rerank candidates using cross-encoder."""
        # Fetch texts for candidates
        texts = []
        valid_ids = []
        
        for chunk_id in candidate_ids:
            chunk_row = self.chunks_df[self.chunks_df["chunk_id"] == chunk_id]
            if not chunk_row.empty:
                texts.append(chunk_row.iloc[0]["text"])
                valid_ids.append(chunk_id)
        
        if not texts:
            return [], []
        
        # Rerank
        reranked_ids, reranked_scores = self.reranker.rerank(
            query, texts, valid_ids, top_n=top_n
        )
        
        return reranked_ids, reranked_scores
    
    def _fetch_chunks(self, chunk_ids: List[str], scores: List[float]) -> pd.DataFrame:
        """Fetch chunk data from DataFrame."""
        if not chunk_ids:
            return pd.DataFrame()
        
        # Create results with scores
        results = []
        for chunk_id, score in zip(chunk_ids, scores):
            chunk_row = self.chunks_df[self.chunks_df["chunk_id"] == chunk_id]
            if not chunk_row.empty:
                row = chunk_row.iloc[0].to_dict()
                row["score"] = score
                results.append(row)
        
        return pd.DataFrame(results)

