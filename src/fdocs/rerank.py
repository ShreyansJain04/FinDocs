"""Cross-encoder re-ranking."""

from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder
from tqdm import tqdm


class Reranker:
    """Cross-encoder re-ranker."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device (cuda, cpu)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        self.model = CrossEncoder(model_name, device=device, max_length=512)
    
    def rerank(
        self,
        query: str,
        texts: List[str],
        chunk_ids: List[str],
        top_n: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """Re-rank candidate texts.
        
        Args:
            query: Query text
            texts: Candidate texts
            chunk_ids: Corresponding chunk IDs
            top_n: Number of top results to return (None = all)
            
        Returns:
            (reranked_chunk_ids, reranked_scores)
        """
        if not texts:
            return [], []
        
        # Create query-document pairs
        pairs = [(query, text) for text in texts]
        
        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 100
        )
        
        # Sort by score (descending)
        sorted_indices = scores.argsort()[::-1]
        
        # Apply top_n filter
        if top_n is not None:
            sorted_indices = sorted_indices[:top_n]
        
        reranked_ids = [chunk_ids[idx] for idx in sorted_indices]
        reranked_scores = [float(scores[idx]) for idx in sorted_indices]
        
        return reranked_ids, reranked_scores

