"""Dense embeddings for RAG with GPU acceleration."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate dense embeddings for retrieval."""
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True
    ):
        """Initialize embedding generator.
        
        Args:
            model_name: Model name/path
            device: Device (cuda, cpu)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (N x D)
        """
        if not texts:
            return np.array([])
        
        # E5 models benefit from instruction prefix for queries (but not for documents)
        # For document embedding, we use texts as-is
        
        # Encode with batching
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query.
        
        E5 models use "query: " prefix for queries.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding (1 x D)
        """
        # Add query prefix for E5 models
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embedding[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.dimension

