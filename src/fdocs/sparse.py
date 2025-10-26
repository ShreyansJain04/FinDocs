"""BM25 sparse retrieval."""

import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 sparse retrieval index."""
    
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        """Initialize BM25 index.
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.index: BM25Okapi = None
        self.chunk_ids: List[str] = []
        self.corpus: List[List[str]] = []
    
    def build(self, texts: List[str], chunk_ids: List[str]):
        """Build BM25 index from corpus.
        
        Args:
            texts: List of text documents
            chunk_ids: Corresponding chunk IDs
        """
        if not texts:
            return
        
        # Tokenize corpus (simple whitespace tokenization)
        self.corpus = [text.lower().split() for text in texts]
        self.chunk_ids = chunk_ids
        
        # Build BM25 index
        self.index = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
    
    def search(self, query: str, k: int = 100) -> Tuple[List[str], List[float]]:
        """Search BM25 index.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            (chunk_ids, scores) tuples
        """
        if self.index is None or not self.chunk_ids:
            return [], []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get scores
        scores = self.index.get_scores(query_tokens)
        
        # Get top-k
        k = min(k, len(scores))
        top_indices = scores.argsort()[-k:][::-1]
        
        top_chunk_ids = [self.chunk_ids[idx] for idx in top_indices]
        top_scores = [float(scores[idx]) for idx in top_indices]
        
        return top_chunk_ids, top_scores
    
    def save(self, index_path: Path):
        """Save BM25 index to disk.
        
        Args:
            index_path: Directory to save index
        """
        index_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "index": self.index,
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus,
            "k1": self.k1,
            "b": self.b
        }
        
        with open(index_path / "bm25.pkl", "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, index_path: Path) -> "BM25Index":
        """Load BM25 index from disk.
        
        Args:
            index_path: Directory containing index
            
        Returns:
            Loaded BM25Index
        """
        with open(index_path / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        
        instance = cls(k1=data["k1"], b=data["b"])
        instance.index = data["index"]
        instance.chunk_ids = data["chunk_ids"]
        instance.corpus = data["corpus"]
        
        return instance
    
    def get_size(self) -> int:
        """Get number of documents in index.
        
        Returns:
            Number of documents
        """
        return len(self.chunk_ids)

