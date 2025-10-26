"""FAISS index for dense vector retrieval."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd


class FAISSIndex:
    """FAISS index manager for dense retrieval."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "IndexFlatIP",
        metric: str = "inner_product"
    ):
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type
            metric: Distance metric
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.chunk_ids: List[str] = []
    
    def add(self, embeddings: np.ndarray, chunk_ids: List[str]):
        """Add vectors to index.
        
        Args:
            embeddings: Embeddings array (N x D)
            chunk_ids: Chunk IDs corresponding to embeddings
        """
        if len(embeddings) == 0:
            return
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # For cosine similarity with IndexFlatIP, normalize vectors
        if self.index_type == "IndexFlatIP" and self.metric == "inner_product":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Add to index
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search index for similar vectors.
        
        Args:
            query_embedding: Query vector (D,)
            k: Number of results
            
        Returns:
            (chunk_ids, scores) tuples
        """
        if self.index.ntotal == 0:
            return [], []
        
        # Ensure shape and type
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        if self.index_type == "IndexFlatIP" and self.metric == "inner_product":
            norm = np.linalg.norm(query)
            if norm > 1e-12:
                query = query / norm
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # Extract results
        chunk_ids = [self.chunk_ids[idx] for idx in indices[0]]
        scores = distances[0].tolist()
        
        return chunk_ids, scores
    
    def save(self, index_path: Path, metadata: Optional[dict] = None):
        """Save index to disk.
        
        Args:
            index_path: Directory to save index
            metadata: Additional metadata to save
        """
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_file = index_path / "index.faiss"
        faiss.write_index(self.index, str(faiss_file))
        
        # Save chunk ID mapping
        mapping_file = index_path / "chunk_ids.json"
        with open(mapping_file, "w") as f:
            json.dump(self.chunk_ids, f)
        
        # Save metadata
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "num_vectors": self.index.ntotal,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        meta_file = index_path / "index_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, index_path: Path) -> "FAISSIndex":
        """Load index from disk.
        
        Args:
            index_path: Directory containing index
            
        Returns:
            Loaded FAISSIndex
        """
        # Load metadata
        meta_file = index_path / "index_meta.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        
        # Create instance
        instance = cls(
            dimension=meta["dimension"],
            index_type=meta["index_type"],
            metric=meta["metric"]
        )
        
        # Load FAISS index
        faiss_file = index_path / "index.faiss"
        instance.index = faiss.read_index(str(faiss_file))
        
        # Load chunk IDs
        mapping_file = index_path / "chunk_ids.json"
        with open(mapping_file, "r") as f:
            instance.chunk_ids = json.load(f)
        
        return instance
    
    def get_size(self) -> int:
        """Get number of vectors in index.
        
        Returns:
            Number of vectors
        """
        return self.index.ntotal


def save_chunks_to_parquet(chunks: List, output_path: Path):
    """Save chunks to Parquet file.
    
    Args:
        chunks: List of Chunk objects
        output_path: Output path
    """
    if not chunks:
        return
    
    # Convert to dicts
    rows = []
    for chunk in chunks:
        if hasattr(chunk, 'model_dump'):
            row = chunk.model_dump()
        elif hasattr(chunk, 'model_dump_json'):
            row = json.loads(chunk.model_dump_json())
        elif hasattr(chunk, '__dict__'):
            row = chunk.__dict__.copy()
        else:
            row = {
                "chunk_id": chunk.chunk_id,
                "company": chunk.company,
                "source_path": chunk.source_path,
                "chunk_idx": chunk.chunk_idx,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "is_table": chunk.is_table,
                "table_id": chunk.table_id,
            }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Append or create
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_parquet(output_path, index=False)

