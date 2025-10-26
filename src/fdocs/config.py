"""Configuration management for FinDocs."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """File system paths."""
    docs_root: str = "docs"
    artifacts_root: str = "artifacts"
    registry_dir: str = "artifacts/registry"
    chunks_dir: str = "artifacts/chunks"
    tables_dir: str = "artifacts/tables"
    index_dir: str = "artifacts/index"
    sparse_dir: str = "artifacts/sparse"
    models_cache: str = "artifacts/models"


class DeviceConfig(BaseModel):
    """GPU/device configuration."""
    gpu_id: int = 0
    use_cuda: bool = True


class ParsingConfig(BaseModel):
    """Document parsing configuration."""
    pdf_extractor: str = "pymupdf+pdfplumber"


class TablesConfig(BaseModel):
    """Table extraction configuration."""
    enabled: bool = True
    extractor: str = "pdfplumber"
    serialize_format: str = "markdown"


class ChunkingConfig(BaseModel):
    """Semantic chunking configuration."""
    target_tokens: int = 300
    overlap_tokens: int = 60
    respect_structure: bool = True
    similarity_threshold: float = 0.7


class SentimentConfig(BaseModel):
    """FinBERT sentiment analysis configuration."""
    model: str = "yiyanghkust/finbert-tone"
    batch_size: int = 16
    max_length: int = 512


class EmbeddingsConfig(BaseModel):
    """Dense embeddings configuration."""
    model: str = "intfloat/e5-large-v2"
    batch_size: int = 32
    normalize: bool = True
    dimension: int = 1024


class BM25Config(BaseModel):
    """BM25 sparse retrieval configuration."""
    enabled: bool = True
    k1: float = 0.9
    b: float = 0.4
    top_k: int = 100


class RerankerConfig(BaseModel):
    """Cross-encoder re-ranking configuration."""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    top_n: int = 200


class HybridConfig(BaseModel):
    """Hybrid retrieval configuration."""
    enabled: bool = True
    fusion_method: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


class FAISSConfig(BaseModel):
    """FAISS index configuration."""
    index_type: str = "IndexFlatIP"
    metric: str = "inner_product"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    structured: bool = True


class OllamaConfig(BaseModel):
    """Settings for Ollama-backed generation."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3"
    temperature: float = 0.0
    max_new_tokens: int = 512
    timeout: float = 120.0


class Config(BaseModel):
    """Main configuration."""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    tables: TablesConfig = Field(default_factory=TablesConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    extraction: Optional[Dict[str, Any]] = None
    active_learning: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML. Defaults to config/default.yaml
        
    Returns:
        Config instance
    """
    if config_path is None:
        config_path = "config/default.yaml"
    
    config_file = Path(config_path)
    if not config_file.exists():
        # Return default config
        return Config()
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def get_device(config: Config) -> str:
    """Get torch device string.
    
    Args:
        config: Configuration
        
    Returns:
        Device string (cuda:0, cpu, etc.)
    """
    if config.device.use_cuda:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{config.device.gpu_id}"
    return "cpu"

