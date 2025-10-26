"""Fine-tuning orchestration stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FinetuneConfig:
    sentiment_lr: float = 2e-5
    sentiment_batch_size: int = 32
    extraction_lr: float = 3e-4
    extraction_batch_size: int = 8
    wandb_project: Optional[str] = None


def finetune_sentiment_model(train_df, val_df, config: FinetuneConfig):
    if train_df.empty:
        raise ValueError("No training data available")
    # Placeholder for actual fine-tuning logic
    return "sentiment_model_checkpoint"


def finetune_extraction_model(datasets, config: FinetuneConfig):
    train_df = datasets.get("train")
    if train_df is None or train_df.empty:
        raise ValueError("No extraction training data")
    # Placeholder for LoRA fine-tuning logic
    return "extraction_model_checkpoint"
