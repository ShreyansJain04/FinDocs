"""Prepare training datasets from feedback."""

from __future__ import annotations

import json
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..feedback import EXTRACTION_FILE, SENTIMENT_FILE


def load_sentiment_feedback(min_confidence: float = 0.0) -> pd.DataFrame:
    if not SENTIMENT_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SENTIMENT_FILE)
    return df[df["original_confidence"] >= min_confidence]


def load_extraction_feedback() -> pd.DataFrame:
    if not EXTRACTION_FILE.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in EXTRACTION_FILE.read_text().splitlines() if line]
    return pd.DataFrame(records)


def build_sentiment_dataset(min_confidence: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_sentiment_feedback(min_confidence)
    if df.empty:
        return df, df, df
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["corrected_label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["corrected_label"])
    return train, val, test


def build_extraction_dataset() -> Dict[str, pd.DataFrame]:
    df = load_extraction_feedback()
    if df.empty:
        return {"train": df, "val": df, "test": df}
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return {"train": train, "val": val, "test": test}
