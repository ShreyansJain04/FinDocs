"""Review queue management for active learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


QUEUE_PATH = Path("artifacts/review_queue.parquet")


@dataclass
class QueueItem:
    document_id: str
    chunk_id: Optional[str]
    field_name: Optional[str]
    uncertainty_score: float
    priority: float
    status: str = "pending"
    metadata: Optional[dict] = None


class ReviewQueue:
    def __init__(self, path: Path = QUEUE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=[
            "document_id",
            "chunk_id",
            "field_name",
            "uncertainty_score",
            "priority",
            "status",
            "metadata",
        ])

    def _save(self):
        self._df.to_parquet(self.path, index=False)

    def add_items(self, items: Iterable[QueueItem]):
        rows = []
        for item in items:
            rows.append({
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "field_name": item.field_name,
                "uncertainty_score": item.uncertainty_score,
                "priority": item.priority,
                "status": item.status,
                "metadata": item.metadata or {},
            })
        if rows:
            self._df = pd.concat([self._df, pd.DataFrame(rows)], ignore_index=True)
            self._save()

    def get_next_batch(self, limit: int = 20, group_by_field: bool = True) -> pd.DataFrame:
        pending = self._df[self._df["status"] == "pending"].copy()
        if pending.empty:
            return pending
        pending.sort_values(by=["priority", "uncertainty_score"], ascending=False, inplace=True)
        if group_by_field:
            pending["group_key"] = pending["field_name"].fillna("general")
            groups = pending.groupby("group_key")
            selected_indices = []
            for _, group in groups:
                selected_indices.extend(group.index.tolist())
                if len(selected_indices) >= limit:
                    break
            selected = pending.loc[selected_indices].head(limit)
            return selected.drop(columns=["group_key"], errors="ignore")
        return pending.head(limit)

    def mark_reviewed(self, indices: Iterable[int], status: str = "reviewed"):
        for idx in indices:
            if idx in self._df.index:
                self._df.at[idx, "status"] = status
        self._save()

    def stats(self) -> dict:
        total = len(self._df)
        pending = int((self._df["status"] == "pending").sum())
        return {
            "total": total,
            "pending": pending,
            "average_uncertainty": float(self._df["uncertainty_score"].mean()) if total else 0.0,
        }
