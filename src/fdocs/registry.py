"""Ingestion registry for tracking processed documents."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel


class RegistryEntry(BaseModel):
    """Single registry entry for a document."""
    company: str
    source_path: str
    sha256: str
    mime_type: str
    size_bytes: int
    parsed_ok: bool
    num_chunks: int
    created_at: datetime
    updated_at: datetime
    error_msg: Optional[str] = None


class IngestionRegistry:
    """Manages ingestion registry for idempotent processing."""
    
    def __init__(self, registry_path: Path):
        """Initialize registry.
        
        Args:
            registry_path: Path to registry parquet file
        """
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None
        self._load()
    
    def _load(self):
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            self._df = pd.read_parquet(self.registry_path)
        else:
            self._df = pd.DataFrame(columns=[
                "company", "source_path", "sha256", "mime_type", "size_bytes",
                "parsed_ok", "num_chunks", "created_at", "updated_at", "error_msg"
            ])
    
    def is_processed(self, company: str, source_path: str, sha256: str) -> bool:
        """Check if document already processed.
        
        Args:
            company: Company name
            source_path: Source file path
            sha256: Content hash
            
        Returns:
            True if already processed successfully
        """
        if self._df is None or len(self._df) == 0:
            return False
        
        mask = (
            (self._df["company"] == company) &
            (self._df["source_path"] == source_path) &
            (self._df["sha256"] == sha256) &
            (self._df["parsed_ok"] == True)
        )
        return mask.any()
    
    def add_entry(self, entry: RegistryEntry):
        """Add or update registry entry.
        
        Args:
            entry: Registry entry to add
        """
        # Remove existing entry for same company/path
        if self._df is not None and len(self._df) > 0:
            mask = (
                (self._df["company"] == entry.company) &
                (self._df["source_path"] == entry.source_path)
            )
            self._df = self._df[~mask]
        
        # Add new entry
        new_row = pd.DataFrame([entry.model_dump()])
        self._df = pd.concat([self._df, new_row], ignore_index=True)
    
    def save(self):
        """Persist registry to disk."""
        if self._df is not None:
            self._df.to_parquet(self.registry_path, index=False)
    
    def get_stats(self) -> dict:
        """Get registry statistics.
        
        Returns:
            Dictionary with counts
        """
        if self._df is None or len(self._df) == 0:
            return {"total": 0, "success": 0, "failed": 0}
        
        return {
            "total": len(self._df),
            "success": int(self._df["parsed_ok"].sum()),
            "failed": int((~self._df["parsed_ok"]).sum()),
        }


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def discover_documents(docs_root: Path, company: str) -> List[Path]:
    """Discover documents for a company.
    
    Args:
        docs_root: Root docs directory
        company: Company name
        
    Returns:
        List of document paths
    """
    company_dir = docs_root / company
    if not company_dir.exists():
        return []
    
    # Find all docs* directories
    doc_files = []
    for docs_dir in company_dir.glob("docs*"):
        if docs_dir.is_dir():
            # Collect supported file types
            for ext in ["*.pdf", "*.docx", "*.pptx", "*.html", "*.txt", "*.md"]:
                doc_files.extend(docs_dir.rglob(ext))
    
    return sorted(doc_files)

