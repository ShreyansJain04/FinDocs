"""High-fidelity table extraction from documents."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pdfplumber
import pymupdf as fitz
from docx import Document as DocxDocument


@dataclass
class TableCell:
    """Single table cell."""
    row_idx: int
    col_idx: int
    text: str
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1)


@dataclass
class ExtractedTable:
    """Extracted table with structure."""
    table_id: str
    page: Optional[int]
    cells: List[TableCell]
    caption: Optional[str] = None
    bbox: Optional[tuple] = None
    num_rows: int = 0
    num_cols: int = 0


class TableExtractor:
    """Extract tables from documents with high fidelity."""
    
    def __init__(self, extractor: str = "pdfplumber", serialize_format: str = "markdown"):
        """Initialize table extractor.
        
        Args:
            extractor: Extraction method (pdfplumber, unstructured)
            serialize_format: How to serialize tables (markdown, csv)
        """
        self.extractor = extractor
        self.serialize_format = serialize_format
    
    def extract_from_pdf(self, file_path: Path) -> List[ExtractedTable]:
        """Extract tables from PDF.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            List of extracted tables
        """
        if self.extractor == "pdfplumber":
            return self._extract_pdfplumber(file_path)
        else:
            return []
    
    def _extract_pdfplumber(self, file_path: Path) -> List[ExtractedTable]:
        """Extract tables using pdfplumber.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if not table_data:
                            continue
                        
                        cells = []
                        num_rows = len(table_data)
                        num_cols = max(len(row) for row in table_data) if table_data else 0
                        
                        for row_idx, row in enumerate(table_data):
                            for col_idx, cell_text in enumerate(row):
                                cell = TableCell(
                                    row_idx=row_idx,
                                    col_idx=col_idx,
                                    text=str(cell_text or "").strip()
                                )
                                cells.append(cell)
                        
                        table = ExtractedTable(
                            table_id=str(uuid.uuid4()),
                            page=page_num,
                            cells=cells,
                            num_rows=num_rows,
                            num_cols=num_cols
                        )
                        tables.append(table)
        
        except Exception as e:
            # Log but don't fail parsing
            pass
        
        return tables
    
    def extract_from_docx(self, file_path: Path) -> List[ExtractedTable]:
        """Extract tables from DOCX.
        
        Args:
            file_path: Path to DOCX
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            doc = DocxDocument(file_path)
            
            for table_idx, table in enumerate(doc.tables):
                cells = []
                num_rows = len(table.rows)
                num_cols = len(table.columns) if table.rows else 0
                
                for row_idx, row in enumerate(table.rows):
                    for col_idx, cell in enumerate(row.cells):
                        cell_obj = TableCell(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            text=cell.text.strip()
                        )
                        cells.append(cell_obj)
                
                extracted = ExtractedTable(
                    table_id=str(uuid.uuid4()),
                    page=None,
                    cells=cells,
                    num_rows=num_rows,
                    num_cols=num_cols
                )
                tables.append(extracted)
        
        except Exception as e:
            pass
        
        return tables
    
    def serialize_table(self, table: ExtractedTable) -> str:
        """Serialize table to text format.
        
        Args:
            table: Extracted table
            
        Returns:
            Serialized table text
        """
        if self.serialize_format == "markdown":
            return self._serialize_markdown(table)
        elif self.serialize_format == "csv":
            return self._serialize_csv(table)
        else:
            return self._serialize_markdown(table)
    
    def _serialize_markdown(self, table: ExtractedTable) -> str:
        """Serialize table to Markdown format.
        
        Args:
            table: Extracted table
            
        Returns:
            Markdown table
        """
        if not table.cells:
            return ""
        
        # Build grid
        grid = {}
        for cell in table.cells:
            grid[(cell.row_idx, cell.col_idx)] = cell.text
        
        lines = []
        for row_idx in range(table.num_rows):
            row_cells = []
            for col_idx in range(table.num_cols):
                cell_text = grid.get((row_idx, col_idx), "")
                row_cells.append(cell_text)
            
            line = "| " + " | ".join(row_cells) + " |"
            lines.append(line)
            
            # Add separator after header
            if row_idx == 0:
                sep = "| " + " | ".join(["-" * max(3, len(c)) for c in row_cells]) + " |"
                lines.append(sep)
        
        return "\n".join(lines)
    
    def _serialize_csv(self, table: ExtractedTable) -> str:
        """Serialize table to CSV format.
        
        Args:
            table: Extracted table
            
        Returns:
            CSV table
        """
        if not table.cells:
            return ""
        
        # Build grid
        grid = {}
        for cell in table.cells:
            grid[(cell.row_idx, cell.col_idx)] = cell.text
        
        lines = []
        for row_idx in range(table.num_rows):
            row_cells = []
            for col_idx in range(table.num_cols):
                cell_text = grid.get((row_idx, col_idx), "")
                # Escape commas and quotes
                if "," in cell_text or "\"" in cell_text:
                    cell_text = "\"" + cell_text.replace("\"", "\"\"") + "\""
                row_cells.append(cell_text)
            
            line = ",".join(row_cells)
            lines.append(line)
        
        return "\n".join(lines)


def save_tables_to_parquet(tables: List[ExtractedTable], company: str, source_path: str, output_path: Path):
    """Save extracted tables to Parquet.
    
    Args:
        tables: List of extracted tables
        company: Company name
        source_path: Source document path
        output_path: Output parquet path
    """
    if not tables:
        return
    
    rows = []
    for table in tables:
        for cell in table.cells:
            row = {
                "table_id": table.table_id,
                "company": company,
                "source_path": source_path,
                "page": table.page,
                "row_idx": cell.row_idx,
                "col_idx": cell.col_idx,
                "text": cell.text,
                "bbox": str(cell.bbox) if cell.bbox else None,
                "caption": table.caption,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Append or create
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_parquet(output_path, index=False)

