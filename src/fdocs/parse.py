"""Document parsing with multi-format support."""

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz
from unstructured.partition.auto import partition


@dataclass
class ParsedElement:
    """Single parsed element from a document."""
    text: str
    element_type: str  # NarrativeText, Title, Table, etc.
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class ParsedDocument:
    """Parsed document with extracted content."""
    source_path: str
    mime_type: str
    elements: List[ParsedElement]
    num_pages: Optional[int] = None
    metadata: Optional[dict] = None


class DocumentParser:
    """Multi-format document parser."""
    
    def __init__(self, pdf_extractor: str = "pymupdf+pdfplumber"):
        """Initialize parser.
        
        Args:
            pdf_extractor: PDF extraction method
        """
        self.pdf_extractor = pdf_extractor
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Parsed document
        """
        mime_type = self._detect_mime_type(file_path)
        
        if mime_type == "application/pdf" and "pymupdf" in self.pdf_extractor:
            return self._parse_pdf_pymupdf(file_path)
        if mime_type in {"text/plain", "text/markdown", "text/x-markdown"}:
            return self._parse_text(file_path, mime_type)
        return self._parse_unstructured(file_path, mime_type)
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def _parse_pdf_pymupdf(self, file_path: Path) -> ParsedDocument:
        """Parse PDF using PyMuPDF.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            Parsed document
        """
        elements = []
        
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            
            for page_num, page in enumerate(doc, start=1):
                # Extract text with structure
                text = page.get_text("text")
                
                if text.strip():
                    element = ParsedElement(
                        text=text,
                        element_type="NarrativeText",
                        page_number=page_num,
                        metadata={"source": "pymupdf"}
                    )
                    elements.append(element)
            
            doc.close()
            
            return ParsedDocument(
                source_path=str(file_path),
                mime_type="application/pdf",
                elements=elements,
                num_pages=num_pages,
                metadata={"parser": "pymupdf"}
            )
        
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with PyMuPDF: {e}")

    def _parse_text(self, file_path: Path, mime_type: str) -> ParsedDocument:
        """Parse plain text or markdown documents."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        elements = []
        normalized = normalize_text(text)
        if normalized:
            elements.append(
                ParsedElement(
                    text=normalized,
                    element_type="NarrativeText",
                    metadata={"source": "text"}
                )
            )

        return ParsedDocument(
            source_path=str(file_path),
            mime_type=mime_type,
            elements=elements,
            metadata={"parser": "text"}
        )
    
    def _parse_unstructured(self, file_path: Path, mime_type: str) -> ParsedDocument:
        """Parse document using unstructured library.
        
        Args:
            file_path: Path to document
            mime_type: MIME type
            
        Returns:
            Parsed document
        """
        try:
            # Use unstructured's auto partition
            raw_elements = partition(filename=str(file_path))
            
            elements = []
            for elem in raw_elements:
                element = ParsedElement(
                    text=str(elem),
                    element_type=elem.category if hasattr(elem, 'category') else "Unknown",
                    page_number=elem.metadata.page_number if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'page_number') else None,
                    metadata={"source": "unstructured"}
                )
                elements.append(element)
            
            return ParsedDocument(
                source_path=str(file_path),
                mime_type=mime_type,
                elements=elements,
                metadata={"parser": "unstructured"}
            )
        
        except Exception as e:
            raise RuntimeError(f"Failed to parse with unstructured: {e}")


def normalize_text(text: str) -> str:
    """Normalize extracted text.
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Unicode normalize to NFC
    text = unicodedata.normalize("NFC", text)
    
    # Collapse excessive whitespace
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if line:
            lines.append(line)
    
    return "\n".join(lines)

