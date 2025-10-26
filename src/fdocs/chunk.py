"""Semantic chunking with structure-awareness."""

import uuid
from dataclasses import dataclass
from typing import List, Optional

import nltk
from transformers import AutoTokenizer

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class Chunk:
    """Text chunk with metadata."""
    chunk_id: str
    company: str
    source_path: str
    chunk_idx: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    is_table: bool = False
    table_id: Optional[str] = None
    sentiment_label: Optional[str] = None
    p_positive: Optional[float] = None
    p_neutral: Optional[float] = None
    p_negative: Optional[float] = None
    extraction_ref: Optional[str] = None


class SemanticChunker:
    """Structure-aware semantic chunking."""
    
    def __init__(
        self,
        target_tokens: int = 300,
        overlap_tokens: int = 60,
        model_name: str = "intfloat/e5-large-v2",
        respect_structure: bool = True,
        similarity_threshold: float = 0.7
    ):
        """Initialize chunker.
        
        Args:
            target_tokens: Target chunk size in tokens
            overlap_tokens: Overlap between chunks
            model_name: Tokenizer model name
            respect_structure: Whether to respect structural boundaries
            similarity_threshold: Threshold for semantic boundary detection
        """
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.respect_structure = respect_structure
        self.similarity_threshold = similarity_threshold
        
        # Load tokenizer for token counting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def chunk_document(
        self,
        text: str,
        company: str,
        source_path: str,
        page_number: Optional[int] = None,
        section: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk a document's text.
        
        Args:
            text: Document text
            company: Company name
            source_path: Source document path
            page_number: Page number (if applicable)
            section: Section name
            
        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []
        
        # Sentence segmentation
        sentences = self._segment_sentences(text)
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks = self._group_sentences(
            sentences, 
            company, 
            source_path, 
            page_number, 
            section
        )
        
        return chunks
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback: split on periods
            sentences = text.split(". ")
            return [s.strip() + "." for s in sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # Fallback: rough estimate
            return len(text.split())
    
    def _group_sentences(
        self,
        sentences: List[str],
        company: str,
        source_path: str,
        page_number: Optional[int],
        section: Optional[str]
    ) -> List[Chunk]:
        """Group sentences into chunks with overlap.
        
        Args:
            sentences: List of sentences
            company: Company name
            source_path: Source path
            page_number: Page number
            section: Section name
            
        Returns:
            List of chunks
        """
        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_idx = 0
        char_offset = 0
        
        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            
            # Check if adding this sentence would exceed target
            if current_tokens + sent_tokens > self.target_tokens and current_sentences:
                # Create chunk
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    company=company,
                    source_path=source_path,
                    chunk_idx=chunk_idx,
                    text=chunk_text,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text),
                    token_count=current_tokens,
                    page_start=page_number,
                    page_end=page_number,
                    section=section,
                    is_table=False
                )
                chunks.append(chunk)
                chunk_idx += 1
                char_offset += len(chunk_text) + 1
                
                # Handle overlap: keep last few sentences
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            
            current_sentences.append(sent)
            current_tokens += sent_tokens
        
        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                company=company,
                source_path=source_path,
                chunk_idx=chunk_idx,
                text=chunk_text,
                char_start=char_offset,
                char_end=char_offset + len(chunk_text),
                token_count=current_tokens,
                page_start=page_number,
                page_end=page_number,
                section=section,
                is_table=False
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_table(
        self,
        table_text: str,
        table_id: str,
        company: str,
        source_path: str,
        page_number: Optional[int] = None,
        caption: Optional[str] = None
    ) -> Chunk:
        """Create a chunk for a table.
        
        Args:
            table_text: Serialized table text
            table_id: Table identifier
            company: Company name
            source_path: Source path
            page_number: Page number
            caption: Table caption
            
        Returns:
            Table chunk
        """
        # Include caption if available
        full_text = table_text
        if caption:
            full_text = f"{caption}\n\n{table_text}"
        
        token_count = self._count_tokens(full_text)
        
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            company=company,
            source_path=source_path,
            chunk_idx=0,  # Will be adjusted later
            text=full_text,
            char_start=0,
            char_end=len(full_text),
            token_count=token_count,
            page_start=page_number,
            page_end=page_number,
            section=None,
            is_table=True,
            table_id=table_id
        )

