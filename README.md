# FinDocs: Financial Document Ingestion + Sentiment + RAG

A production-grade pipeline for ingesting financial documents (earnings calls, filings, reports), performing sentiment analysis with FinBERT, and building a hybrid RAG system with dense + sparse retrieval and cross-encoder re-ranking.

## Features

- **Multi-format parsing**: PDF, DOCX, PPTX, HTML, TXT, MD
- **High-fidelity table extraction**: Using `pdfplumber` + `pymupdf` for accurate table structure
- **Semantic chunking**: Structure-aware chunking (~300 tokens) with overlap for context preservation
- **FinBERT sentiment**: Per-chunk sentiment analysis with calibrated probabilities
- **Hybrid retrieval**:
  - Dense: E5-large-v2 embeddings + FAISS (cosine similarity)
  - Sparse: BM25 for keyword/ticker matching
  - Fusion: Reciprocal Rank Fusion (RRF) or weighted score fusion
- **Cross-encoder re-ranking**: MiniLM reranker for precision@k
- **Idempotent ingestion**: Content-hash based deduplication
- **GPU-accelerated**: Single T4/L4 GPU support

## Design Decisions & Rationale

### Why `pymupdf` + `pdfplumber` over just `unstructured`?

**Table fidelity**: Financial documents contain complex tables (balance sheets, cash flow statements, footnotes). `pdfplumber` provides:
- Explicit cell boundaries with coordinates (bbox)
- Row/column structure for numeric analysis
- Higher accuracy on multi-page tables
- Fallback to `pymupdf` for robust text extraction

`unstructured` is great for general documents but can miss table structure nuances critical for financial analytics.

### Why 300-token chunks with 60-token overlap?

**Token budget**: Both FinBERT and E5-large-v2 have 512-token limits.
- 300 tokens: Safe margin for special tokens and long sentences
- 60-token overlap: Preserves context across boundaries (20% overlap standard)
- Prevents cutting mid-sentence/paragraph, improving both sentiment accuracy and retrieval coherence

### Why FinBERT (`yiyanghkust/finbert-tone`)?

**Domain specificity**: General sentiment models fail on financial language:
- "beat expectations" → positive (not literal violence)
- "headwinds" → negative (financial jargon)
- "flat earnings" → neutral/negative context

FinBERT is trained on 10K+ financial texts (filings, news), providing accurate labels + probabilities for analyst workflows.

### Why E5-large-v2 embeddings?

**Retrieval quality**: E5-large-v2 excels at:
- Instruction-tuned: Better query-document matching
- Strong zero-shot: Works without fine-tuning on your corpus
- 1024 dimensions: Good balance of capacity vs speed on T4/L4
- Widely validated: MTEB leaderboard, production deployments

Alternative considered: `BAAI/bge-large-en-v1.5` (comparable performance, can swap via config).

### Why BM25 + Dense hybrid?

**Complementary strengths**:
- **BM25**: Exact keyword matching for tickers (AAPL, MSFT), acronyms (EPS, EBITDA), rare terms
- **Dense**: Semantic matching for paraphrases ("revenue growth" ≈ "top-line expansion")
- **Hybrid**: Captures both exact and semantic matches, critical for financial Q&A

**Evidence**: BEIR benchmark shows hybrid outperforms either alone by 5-15% on domain-specific tasks.

### Why cross-encoder re-ranking?

**Precision**: Bi-encoders (E5, BM25) are fast but limited to independent encoding. Cross-encoders:
- Attend across query-document pairs → better relevance modeling
- `ms-marco-MiniLM-L-6-v2`: 6 layers, fast on GPU (~50ms per batch of 32)
- Typical setup: retrieve 200 candidates, rerank to top-10 → 2-3x precision gain

Trade-off: Can't precompute (must run at query time), but worth it for financial analysts needing high-precision results.

### Why FAISS IndexFlatIP + L2 normalization?

**Cosine via inner product**:
- Cosine similarity: `cos(u, v) = (u · v) / (||u|| ||v||)`
- If vectors pre-normalized (||u|| = 1), then: `cos(u, v) = u · v`
- IndexFlatIP computes inner product → equivalent to cosine, faster

**Upgrade path**: Start with FlatIP (exact search, 100% recall). When corpus grows:
- IVF (Inverted File): ~10x faster, 95%+ recall
- HNSW: Graph-based, even faster approximate search

### Why Parquet for metadata?

**Analytics-friendly**:
- Columnar: Fast filtering (e.g., `WHERE company = 'AAPL'`)
- Compression: 5-10x smaller than JSON
- Schema evolution: Add fields without breaking existing data
- Pandas integration: Native support

Critical for financial pipelines where analysts query by date, sentiment, ticker, etc.

### Why content-hash (SHA256) registry?

**Auditability**: Financial workflows require:
- **Idempotency**: Re-running ingestion doesn't duplicate data
- **Reproducibility**: Know exactly which document version was processed
- **Audit trail**: (company, path, sha256, timestamp) → full lineage

SHA256 ensures even 1-byte change triggers re-processing.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### 1. Ingest documents

Place documents under `docs/<company>/docs*/`:
```
docs/
  AcmeCo/
    docs_q3_2025/
      earnings_call.pdf
      10q.pdf
```

Run ingestion:
```bash
python -m src.fdocs.cli ingest --company AcmeCo
```

This will:
- Parse documents (PDF, DOCX, etc.)
- Extract tables with high fidelity
- Chunk text semantically (~300 tokens)
- Run FinBERT sentiment analysis
- Generate E5 embeddings
- Build FAISS + BM25 indexes

### 2. Query the index

```bash
# Dense retrieval
python -m src.fdocs.cli query --company AcmeCo --query "What were the revenue drivers?" --mode dense

# Sparse (BM25) retrieval
python -m src.fdocs.cli query --company AcmeCo --query "EPS guidance" --mode sparse

# Hybrid with re-ranking (default, best quality)
python -m src.fdocs.cli query --company AcmeCo --query "margin pressure" --mode hybrid --rerank
```

### 3. Rebuild indexes

If you modify config (e.g., change embedding model):
```bash
python -m src.fdocs.cli rebuild-index
```

## Configuration

Edit `config/default.yaml`:
- **Paths**: Artifact directories
- **Device**: GPU ID
- **Chunking**: Token sizes, overlap
- **Models**: FinBERT, E5, reranker names
- **Hybrid**: Fusion method (rrf, weighted), weights
- **Tables**: Enable/disable, extractor choice

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  docs/<company>/docs*/*.pdf,docx,pptx,html                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │  Parsing (pymupdf/     │
        │  pdfplumber/unstructured)│
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Table Extraction      │
        │  (pdfplumber)          │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Semantic Chunking     │
        │  (~300 tokens, overlap)│
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  FinBERT Sentiment     │
        │  (GPU batched)         │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  E5 Embeddings         │
        │  (GPU, L2-normalized)  │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  FAISS + BM25 Indexes  │
        │  (artifacts/)          │
        └────────────────────────┘

Query Flow:
  Query → [FAISS (dense) + BM25 (sparse)] → RRF Fusion → 
  Cross-Encoder Rerank → Top-K Results (with sentiment)
```

## File Structure

```
.
├── config/
│   └── default.yaml          # Configuration
├── src/fdocs/
│   ├── config.py             # Config loading
│   ├── registry.py           # Ingestion ledger
│   ├── parse.py              # Multi-format parsing
│   ├── tables.py             # Table extraction
│   ├── chunk.py              # Semantic chunking
│   ├── sentiment.py          # FinBERT sentiment
│   ├── embed.py              # E5 embeddings
│   ├── index.py              # FAISS index
│   ├── sparse.py             # BM25 index
│   ├── rerank.py             # Cross-encoder reranking
│   ├── retrieval.py          # Hybrid orchestration
│   └── cli.py                # CLI entrypoints
├── artifacts/                # Generated artifacts
│   ├── registry/             # Ingestion ledger
│   ├── chunks/               # Chunk metadata (Parquet)
│   ├── tables/               # Extracted tables (Parquet)
│   ├── index/                # FAISS index
│   └── sparse/               # BM25 index
├── docs/                     # Input documents
└── requirements.txt          # Dependencies
```

## Performance Notes

**GPU Memory** (T4 16GB / L4 24GB):
- FinBERT: ~1.5 GB
- E5-large-v2: ~3 GB
- Reranker (MiniLM): ~0.5 GB
- Comfortable to load sequentially or pin all on L4

**Throughput** (on L4):
- Parsing: ~2-5 pages/sec (PDF)
- FinBERT: ~50-100 chunks/sec (batch=16)
- Embeddings: ~100-200 chunks/sec (batch=32)
- Reranking: ~500-1000 pairs/sec (batch=32)

**Scaling**:
- Up to ~10M chunks: FlatIP works well
- Beyond: Switch to IVF or HNSW (config change + rebuild)

## Testing

Run basic tests:
```bash
python -m pytest tests/ -v
```

Tests cover:
- Chunking boundary preservation
- Sentiment label mapping
- Table serialization round-trip
- RRF fusion correctness

## Future Extensions

This pipeline is designed to evolve into a full financial intelligence tool:
- **Alpha generation**: Correlate sentiment trends with price movements
- **Event detection**: Earnings surprise, guidance changes
- **Entity linking**: Map mentions to tickers, people, products
- **Time-series analysis**: Track sentiment over quarters
- **Multi-document reasoning**: Compare companies, competitive analysis

The current design (chunking, sentiment, hybrid retrieval, metadata schema) supports all these extensions without refactoring.

## Why These Choices Matter

In production financial systems, **quality and auditability** trump raw speed:
- **Table fidelity**: Analysts need accurate numbers, not "table detected"
- **Sentiment probabilities**: Not just labels—probabilities enable confidence thresholds and risk modeling
- **Hybrid retrieval**: Catches both "AAPL guidance" (BM25) and "Apple's forward outlook" (dense)
- **Content hashes**: Compliance requires knowing exactly what was processed
- **Parquet metadata**: Enables SQL-like analytics over chunks (filter by sentiment, date, section)

Every design decision supports **defensibility**: when an analyst or auditor asks "why did this chunk rank here?", you can trace through BM25 scores, embedding distances, reranker logits, and sentiment probabilities.

## License

MIT

## Questions?

For design rationale or implementation details, see inline comments in source files. Key modules:
- `chunk.py`: Structure-aware sentence grouping
- `sentiment.py`: FinBERT probability calibration
- `retrieval.py`: RRF fusion logic
- `tables.py`: pdfplumber table serialization

