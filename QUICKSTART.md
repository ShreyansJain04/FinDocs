# FinDocs Quickstart Guide

This guide walks you through ingesting your first financial document and querying the system.

## Prerequisites

- Python 3.9+
- NVIDIA GPU (T4/L4) with CUDA support (or CPU fallback)
- ~20GB disk space for models and artifacts

## Setup (5 minutes)

### 1. Install dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Verify GPU (optional but recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is unavailable, the system will fall back to CPU (slower but functional).

## Ingest Sample Document (2 minutes)

We've included a sample earnings call transcript in `docs/ExampleCo/docs_q3_2025/sample.txt`.

```bash
python -m src.fdocs.cli ingest --company ExampleCo
```

**What happens:**
1. Discovers `sample.txt` in `docs/ExampleCo/docs_q3_2025/`
2. Parses and normalizes text
3. Chunks into ~300-token segments with overlap
4. Runs FinBERT sentiment analysis (GPU accelerated)
5. Generates E5 embeddings (GPU accelerated)
6. Builds FAISS (dense) and BM25 (sparse) indexes
7. Saves to `artifacts/`

**Expected output:**
```
Starting ingestion for company: ExampleCo
Using device: cuda:0
Found 1 documents

Processing: sample.txt
  Parsed 1 elements
  Created 5 chunks

Total chunks created: 5

Running sentiment analysis...
Sentiment analysis: 100%|████████| 1/1 [00:02<00:00]
Sentiment analysis complete

Generating embeddings...
Batches: 100%|████████| 1/1 [00:01<00:00]
Generated 5 embeddings

Building FAISS index...
FAISS index saved with 5 vectors

Building BM25 index...
BM25 index saved with 5 documents

Saving chunks...
Chunks saved to artifacts/chunks/chunks.parquet

============================================================
Ingestion complete!
  Documents processed: 1/1
  Total chunks: 5
  Total tables: 0
============================================================
```

## Query the System (30 seconds)

### Example 1: Dense semantic retrieval

```bash
python -m src.fdocs.cli query \
  --company ExampleCo \
  --query "What were the revenue numbers?" \
  --mode dense \
  --top-k 3
```

**Why dense?** Captures semantic meaning ("revenue numbers" matches "Total revenue", "financial performance").

### Example 2: Sparse keyword retrieval

```bash
python -m src.fdocs.cli query \
  --company ExampleCo \
  --query "EPS margin" \
  --mode sparse \
  --top-k 3
```

**Why sparse?** Exact keyword matching for acronyms and specific terms.

### Example 3: Hybrid with re-ranking (best quality)

```bash
python -m src.fdocs.cli query \
  --company ExampleCo \
  --query "What risks does the company face?" \
  --mode hybrid \
  --rerank \
  --top-k 5
```

**Why hybrid + rerank?**
- Combines dense (semantic) + sparse (keyword) → better recall
- Cross-encoder reranks top candidates → better precision

**Sample output:**
```
Querying: What risks does the company face?
Mode: hybrid, Rerank: True
Loaded 5 chunks for ExampleCo

Loading indexes...
Retrieving...

================================================================================
Top 5 results:
================================================================================

[1] Score: 0.8523
    Source: sample.txt
    Page: N/A
    Sentiment: negative (pos: 0.12, neu: 0.25, neg: 0.63)
    Text: While we are optimistic about our trajectory, we face headwinds from 
    increased competition in certain markets. Additionally, macroeconomic 
    uncertainties could impact customer spending patterns...

[2] Score: 0.7341
    Source: sample.txt
    Page: N/A
    Sentiment: neutral (pos: 0.35, neu: 0.52, neg: 0.13)
    Text: Risk Factors: While we are optimistic about our trajectory...

...
```

## Add Your Own Documents

### Step 1: Organize documents

Create a directory structure:
```
docs/
  <CompanyName>/
    docs_<descriptor>/
      file1.pdf
      file2.docx
      ...
```

Example:
```
docs/
  Tesla/
    docs_q4_2024/
      earnings_call.pdf
      10k.pdf
    docs_q1_2025/
      shareholder_letter.pdf
```

### Step 2: Ingest

```bash
python -m src.fdocs.cli ingest --company Tesla
```

### Step 3: Query

```bash
python -m src.fdocs.cli query \
  --company Tesla \
  --query "vehicle deliveries and production" \
  --mode hybrid \
  --rerank
```

## Configuration

Edit `config/default.yaml` to customize:

### Change chunk size
```yaml
chunking:
  target_tokens: 500  # Larger chunks (default: 300)
  overlap_tokens: 100  # More overlap (default: 60)
```

### Disable tables
```yaml
tables:
  enabled: false
```

### Adjust batch sizes (for memory constraints)
```yaml
sentiment:
  batch_size: 8  # Reduce if GPU OOM (default: 16)

embeddings:
  batch_size: 16  # Reduce if GPU OOM (default: 32)
```

### Change models
```yaml
sentiment:
  model: "ProsusAI/finbert"  # Alternative FinBERT

embeddings:
  model: "BAAI/bge-large-en-v1.5"  # Alternative embedder
```

### Disable BM25 or reranking
```yaml
bm25:
  enabled: false

reranker:
  enabled: false
```

## Common Issues

### "CUDA out of memory"

**Solution 1:** Reduce batch sizes in `config/default.yaml`:
```yaml
sentiment:
  batch_size: 4
embeddings:
  batch_size: 8
reranker:
  batch_size: 16
```

**Solution 2:** Use CPU (slower):
```yaml
device:
  use_cuda: false
```

### "No module named 'unstructured'"

```bash
pip install unstructured
```

### "NLTK punkt not found"

```bash
python -c "import nltk; nltk.download('punkt')"
```

### Slow ingestion on CPU

This is expected. CPU inference is 10-50x slower than GPU. Consider:
- Using a GPU instance
- Processing documents in smaller batches
- Disabling features (tables, reranking) for faster iteration

## Next Steps

1. **Ingest more documents**: Add PDFs, DOCX, presentations to `docs/<company>/`
2. **Experiment with queries**: Try different modes (dense, sparse, hybrid)
3. **Explore sentiment**: Filter results by sentiment in the Parquet files
4. **Analyze tables**: Check `artifacts/tables/tables.parquet` for extracted data
5. **Build analytics**: Load `artifacts/chunks/chunks.parquet` in Pandas/DuckDB for analysis

## Example: Analyze sentiment distribution

```python
import pandas as pd

chunks = pd.read_parquet("artifacts/chunks/chunks.parquet")

# Filter by company
tesla_chunks = chunks[chunks["company"] == "Tesla"]

# Sentiment distribution
print(tesla_chunks["sentiment_label"].value_counts())

# High-confidence negative chunks
negative = tesla_chunks[
    (tesla_chunks["sentiment_label"] == "negative") & 
    (tesla_chunks["p_negative"] > 0.8)
]
print(negative[["text", "p_negative"]])
```

## Performance Expectations

**On L4 GPU:**
- Small doc (10 pages): ~30 seconds
- Medium doc (50 pages): ~2 minutes
- Large doc (200 pages): ~8 minutes

**Bottlenecks:**
1. PDF parsing (CPU bound)
2. FinBERT inference (GPU)
3. Embedding generation (GPU)

**Scaling:** For production (1000+ docs), consider:
- Distributed ingestion (Ray, Dask)
- Pre-built indexes (ingest once, query many times)
- IVF/HNSW indexes for faster search

## Support

For issues or questions:
1. Check inline documentation in source files
2. Review design rationale in `README.md`
3. Inspect artifacts in `artifacts/` for debugging

## Key Files

- `config/default.yaml`: Configuration
- `artifacts/chunks/chunks.parquet`: All chunks with metadata
- `artifacts/index/index.faiss`: Dense vector index
- `artifacts/sparse/bm25.pkl`: Sparse BM25 index
- `artifacts/registry/ingestion_registry.parquet`: Processing history

