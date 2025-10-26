"""Benchmark script to measure actual performance metrics."""

import time
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fdocs.config import load_config, get_device
from fdocs.sentiment import SentimentAnalyzer
from fdocs.embed import EmbeddingGenerator
from fdocs.chunk import SemanticChunker
from fdocs.rerank import Reranker
from fdocs.index import FAISSIndex
from fdocs.sparse import BM25Index


def benchmark_sentiment(device: str, batch_size: int = 16):
    """Benchmark FinBERT sentiment analysis."""
    print(f"\n{'='*60}")
    print("Benchmarking FinBERT Sentiment Analysis")
    print(f"{'='*60}")
    
    analyzer = SentimentAnalyzer(
        model_name="yiyanghkust/finbert-tone",
        device=device,
        batch_size=batch_size,
        max_length=512
    )
    
    # Sample texts (financial domain)
    sample_text = (
        "The company reported strong earnings growth of 15% year-over-year, "
        "driven by robust demand in core markets. However, management expressed "
        "concerns about rising input costs and supply chain headwinds that could "
        "impact margins in the coming quarters. Despite these challenges, the "
        "outlook remains cautiously optimistic with continued investment in "
        "innovation and market expansion."
    )
    
    # Warmup
    _ = analyzer.analyze_batch([sample_text])
    
    # Benchmark different batch sizes
    test_sizes = [10, 50, 100, 200]
    
    for n in test_sizes:
        texts = [sample_text] * n
        
        start = time.perf_counter()
        results = analyzer.analyze_batch(texts)
        elapsed = time.perf_counter() - start
        
        throughput = n / elapsed
        print(f"  Texts: {n:4d} | Time: {elapsed:.3f}s | Throughput: {throughput:.1f} texts/sec")
    
    return throughput


def benchmark_embeddings(device: str, batch_size: int = 32):
    """Benchmark E5 embedding generation."""
    print(f"\n{'='*60}")
    print("Benchmarking E5 Embedding Generation")
    print(f"{'='*60}")
    
    embedder = EmbeddingGenerator(
        model_name="intfloat/e5-large-v2",
        device=device,
        batch_size=batch_size,
        normalize=True
    )
    
    sample_text = (
        "The company reported strong earnings growth of 15% year-over-year, "
        "driven by robust demand in core markets."
    )
    
    # Warmup
    _ = embedder.embed_batch([sample_text])
    
    test_sizes = [10, 50, 100, 200, 500]
    
    for n in test_sizes:
        texts = [sample_text] * n
        
        start = time.perf_counter()
        embeddings = embedder.embed_batch(texts)
        elapsed = time.perf_counter() - start
        
        throughput = n / elapsed
        print(f"  Texts: {n:4d} | Time: {elapsed:.3f}s | Throughput: {throughput:.1f} texts/sec | Shape: {embeddings.shape}")
    
    return throughput


def benchmark_chunking():
    """Benchmark semantic chunking."""
    print(f"\n{'='*60}")
    print("Benchmarking Semantic Chunking")
    print(f"{'='*60}")
    
    chunker = SemanticChunker(
        target_tokens=300,
        overlap_tokens=60,
        model_name="intfloat/e5-large-v2"
    )
    
    # Generate sample document (5000 words)
    paragraph = (
        "The company delivered exceptional results in Q3 2024, with revenue reaching $2.5 billion, "
        "representing a 22% increase year-over-year. Net income grew 18% to $450 million, while "
        "operating margins expanded by 150 basis points to 23.5%. Management attributed the strong "
        "performance to robust demand across all geographic segments and continued market share gains. "
        "Looking ahead, the company reaffirmed full-year guidance and expressed confidence in sustained "
        "momentum despite ongoing macroeconomic uncertainties. "
    )
    document = " ".join([paragraph] * 50)  # ~5000 words
    
    # Warmup
    _ = chunker.chunk_document(
        text=document[:1000],
        company="TestCo",
        source_path="test.pdf"
    )
    
    # Benchmark
    start = time.perf_counter()
    chunks = chunker.chunk_document(
        text=document,
        company="TestCo",
        source_path="test.pdf"
    )
    elapsed = time.perf_counter() - start
    
    word_count = len(document.split())
    throughput = word_count / elapsed
    
    print(f"  Document: {word_count:,} words")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} words/sec")
    print(f"  Avg chunk size: {sum(c.token_count for c in chunks) / len(chunks):.1f} tokens")


def benchmark_reranking(device: str, batch_size: int = 32):
    """Benchmark cross-encoder reranking."""
    print(f"\n{'='*60}")
    print("Benchmarking Cross-Encoder Reranking")
    print(f"{'='*60}")
    
    reranker = Reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device,
        batch_size=batch_size
    )
    
    query = "What were the revenue numbers?"
    
    candidate_text = (
        "The company reported total revenue of $2.5 billion for Q3 2024, "
        "representing strong growth of 22% year-over-year driven by increased demand."
    )
    
    # Warmup
    _ = reranker.rerank(query, [candidate_text] * 10, [f"chunk_{i}" for i in range(10)])
    
    test_sizes = [50, 100, 200, 500]
    
    for n in test_sizes:
        texts = [candidate_text] * n
        chunk_ids = [f"chunk_{i}" for i in range(n)]
        
        start = time.perf_counter()
        reranked_ids, scores = reranker.rerank(query, texts, chunk_ids)
        elapsed = time.perf_counter() - start
        
        throughput = n / elapsed
        print(f"  Pairs: {n:4d} | Time: {elapsed:.3f}s | Throughput: {throughput:.1f} pairs/sec")
    
    return throughput


def benchmark_faiss_search():
    """Benchmark FAISS search."""
    print(f"\n{'='*60}")
    print("Benchmarking FAISS Search")
    print(f"{'='*60}")
    
    dimension = 1024
    
    # Build index with different sizes
    test_sizes = [1000, 10000, 100000]
    
    for n_vectors in test_sizes:
        # Create random vectors
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        chunk_ids = [f"chunk_{i}" for i in range(n_vectors)]
        
        # Build index
        index = FAISSIndex(dimension=dimension, index_type="IndexFlatIP", metric="inner_product")
        
        build_start = time.perf_counter()
        index.add(vectors, chunk_ids)
        build_time = time.perf_counter() - build_start
        
        # Benchmark search
        query_vector = np.random.randn(dimension).astype(np.float32)
        
        # Warmup
        _ = index.search(query_vector, k=10)
        
        # Benchmark multiple queries
        n_queries = 1000
        start = time.perf_counter()
        for _ in range(n_queries):
            _ = index.search(query_vector, k=10)
        elapsed = time.perf_counter() - start
        
        throughput = n_queries / elapsed
        
        print(f"  Index size: {n_vectors:,} vectors")
        print(f"    Build time: {build_time:.3f}s ({n_vectors/build_time:.0f} vec/sec)")
        print(f"    Search: {n_queries} queries in {elapsed:.3f}s ({throughput:.0f} qps)")


def benchmark_bm25_search():
    """Benchmark BM25 search."""
    print(f"\n{'='*60}")
    print("Benchmarking BM25 Search")
    print(f"{'='*60}")
    
    sample_text = (
        "The company reported strong earnings growth of 15% year-over-year, "
        "driven by robust demand in core markets and improved operational efficiency."
    )
    
    test_sizes = [1000, 10000, 50000]
    
    for n_docs in test_sizes:
        texts = [sample_text] * n_docs
        chunk_ids = [f"chunk_{i}" for i in range(n_docs)]
        
        # Build index
        index = BM25Index(k1=0.9, b=0.4)
        
        build_start = time.perf_counter()
        index.build(texts, chunk_ids)
        build_time = time.perf_counter() - build_start
        
        # Benchmark search
        query = "revenue growth earnings"
        
        # Warmup
        _ = index.search(query, k=10)
        
        # Benchmark multiple queries
        n_queries = 1000
        start = time.perf_counter()
        for _ in range(n_queries):
            _ = index.search(query, k=100)
        elapsed = time.perf_counter() - start
        
        throughput = n_queries / elapsed
        
        print(f"  Index size: {n_docs:,} documents")
        print(f"    Build time: {build_time:.3f}s ({n_docs/build_time:.0f} docs/sec)")
        print(f"    Search: {n_queries} queries in {elapsed:.3f}s ({throughput:.0f} qps)")


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("FinDocs Performance Benchmark")
    print("="*60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run benchmarks
    try:
        benchmark_chunking()
        sentiment_throughput = benchmark_sentiment(device, batch_size=16)
        embedding_throughput = benchmark_embeddings(device, batch_size=32)
        rerank_throughput = benchmark_reranking(device, batch_size=32)
        benchmark_faiss_search()
        benchmark_bm25_search()
        
        # Summary
        print(f"\n{'='*60}")
        print("Summary (Peak Throughput)")
        print(f"{'='*60}")
        print(f"  FinBERT Sentiment:     ~{sentiment_throughput:.0f} texts/sec")
        print(f"  E5 Embeddings:         ~{embedding_throughput:.0f} texts/sec")
        print(f"  Cross-encoder Rerank:  ~{rerank_throughput:.0f} pairs/sec")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


