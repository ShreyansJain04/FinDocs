"""Command-line interface for FinDocs."""

import sys
from pathlib import Path
from datetime import datetime

import click
import pandas as pd

from .config import Config, load_config, get_device
from .registry import IngestionRegistry, RegistryEntry, compute_file_hash, discover_documents
from .parse import DocumentParser, normalize_text
from .tables import TableExtractor, save_tables_to_parquet
from collections import defaultdict

from .active_learning import score_document, UncertaintyWeights
from .chunk import SemanticChunker
def _group_chunks_by_section(chunks):
    groups = defaultdict(list)
    for chunk in chunks:
        key = chunk.section or "general"
        groups[key].append(chunk.text)
    return {section: texts for section, texts in groups.items() if texts}

from .sentiment import SentimentAnalyzer
from .embed import EmbeddingGenerator
from .index import FAISSIndex, save_chunks_to_parquet
from .sparse import BM25Index
from .rerank import Reranker
from .retrieval import HybridRetriever
from .extract import ExtractionConfig, StructureExtractor


@click.group()
def cli():
    """FinDocs: Financial Document Ingestion and RAG."""
    pass


@cli.command()
@click.option("--company", required=True, help="Company name")
@click.option("--config", default="config/default.yaml", help="Config file path")
@click.option("--force", is_flag=True, help="Force re-processing")
def ingest(company: str, config: str, force: bool):
    """Ingest documents for a company."""
    click.echo(f"Starting ingestion for company: {company}")
    
    # Load config
    cfg = load_config(config)
    device = get_device(cfg)
    click.echo(f"Using device: {device}")
    
    # Initialize components
    docs_root = Path(cfg.paths.docs_root)
    registry_path = Path(cfg.paths.registry_dir) / "ingestion_registry.parquet"
    chunks_path = Path(cfg.paths.chunks_dir) / "chunks.parquet"
    tables_path = Path(cfg.paths.tables_dir) / "tables.parquet"
    index_path = Path(cfg.paths.index_dir)
    sparse_path = Path(cfg.paths.sparse_dir)
    
    registry = IngestionRegistry(registry_path)
    parser = DocumentParser(pdf_extractor=cfg.parsing.pdf_extractor)
    table_extractor = TableExtractor(
        extractor=cfg.tables.extractor,
        serialize_format=cfg.tables.serialize_format
    ) if cfg.tables.enabled else None

    chunker = SemanticChunker(
        target_tokens=cfg.chunking.target_tokens,
        overlap_tokens=cfg.chunking.overlap_tokens,
        model_name=cfg.embeddings.model,
        respect_structure=cfg.chunking.respect_structure,
        similarity_threshold=cfg.chunking.similarity_threshold
    )

    enqueue_cfg = cfg.active_learning or {}

    extractor = None
    if cfg.extraction is not None:
        extraction_cfg = ExtractionConfig(
            primary_endpoint=cfg.extraction.get("primary_endpoint"),
            fallback_endpoint=cfg.extraction.get("fallback_endpoint"),
            max_retries=cfg.extraction.get("max_retries", 2),
            extraction_version=cfg.extraction.get("extraction_version", "v1"),
            mode=cfg.extraction.get("mode", "remote"),
        )
        extractor = StructureExtractor(extraction_cfg)
    
    # Discover documents
    doc_files = discover_documents(docs_root, company)
    click.echo(f"Found {len(doc_files)} documents")
    
    if not doc_files:
        click.echo("No documents found. Exiting.")
        return
    
    # Process documents
    all_chunks = []
    all_tables = []
    doc_records = []
    
    for doc_path in doc_files:
        click.echo(f"\nProcessing: {doc_path.name}")
        
        # Check if already processed
        file_hash = compute_file_hash(doc_path)
        if not force and registry.is_processed(company, str(doc_path), file_hash):
            click.echo("  Already processed (skip)")
            continue
        
        try:
            doc_tables = []
            start_idx = len(all_chunks)
            # Parse document
            parsed_doc = parser.parse(doc_path)
            click.echo(f"  Parsed {len(parsed_doc.elements)} elements")
            
            # Extract tables
            if table_extractor and doc_path.suffix == ".pdf":
                tables = table_extractor.extract_from_pdf(doc_path)
                click.echo(f"  Extracted {len(tables)} tables")
                all_tables.extend(tables)
                doc_tables.extend(tables)
                
                # Create table chunks
                for table in tables:
                    table_text = table_extractor.serialize_table(table)
                    table_chunk = chunker.chunk_table(
                        table_text=table_text,
                        table_id=table.table_id,
                        company=company,
                        source_path=str(doc_path),
                        page_number=table.page,
                        caption=table.caption
                    )
                    all_chunks.append(table_chunk)
            elif table_extractor and doc_path.suffix == ".docx":
                tables = table_extractor.extract_from_docx(doc_path)
                click.echo(f"  Extracted {len(tables)} tables")
                all_tables.extend(tables)
                doc_tables.extend(tables)
                
                for table in tables:
                    table_text = table_extractor.serialize_table(table)
                    table_chunk = chunker.chunk_table(
                        table_text=table_text,
                        table_id=table.table_id,
                        company=company,
                        source_path=str(doc_path),
                        caption=table.caption
                    )
                    all_chunks.append(table_chunk)
            
            # Chunk text elements
            for element in parsed_doc.elements:
                if element.element_type in ["NarrativeText", "Title"]:
                    text = normalize_text(element.text)
                    if text:
                        chunks = chunker.chunk_document(
                            text=text,
                            company=company,
                            source_path=str(doc_path),
                            page_number=element.page_number,
                            section=element.section
                        )
                        all_chunks.extend(chunks)
            
            doc_chunks = all_chunks[start_idx:]

            # Update registry
            entry = RegistryEntry(
                company=company,
                source_path=str(doc_path),
                sha256=file_hash,
                mime_type=parsed_doc.mime_type,
                size_bytes=doc_path.stat().st_size,
                parsed_ok=True,
                num_chunks=len(doc_chunks),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            registry.add_entry(entry)
            click.echo(f"  Created {entry.num_chunks} chunks")

            doc_records.append({
                "path": str(doc_path),
                "hash": file_hash,
                "chunks": doc_chunks,
                "tables": doc_tables,
            })
        
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            entry = RegistryEntry(
                company=company,
                source_path=str(doc_path),
                sha256=file_hash,
                mime_type="unknown",
                size_bytes=doc_path.stat().st_size,
                parsed_ok=False,
                num_chunks=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                error_msg=str(e)
            )
            registry.add_entry(entry)
    
    # Save registry
    registry.save()
    click.echo(f"\nTotal chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        click.echo("No chunks to process. Exiting.")
        return
    
    # Run sentiment analysis
    click.echo("\nRunning sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer(
        model_name=cfg.sentiment.model,
        device=device,
        batch_size=cfg.sentiment.batch_size,
        max_length=cfg.sentiment.max_length
    )
    
    chunk_texts = [chunk.text for chunk in all_chunks]
    sentiment_results = sentiment_analyzer.analyze_batch(chunk_texts)
    
    # Add sentiment to chunks
    for chunk, (label, p_pos, p_neu, p_neg) in zip(all_chunks, sentiment_results):
        chunk.sentiment_label = label
        chunk.p_positive = p_pos
        chunk.p_neutral = p_neu
        chunk.p_negative = p_neg
    
    click.echo("Sentiment analysis complete")

    # Structured extraction
    queue_items = []
    if extractor and doc_records:
        click.echo("\nRunning structured extraction...")
        for record in doc_records:
            section_groups = _group_chunks_by_section(record["chunks"])
            if not section_groups:
                continue
            try:
                report = extractor.extract(
                    company=company,
                    document_id=record["hash"],
                    document_path=record["path"],
                    sections=section_groups,
                    metadata={"chunk_count": len(record["chunks"])}
                )
            except Exception as exc:
                click.echo(f"  Structured extraction failed: {exc}", err=True)
                continue
            for chunk in record["chunks"]:
                chunk.extraction_ref = getattr(report, "document_id", None)
            weights = UncertaintyWeights()
            uncertainty_items = score_document(
                document_id=record["hash"],
                chunks=record["chunks"],
                report=report,
                weights=weights,
            )
            queue_items.extend(uncertainty_items)
        click.echo("Structured extraction complete")
    
    # Generate embeddings
    click.echo("\nGenerating embeddings...")
    embedding_gen = EmbeddingGenerator(
        model_name=cfg.embeddings.model,
        device=device,
        batch_size=cfg.embeddings.batch_size,
        normalize=cfg.embeddings.normalize
    )
    
    embeddings = embedding_gen.embed_batch(chunk_texts)
    click.echo(f"Generated {len(embeddings)} embeddings")
    
    # Build FAISS index
    click.echo("\nBuilding FAISS index...")
    faiss_index = FAISSIndex(
        dimension=embedding_gen.get_dimension(),
        index_type=cfg.faiss.index_type,
        metric=cfg.faiss.metric
    )
    
    chunk_ids = [chunk.chunk_id for chunk in all_chunks]
    faiss_index.add(embeddings, chunk_ids)
    faiss_index.save(index_path, metadata={"company": company, "num_chunks": len(all_chunks)})
    click.echo(f"FAISS index saved with {faiss_index.get_size()} vectors")
    
    # Build BM25 index
    if cfg.bm25.enabled:
        click.echo("\nBuilding BM25 index...")
        bm25_index = BM25Index(k1=cfg.bm25.k1, b=cfg.bm25.b)
        bm25_index.build(chunk_texts, chunk_ids)
        bm25_index.save(sparse_path)
        click.echo(f"BM25 index saved with {bm25_index.get_size()} documents")
    
    # Save chunks
    click.echo("\nSaving chunks...")
    save_chunks_to_parquet(all_chunks, chunks_path)
    click.echo(f"Chunks saved to {chunks_path}")
    
    # Save tables
    if all_tables and cfg.tables.enabled:
        click.echo("\nSaving tables...")
        save_tables_to_parquet(all_tables, company, "batch", tables_path)
        click.echo(f"Tables saved to {tables_path}")
    
    if extractor and doc_records:
        click.echo("\nRunning structured extraction...")
        for record in doc_records:
            section_groups = _group_chunks_by_section(record["chunks"])
            if not section_groups:
                continue
            try:
                report = extractor.extract(
                    company=company,
                    document_id=record["hash"],
                    document_path=record["path"],
                    sections=section_groups,
                    metadata={"chunk_count": len(record["chunks"])}
                )
            except Exception as exc:
                click.echo(f"  Structured extraction failed: {exc}", err=True)
                continue
            for chunk in record["chunks"]:
                chunk.extraction_ref = getattr(report, "document_id", None)
        click.echo("Structured extraction complete")

    if queue_items:
        from .queue import QueueItem, ReviewQueue

        click.echo("\nQueueing uncertain items...")
        queue = ReviewQueue()
        limit = enqueue_cfg.get("enqueue_limit", len(queue_items))
        queue.add_items(
            QueueItem(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                field_name=item.field_name,
                uncertainty_score=item.score,
                priority=item.score,
                metadata=item.metadata,
            )
            for item in queue_items[:limit]
        )
        click.echo(f"Added {min(limit, len(queue_items))} items to review queue")

    # Stats
    stats = registry.get_stats()
    click.echo(f"\n{'='*60}")
    click.echo("Ingestion complete!")
    click.echo(f"  Documents processed: {stats['success']}/{stats['total']}")
    click.echo(f"  Total chunks: {len(all_chunks)}")
    click.echo(f"  Total tables: {len(all_tables)}")
    click.echo(f"{'='*60}")


@cli.command()
@click.option("--config", default="config/default.yaml", help="Config file path")
def rebuild_index(config: str):
    """Rebuild indexes from existing chunks."""
    click.echo("Rebuilding indexes...")
    
    cfg = load_config(config)
    device = get_device(cfg)
    
    chunks_path = Path(cfg.paths.chunks_dir) / "chunks.parquet"
    index_path = Path(cfg.paths.index_dir)
    sparse_path = Path(cfg.paths.sparse_dir)
    
    if not chunks_path.exists():
        click.echo(f"Chunks file not found: {chunks_path}", err=True)
        return
    
    # Load chunks
    click.echo(f"Loading chunks from {chunks_path}")
    chunks_df = pd.read_parquet(chunks_path)
    click.echo(f"Loaded {len(chunks_df)} chunks")
    
    # Generate embeddings
    click.echo("\nGenerating embeddings...")
    embedding_gen = EmbeddingGenerator(
        model_name=cfg.embeddings.model,
        device=device,
        batch_size=cfg.embeddings.batch_size,
        normalize=cfg.embeddings.normalize
    )
    
    texts = chunks_df["text"].tolist()
    embeddings = embedding_gen.embed_batch(texts)
    
    # Build FAISS
    click.echo("\nBuilding FAISS index...")
    faiss_index = FAISSIndex(
        dimension=embedding_gen.get_dimension(),
        index_type=cfg.faiss.index_type,
        metric=cfg.faiss.metric
    )
    
    chunk_ids = chunks_df["chunk_id"].tolist()
    faiss_index.add(embeddings, chunk_ids)
    faiss_index.save(index_path)
    click.echo(f"FAISS index rebuilt: {faiss_index.get_size()} vectors")
    
    # Build BM25
    if cfg.bm25.enabled:
        click.echo("\nBuilding BM25 index...")
        bm25_index = BM25Index(k1=cfg.bm25.k1, b=cfg.bm25.b)
        bm25_index.build(texts, chunk_ids)
        bm25_index.save(sparse_path)
        click.echo(f"BM25 index rebuilt: {bm25_index.get_size()} documents")
    
    click.echo("\nRebuild complete!")


@cli.command()
@click.option("--company", required=True, help="Company name")
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--mode", default="hybrid", type=click.Choice(["dense", "sparse", "hybrid"]), help="Retrieval mode")
@click.option("--rerank", is_flag=True, default=True, help="Enable reranking")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--config", default="config/default.yaml", help="Config file path")
def query(company: str, query: str, mode: str, rerank: bool, top_k: int, config: str):
    """Query the document index."""
    click.echo(f"Querying: {query}")
    click.echo(f"Mode: {mode}, Rerank: {rerank}")
    
    cfg = load_config(config)
    device = get_device(cfg)
    
    chunks_path = Path(cfg.paths.chunks_dir) / "chunks.parquet"
    index_path = Path(cfg.paths.index_dir)
    sparse_path = Path(cfg.paths.sparse_dir)
    
    # Load chunks
    if not chunks_path.exists():
        click.echo(f"Chunks file not found: {chunks_path}", err=True)
        return
    
    chunks_df = pd.read_parquet(chunks_path)
    
    # Filter by company
    chunks_df = chunks_df[chunks_df["company"] == company]
    click.echo(f"Loaded {len(chunks_df)} chunks for {company}")
    
    if len(chunks_df) == 0:
        click.echo("No chunks found for this company.", err=True)
        return
    
    # Load indexes
    click.echo("\nLoading indexes...")
    faiss_index = FAISSIndex.load(index_path)
    
    bm25_index = None
    if cfg.bm25.enabled and mode in ["sparse", "hybrid"]:
        bm25_index = BM25Index.load(sparse_path)
    
    # Initialize reranker
    reranker_model = None
    if rerank and cfg.reranker.enabled:
        reranker_model = Reranker(
            model_name=cfg.reranker.model,
            device=device,
            batch_size=cfg.reranker.batch_size
        )
    
    # Generate query embedding
    embedding_gen = EmbeddingGenerator(
        model_name=cfg.embeddings.model,
        device=device,
        batch_size=1,
        normalize=cfg.embeddings.normalize
    )
    query_embedding = embedding_gen.embed_query(query)
    
    # Retrieve
    click.echo("\nRetrieving...")
    retriever = HybridRetriever(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        reranker=reranker_model,
        chunks_df=chunks_df,
        fusion_method=cfg.hybrid.fusion_method,
        rrf_k=cfg.hybrid.rrf_k,
        dense_weight=cfg.hybrid.dense_weight,
        sparse_weight=cfg.hybrid.sparse_weight,
        rerank_top_n=cfg.reranker.top_n
    )
    
    results_df = retriever.retrieve(
        query=query,
        query_embedding=query_embedding,
        mode=mode,
        k=top_k,
        rerank=rerank
    )
    
    # Display results
    click.echo(f"\n{'='*80}")
    click.echo(f"Top {len(results_df)} results:")
    click.echo(f"{'='*80}\n")
    
    for idx, row in results_df.iterrows():
        click.echo(f"[{idx + 1}] Score: {row['score']:.4f}")
        click.echo(f"    Source: {Path(row['source_path']).name}")
        click.echo(f"    Page: {row.get('page_start', 'N/A')}")
        click.echo(f"    Sentiment: {row.get('sentiment_label', 'N/A')} "
                   f"(pos: {row.get('p_positive', 0):.2f}, "
                   f"neu: {row.get('p_neutral', 0):.2f}, "
                   f"neg: {row.get('p_negative', 0):.2f})")
        click.echo(f"    Text: {row['text'][:200]}...")
        click.echo()


if __name__ == "__main__":
    cli()

