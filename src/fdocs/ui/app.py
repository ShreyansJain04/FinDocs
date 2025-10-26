"""Streamlit human-in-the-loop interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fdocs.config import Config, load_config
from fdocs.embed import EmbeddingGenerator
from fdocs.extract import ExtractionConfig, StructureExtractor
from fdocs.index import FAISSIndex
from fdocs.queue import QueueItem, ReviewQueue
from fdocs.retrieval import HybridRetriever
from fdocs.registry import IngestionRegistry
from fdocs.rerank import Reranker
from fdocs.schemas import ExtractedReport
from fdocs.sentiment import SentimentAnalyzer
from fdocs.sparse import BM25Index


def _load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _make_chunk_label(row: pd.Series) -> str:
    """Create a friendly chunk label from row data."""
    file_name = Path(row.get("source_path", "unknown")).stem  # Remove extension
    page_start = row.get("page_start", "?")
    page_end = row.get("page_end")
    
    # Build page range
    if page_end and page_end != page_start:
        page_str = f"p{page_start}-{page_end}"
    else:
        page_str = f"p{page_start}"
    
    # Get first meaningful words from text as preview
    text = row.get("text", "")
    words = text.split()[:8]  # First 8 words
    preview = " ".join(words)
    if len(text) > len(preview):
        preview += "..."
    
    return f"{file_name} {page_str}: {preview}"


def _load_report(path: Path) -> Optional[ExtractedReport]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ExtractedReport(**data)


def _load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _list_extraction_reports(extractions_dir: Path) -> Dict[str, Path]:
    if not extractions_dir.exists():
        return {}
    reports: Dict[str, Path] = {}
    for json_path in sorted(extractions_dir.glob("*.json")):
        label = json_path.stem
        reports[label] = json_path
    return reports


def _build_friendly_report_labels(
    reports: Dict[str, Path],
    registry_df: pd.DataFrame
) -> Dict[str, Path]:
    """Build friendly labels for extraction reports using registry data.
    
    Returns a dict mapping friendly labels to file paths.
    """
    friendly_mapping: Dict[str, Path] = {}
    
    for stem, path in reports.items():
        # Skip versioned files (those with timestamps)
        if "_202" in stem and len(stem.split("_")) > 2:
            continue
            
        # Try to extract company and document_id from stem
        parts = stem.split("_", 1)
        if len(parts) >= 2:
            company = parts[0]
            document_id = parts[1]
        else:
            # Fallback: just use stem
            friendly_mapping[stem] = path
            continue
        
        # Look up in registry
        if not registry_df.empty:
            match = registry_df[registry_df["sha256"] == document_id]
            if not match.empty:
                row = match.iloc[0]
                file_name = Path(row.get("source_path", "unknown")).stem  # Remove extension
                created_at = str(row.get("created_at", ""))[:10]  # YYYY-MM-DD
                friendly_label = f"{company} - {file_name} ({created_at})"
                friendly_mapping[friendly_label] = path
                continue
        
        # Fallback: skip if hash is too long (likely incomplete extraction)
        if len(document_id) > 16:
            continue
        
        friendly_mapping[f"{company} - {document_id}"] = path
    
    return friendly_mapping


@st.cache_resource(show_spinner=False)
def _cached_embedding_generator(model_name: str, batch_size: int, normalize: bool, device: str) -> EmbeddingGenerator:
    return EmbeddingGenerator(model_name=model_name, batch_size=batch_size, normalize=normalize, device=device)


@st.cache_resource(show_spinner=False)
def _cached_faiss_index(index_dir: Path) -> Optional[FAISSIndex]:
    if not index_dir.exists() or not (index_dir / "index.faiss").exists():
        return None
    return FAISSIndex.load(index_dir)


@st.cache_resource(show_spinner=False)
def _cached_bm25_index(sparse_dir: Path) -> Optional[BM25Index]:
    if not sparse_dir.exists() or not (sparse_dir / "bm25.pkl").exists():
        return None
    return BM25Index.load(sparse_dir)


@st.cache_resource(show_spinner=False)
def _cached_reranker(model: str, batch_size: int, device: str) -> Optional[Reranker]:
    try:
        return Reranker(model_name=model, device=device, batch_size=batch_size)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_extractor(cfg: ExtractionConfig) -> Optional[StructureExtractor]:
    try:
        return StructureExtractor(cfg)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_sentiment(model: str, batch_size: int, max_length: int, device: str) -> SentimentAnalyzer:
    return SentimentAnalyzer(model_name=model, batch_size=batch_size, max_length=max_length, device=device)


def _refresh_sentiment(chunks: pd.DataFrame, analyzer: SentimentAnalyzer) -> pd.DataFrame:
    texts = chunks["text"].tolist()
    labels = analyzer.analyze_batch(texts)
    columns = ["sentiment_label", "p_positive", "p_neutral", "p_negative"]
    chunks.loc[:, columns] = labels
    return chunks


def render_document_viewer(chunks_df: pd.DataFrame, selected_company: str):
    st.subheader("Document Viewer")
    if chunks_df.empty:
        st.info("No chunks available. Run ingestion first.")
        return
    
    # Filter to selected company
    company_df = chunks_df[chunks_df["company"] == selected_company].copy()
    
    # Sort by source_path, page_start, chunk_idx
    sort_cols = []
    if "source_path" in company_df.columns:
        sort_cols.append("source_path")
    if "page_start" in company_df.columns:
        sort_cols.append("page_start")
    if "chunk_idx" in company_df.columns:
        sort_cols.append("chunk_idx")
    if sort_cols:
        company_df = company_df.sort_values(sort_cols)
    
    # Chunk filter
    filter_text = st.text_input("Filter chunks (by file, section, or text)", key="chunk_filter")
    if filter_text:
        filter_lower = filter_text.lower()
        company_df = company_df[
            company_df.apply(
                lambda r: (
                    filter_lower in str(r.get("source_path", "")).lower()
                    or filter_lower in str(r.get("section", "")).lower()
                    or filter_lower in str(r.get("text", "")).lower()
                ),
                axis=1
            )
        ]
    
    active = st.session_state.get("active_chunk")
    for _, row in company_df.iterrows():
        sentiment = row.get("sentiment_label") or "unknown"
        confidence = max(
            [value for value in (row.get("p_positive"), row.get("p_neutral"), row.get("p_negative")) if value is not None],
            default=0.0,
        )
        label = _make_chunk_label(row)
        header = f"{label} — {sentiment} ({confidence:.2f})"
        expanded = active == row["chunk_id"]
        with st.expander(header, expanded=expanded):
            st.write(row.get("text", ""))
            new_label = st.selectbox(
                "Sentiment Label",
                options=["positive", "neutral", "negative", "mixed"],
                index=["positive", "neutral", "negative", "mixed"].index(sentiment) if sentiment in {"positive", "neutral", "negative", "mixed"} else 1,
                key=f"sentiment_{row['chunk_id']}",
            )
            if new_label != sentiment:
                st.session_state.setdefault("sentiment_updates", {})[row["chunk_id"]] = new_label
            notes_key = f"notes_{row['chunk_id']}"
            notes = st.text_area("Reviewer Notes", key=notes_key, height=80)
            if st.button("Add to Review Queue", key=f"mark_{row['chunk_id']}"):
                st.session_state["active_chunk"] = row["chunk_id"]
                queue = st.session_state.setdefault("queued_items", [])
                queue.append(
                    QueueItem(
                        document_id=row.get("company", "unknown"),
                        chunk_id=row["chunk_id"],
                        field_name="sentiment",
                        uncertainty_score=confidence,
                        priority=confidence,
                        metadata={
                            "notes": notes.strip() if notes else None,
                            "sentiment": new_label,
                            "source_path": row.get("source_path"),
                        },
                    )
                )
                st.success("Queued for manual review")


def render_extraction_editor(report: Optional[ExtractedReport]):
    st.subheader("Extraction Editor")
    if report is None:
        st.info("No extraction report available")
        return
    st.caption(
        f"Company: {report.company or 'n/a'} | Document ID: {report.document_id or 'n/a'} | Version: {report.extraction_version or 'n/a'}"
    )
    
    sections = {
        "Guidance": report.guidance,
        "Risk Factors": report.risk_factors,
        "Financial Metrics": report.financial_metrics,
        "Management Tone": report.management_tone,
        "Analyst Concerns": report.analyst_concerns,
    }
    
    # Check if all sections are empty
    all_empty = all(len(items) == 0 for items in sections.values())
    
    if all_empty and report.raw_sections:
        st.warning("No structured items extracted. Showing raw section content below.")
        with st.expander("Raw Sections", expanded=True):
            for section_name, section_content in report.raw_sections.items():
                st.markdown(f"**{section_name}**")
                if isinstance(section_content, dict):
                    st.text_area(
                        f"Content",
                        value=section_content.get("raw_text", str(section_content)),
                        height=150,
                        key=f"raw_{section_name}",
                        disabled=True
                    )
                else:
                    st.text_area(
                        f"Content",
                        value=str(section_content),
                        height=150,
                        key=f"raw_{section_name}",
                        disabled=True
                    )
        
        if st.button("Regenerate items (local)", key="regenerate_local"):
            with st.spinner("Regenerating items using local extraction..."):
                try:
                    from fdocs.extract import ExtractionConfig, StructureExtractor
                    local_cfg = ExtractionConfig(
                        mode="local",
                        extraction_version=report.extraction_version or "v1"
                    )
                    extractor = StructureExtractor(local_cfg)
                    
                    # Convert raw_sections to the format expected by extract
                    sections_dict = {}
                    for section_name, section_content in report.raw_sections.items():
                        if isinstance(section_content, dict):
                            text = section_content.get("raw_text", str(section_content))
                        else:
                            text = str(section_content)
                        sections_dict[section_name] = [text]
                    
                    regenerated_report = extractor._extract_local(
                        company=report.company or "unknown",
                        document_id=report.document_id or "unknown",
                        document_path=report.document_path or "",
                        sections=sections_dict,
                        metadata=report.metadata
                    )
                    
                    # Update current report in session
                    st.session_state["regenerated_report"] = regenerated_report
                    st.success("Items regenerated! Reload to see changes.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Regeneration failed: {exc}")
    
    tabs = st.tabs(list(sections.keys()))
    for tab, (name, items) in zip(tabs, sections.items()):
        with tab:
            items_data = [item.model_dump() for item in items]
            if items_data:
                df = pd.DataFrame(items_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No items extracted for this section")
            edit_key = f"edit_{name}"
            edited_json = st.text_area("JSON Editor", value=json.dumps(items_data, indent=2), height=200, key=edit_key)
            if st.button(f"Save {name}", key=f"save_{name}"):
                try:
                    parsed = json.loads(edited_json or "[]")
                    st.session_state.setdefault("extraction_updates", {})[name] = parsed
                    st.success("Updates captured in session state")
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")
    
    # Persist controls at bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Apply & Save Report", key="apply_save_report"):
            try:
                # Apply staged updates from session
                updates = st.session_state.get("extraction_updates", {})
                if updates:
                    from fdocs.schemas import GuidanceItem, RiskFactor, FinancialMetric, ManagementTone, AnalystConcern
                    section_map = {
                        "Guidance": ("guidance", GuidanceItem),
                        "Risk Factors": ("risk_factors", RiskFactor),
                        "Financial Metrics": ("financial_metrics", FinancialMetric),
                        "Management Tone": ("management_tone", ManagementTone),
                        "Analyst Concerns": ("analyst_concerns", AnalystConcern),
                    }
                    for section_name, items_json in updates.items():
                        if section_name in section_map:
                            field_name, model_class = section_map[section_name]
                            setattr(report, field_name, [model_class(**item) for item in items_json])
                
                # Save to disk
                output_file = Path("artifacts/extractions") / f"{report.company}_{report.document_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                st.success(f"Report saved to {output_file}")
                st.session_state.pop("extraction_updates", None)
            except Exception as exc:
                st.error(f"Save failed: {exc}")
    
    with col2:
        # Download button
        report_json = report.model_dump_json(indent=2)
        st.download_button(
            label="Download JSON",
            data=report_json,
            file_name=f"{report.company}_{report.document_id}.json",
            mime="application/json",
            key="download_report"
        )
    
    with col3:
        if st.button("Save as new version", key="save_version"):
            try:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path("artifacts/extractions") / f"{report.company}_{report.document_id}_{timestamp}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                st.success(f"New version saved: {output_file.name}")
            except Exception as exc:
                st.error(f"Save failed: {exc}")


def _build_hybrid_retriever(
    cfg: Config,
    chunks_df: pd.DataFrame,
    device: str,
) -> Optional[HybridRetriever]:
    faiss_path = Path(cfg.paths.index_dir)
    faiss_index = _cached_faiss_index(faiss_path)
    bm25_index = _cached_bm25_index(Path(cfg.paths.sparse_dir)) if cfg.bm25.enabled else None
    reranker = _cached_reranker(cfg.reranker.model, cfg.reranker.batch_size, device) if cfg.reranker.enabled else None
    if faiss_index is None:
        return None
    extractor_cfg = cfg.extraction or {}
    extractor = None
    if extractor_cfg and (extractor_cfg.get("mode", "remote").lower() == "remote"):
        extractor = _cached_extractor(
            ExtractionConfig(
                primary_endpoint=extractor_cfg.get("primary_endpoint"),
                fallback_endpoint=extractor_cfg.get("fallback_endpoint"),
                max_retries=extractor_cfg.get("max_retries", 2),
                extraction_version=extractor_cfg.get("extraction_version", "v1"),
                mode="remote",
            )
        )
    return HybridRetriever(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        reranker=reranker,
        chunks_df=chunks_df,
        fusion_method=cfg.hybrid.fusion_method,
        rrf_k=cfg.hybrid.rrf_k,
        dense_weight=cfg.hybrid.dense_weight,
        sparse_weight=cfg.hybrid.sparse_weight,
        rerank_top_n=cfg.reranker.top_n,
        extractor=extractor,
    )


def render_rag_chat(
    chunks_df: pd.DataFrame,
    retriever: Optional[HybridRetriever],
    embedder: EmbeddingGenerator,
    selected_company: str,
) -> None:
    st.subheader("RAG Chat")
    
    if chunks_df.empty:
        st.info("No chunks available. Run ingestion first.")
        return
    
    if retriever is None:
        st.warning("Retriever not available. Ingest data first.")
        return
    
    query = st.text_area("Ask a question", placeholder="e.g. What guidance did management provide?", height=80, key="rag_query")
    
    if not query:
        st.info("Enter a question above to search.")
        return
    
    # Retrieve
    query_embedding = embedder.embed_batch([query])[0]
    results = retriever.retrieve(query, query_embedding, mode="hybrid", k=20, rerank=True)
    
    # Always scope to selected company
    results = results[results["company"] == selected_company]
    results = results.head(5)
    
    if results.empty:
        st.info(f"No supporting chunks found for {selected_company}.")
        return
    
    # Build citations
    st.markdown("### Answer")
    contexts: List[str] = []
    citations: List[Dict[str, Any]] = []
    
    for idx, (_, row) in enumerate(results.iterrows(), start=1):
        contexts.append(row['text'])
        file_name = Path(row.get("source_path", "unknown")).stem
        page = row.get("page_start", "?")
        citations.append({
            "num": idx,
            "file": file_name,
            "page": page,
            "text": row['text']
        })
    
    # Build a simple prompt for synthesis
    context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""Based on the following context from {selected_company} documents, answer the question concisely.

Question: {query}

Context:
{context_text}

Answer (2-4 sentences with citations [1], [2], etc):"""
    
    # Try to use local LLM or fallback to extractive
    try:
        # Check if there's an extractor available (has LLM access)
        extractor_cfg = st.session_state.get("extractor_config")
        if extractor_cfg:
            extractor = _cached_extractor(extractor_cfg)
            if extractor and extractor.mode == "remote" and extractor.primary:
                # Use LLM to generate answer
                with st.spinner("Generating answer..."):
                    llm_response = extractor.primary.generate(prompt, temperature=0.3, max_tokens=300)
                    st.write(llm_response)
            else:
                # Fallback: extractive
                _render_extractive_answer(contexts, citations)
        else:
            _render_extractive_answer(contexts, citations)
    except Exception as e:
        st.warning(f"LLM generation failed: {e}. Showing extractive summary.")
        _render_extractive_answer(contexts, citations)
    
    # Show citations
    st.markdown("### Citations")
    for cite in citations:
        st.caption(f"[{cite['num']}] {cite['file']} (p.{cite['page']})")
        with st.expander(f"View full context"):
            st.write(cite['text'])


def _render_extractive_answer(contexts: List[str], citations: List[Dict[str, Any]]):
    """Fallback extractive answer when LLM unavailable."""
    # Take key sentences from top contexts
    answer_parts = []
    for i, ctx in enumerate(contexts[:3], start=1):
        sentences = ctx.split(". ")
        if sentences:
            answer_parts.append(f"{sentences[0].strip()} [{i}]")
    
    answer = ". ".join(answer_parts)
    if not answer.endswith("."):
        answer += "."
    st.write(answer)


def render_queue(queue: ReviewQueue):
    st.subheader("Review Queue")
    batch = queue.get_next_batch()
    if batch.empty:
        st.info("Queue empty")
        return
    st.dataframe(batch)
    if st.button("Mark Reviewed"):
        queue.mark_reviewed(batch.index)
        st.success("Marked selected items as reviewed")
    queued = st.session_state.get("queued_items", [])
    if queued:
        st.caption(f"Queued this session: {len(queued)} chunks")
        if st.button("Commit Session Queue"):
            queue.add_items(queued)
            st.session_state.pop("queued_items", None)
            st.success("Session queue committed")


def main() -> None:
    st.set_page_config(layout="wide", page_title="FinDocs HITL")
    st.title("FinDocs Review Console")

    cfg = load_config()
    device = "cpu"
    chunks_df = _load_chunks(Path(cfg.paths.chunks_dir) / "chunks.parquet")
    registry_df = _load_registry(Path(cfg.paths.registry_dir) / "ingestion_registry.parquet")
    queue = ReviewQueue()

    # Global company selector at top of sidebar
    st.sidebar.title("Settings")
    companies = sorted(chunks_df["company"].unique()) if not chunks_df.empty else []
    if companies:
        if len(companies) <= 6:
            selected_company = st.sidebar.radio("📊 Company", companies, key="global_company")
        else:
            selected_company = st.sidebar.selectbox("📊 Company", companies, key="global_company")
    else:
        st.sidebar.warning("No companies found. Run ingestion first.")
        selected_company = None
    
    st.sidebar.markdown("---")
    
    queue_stats = queue.stats()
    st.sidebar.metric("Pending Reviews", queue_stats["pending"])
    st.sidebar.metric("Average Uncertainty", f"{queue_stats['average_uncertainty']:.2f}")
    st.sidebar.metric("Total Documents", len(registry_df))
    st.sidebar.metric("Queued (session)", len(st.session_state.get("queued_items", [])))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 LLM Settings")
    llm_backend = st.sidebar.radio("Backend", ["Ollama", "HTTP"], key="llm_backend")
    if llm_backend == "Ollama":
        ollama_model = st.sidebar.text_input("Model", value="llama3.1", key="ollama_model")
        if st.sidebar.button("Apply LLM", key="apply_llm_ollama"):
            st.session_state["extractor_config"] = ExtractionConfig(
                primary_endpoint=f"ollama:{ollama_model}",
                fallback_endpoint=None,
                max_retries=2,
                extraction_version="v1",
                mode="remote",
            )
            st.sidebar.success(f"Using Ollama model: {ollama_model}")
    else:
        http_endpoint = st.sidebar.text_input("HTTP Endpoint", value="http://localhost:8000", key="http_llm_endpoint")
        if st.sidebar.button("Apply LLM", key="apply_llm_http"):
            st.session_state["extractor_config"] = ExtractionConfig(
                primary_endpoint=http_endpoint,
                fallback_endpoint=None,
                max_retries=2,
                extraction_version="v1",
                mode="remote",
            )
            st.sidebar.success(f"Using HTTP endpoint: {http_endpoint}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Extraction Reports")
    
    extraction_reports = _list_extraction_reports(Path(cfg.paths.artifacts_root) / "extractions")
    if extraction_reports:
        # Filter reports by selected company
        if selected_company:
            extraction_reports = {
                stem: path for stem, path in extraction_reports.items() if stem.startswith(f"{selected_company}_")
            }
        friendly_reports = _build_friendly_report_labels(extraction_reports, registry_df)
        
        # Add search filter if many reports
        if len(friendly_reports) > 5:
            search_filter = st.sidebar.text_input("🔍 Search", key="report_search", placeholder="Filter reports...")
            if search_filter:
                search_lower = search_filter.lower()
                friendly_reports = {
                    label: path for label, path in friendly_reports.items()
                    if search_lower in label.lower()
                }
        
        if friendly_reports:
            labels = sorted(list(friendly_reports.keys()))
            if len(labels) <= 6:
                selected_label = st.sidebar.radio("Select report", labels, key="report_selector")
            else:
                selected_label = st.sidebar.selectbox("Select report", labels, key="report_selector")
            
            report_path = friendly_reports.get(selected_label)
            report = _load_report(report_path) if report_path else None
            
            # Check if we have a regenerated report in session
            if st.session_state.get("regenerated_report"):
                report = st.session_state["regenerated_report"]
        else:
            st.sidebar.info("No reports match search")
            report = None
    else:
        st.sidebar.info("No extraction reports found")
        report = None

    sentiment_analyzer = _cached_sentiment(
        cfg.sentiment.model,
        cfg.sentiment.batch_size,
        cfg.sentiment.max_length,
        device,
    )
    if st.sidebar.button("Refresh Sentiment") and not chunks_df.empty:
        with st.spinner("Recomputing sentiment..."):
            chunks_df = _refresh_sentiment(chunks_df, sentiment_analyzer)
            chunks_path = Path(cfg.paths.chunks_dir) / "chunks.parquet"
            chunks_df.to_parquet(chunks_path, index=False)
            st.success("Sentiment updated")

    # Store extractor config in session for RAG
    if cfg.extraction:
        st.session_state["extractor_config"] = ExtractionConfig(
            primary_endpoint=cfg.extraction.get("primary_endpoint"),
            fallback_endpoint=cfg.extraction.get("fallback_endpoint"),
            max_retries=cfg.extraction.get("max_retries", 2),
            extraction_version=cfg.extraction.get("extraction_version", "v1"),
            mode=cfg.extraction.get("mode", "remote"),
        )
    
    retriever = _build_hybrid_retriever(cfg, chunks_df, device)
    embedder = _cached_embedding_generator(
        cfg.embeddings.model,
        cfg.embeddings.batch_size,
        cfg.embeddings.normalize,
        device,
    )

    if not chunks_df.empty:
        chunks_df = chunks_df.copy()
        if {"sentiment_label", "p_positive", "p_neutral", "p_negative"} - set(chunks_df.columns):
            chunks_df = _refresh_sentiment(chunks_df, sentiment_analyzer)

    # Only show content if company is selected
    if selected_company:
        col1, col2, col3 = st.columns([1.5, 1.2, 1])
        with col1:
            render_document_viewer(chunks_df, selected_company)
        with col2:
            render_extraction_editor(report)
        with col3:
            render_rag_chat(chunks_df, retriever, embedder, selected_company)

        render_queue(queue)
    else:
        st.info("👆 Select a company from the sidebar to begin.")


if __name__ == "__main__":
    main()
