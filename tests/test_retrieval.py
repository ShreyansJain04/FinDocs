"""Test retrieval and fusion functionality."""

import pytest
from src.fdocs.retrieval import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_fusion():
    """Test Reciprocal Rank Fusion."""
    # Two ranking lists
    results1 = ["doc1", "doc2", "doc3", "doc4"]
    results2 = ["doc3", "doc1", "doc5", "doc2"]
    
    scores1 = [0.9, 0.8, 0.7, 0.6]
    scores2 = [0.95, 0.85, 0.75, 0.65]
    
    fused_ids, fused_scores = reciprocal_rank_fusion(
        [results1, results2],
        [scores1, scores2],
        k=60
    )
    
    # Check we got results
    assert len(fused_ids) > 0
    assert len(fused_ids) == len(fused_scores)
    
    # doc1 and doc3 appear in both lists, should rank high
    assert "doc1" in fused_ids[:3]
    assert "doc3" in fused_ids[:3]


def test_weighted_fusion():
    """Test weighted score fusion."""
    results1 = ["doc1", "doc2", "doc3"]
    results2 = ["doc3", "doc4", "doc1"]
    
    scores1 = [0.9, 0.8, 0.7]
    scores2 = [0.95, 0.85, 0.75]
    
    fused_ids, fused_scores = weighted_score_fusion(
        [results1, results2],
        [scores1, scores2],
        weights=[0.6, 0.4]
    )
    
    assert len(fused_ids) > 0
    assert len(fused_ids) == len(fused_scores)
    
    # Scores should be normalized and weighted
    assert all(score >= 0 for score in fused_scores)


def test_rrf_single_list():
    """Test RRF with single list."""
    results = ["doc1", "doc2", "doc3"]
    scores = [0.9, 0.8, 0.7]
    
    fused_ids, fused_scores = reciprocal_rank_fusion(
        [results],
        [scores],
        k=60
    )
    
    # Should preserve order
    assert fused_ids == results
    assert len(fused_scores) == len(results)


def test_empty_results():
    """Test fusion with empty results."""
    fused_ids, fused_scores = reciprocal_rank_fusion(
        [[], []],
        [[], []],
        k=60
    )
    
    assert len(fused_ids) == 0
    assert len(fused_scores) == 0

