"""Tests for multi-CSV schema merger tool."""
import pytest
import pandas as pd
from tools.schema_merger import SchemaMerger


@pytest.fixture
def merger():
    return SchemaMerger()


@pytest.fixture
def compatible_dfs():
    df1 = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "revenue": [1000, 2000, 1500],
        "region": ["North", "South", "East"],
    })
    df2 = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "segment": ["Enterprise", "SMB", "Mid-Market"],
        "country": ["USA", "UK", "Germany"],
    })
    return df1, df2


def test_recommend_merge_finds_key(merger, compatible_dfs):
    df1, df2 = compatible_dfs
    recs = merger.recommend_merge({"sales": df1, "customers": df2})
    assert len(recs) > 0
    assert "customer_id" in recs[0].join_keys


def test_no_recommendation_for_single_df(merger, compatible_dfs):
    df1, _ = compatible_dfs
    recs = merger.recommend_merge({"only": df1})
    assert recs == []


def test_execute_merge_produces_correct_shape(merger, compatible_dfs):
    df1, df2 = compatible_dfs
    recs = merger.recommend_merge({"sales": df1, "customers": df2})
    assert recs
    merged = merger.execute_merge({"sales": df1, "customers": df2}, recs[0])
    assert len(merged) == 3
    assert "revenue" in merged.columns
    assert "segment" in merged.columns


def test_execute_merge_missing_key_raises(merger, compatible_dfs):
    df1, df2 = compatible_dfs
    from tools.schema_merger import MergeRecommendation
    bad_rec = MergeRecommendation("sales", "customers", ["nonexistent_key"], "inner", 0.5, "test")
    with pytest.raises(ValueError):
        merger.execute_merge({"sales": df1, "customers": df2}, bad_rec)
