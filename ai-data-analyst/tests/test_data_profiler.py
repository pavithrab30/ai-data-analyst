"""Tests for the data profiler tool."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from tools.data_profiler import DataProfiler


@pytest.fixture
def profiler():
    return DataProfiler()


def test_profile_basic_structure(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df, "test_sales")
    assert profile.dataset_name == "test_sales"
    assert profile.row_count == len(sample_sales_df)
    assert profile.column_count == len(sample_sales_df.columns)
    assert len(profile.column_profiles) == len(sample_sales_df.columns)


def test_numeric_column_profiled(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df)
    rev_profile = next((c for c in profile.column_profiles if c.name == "revenue"), None)
    assert rev_profile is not None
    assert rev_profile.mean is not None
    assert rev_profile.min_value is not None
    assert rev_profile.max_value is not None


def test_categorical_column_profiled(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df)
    region_profile = next((c for c in profile.column_profiles if c.name == "region"), None)
    assert region_profile is not None
    assert region_profile.top_value is not None
    assert region_profile.value_counts is not None


def test_correlations_computed(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df)
    assert isinstance(profile.top_correlations, list)


def test_memory_usage_positive(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df)
    assert profile.memory_usage_bytes > 0


def test_schema_summary_output(profiler, sample_sales_df):
    profile = profiler.profile(sample_sales_df)
    summary = profiler.generate_schema_summary(profile)
    assert "test_sales" in summary or "dataset" in summary
    assert "rows" in summary.lower() or "shape" in summary.lower()
