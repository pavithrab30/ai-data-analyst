"""Tests for the data validator tool."""
import pytest
import pandas as pd
import numpy as np
from tools.data_validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


def test_clean_dataframe_high_score(validator):
    df = pd.DataFrame({
        "id": range(100),
        "value": np.random.uniform(0, 100, 100),
        "category": list("ABCD") * 25,
    })
    report = validator.validate(df)
    assert report.overall_quality_score >= 80
    assert report.duplicate_rows == 0


def test_duplicate_rows_detected(validator):
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})
    report = validator.validate(df)
    assert report.duplicate_rows == 2


def test_null_detection(validator, sample_df_with_nulls):
    report = validator.validate(sample_df_with_nulls)
    assert len(report.columns_with_nulls) > 0
    assert "value" in report.columns_with_nulls


def test_constant_column_detection(validator):
    df = pd.DataFrame({
        "id": range(10),
        "constant": ["SAME"] * 10,
        "value": range(10),
    })
    report = validator.validate(df)
    assert "constant" in report.constant_columns


def test_schema_compatibility_common_cols(validator):
    df1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "value": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "extra": [100, 200]})
    result = validator.validate_schema_compatibility(df1, df2)
    assert result["compatible"] is True
    assert "id" in result["common_columns"]


def test_memory_usage_reported(validator):
    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    report = validator.validate(df)
    assert report.memory_usage_bytes > 0
