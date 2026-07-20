"""Tests for the secure execution service (AST validation)."""
import pytest
import pandas as pd
import numpy as np
from services.execution_service import (
    ExecutionService,
    SecurityViolationError,
    ASTValidator,
)


@pytest.fixture
def service():
    return ExecutionService()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East"],
        "revenue": [1000.0, 2000.0, 1500.0],
        "quantity": [10, 20, 15],
    })


def test_safe_aggregation(service, sample_df):
    code = "result = df['revenue'].sum()"
    out = service.execute_pandas_code(code, sample_df)
    assert out["success"] is True
    assert out["result"] == pytest.approx(4500.0)


def test_safe_groupby(service, sample_df):
    code = "result = df.groupby('region')['revenue'].sum().reset_index()"
    out = service.execute_pandas_code(code, sample_df)
    assert out["success"] is True
    assert isinstance(out["result"], pd.DataFrame)


def test_import_blocked(service, sample_df):
    code = "import os\nresult = os.getcwd()"
    with pytest.raises((SecurityViolationError, SyntaxError)):
        service.execute_pandas_code(code, sample_df)


def test_exec_blocked(service, sample_df):
    code = "exec('result = 1')"
    with pytest.raises(SecurityViolationError):
        service.execute_pandas_code(code, sample_df)


def test_open_blocked(service, sample_df):
    code = "result = open('/etc/passwd').read()"
    with pytest.raises(SecurityViolationError):
        service.execute_pandas_code(code, sample_df)


def test_syntax_error_raised(service, sample_df):
    code = "result = df['revenue'].sum( =="
    with pytest.raises(SyntaxError):
        service.execute_pandas_code(code, sample_df)


def test_result_none_on_no_assignment(service, sample_df):
    code = "x = df['revenue'].mean()"
    out = service.execute_pandas_code(code, sample_df)
    assert out["success"] is True
    assert out["result"] is None  # No 'result' variable assigned


def test_original_df_not_modified(service, sample_df):
    original_len = len(sample_df)
    code = "df = df.head(1)\nresult = df"
    service.execute_pandas_code(code, sample_df)
    assert len(sample_df) == original_len  # Original unchanged
