"""Tests for the CSV loader tool."""
import io
import pytest
import pandas as pd
from tools.csv_loader import CSVLoader, CSVLoadError


@pytest.fixture
def loader():
    return CSVLoader()


def test_load_valid_csv_bytes(loader):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    result_df, name = loader.load_from_bytes(buf.getvalue(), "test.csv")
    assert len(result_df) == 3
    assert "a" in result_df.columns
    assert name == "test.csv"


def test_load_invalid_extension(loader):
    with pytest.raises(CSVLoadError, match="not supported"):
        loader._validate_extension("data.xlsx")


def test_load_empty_csv_raises(loader):
    empty_csv = b"a,b,c\n"
    with pytest.raises(CSVLoadError, match="empty"):
        loader.load_from_bytes(empty_csv, "empty.csv")


def test_column_sanitization(loader):
    csv_content = b"First Name,Last Name,Total $,123col\n Alice,Smith,1000,1\n"
    df, _ = loader.load_from_bytes(csv_content, "test.csv")
    for col in df.columns:
        assert col[0].isalpha() or col[0] == "_"
        assert " " not in col


def test_file_size_validation(loader):
    """Test that files exceeding the size limit raise CSVLoadError."""
    loader._max_file_size_bytes = 20  # Set tiny limit
    # Build content that exceeds the limit
    content = b"a,b,c\n1,2,3\n" * 100  # ~1200 bytes, well over 20

    with pytest.raises(CSVLoadError, match="exceeds"):
        loader.load_from_bytes(content, "big.csv")

    # Reset to default
    loader._max_file_size_bytes = 50 * 1024 * 1024


def test_dtype_inference(loader):
    csv_content = b"id,amount,date\n1,100.50,2023-01-01\n2,200.00,2023-02-01\n"
    df, _ = loader.load_from_bytes(csv_content, "test.csv")
    assert pd.api.types.is_numeric_dtype(df["amount"])
