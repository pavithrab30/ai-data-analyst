"""Shared pytest fixtures for AI Data Analyst tests."""
import io
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_sales_df():
    """Returns a small sample sales DataFrame for testing."""
    return pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(1, 21)],
        "order_date": pd.date_range("2023-01-01", periods=20, freq="15D").strftime("%Y-%m-%d"),
        "customer_id": [f"Customer_{i:03d}" for i in range(1, 21)],
        "product": ["Widget Pro", "Gadget Plus"] * 10,
        "region": ["North", "South", "East", "West", "Central"] * 4,
        "revenue": np.random.uniform(100, 10000, 20).round(2),
        "cost": np.random.uniform(50, 5000, 20).round(2),
        "quantity": np.random.randint(1, 50, 20),
    })

@pytest.fixture
def sample_df_with_nulls():
    """DataFrame with known null values for validation testing."""
    df = pd.DataFrame({
        "id": range(100),
        "value": np.random.uniform(0, 100, 100),
        "category": ["A", "B", "C", "D"] * 25,
        "amount": np.random.uniform(1000, 50000, 100),
    })
    df.loc[df.sample(frac=0.1).index, "value"] = None
    df.loc[df.sample(frac=0.05).index, "category"] = None
    return df

@pytest.fixture
def sample_csv_bytes():
    """Returns CSV content as bytes for upload testing."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "value": [100, 200, 300],
        "category": ["A", "B", "A"],
    })
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()
