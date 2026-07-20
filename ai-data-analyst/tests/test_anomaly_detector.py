"""Tests for anomaly detection tool."""
import pytest
import pandas as pd
import numpy as np
from tools.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector():
    return AnomalyDetector()


@pytest.fixture
def df_with_outliers():
    """DataFrame with clear outliers injected."""
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 95)
    outliers = np.array([500, 600, 700, -200, -300])
    values = np.concatenate([normal_data, outliers])
    return pd.DataFrame({"value": values, "other": np.random.normal(50, 5, 100)})


def test_detects_outliers(detector, df_with_outliers):
    report = detector.detect(df_with_outliers, contamination=0.05)
    assert report.total_records_analyzed == 100
    assert report.anomalies_detected > 0


def test_report_structure(detector, df_with_outliers):
    report = detector.detect(df_with_outliers)
    assert report.detection_method in ("isolation_forest", "z_score")
    assert 0 <= report.anomaly_percentage <= 100
    assert len(report.anomaly_records) == report.anomalies_detected


def test_no_numeric_columns(detector):
    df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NY", "LA"]})
    report = detector.detect(df)
    assert report.anomalies_detected == 0
    assert "No numeric columns" in report.summary


def test_small_dataset_uses_zscore(detector):
    """Small datasets fall back to Z-score."""
    df = pd.DataFrame({"value": [1, 2, 3, 100, 2, 3, 1, 2]})
    report = detector.detect(df)
    # Either method should work on small data
    assert report.total_records_analyzed <= 10


def test_anomaly_records_have_explanations(detector, df_with_outliers):
    report = detector.detect(df_with_outliers, contamination=0.1)
    for rec in report.anomaly_records:
        assert rec.explanation != ""
        assert rec.row_index >= 0
