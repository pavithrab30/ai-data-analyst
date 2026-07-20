"""
Anomaly detection tool.

Identifies statistical outliers in datasets using a two-tier approach:
1. Primary: Isolation Forest — effective for multivariate anomalies
2. Fallback: Z-score per column — simpler, interpretable, always available

For each anomaly, the tool:
- Records the anomaly score
- Computes Z-scores for all numeric columns in that record
- Identifies which columns contributed most to the anomaly
- Generates a human-readable explanation

The full report is returned as an AnomalyReport model.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from models.analysis_models import AnomalyRecord, AnomalyReport, ConfidenceLevel
from services.llm_service import LLMService
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

_ANOMALY_SUMMARY_PROMPT = """You are a data analyst. The following anomalies were detected in a dataset.
Provide a concise business-friendly summary explaining what was found and potential causes.

Dataset columns analyzed: {columns}
Total records: {total_records}
Anomalies found: {anomaly_count} ({anomaly_pct:.1f}%)
Detection method: {method}

Top anomalous records:
{sample_records}

Write a 3-5 sentence summary suitable for a business report.
Focus on: what was unusual, which columns showed the most deviation, and potential business implications."""


class AnomalyDetector:
    """
    Detects and explains outliers in tabular datasets.

    Uses sklearn's IsolationForest as the primary detector with
    Z-score analysis as a fallback when sklearn is unavailable
    or the dataset is too small for IsolationForest.

    Minimum rows for IsolationForest: 20
    """

    MIN_ROWS_FOR_ISOLATION_FOREST = 20
    Z_SCORE_THRESHOLD = 3.0  # Standard deviations from mean
    MAX_ANOMALIES_TO_EXPLAIN = 20

    def __init__(self, llm_service_arg: Optional[LLMService] = None) -> None:
        self._llm = llm_service_arg or LLMService()

    def detect(
        self,
        df: pd.DataFrame,
        contamination: float = 0.05,
        columns: Optional[list[str]] = None,
    ) -> AnomalyReport:
        """
        Run anomaly detection on the DataFrame.

        Args:
            df: DataFrame to analyze.
            contamination: Expected proportion of anomalies (0.01 to 0.5).
            columns: Specific numeric columns to analyze. Defaults to all numeric.

        Returns:
            AnomalyReport with detected anomalies and business summary.
        """
        logger.info(
            "Anomaly detection started",
            rows=len(df),
            contamination=contamination,
        )

        # Limit dataset size for performance
        if len(df) > settings.max_anomaly_samples:
            df = df.sample(n=settings.max_anomaly_samples, random_state=42)
            logger.info("Dataset sampled for anomaly detection", sample_size=settings.max_anomaly_samples)

        # Select numeric columns
        if columns:
            numeric_df = df[columns].select_dtypes(include=["number"])
        else:
            numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.empty or len(numeric_df.columns) == 0:
            return AnomalyReport(
                total_records_analyzed=len(df),
                anomalies_detected=0,
                anomaly_percentage=0.0,
                detection_method="none",
                summary="No numeric columns found for anomaly detection.",
                columns_analyzed=[],
            )

        columns_analyzed = list(numeric_df.columns)

        # Fill nulls with median for detection (don't modify original df)
        clean_numeric = numeric_df.copy()
        for col in clean_numeric.columns:
            clean_numeric[col] = clean_numeric[col].fillna(clean_numeric[col].median())

        # Choose detection method
        if len(df) >= self.MIN_ROWS_FOR_ISOLATION_FOREST:
            anomaly_labels, anomaly_scores, method = self._isolation_forest(
                clean_numeric, contamination
            )
        else:
            anomaly_labels, anomaly_scores, method = self._z_score_detection(
                clean_numeric
            )

        # Compute Z-scores for all numeric columns
        z_score_matrix = self._compute_z_scores(clean_numeric)

        # Build anomaly records
        anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
        anomaly_records = self._build_anomaly_records(
            df, anomaly_indices, anomaly_scores, z_score_matrix, columns_analyzed
        )

        # Generate summary
        summary = self._generate_summary(
            total_records=len(df),
            anomaly_count=len(anomaly_records),
            method=method,
            columns=columns_analyzed,
            anomaly_records=anomaly_records,
        )

        report = AnomalyReport(
            total_records_analyzed=len(df),
            anomalies_detected=len(anomaly_records),
            anomaly_percentage=round(len(anomaly_records) / max(len(df), 1) * 100, 2),
            detection_method=method,
            contamination_rate=contamination,
            anomaly_records=anomaly_records,
            summary=summary,
            columns_analyzed=columns_analyzed,
        )

        logger.info(
            "Anomaly detection complete",
            method=method,
            anomalies=len(anomaly_records),
            pct=report.anomaly_percentage,
        )

        return report

    # ── Detection Methods ──────────────────────────────────────────────────────

    def _isolation_forest(
        self,
        numeric_df: pd.DataFrame,
        contamination: float,
    ) -> tuple[list[int], list[float], str]:
        """
        Run Isolation Forest anomaly detection.

        Returns:
            Tuple of (labels, scores, method_name)
            labels: -1 for anomaly, 1 for normal
            scores: anomaly score per row (lower = more anomalous)
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df.values)

            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            labels = clf.fit_predict(scaled).tolist()
            scores = clf.score_samples(scaled).tolist()

            return labels, scores, "isolation_forest"

        except ImportError:
            logger.warning("scikit-learn not available, falling back to Z-score")
            return self._z_score_detection(numeric_df)
        except Exception as exc:
            logger.warning("Isolation Forest failed, falling back to Z-score", error=str(exc))
            return self._z_score_detection(numeric_df)

    def _z_score_detection(
        self,
        numeric_df: pd.DataFrame,
    ) -> tuple[list[int], list[float], str]:
        """
        Fallback anomaly detection using Z-scores.

        A row is flagged as anomalous if ANY column has |z-score| > threshold.

        Returns:
            Tuple of (labels, scores, method_name)
        """
        z_scores = np.abs(stats.zscore(numeric_df.values, nan_policy="omit"))
        # Max Z-score per row as the anomaly score (negated to match IF convention)
        row_max_z = np.nanmax(z_scores, axis=1)
        labels = [-1 if z > self.Z_SCORE_THRESHOLD else 1 for z in row_max_z]
        scores = (-row_max_z).tolist()

        return labels, scores, "z_score"

    # ── Support Methods ────────────────────────────────────────────────────────

    def _compute_z_scores(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Z-scores for all columns, returning a DataFrame of the same shape."""
        try:
            z_array = stats.zscore(numeric_df.values, nan_policy="omit")
            return pd.DataFrame(z_array, columns=numeric_df.columns, index=numeric_df.index)
        except Exception:
            return pd.DataFrame(
                np.zeros_like(numeric_df.values),
                columns=numeric_df.columns,
                index=numeric_df.index,
            )

    def _build_anomaly_records(
        self,
        original_df: pd.DataFrame,
        anomaly_indices: list[int],
        anomaly_scores: list[float],
        z_score_matrix: pd.DataFrame,
        columns_analyzed: list[str],
    ) -> list[AnomalyRecord]:
        """Build detailed AnomalyRecord objects for each detected anomaly."""
        records: list[AnomalyRecord] = []

        for idx in anomaly_indices[: self.MAX_ANOMALIES_TO_EXPLAIN]:
            if idx >= len(original_df):
                continue

            row = original_df.iloc[idx]
            score = anomaly_scores[idx] if idx < len(anomaly_scores) else 0.0

            # Z-scores for this row's numeric columns
            z_scores_row: dict[str, float] = {}
            flagged_cols: list[str] = []

            if idx < len(z_score_matrix):
                for col in columns_analyzed:
                    if col in z_score_matrix.columns:
                        z = float(z_score_matrix.iloc[idx][col])
                        if not np.isnan(z):
                            z_scores_row[col] = round(z, 3)
                            if abs(z) > self.Z_SCORE_THRESHOLD:
                                flagged_cols.append(col)

            # Sort flagged columns by abs Z-score
            flagged_cols.sort(
                key=lambda c: abs(z_scores_row.get(c, 0)), reverse=True
            )

            # Generate explanation
            explanation = self._explain_anomaly(
                row_data={k: _safe_val(v) for k, v in row.items()},
                z_scores=z_scores_row,
                flagged_columns=flagged_cols,
                score=score,
            )

            records.append(
                AnomalyRecord(
                    row_index=int(original_df.index[idx]),
                    anomaly_score=round(score, 4),
                    z_scores=z_scores_row,
                    flagged_columns=flagged_cols,
                    record_data={k: _safe_val(v) for k, v in row.items()},
                    explanation=explanation,
                )
            )

        return records

    def _explain_anomaly(
        self,
        row_data: dict,
        z_scores: dict[str, float],
        flagged_columns: list[str],
        score: float,
    ) -> str:
        """Generate a concise human-readable explanation for a single anomaly."""
        if not flagged_columns:
            return "This record has an unusual combination of values across multiple columns."

        top_cols = flagged_columns[:3]
        parts: list[str] = []

        for col in top_cols:
            z = z_scores.get(col, 0)
            val = row_data.get(col, "N/A")
            direction = "above" if z > 0 else "below"
            parts.append(
                f"'{col}' is {abs(z):.1f} standard deviations {direction} the mean (value: {val})"
            )

        return "Anomaly detected: " + "; ".join(parts) + "."

    def _generate_summary(
        self,
        total_records: int,
        anomaly_count: int,
        method: str,
        columns: list[str],
        anomaly_records: list[AnomalyRecord],
    ) -> str:
        """Generate a business-friendly summary using the LLM."""
        if anomaly_count == 0:
            return "No anomalies were detected in the dataset. All records appear to fall within expected statistical ranges."

        # Build sample records text
        sample_lines: list[str] = []
        for rec in anomaly_records[:5]:
            flagged = ", ".join(
                f"{c}={rec.record_data.get(c, 'N/A')}" for c in rec.flagged_columns[:3]
            )
            sample_lines.append(
                f"  Row {rec.row_index}: {flagged} (score: {rec.anomaly_score:.3f})"
            )
        sample_text = "\n".join(sample_lines)

        prompt = _ANOMALY_SUMMARY_PROMPT.format(
            columns=", ".join(columns),
            total_records=total_records,
            anomaly_count=anomaly_count,
            anomaly_pct=anomaly_count / max(total_records, 1) * 100,
            method=method,
            sample_records=sample_text,
        )

        try:
            return self._llm.generate(prompt, temperature_override=0.2)
        except Exception as exc:
            logger.warning("Anomaly summary generation failed", error=str(exc))
            return (
                f"{anomaly_count} anomalies detected ({anomaly_count/max(total_records,1)*100:.1f}% of records) "
                f"using {method}. Most anomalous columns: {', '.join(columns[:3])}."
            )


def _safe_val(v) -> object:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


# Module-level singleton
anomaly_detector = AnomalyDetector()


