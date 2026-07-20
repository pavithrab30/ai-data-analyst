"""
Automatic dataset profiling tool.

Generates a comprehensive DataProfile after CSV upload.
The profile is displayed on the main dashboard and also
injected into LLM prompts so the agent understands the
dataset's structure without needing to re-analyze it.

Profiling covers:
- Row/column counts and memory usage
- Dtype distribution
- Per-column statistics (numeric, categorical, datetime)
- Missing value patterns
- Top correlations between numeric columns
- Sample values for each column
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd

from models.analysis_models import ColumnProfile, DataProfile, DataQualityReport
from tools.data_validator import data_validator
from utils.logger import get_logger

logger = get_logger(__name__)

# Suppress pandas performance warnings during profiling
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class DataProfiler:
    """
    Generates detailed statistical profiles of DataFrames.

    All profiling is computed eagerly at upload time so that:
    1. The dashboard loads instantly when a user opens the app.
    2. The LLM planner has schema context without additional API calls.
    3. Slow operations (correlation, type inference) run once, not per query.
    """

    # Max unique values to include in categorical value_counts
    MAX_VALUE_COUNTS = 20
    # Max sample values to show per column
    MAX_SAMPLE_VALUES = 5
    # Max columns for correlation matrix (performance guard)
    MAX_CORR_COLUMNS = 50
    # Top N correlations to surface
    TOP_N_CORRELATIONS = 10

    def profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataProfile:
        """
        Build a complete DataProfile for the given DataFrame.

        Args:
            df: DataFrame to profile.
            dataset_name: Logical name used in display and logging.

        Returns:
            Fully populated DataProfile model.
        """
        logger.info(
            "Starting dataset profiling",
            dataset=dataset_name,
            rows=len(df),
            columns=len(df.columns),
        )

        # ── Structural metadata ────────────────────────────────────────────────
        row_count = len(df)
        column_count = len(df.columns)
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())

        # ── Column type categorization ─────────────────────────────────────────
        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
        categorical_columns = list(df.select_dtypes(include=["object", "category"]).columns)
        date_columns = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        boolean_columns = list(df.select_dtypes(include=["bool"]).columns)
        
        # Detect date-like columns stored as strings
        common_date_patterns = {"date", "time", "timestamp", "created", "updated", "posted", 
                               "order_date", "sale_date", "transaction_date", "datetime",
                               "date_time", "year", "month", "day"}
        
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in date_columns:
                col_lower = col.lower()
                # Check if column name suggests it's a date
                is_likely_date = any(pattern in col_lower for pattern in common_date_patterns)
                
                if is_likely_date or col not in categorical_columns:
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        valid_ratio = parsed.notna().sum() / max(len(df), 1)
                        # Use lower threshold for named date columns, higher for others
                        threshold = 0.4 if is_likely_date else 0.5
                        if valid_ratio > threshold:
                            date_columns.append(col)
                            if col in categorical_columns:
                                categorical_columns.remove(col)
                    except Exception:
                        pass

        dtypes_summary = {
            "numeric": len(numeric_columns),
            "categorical": len(categorical_columns),
            "datetime": len(date_columns),
            "boolean": len(boolean_columns),
        }

        # ── Per-column profiles ────────────────────────────────────────────────
        column_profiles = [
            self._profile_column(df, col) for col in df.columns
        ]

        # ── Correlation analysis ───────────────────────────────────────────────
        top_correlations = self._compute_top_correlations(df, numeric_columns)

        # ── Data quality report ────────────────────────────────────────────────
        quality_report = data_validator.validate(df, dataset_name)

        profile = DataProfile(
            dataset_name=dataset_name,
            row_count=row_count,
            column_count=column_count,
            memory_usage_bytes=memory_usage_bytes,
            dtypes_summary=dtypes_summary,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            date_columns=date_columns,
            boolean_columns=boolean_columns,
            column_profiles=column_profiles,
            quality_report=quality_report,
            top_correlations=top_correlations,
            profiled_at=datetime.utcnow(),
        )

        logger.info(
            "Dataset profiling complete",
            dataset=dataset_name,
            quality_score=quality_report.overall_quality_score,
            numeric_cols=len(numeric_columns),
            categorical_cols=len(categorical_columns),
            date_cols=len(date_columns),
        )

        return profile

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """
        Compute statistics for a single column.

        Dispatches to numeric, categorical, or datetime profiler
        based on the column's dtype.

        Args:
            df: Source DataFrame.
            col: Column name to profile.

        Returns:
            Populated ColumnProfile.
        """
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = null_count / max(len(df), 1)
        unique_count = int(series.nunique(dropna=True))
        unique_pct = unique_count / max(len(df), 1)

        # Sample up to MAX_SAMPLE_VALUES non-null values
        sample_values = (
            series.dropna()
            .head(self.MAX_SAMPLE_VALUES)
            .tolist()
        )
        sample_values = [_safe_serialize(v) for v in sample_values]

        base = dict(
            name=col,
            dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=round(null_pct * 100, 2),
            unique_count=unique_count,
            unique_percentage=round(unique_pct * 100, 2),
            sample_values=sample_values,
        )

        # Dispatch by dtype
        if pd.api.types.is_numeric_dtype(series):
            base.update(self._numeric_stats(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            base.update(self._datetime_stats(series))
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            base.update(self._categorical_stats(series))

        return ColumnProfile(**base)

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute descriptive statistics for a numeric series."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        try:
            desc = clean.describe()
            return {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min_value": _safe_float(desc.get("min")),
                "max_value": _safe_float(desc.get("max")),
                "median": _safe_float(clean.median()),
                "q25": _safe_float(desc.get("25%")),
                "q75": _safe_float(desc.get("75%")),
            }
        except Exception as exc:
            logger.debug("Numeric stats computation failed", error=str(exc))
            return {}

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute frequency statistics for a categorical series."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        try:
            vc = clean.value_counts()
            top_value = str(vc.index[0]) if len(vc) > 0 else None
            top_frequency = int(vc.iloc[0]) if len(vc) > 0 else None
            value_counts = {
                str(k): int(v)
                for k, v in vc.head(self.MAX_VALUE_COUNTS).items()
            }
            return {
                "top_value": top_value,
                "top_frequency": top_frequency,
                "value_counts": value_counts,
            }
        except Exception as exc:
            logger.debug("Categorical stats computation failed", error=str(exc))
            return {}

    def _datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        """Compute range statistics for a datetime series."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        try:
            min_date = clean.min()
            max_date = clean.max()
            date_range_days = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None

            return {
                "min_date": str(min_date.date()) if pd.notna(min_date) else None,
                "max_date": str(max_date.date()) if pd.notna(max_date) else None,
                "date_range_days": date_range_days,
            }
        except Exception as exc:
            logger.debug("Datetime stats computation failed", error=str(exc))
            return {}

    def _compute_top_correlations(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
    ) -> list[dict[str, Any]]:
        """
        Compute the top N most correlated column pairs.

        Limits computation to MAX_CORR_COLUMNS columns to avoid
        performance issues on wide datasets.

        Args:
            df: Source DataFrame.
            numeric_columns: List of numeric column names.

        Returns:
            List of dicts: [{col1, col2, correlation}] sorted by abs correlation desc.
        """
        if len(numeric_columns) < 2:
            return []

        # Limit columns for performance
        cols_to_use = numeric_columns[: self.MAX_CORR_COLUMNS]
        numeric_df = df[cols_to_use].select_dtypes(include=["number"])

        if len(numeric_df.columns) < 2:
            return []

        try:
            corr_matrix = numeric_df.corr()
        except Exception as exc:
            logger.warning("Correlation computation failed", error=str(exc))
            return []

        correlations: list[dict[str, Any]] = []
        cols = corr_matrix.columns.tolist()

        for i, col1 in enumerate(cols):
            for col2 in cols[i + 1 :]:
                corr_value = corr_matrix.loc[col1, col2]
                if pd.notna(corr_value) and col1 != col2:
                    correlations.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(float(corr_value), 4),
                            "abs_correlation": abs(float(corr_value)),
                        }
                    )

        # Sort by absolute correlation descending
        correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return correlations[: self.TOP_N_CORRELATIONS]

    def generate_schema_summary(self, profile: DataProfile) -> str:
        """
        Generate a concise text schema summary suitable for LLM prompts.

        This summary is injected into every LLM call so the model
        understands the dataset without seeing the raw data.

        Args:
            profile: Pre-computed DataProfile.

        Returns:
            Formatted multi-line string describing the dataset schema.
        """
        lines = [
            f"Dataset: {profile.dataset_name}",
            f"Shape: {profile.row_count:,} rows × {profile.column_count} columns",
            f"Memory: {profile.memory_usage_bytes / 1024 / 1024:.1f} MB",
            f"Quality Score: {profile.quality_report.overall_quality_score}/100",
            "",
            "Columns:",
        ]

        for col_profile in profile.column_profiles:
            null_info = (
                f", {col_profile.null_percentage:.1f}% null"
                if col_profile.null_percentage > 0
                else ""
            )
            lines.append(
                f"  - {col_profile.name} ({col_profile.dtype}{null_info})"
            )

        if profile.top_correlations:
            lines.append("")
            lines.append("Top correlations:")
            for corr in profile.top_correlations[:5]:
                lines.append(
                    f"  - {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.2f}"
                )

        return "\n".join(lines)


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float safely, returning None on failure."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_serialize(value: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


# Module-level singleton
data_profiler = DataProfiler()
