"""
Dataset validation tool using Pandera and custom heuristics.

Performs comprehensive data quality checks after every CSV upload:
- Missing value analysis
- Duplicate row and column detection
- Data type consistency
- Date format validation
- Constant and high-cardinality column detection
- Schema consistency across multiple files
- Memory usage reporting
- Overall quality scoring

Returns a DataQualityReport model that feeds the dashboard's
quality summary and gates downstream analysis steps.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from models.analysis_models import DataQualityReport
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Runs a comprehensive quality check on any uploaded DataFrame.

    The validator is intentionally read-only: it inspects data without
    modifying it, producing a DataQualityReport that callers can act upon.

    Quality score calculation:
    - Starts at 100
    - -10 for each column with >20% nulls
    - -5 for each column with >5% nulls
    - -10 if duplicate rows > 1% of total
    - -5 for each constant column
    - -3 for each type inconsistency detected
    """

    HIGH_NULL_THRESHOLD = 0.20       # Columns with >20% nulls trigger major penalty
    LOW_NULL_THRESHOLD = 0.05        # Columns with >5% nulls trigger minor penalty
    HIGH_CARDINALITY_THRESHOLD = 0.50  # >50% unique values in a string col
    DATE_INFERENCE_THRESHOLD = 0.80  # >80% parseable values = date column

    def validate(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataQualityReport:
        """
        Run all validation checks and return a structured report.

        Args:
            df: DataFrame to validate.
            dataset_name: Name used in log messages.

        Returns:
            DataQualityReport with all findings and quality score.
        """
        logger.info(
            "Starting data validation",
            dataset=dataset_name,
            rows=len(df),
            columns=len(df.columns),
        )

        quality_issues: list[str] = []
        quality_score = 100.0

        # ── 1. Basic shape ─────────────────────────────────────────────────────
        total_rows = len(df)
        total_columns = len(df.columns)

        # ── 2. Duplicate rows ──────────────────────────────────────────────────
        duplicate_rows = int(df.duplicated().sum())
        dup_row_pct = duplicate_rows / max(total_rows, 1)
        if dup_row_pct > 0.01:
            quality_score -= 10
            quality_issues.append(
                f"High duplicate row rate: {duplicate_rows:,} rows "
                f"({dup_row_pct:.1%} of total)"
            )
        elif duplicate_rows > 0:
            quality_issues.append(
                f"Minor duplicates: {duplicate_rows} duplicate rows detected"
            )

        # ── 3. Duplicate columns ───────────────────────────────────────────────
        duplicate_columns = self._find_duplicate_columns(df)
        if duplicate_columns:
            quality_score -= 5 * len(duplicate_columns)
            quality_issues.append(
                f"Duplicate columns detected: {duplicate_columns}"
            )

        # ── 4. Missing values ──────────────────────────────────────────────────
        columns_with_nulls: dict[str, float] = {}
        for col in df.columns:
            null_pct = df[col].isna().sum() / max(total_rows, 1)
            if null_pct > 0:
                columns_with_nulls[col] = round(null_pct * 100, 2)
                if null_pct > self.HIGH_NULL_THRESHOLD:
                    quality_score -= 10
                    quality_issues.append(
                        f"Column '{col}' has {null_pct:.1%} missing values (high)"
                    )
                elif null_pct > self.LOW_NULL_THRESHOLD:
                    quality_score -= 5
                    quality_issues.append(
                        f"Column '{col}' has {null_pct:.1%} missing values"
                    )

        # ── 5. Constant columns ────────────────────────────────────────────────
        constant_columns = [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]
        if constant_columns:
            quality_score -= 5 * len(constant_columns)
            quality_issues.append(
                f"Constant columns (no analytical value): {constant_columns}"
            )

        # ── 6. High-cardinality string columns ─────────────────────────────────
        high_cardinality_columns = []
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / max(total_rows, 1)
            if unique_ratio > self.HIGH_CARDINALITY_THRESHOLD and total_rows > 50:
                high_cardinality_columns.append(col)

        if high_cardinality_columns:
            quality_issues.append(
                f"High-cardinality string columns (may be IDs): {high_cardinality_columns}"
            )

        # ── 7. Type inconsistencies ────────────────────────────────────────────
        type_inconsistencies = self._detect_type_inconsistencies(df)
        if type_inconsistencies:
            quality_score -= 3 * len(type_inconsistencies)
            for col, detail in type_inconsistencies.items():
                quality_issues.append(f"Type inconsistency in '{col}': {detail}")

        # ── 8. Memory usage ────────────────────────────────────────────────────
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())

        # Clamp score
        quality_score = max(0.0, min(100.0, quality_score))

        report = DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows=duplicate_rows,
            duplicate_row_percentage=round(dup_row_pct * 100, 2),
            duplicate_columns=duplicate_columns,
            columns_with_nulls=columns_with_nulls,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            type_inconsistencies=type_inconsistencies,
            memory_usage_bytes=memory_usage_bytes,
            overall_quality_score=round(quality_score, 1),
            quality_issues=quality_issues,
        )

        logger.info(
            "Data validation complete",
            dataset=dataset_name,
            quality_score=quality_score,
            issues=len(quality_issues),
        )

        return report

    def _find_duplicate_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Identify columns that are exact duplicates of another column.

        Returns a list of duplicate column names (not the originals).
        """
        seen_hashes: dict[str, str] = {}
        duplicates: list[str] = []

        for col in df.columns:
            try:
                col_hash = pd.util.hash_pandas_object(df[col]).sum()
                col_hash_str = str(col_hash)
                if col_hash_str in seen_hashes:
                    duplicates.append(col)
                else:
                    seen_hashes[col_hash_str] = col
            except Exception:
                # Unhashable types: skip
                pass

        return duplicates

    def _detect_type_inconsistencies(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Detect columns where stored dtype doesn't match the actual data.

        For example: a column stored as 'object' that contains
        mostly numeric values is a type inconsistency.

        Returns:
            Dict mapping column name to inconsistency description.
        """
        inconsistencies: dict[str, str] = {}

        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna()
            if len(sample) == 0:
                continue

            # Check if mostly numeric
            numeric_attempt = pd.to_numeric(sample, errors="coerce")
            numeric_ratio = numeric_attempt.notna().sum() / len(sample)
            if numeric_ratio > 0.90:
                inconsistencies[col] = (
                    f"Stored as 'object' but {numeric_ratio:.0%} of values are numeric"
                )
                continue

            # Check if mostly datetime
            try:
                date_attempt = pd.to_datetime(sample, errors="coerce", infer_format=True)
                date_ratio = date_attempt.notna().sum() / len(sample)
                if date_ratio > self.DATE_INFERENCE_THRESHOLD:
                    inconsistencies[col] = (
                        f"Stored as 'object' but {date_ratio:.0%} of values appear to be dates"
                    )
            except Exception:
                pass

        return inconsistencies

    def validate_schema_compatibility(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str = "dataset1",
        name2: str = "dataset2",
    ) -> dict[str, object]:
        """
        Check whether two DataFrames have compatible schemas for merging.

        Args:
            df1: First DataFrame.
            df2: Second DataFrame.
            name1: Display name for df1.
            name2: Display name for df2.

        Returns:
            Dict with keys: compatible (bool), common_columns (list),
            only_in_first (list), only_in_second (list), type_conflicts (dict)
        """
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)

        common_columns = list(cols1 & cols2)
        only_in_first = list(cols1 - cols2)
        only_in_second = list(cols2 - cols1)

        type_conflicts: dict[str, str] = {}
        for col in common_columns:
            t1 = str(df1[col].dtype)
            t2 = str(df2[col].dtype)
            if t1 != t2:
                type_conflicts[col] = f"{name1}:{t1} vs {name2}:{t2}"

        compatible = len(common_columns) > 0

        logger.info(
            "Schema compatibility check",
            common_columns=len(common_columns),
            type_conflicts=len(type_conflicts),
        )

        return {
            "compatible": compatible,
            "common_columns": common_columns,
            "only_in_first": only_in_first,
            "only_in_second": only_in_second,
            "type_conflicts": type_conflicts,
        }


# Module-level singleton
data_validator = DataValidator()
