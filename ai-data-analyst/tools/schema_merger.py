"""
Multi-CSV schema inspection and join recommendation tool.

When users upload multiple CSV files, this tool:
1. Inspects all schemas to find common columns.
2. Scores potential join keys by type compatibility and uniqueness.
3. Recommends the best join strategy with reasoning.
4. Executes the approved merge and returns the combined DataFrame.

Design: The merge is only executed after explicit user approval.
The tool exposes two methods:
  - recommend_merge() — analysis only, no side effects
  - execute_merge()   — performs the actual join
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class MergeRecommendation:
    """
    Data class holding a merge recommendation.

    Not a Pydantic model because it is internal to this tool
    and doesn't cross service boundaries.
    """

    def __init__(
        self,
        left_dataset: str,
        right_dataset: str,
        join_keys: list[str],
        join_type: str,
        confidence: float,
        reasoning: str,
        expected_rows_estimate: Optional[int] = None,
    ) -> None:
        self.left_dataset = left_dataset
        self.right_dataset = right_dataset
        self.join_keys = join_keys
        self.join_type = join_type
        self.confidence = confidence
        self.reasoning = reasoning
        self.expected_rows_estimate = expected_rows_estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_dataset": self.left_dataset,
            "right_dataset": self.right_dataset,
            "join_keys": self.join_keys,
            "join_type": self.join_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_rows_estimate": self.expected_rows_estimate,
        }


class SchemaMerger:
    """
    Inspects multiple DataFrames and recommends merge strategies.

    Join key scoring criteria:
    - Column name similarity between datasets (exact match = 1.0)
    - Data type compatibility (same type = bonus)
    - Key uniqueness in left dataset (higher unique ratio = better key)
    - Value overlap percentage (join cardinality estimate)
    """

    # Minimum overlap percentage to consider a join viable
    MIN_OVERLAP_THRESHOLD = 0.10

    def recommend_merge(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> list[MergeRecommendation]:
        """
        Analyze all pairs of datasets and return merge recommendations.

        Args:
            datasets: Dict mapping dataset name to DataFrame.

        Returns:
            List of MergeRecommendation objects sorted by confidence desc.
        """
        names = list(datasets.keys())
        if len(names) < 2:
            return []

        recommendations: list[MergeRecommendation] = []

        # Evaluate every pair
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                df1 = datasets[name1]
                df2 = datasets[name2]

                rec = self._analyze_pair(name1, df1, name2, df2)
                if rec is not None:
                    recommendations.append(rec)

        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations

    def execute_merge(
        self,
        datasets: dict[str, pd.DataFrame],
        recommendation: MergeRecommendation,
        suffix_left: str = "_left",
        suffix_right: str = "_right",
    ) -> pd.DataFrame:
        """
        Execute a merge based on a recommendation.

        Args:
            datasets: Dict mapping dataset name to DataFrame.
            recommendation: The approved MergeRecommendation.
            suffix_left: Suffix for left DataFrame conflicting columns.
            suffix_right: Suffix for right DataFrame conflicting columns.

        Returns:
            Merged DataFrame.

        Raises:
            KeyError: If a dataset name in the recommendation is not in datasets.
            ValueError: If join keys are not present in both DataFrames.
        """
        left_name = recommendation.left_dataset
        right_name = recommendation.right_dataset

        if left_name not in datasets:
            raise KeyError(f"Dataset '{left_name}' not found.")
        if right_name not in datasets:
            raise KeyError(f"Dataset '{right_name}' not found.")

        df_left = datasets[left_name]
        df_right = datasets[right_name]
        join_keys = recommendation.join_keys
        join_type = recommendation.join_type

        # Validate join keys exist
        missing_left = [k for k in join_keys if k not in df_left.columns]
        missing_right = [k for k in join_keys if k not in df_right.columns]

        if missing_left:
            raise ValueError(
                f"Join keys {missing_left} not found in '{left_name}'."
            )
        if missing_right:
            raise ValueError(
                f"Join keys {missing_right} not found in '{right_name}'."
            )

        logger.info(
            "Executing dataset merge",
            left=left_name,
            right=right_name,
            join_keys=join_keys,
            join_type=join_type,
        )

        merged = pd.merge(
            df_left,
            df_right,
            on=join_keys,
            how=join_type,
            suffixes=(suffix_left, suffix_right),
        )

        logger.info(
            "Merge complete",
            result_rows=len(merged),
            result_columns=len(merged.columns),
        )

        return merged

    # ── Private Methods ────────────────────────────────────────────────────────

    def _analyze_pair(
        self,
        name1: str,
        df1: pd.DataFrame,
        name2: str,
        df2: pd.DataFrame,
    ) -> Optional[MergeRecommendation]:
        """
        Analyze a pair of DataFrames for potential join compatibility.

        Returns a MergeRecommendation if a viable join is found,
        None if no compatible join keys are identified.
        """
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        common = cols1 & cols2

        if not common:
            return None

        best_keys: list[str] = []
        best_score = 0.0
        reasoning_parts: list[str] = []

        for col in common:
            score, reason = self._score_join_key(df1, df2, col)
            if score > best_score:
                best_score = score
                best_keys = [col]
                reasoning_parts = [reason]
            elif score == best_score and score > 0:
                best_keys.append(col)

        if not best_keys or best_score < self.MIN_OVERLAP_THRESHOLD:
            return None

        # Determine join type based on cardinality
        join_type = self._recommend_join_type(df1, df2, best_keys[0])

        # Estimate result row count
        expected_rows = self._estimate_merge_rows(df1, df2, best_keys[0], join_type)

        reasoning = (
            f"Found {len(common)} common column(s): {list(common)}. "
            f"Best join key: '{best_keys[0]}' with compatibility score {best_score:.2f}. "
            + "; ".join(reasoning_parts)
        )

        return MergeRecommendation(
            left_dataset=name1,
            right_dataset=name2,
            join_keys=best_keys[:1],  # Use single best key by default
            join_type=join_type,
            confidence=min(best_score, 1.0),
            reasoning=reasoning,
            expected_rows_estimate=expected_rows,
        )

    def _score_join_key(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        col: str,
    ) -> tuple[float, str]:
        """
        Score a column as a join key (0.0 to 1.0).

        Criteria:
        - Type compatibility: same dtype = +0.3
        - Value overlap ratio: percentage of df1 values in df2 = up to +0.5
        - Uniqueness in df1: higher unique ratio = +0.2

        Returns:
            Tuple of (score, reasoning string)
        """
        score = 0.0
        reasons: list[str] = []

        # Type compatibility
        t1 = str(df1[col].dtype)
        t2 = str(df2[col].dtype)
        if t1 == t2:
            score += 0.3
            reasons.append(f"same dtype ({t1})")
        else:
            reasons.append(f"dtype mismatch ({t1} vs {t2})")

        # Value overlap
        try:
            vals1 = set(df1[col].dropna().astype(str))
            vals2 = set(df2[col].dropna().astype(str))
            if vals1 and vals2:
                overlap = len(vals1 & vals2) / len(vals1)
                score += overlap * 0.5
                reasons.append(f"{overlap:.0%} value overlap")
        except Exception:
            pass

        # Uniqueness in left
        try:
            unique_ratio = df1[col].nunique() / max(len(df1), 1)
            score += min(unique_ratio, 1.0) * 0.2
            reasons.append(f"{unique_ratio:.0%} unique in left dataset")
        except Exception:
            pass

        return score, "; ".join(reasons)

    def _recommend_join_type(
        self, df1: pd.DataFrame, df2: pd.DataFrame, key: str
    ) -> str:
        """
        Choose between inner / left / outer join based on key overlap.

        - >90% overlap both ways → inner join
        - df1 has all values → left join
        - otherwise → left join (conservative default)
        """
        try:
            vals1 = set(df1[key].dropna().astype(str))
            vals2 = set(df2[key].dropna().astype(str))

            if not vals1 or not vals2:
                return "left"

            overlap_ratio_1_in_2 = len(vals1 & vals2) / len(vals1)
            overlap_ratio_2_in_1 = len(vals1 & vals2) / len(vals2)

            if overlap_ratio_1_in_2 > 0.9 and overlap_ratio_2_in_1 > 0.9:
                return "inner"
            return "left"
        except Exception:
            return "left"

    def _estimate_merge_rows(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key: str,
        join_type: str,
    ) -> int:
        """Estimate the number of rows in the merged result."""
        try:
            if join_type == "inner":
                vals1 = set(df1[key].dropna().astype(str))
                vals2 = set(df2[key].dropna().astype(str))
                common = vals1 & vals2
                # Average matches per key
                avg_matches = len(df2[df2[key].astype(str).isin(common)]) / max(len(common), 1)
                return int(len(df1[df1[key].astype(str).isin(common)]) * avg_matches)
            return len(df1)
        except Exception:
            return len(df1)


# Module-level singleton
schema_merger = SchemaMerger()
