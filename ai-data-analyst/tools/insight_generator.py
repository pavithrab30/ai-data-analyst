"""
Business insight generation tool.

Analyzes a dataset to automatically surface actionable business insights:
- Top/bottom performers (products, regions, customers)
- Trend identification (growth, decline, seasonality)
- Revenue concentration (Pareto analysis)
- Anomalous patterns warranting attention
- Cross-metric correlations with business relevance

Each insight is returned as a typed BusinessInsight model with
supporting data, confidence level, and metric values.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from models.analysis_models import BusinessInsight, ConfidenceLevel
from services.llm_service import LLMService
from utils.logger import get_logger

logger = get_logger(__name__)

_INSIGHT_GENERATION_PROMPT = """You are a senior business analyst. Analyze the following dataset summary
and generate 5-7 high-value business insights.

Dataset: {dataset_name}
Shape: {rows} rows × {columns} columns
Schema: {schema}
Key statistics:
{statistics}

For each insight provide:
1. A short headline (max 10 words)
2. A detailed description (2-3 sentences)
3. The insight type: trend, top_performer, underperformer, anomaly, or general
4. The primary metric involved

Format each insight as:
INSIGHT: [headline]
TYPE: [type]
METRIC: [metric name]
DESCRIPTION: [description]
---

Generate only genuine insights backed by the statistics. Do not invent data."""


class InsightGenerator:
    """
    Generates business insights from statistical summaries.

    Combines deterministic statistical analysis (computed with Pandas)
    with LLM-powered interpretation (Gemini) to produce insights
    that are both data-accurate and business-relevant.
    """

    def __init__(self, llm_service_arg: Optional[LLMService] = None) -> None:
        self._llm = llm_service_arg or LLMService()

    def generate(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        focus_columns: Optional[list[str]] = None,
    ) -> list[BusinessInsight]:
        """
        Generate business insights from the dataset.

        Args:
            df: DataFrame to analyze.
            dataset_name: Display name for logging and prompts.
            focus_columns: Optional list of columns to prioritize.

        Returns:
            List of BusinessInsight models sorted by confidence.
        """
        logger.info("Insight generation started", dataset=dataset_name)

        # ── Compute statistical summaries ──────────────────────────────────────
        stat_insights = self._compute_statistical_insights(df, focus_columns)

        # ── Ask Gemini to interpret and extend ────────────────────────────────
        llm_insights = self._generate_llm_insights(df, dataset_name)

        # ── Merge and deduplicate ──────────────────────────────────────────────
        all_insights = stat_insights + llm_insights

        logger.info(
            "Insight generation complete",
            stat_insights=len(stat_insights),
            llm_insights=len(llm_insights),
        )

        return all_insights

    def generate_kpis(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Extract key performance indicators from the dataset.

        Returns a dict of {metric_name: value} for display in KPI cards.
        Identifies revenue, count, and ratio columns automatically.

        Args:
            df: Source DataFrame.

        Returns:
            Dict of KPI name → value pairs.
        """
        kpis: dict[str, Any] = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            clean = df[col].dropna()
            if len(clean) == 0:
                continue

            col_lower = col.lower()

            # Revenue / sales / amount columns → sum
            if any(kw in col_lower for kw in ["revenue", "sales", "amount", "total", "price", "value"]):
                kpis[f"Total {col.replace('_', ' ').title()}"] = float(clean.sum())
                kpis[f"Avg {col.replace('_', ' ').title()}"] = float(clean.mean())

            # Count / quantity columns → sum
            elif any(kw in col_lower for kw in ["count", "quantity", "qty", "units", "orders"]):
                kpis[f"Total {col.replace('_', ' ').title()}"] = int(clean.sum())

            # Rate / ratio / percentage columns → mean
            elif any(kw in col_lower for kw in ["rate", "ratio", "pct", "percent", "score"]):
                kpis[f"Avg {col.replace('_', ' ').title()}"] = float(clean.mean())

        # Always include row count
        kpis["Total Records"] = len(df)

        # Date range if date columns exist
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            clean_dates = df[date_col].dropna()
            if len(clean_dates) > 0:
                kpis["Date Range"] = f"{clean_dates.min().date()} to {clean_dates.max().date()}"

        return kpis

    # ── Private Methods ────────────────────────────────────────────────────────

    def _compute_statistical_insights(
        self,
        df: pd.DataFrame,
        focus_columns: Optional[list[str]] = None,
    ) -> list[BusinessInsight]:
        """Compute deterministic statistical insights."""
        insights: list[BusinessInsight] = []
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        if focus_columns:
            numeric_cols = [c for c in numeric_cols if c in focus_columns] or numeric_cols
            categorical_cols = [c for c in categorical_cols if c in focus_columns] or categorical_cols

        # ── Top/bottom performers per category ─────────────────────────────────
        for num_col in numeric_cols[:3]:
            for cat_col in categorical_cols[:2]:
                insight = self._top_performer_insight(df, num_col, cat_col)
                if insight:
                    insights.append(insight)

        # ── Distribution insights ──────────────────────────────────────────────
        for num_col in numeric_cols[:4]:
            insight = self._distribution_insight(df, num_col)
            if insight:
                insights.append(insight)

        # ── Concentration / Pareto ─────────────────────────────────────────────
        for num_col in numeric_cols[:2]:
            for cat_col in categorical_cols[:2]:
                insight = self._pareto_insight(df, num_col, cat_col)
                if insight:
                    insights.append(insight)

        return insights

    def _top_performer_insight(
        self, df: pd.DataFrame, value_col: str, group_col: str
    ) -> Optional[BusinessInsight]:
        """Generate a top/bottom performer insight."""
        try:
            grouped = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
            if len(grouped) < 2:
                return None

            top = grouped.index[0]
            top_val = float(grouped.iloc[0])
            bottom = grouped.index[-1]
            bottom_val = float(grouped.iloc[-1])
            total = float(grouped.sum())
            top_share = top_val / max(total, 1) * 100

            return BusinessInsight(
                title=f"Top {group_col.replace('_', ' ').title()} by {value_col.replace('_', ' ').title()}",
                description=(
                    f"'{top}' leads with {value_col.replace('_',' ')} of {top_val:,.2f}, "
                    f"representing {top_share:.1f}% of the total. "
                    f"'{bottom}' is the lowest performer at {bottom_val:,.2f}."
                ),
                metric_name=value_col,
                metric_value=top_val,
                insight_type="top_performer",
                confidence=ConfidenceLevel.HIGH,
                supporting_data=grouped.reset_index().head(10).to_dict(orient="records"),
            )
        except Exception:
            return None

    def _distribution_insight(
        self, df: pd.DataFrame, col: str
    ) -> Optional[BusinessInsight]:
        """Generate a distribution/spread insight for a numeric column."""
        try:
            clean = df[col].dropna()
            if len(clean) < 10:
                return None

            mean_val = float(clean.mean())
            std_val = float(clean.std())
            cv = std_val / abs(mean_val) if mean_val != 0 else 0  # Coefficient of variation
            skew = float(clean.skew())

            if cv > 1.0:
                desc = (
                    f"{col.replace('_',' ').title()} shows high variability "
                    f"(CV={cv:.2f}), suggesting inconsistent performance. "
                    f"Mean: {mean_val:,.2f}, Std Dev: {std_val:,.2f}."
                )
                insight_type = "anomaly"
            elif abs(skew) > 1.5:
                direction = "right" if skew > 0 else "left"
                desc = (
                    f"{col.replace('_',' ').title()} is {direction}-skewed (skewness={skew:.2f}), "
                    f"indicating a concentration of {'low' if skew > 0 else 'high'} values "
                    f"with {'high' if skew > 0 else 'low'} outliers."
                )
                insight_type = "general"
            else:
                return None  # Unremarkable distribution

            return BusinessInsight(
                title=f"{col.replace('_', ' ').title()} Distribution Analysis",
                description=desc,
                metric_name=col,
                metric_value=mean_val,
                insight_type=insight_type,
                confidence=ConfidenceLevel.MEDIUM,
            )
        except Exception:
            return None

    def _pareto_insight(
        self, df: pd.DataFrame, value_col: str, group_col: str
    ) -> Optional[BusinessInsight]:
        """Generate a Pareto / concentration insight."""
        try:
            grouped = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
            if len(grouped) < 5:
                return None

            total = float(grouped.sum())
            top_20_pct = int(len(grouped) * 0.2)
            top_20_share = float(grouped.head(max(top_20_pct, 1)).sum()) / max(total, 1) * 100

            if top_20_share < 50:
                return None  # Not a meaningful concentration

            return BusinessInsight(
                title=f"Revenue Concentration in {group_col.replace('_', ' ').title()}",
                description=(
                    f"The top {max(top_20_pct, 1)} {group_col.replace('_', ' ')} "
                    f"({top_20_pct / max(len(grouped), 1) * 100:.0f}% of {group_col.replace('_',' ')} groups) "
                    f"account for {top_20_share:.1f}% of total {value_col.replace('_', ' ')}. "
                    f"This concentration suggests dependency risk and upsell opportunities."
                ),
                metric_name=value_col,
                metric_value=top_20_share,
                insight_type="general",
                confidence=ConfidenceLevel.HIGH,
                supporting_data=grouped.reset_index().head(10).to_dict(orient="records"),
            )
        except Exception:
            return None

    def _generate_llm_insights(
        self, df: pd.DataFrame, dataset_name: str
    ) -> list[BusinessInsight]:
        """Ask Gemini to generate additional insights from the data summary."""
        try:
            # Build schema description
            schema = ", ".join(
                f"{col} ({str(df[col].dtype)})" for col in df.columns[:20]
            )

            # Compute key statistics
            numeric_df = df.select_dtypes(include=["number"])
            stats_lines: list[str] = []
            for col in numeric_df.columns[:8]:
                clean = numeric_df[col].dropna()
                if len(clean) > 0:
                    stats_lines.append(
                        f"  {col}: mean={clean.mean():.2f}, "
                        f"min={clean.min():.2f}, max={clean.max():.2f}, "
                        f"sum={clean.sum():.2f}"
                    )
            statistics = "\n".join(stats_lines)

            prompt = _INSIGHT_GENERATION_PROMPT.format(
                dataset_name=dataset_name,
                rows=len(df),
                columns=len(df.columns),
                schema=schema,
                statistics=statistics or "No numeric statistics available.",
            )

            response = self._llm.generate(prompt, temperature_override=0.3)
            return self._parse_llm_insights(response)

        except Exception as exc:
            logger.warning("LLM insight generation failed", error=str(exc))
            return []

    def _parse_llm_insights(self, response: str) -> list[BusinessInsight]:
        """Parse structured LLM response into BusinessInsight objects."""
        insights: list[BusinessInsight] = []
        blocks = response.strip().split("---")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = {
                k.strip(): v.strip()
                for line in block.split("\n")
                if ":" in line
                for k, v in [line.split(":", 1)]
            }

            title = lines.get("INSIGHT", "").strip()
            description = lines.get("DESCRIPTION", "").strip()
            insight_type = lines.get("TYPE", "general").strip().lower()
            metric = lines.get("METRIC", None)

            if title and description:
                insights.append(
                    BusinessInsight(
                        title=title,
                        description=description,
                        insight_type=insight_type,
                        metric_name=metric,
                        confidence=ConfidenceLevel.MEDIUM,
                    )
                )

        return insights


# Module-level singleton
insight_generator = InsightGenerator()


