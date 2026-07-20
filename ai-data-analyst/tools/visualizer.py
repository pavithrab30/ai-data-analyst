"""
Intelligent visualization tool.

Automatically selects the most appropriate chart type based on
the user's question and data characteristics, then generates
a Plotly figure with consistent styling.

Chart type selection logic:
- Comparison over categories → bar chart
- Trend over time → line chart
- Part-of-whole / composition → pie chart
- Relationship between numerics → scatter plot
- Distribution of single numeric → histogram
- Distribution with outliers → box plot
- Correlation matrix → heatmap
- Cumulative / area trends → area chart
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.chart_models import ChartConfig, ChartResult, ChartType
from services.llm_service import LLMService
from utils.logger import get_logger

logger = get_logger(__name__)

# Keywords in the question that signal specific chart types
_CHART_KEYWORD_MAP: dict[str, list[str]] = {
    ChartType.LINE: [
        "trend", "over time", "monthly", "weekly", "daily", "yearly",
        "timeline", "time series", "growth", "change over",
    ],
    ChartType.PIE: [
        "proportion", "percentage", "share", "distribution", "breakdown",
        "composition", "percent of", "part of",
    ],
    ChartType.SCATTER: [
        "correlation", "relationship", "scatter", "vs", "versus",
        "impact of", "effect of",
    ],
    ChartType.HISTOGRAM: [
        "distribution", "frequency", "histogram", "spread", "how many",
        "range of",
    ],
    ChartType.BOX: [
        "box plot", "outlier", "quartile", "median", "variance", "spread by",
    ],
    ChartType.HEATMAP: [
        "heatmap", "correlation matrix", "heat map",
    ],
    ChartType.BAR: [
        "compare", "top", "bottom", "highest", "lowest", "rank", "best",
        "worst", "most", "least", "by region", "by category", "by product",
        "by customer",
    ],
}

_CHART_CONFIG_PROMPT = """You are a data visualization expert. Given a dataset schema and a user question,
determine the best chart configuration.

Schema:
{schema}

User question: {question}
Suggested chart type: {suggested_type}

Respond with a JSON object containing these fields:
- chart_type: one of bar, line, pie, scatter, histogram, box, heatmap, area
- x_column: column name for x axis (or null)
- y_column: column name for y axis (or null)
- y_columns: list of column names for multi-series (or empty list)
- color_column: column for color grouping (or null)
- aggregation: one of sum, mean, count, max, min (or null)
- sort_by: column to sort by (or null)
- sort_ascending: true or false
- top_n: integer limit (or null)
- title: descriptive chart title
- x_label: x axis label (or null)
- y_label: y axis label (or null)
- reasoning: why this chart type was chosen

Return ONLY valid JSON, no markdown fences."""


class Visualizer:
    """
    Generates Plotly charts from DataFrames based on natural language questions.

    Uses a two-step approach:
    1. Keyword-based heuristic for fast chart type pre-selection.
    2. LLM-based column and configuration selection.
    """

    def __init__(self, llm_service_arg: Optional[LLMService] = None) -> None:
        self._llm = llm_service_arg or LLMService()

    def create_chart(
        self,
        question: str,
        df: pd.DataFrame,
        chart_type_hint: Optional[ChartType] = None,
    ) -> ChartResult:
        """
        Generate a Plotly chart answering the user's question.

        Args:
            question: Natural language question to visualize.
            df: Source DataFrame.
            chart_type_hint: Optional override for chart type.

        Returns:
            ChartResult with config and rendered Plotly figure dict.
        """
        logger.info("Chart generation started", question=question[:100])

        # Step 1: Suggest chart type via keyword matching
        suggested_type = chart_type_hint or self._suggest_chart_type(question)

        # Step 2: Get full config from LLM
        config = self._get_chart_config(question, df, suggested_type)

        # Step 3: Render the chart
        result = self._render_chart(config, df, question)

        logger.info(
            "Chart generated",
            chart_type=config.chart_type,
            success=result.success,
        )

        return result

    def create_correlation_heatmap(self, df: pd.DataFrame) -> ChartResult:
        """
        Generate a correlation heatmap for all numeric columns.

        Args:
            df: Source DataFrame.

        Returns:
            ChartResult with heatmap figure.
        """
        numeric_df = df.select_dtypes(include=["number"])
        if len(numeric_df.columns) < 2:
            return ChartResult(
                config=ChartConfig(
                    chart_type=ChartType.HEATMAP,
                    title="Correlation Matrix",
                ),
                success=False,
                error_message="Need at least 2 numeric columns for a correlation heatmap.",
            )

        corr = numeric_df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Correlation Matrix",
            height=500,
            template="plotly_white",
        )

        config = ChartConfig(
            chart_type=ChartType.HEATMAP,
            title="Correlation Matrix",
            reasoning="Heatmap selected to show relationships between all numeric columns.",
        )

        return ChartResult(
            config=config,
            figure_dict=fig.to_dict(),
            question="Correlation heatmap",
            interpretation="Darker red = strong positive correlation. Darker blue = strong negative correlation.",
            success=True,
        )

    # ── Private Methods ────────────────────────────────────────────────────────

    def _suggest_chart_type(self, question: str) -> ChartType:
        """
        Use keyword matching to suggest a chart type from the question.

        Defaults to bar chart if no keywords match.

        Args:
            question: User question string.

        Returns:
            Suggested ChartType enum value.
        """
        question_lower = question.lower()

        for chart_type, keywords in _CHART_KEYWORD_MAP.items():
            if any(kw in question_lower for kw in keywords):
                return ChartType(chart_type)

        return ChartType.BAR  # Safe default

    def _get_chart_config(
        self,
        question: str,
        df: pd.DataFrame,
        suggested_type: ChartType,
    ) -> ChartConfig:
        """
        Ask Gemini for chart configuration details.

        Falls back to a simple default configuration on LLM failure.

        Args:
            question: User question.
            df: Source DataFrame.
            suggested_type: Pre-selected chart type.

        Returns:
            Populated ChartConfig.
        """
        schema = self._build_schema(df)
        prompt = _CHART_CONFIG_PROMPT.format(
            schema=schema,
            question=question,
            suggested_type=suggested_type.value,
        )

        try:
            import json
            raw = self._llm.generate(prompt, temperature_override=0.0)
            # Extract JSON even if surrounded by text
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")

            config_dict = json.loads(json_match.group())

            # Map chart_type string to enum
            chart_type_str = config_dict.get("chart_type", suggested_type.value)
            try:
                chart_type = ChartType(chart_type_str.lower())
            except ValueError:
                chart_type = suggested_type

            return ChartConfig(
                chart_type=chart_type,
                title=config_dict.get("title", question[:60]),
                x_column=config_dict.get("x_column"),
                y_column=config_dict.get("y_column"),
                y_columns=config_dict.get("y_columns", []),
                color_column=config_dict.get("color_column"),
                aggregation=config_dict.get("aggregation"),
                sort_by=config_dict.get("sort_by"),
                sort_ascending=config_dict.get("sort_ascending", False),
                top_n=config_dict.get("top_n"),
                x_label=config_dict.get("x_label"),
                y_label=config_dict.get("y_label"),
                reasoning=config_dict.get("reasoning", ""),
            )

        except Exception as exc:
            logger.warning("Chart config LLM call failed, using defaults", error=str(exc))
            return self._default_config(df, suggested_type, question)

    def _default_config(
        self,
        df: pd.DataFrame,
        chart_type: ChartType,
        question: str,
    ) -> ChartConfig:
        """Build a sensible default chart config when LLM fails."""
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        x_col = categorical_cols[0] if categorical_cols else (numeric_cols[0] if numeric_cols else None)
        y_col = numeric_cols[0] if numeric_cols else None

        return ChartConfig(
            chart_type=chart_type,
            title=question[:80],
            x_column=x_col,
            y_column=y_col,
            reasoning="Default configuration (LLM unavailable)",
        )

    def _render_chart(
        self,
        config: ChartConfig,
        df: pd.DataFrame,
        question: str,
    ) -> ChartResult:
        """
        Render a Plotly figure based on the chart configuration.

        Args:
            config: Populated ChartConfig.
            df: Source DataFrame.
            question: Original question for the result object.

        Returns:
            ChartResult with Plotly figure dict.
        """
        try:
            # Prepare data
            plot_df = self._prepare_data(df, config)

            # Dispatch to appropriate renderer
            render_map = {
                ChartType.BAR: self._render_bar,
                ChartType.LINE: self._render_line,
                ChartType.PIE: self._render_pie,
                ChartType.SCATTER: self._render_scatter,
                ChartType.HISTOGRAM: self._render_histogram,
                ChartType.BOX: self._render_box,
                ChartType.HEATMAP: self._render_heatmap_data,
                ChartType.AREA: self._render_area,
            }

            renderer = render_map.get(config.chart_type, self._render_bar)
            fig = renderer(plot_df, config)

            # Apply consistent styling
            fig.update_layout(
                template="plotly_white",
                height=config.height,
                title={"text": config.title, "x": 0.5, "xanchor": "center"},
                showlegend=config.show_legend,
                margin=dict(t=60, b=40, l=40, r=40),
            )

            if config.x_label:
                fig.update_xaxes(title_text=config.x_label)
            if config.y_label:
                fig.update_yaxes(title_text=config.y_label)

            # Store small data snapshot in config for export
            config.data_snapshot = plot_df.head(500).to_dict(orient="records")

            return ChartResult(
                config=config,
                figure_dict=fig.to_dict(),
                question=question,
                interpretation=f"Chart showing {config.title}",
                success=True,
            )

        except Exception as exc:
            logger.error("Chart rendering failed", error=str(exc), chart_type=str(config.chart_type))
            return ChartResult(
                config=config,
                success=False,
                error_message=str(exc),
                question=question,
            )

    def _prepare_data(self, df: pd.DataFrame, config: ChartConfig) -> pd.DataFrame:
        """Apply aggregation, sorting, and top-N filtering to the DataFrame."""
        plot_df = df.copy()

        # Aggregation
        if config.aggregation and config.x_column and config.y_column:
            agg_func = config.aggregation
            group_cols = [config.x_column]
            if config.color_column and config.color_column in plot_df.columns:
                group_cols.append(config.color_column)

            if config.y_column in plot_df.columns:
                plot_df = (
                    plot_df.groupby(group_cols)[config.y_column]
                    .agg(agg_func)
                    .reset_index()
                )

        # Sorting
        sort_col = config.sort_by or config.y_column
        if sort_col and sort_col in plot_df.columns:
            plot_df = plot_df.sort_values(sort_col, ascending=config.sort_ascending)

        # Top N
        if config.top_n and len(plot_df) > config.top_n:
            plot_df = plot_df.head(config.top_n)

        return plot_df

    def _render_bar(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        x = config.x_column or df.columns[0]
        y = config.y_column or (df.select_dtypes(include=["number"]).columns[0] if len(df.select_dtypes(include=["number"]).columns) > 0 else df.columns[-1])
        color = config.color_column if config.color_column and config.color_column in df.columns else None

        return px.bar(
            df,
            x=x if x in df.columns else df.columns[0],
            y=y if y in df.columns else df.columns[-1],
            color=color,
            orientation=config.orientation,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=config.title,
        )

    def _render_line(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        x = config.x_column or df.columns[0]
        y_cols = config.y_columns if config.y_columns else ([config.y_column] if config.y_column else None)
        color = config.color_column if config.color_column and config.color_column in df.columns else None

        if y_cols and len(y_cols) > 1:
            fig = go.Figure()
            for col in y_cols:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=col))
            return fig

        y = (y_cols[0] if y_cols else None) or df.select_dtypes(include=["number"]).columns[0]
        return px.line(
            df,
            x=x if x in df.columns else df.columns[0],
            y=y if y in df.columns else df.columns[-1],
            color=color,
            markers=True,
            title=config.title,
        )

    def _render_pie(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        names_col = config.x_column or df.columns[0]
        values_col = config.y_column or (df.select_dtypes(include=["number"]).columns[0] if len(df.select_dtypes(include=["number"]).columns) > 0 else df.columns[-1])
        return px.pie(
            df,
            names=names_col if names_col in df.columns else df.columns[0],
            values=values_col if values_col in df.columns else df.columns[-1],
            title=config.title,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

    def _render_scatter(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        x = config.x_column or (numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0])
        y = config.y_column or (numeric_cols[1] if len(numeric_cols) > 1 else df.columns[-1])
        color = config.color_column if config.color_column and config.color_column in df.columns else None
        size = config.size_column if config.size_column and config.size_column in df.columns else None

        return px.scatter(
            df,
            x=x if x in df.columns else df.columns[0],
            y=y if y in df.columns else df.columns[-1],
            color=color,
            size=size,
            title=config.title,
            opacity=0.7,
        )

    def _render_histogram(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        x = config.x_column or (numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0])
        color = config.color_column if config.color_column and config.color_column in df.columns else None

        return px.histogram(
            df,
            x=x if x in df.columns else df.columns[0],
            color=color,
            nbins=config.nbins or 30,
            title=config.title,
            opacity=0.8,
        )

    def _render_box(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        y = config.y_column or (numeric_cols[0] if len(numeric_cols) > 0 else df.columns[-1])
        x = config.x_column if config.x_column and config.x_column in df.columns else None

        return px.box(
            df,
            x=x,
            y=y if y in df.columns else df.columns[-1],
            color=x,
            title=config.title,
        )

    def _render_heatmap_data(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        """Render a heatmap from data (not correlation)."""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return px.bar(df, title=config.title)

        corr = numeric_df.corr()
        return go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
            )
        )

    def _render_area(self, df: pd.DataFrame, config: ChartConfig) -> go.Figure:
        x = config.x_column or df.columns[0]
        y = config.y_column or (df.select_dtypes(include=["number"]).columns[0] if len(df.select_dtypes(include=["number"]).columns) > 0 else df.columns[-1])

        return px.area(
            df,
            x=x if x in df.columns else df.columns[0],
            y=y if y in df.columns else df.columns[-1],
            title=config.title,
        )

    def _build_schema(self, df: pd.DataFrame) -> str:
        """Build a schema description string for the LLM prompt."""
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample)
            lines.append(f"  {col} ({dtype}): [{sample_str}]")
        return "\n".join(lines)


# Module-level singleton
visualizer = Visualizer()


