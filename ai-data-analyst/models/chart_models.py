"""
Pydantic models for chart configuration and results.

Decouples chart specification from rendering so the visualization
tool can produce configurations that the UI layer renders with Plotly.
This separation makes charts serializable, exportable, and testable
without requiring a display environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ChartType(str, Enum):
    """Supported Plotly chart types."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    AREA = "area"
    FUNNEL = "funnel"
    TREEMAP = "treemap"


class ChartConfig(BaseModel):
    """
    Complete specification for generating a Plotly chart.

    The visualizer tool populates this model; the UI layer calls
    the appropriate Plotly function using its fields. This means
    chart logic never leaks into UI components.
    """

    chart_type: ChartType = Field(..., description="Type of Plotly chart to render")
    title: str = Field(..., description="Chart title displayed at the top")
    x_column: Optional[str] = Field(
        default=None, description="Column to use for the X axis"
    )
    y_column: Optional[str] = Field(
        default=None, description="Column to use for the Y axis (primary)"
    )
    y_columns: list[str] = Field(
        default_factory=list,
        description="Multiple Y columns for multi-series charts",
    )
    color_column: Optional[str] = Field(
        default=None, description="Column for color encoding / legend grouping"
    )
    size_column: Optional[str] = Field(
        default=None, description="Column for bubble size in scatter plots"
    )
    aggregation: Optional[str] = Field(
        default=None,
        description="Aggregation applied before plotting: 'sum', 'mean', 'count', 'max', 'min'",
    )
    sort_by: Optional[str] = Field(
        default=None, description="Column to sort data by before plotting"
    )
    sort_ascending: bool = Field(default=False)
    top_n: Optional[int] = Field(
        default=None, description="Limit to top N records after sorting"
    )
    x_label: Optional[str] = Field(default=None, description="X axis label override")
    y_label: Optional[str] = Field(default=None, description="Y axis label override")
    color_scheme: str = Field(
        default="plotly", description="Plotly color scheme name"
    )
    height: int = Field(default=450, description="Chart height in pixels")
    show_legend: bool = Field(default=True)
    orientation: str = Field(
        default="v", description="Bar chart orientation: 'v' (vertical) or 'h' (horizontal)"
    )
    nbins: Optional[int] = Field(
        default=None, description="Number of bins for histograms"
    )
    reasoning: str = Field(
        default="",
        description="Why this chart type was chosen for the question",
    )
    data_snapshot: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Serialized data used to build the chart (max 500 rows)",
    )


class ChartResult(BaseModel):
    """
    Output of the visualizer tool including the chart config
    and a Plotly figure dict for direct rendering.
    """

    config: ChartConfig
    figure_dict: dict[str, Any] = Field(
        default_factory=dict,
        description="Plotly figure serialized as dict for JSON transfer",
    )
    question: str = Field(default="", description="Question that prompted this chart")
    interpretation: str = Field(
        default="",
        description="Natural language interpretation of what the chart shows",
    )
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)
