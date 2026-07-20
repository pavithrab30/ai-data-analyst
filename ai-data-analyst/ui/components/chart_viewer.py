"""
Chart viewer with user-selectable chart type dropdown.
This is the key component for the "choosable chart type" feature.
"""

from __future__ import annotations
from typing import Optional
import streamlit as st
import plotly.graph_objects as go

from models.chart_models import ChartResult, ChartType

# All supported chart types shown in the dropdown
CHART_TYPE_OPTIONS: dict[str, str] = {
    "auto":      "🤖 Auto-detect",
    "bar":       "📊 Bar Chart",
    "line":      "📈 Line Chart",
    "pie":       "🥧 Pie Chart",
    "scatter":   "🔵 Scatter Plot",
    "histogram": "📉 Histogram",
    "box":       "📦 Box Plot",
    "area":      "🌊 Area Chart",
    "heatmap":   "🌡 Heatmap",
}

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"),
    margin=dict(t=44, b=36, l=52, r=20),
    height=380,
    xaxis=dict(gridcolor="#1e2640", linecolor="#1e2640", zerolinecolor="#1e2640"),
    yaxis=dict(gridcolor="#1e2640", linecolor="#1e2640", zerolinecolor="#1e2640"),
    title=dict(font=dict(color="#e2e8f0", size=14)),
)

def _layout(**extra):
    """Merge base layout with extra kwargs safely."""
    return {**_BASE_LAYOUT, **extra}

# Keep for backward compatibility
_PLOTLY_LAYOUT = _BASE_LAYOUT


def render_chart_selector(key: str = "chart_type") -> Optional[str]:
    """
    Render the chart type dropdown and return selected value.

    Args:
        key: Streamlit widget key (must be unique per page location).

    Returns:
        Selected chart type string, e.g. "bar", "line", or "auto".
    """
    st.markdown(
        """
        <div class="chart-selector-wrap">
          <div class="chart-selector-title">📊 Chart Type</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.selectbox(
        "Chart type",
        options=list(CHART_TYPE_OPTIONS.keys()),
        format_func=lambda v: CHART_TYPE_OPTIONS[v],
        index=0,
        key=key,
        label_visibility="collapsed",
        help="Choose the visualization type, or let the AI decide.",
    )
    return selected if selected != "auto" else None


def render_chart(
    chart_result: ChartResult,
    key: str = "",
    show_export: bool = True,
) -> None:
    """
    Render a ChartResult as a styled Plotly chart.

    Args:
        chart_result: ChartResult produced by the visualizer tool.
        key: Unique key suffix for Streamlit widget deduplication.
        show_export: Whether to show the JSON download button.
    """
    if not chart_result.success:
        st.warning(f"⚠️ Chart could not be rendered: {chart_result.error_message}")
        return

    if not chart_result.figure_dict:
        st.info("No chart data available.")
        return

    try:
        fig = go.Figure(chart_result.figure_dict)

        # Apply dark theme layout (no duplicate legend key)
        fig.update_layout(**_layout(
            title=dict(
                text=chart_result.config.title,
                x=0.5, xanchor="center",
                font=dict(color="#e2e8f0", size=14, family="Inter"),
            )
        ))

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_{key}",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"chart_{key}",
                    "scale": 2,
                },
            },
        )

        # Interpretation caption
        if chart_result.interpretation:
            st.markdown(
                f'<div style="font-size:0.78rem;color:#475569;'
                f'margin-top:-8px;padding:0 4px;">💡 {chart_result.interpretation}</div>',
                unsafe_allow_html=True,
            )

        # Reasoning + export row
        col_r, col_e = st.columns([4, 1])
        with col_r:
            if chart_result.config.reasoning:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#334155;padding:4px 0;">'
                    f'Why this chart: {chart_result.config.reasoning}</div>',
                    unsafe_allow_html=True,
                )
        with col_e:
            if show_export:
                import json
                st.download_button(
                    "⬇ Export",
                    data=json.dumps(chart_result.figure_dict, default=str),
                    file_name=f"chart_{key or 'export'}.json",
                    mime="application/json",
                    key=f"chart_export_{key}",
                    use_container_width=True,
                )

    except Exception as exc:
        st.error(f"Chart render error: {exc}")
