"""Business insight cards — dark corporate theme."""

from __future__ import annotations
import streamlit as st
from models.analysis_models import BusinessInsight

_CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}

_TYPE_COLORS = {
    "trend":          ("#3b82f6", "badge-trend"),
    "top_performer":  ("#10b981", "badge-top_performer"),
    "underperformer": ("#f59e0b", "badge-underperformer"),
    "anomaly":        ("#ef4444", "badge-anomaly"),
    "general":        ("#8b5cf6", "badge-general"),
}


def render_insight_cards(insights: list[BusinessInsight]) -> None:
    """Render business insights as styled dark cards."""
    if not insights:
        st.info("No insights available. Ask a question or click 'Generate Insights'.")
        return

    st.markdown(
        f'<div class="section-title">💡 Business Insights '
        f'<span style="color:#334155;font-weight:400;">({len(insights)} found)</span></div>',
        unsafe_allow_html=True,
    )

    cols_per_row = 2
    for i in range(0, len(insights), cols_per_row):
        row = insights[i : i + cols_per_row]
        cols = st.columns(len(row))

        for col, insight in zip(cols, row):
            with col:
                color, badge_class = _TYPE_COLORS.get(
                    insight.insight_type, ("#8b5cf6", "badge-general")
                )
                conf_icon = _CONF_ICON.get(insight.confidence.value, "⚪")

                metric_html = ""
                if insight.metric_name and insight.metric_value is not None:
                    v = insight.metric_value
                    if isinstance(v, float):
                        v_str = f"{v:,.2f}"
                    elif isinstance(v, int):
                        v_str = f"{v:,}"
                    else:
                        v_str = str(v)
                    metric_html = (
                        f'<div class="insight-metric">'
                        f'<span style="color:#475569;">{insight.metric_name}:</span> '
                        f'<b style="color:#e2e8f0;">{v_str}</b>'
                        f'</div>'
                    )

                st.markdown(
                    f"""
                    <div class="insight-card" style="border-top-color:{color};">
                      <div style="display:flex;align-items:center;
                                  justify-content:space-between;margin-bottom:8px;">
                        <span class="insight-type-badge {badge_class}">
                          {insight.insight_type.replace("_"," ")}
                        </span>
                        <span title="Confidence: {insight.confidence.value}">{conf_icon}</span>
                      </div>
                      <div class="insight-title">{insight.title}</div>
                      <div class="insight-desc">{insight.description}</div>
                      {metric_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
