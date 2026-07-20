"""KPI cards — dark corporate style."""

from __future__ import annotations
from typing import Any
import streamlit as st


def render_kpi_cards(kpis: dict[str, Any]) -> None:
    """Render KPI metric cards in a responsive grid."""
    if not kpis:
        return

    items = list(kpis.items())[:8]
    cols_per_row = min(4, len(items))

    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start : row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            with col:
                if isinstance(value, float):
                    if value >= 1_000_000:
                        display = f"{value/1_000_000:.2f}M"
                    elif value >= 1_000:
                        display = f"{value:,.0f}"
                    else:
                        display = f"{value:,.2f}"
                elif isinstance(value, int):
                    display = f"{value:,}"
                else:
                    display = str(value)

                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-value">{display}</div>
                      <div class="kpi-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
