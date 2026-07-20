"""Dataset preview table component — dark corporate theme."""

from __future__ import annotations
import io
import streamlit as st
import pandas as pd
from config.settings import settings
from utils.formatters import format_dataframe_for_display


def render_data_preview(df: pd.DataFrame, title: str = "Data Preview") -> None:
    """Render a styled, downloadable data preview table."""
    st.markdown(
        f'<div class="section-title">🗃 {title}</div>',
        unsafe_allow_html=True,
    )

    total_rows = len(df)
    total_cols = len(df.columns)
    display_df = format_dataframe_for_display(df, max_rows=settings.max_rows_display)

    # Header row with stats and download
    col_info, col_dl = st.columns([4, 1])
    with col_info:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#475569;margin-bottom:6px;">'
            f'Showing <b style="color:#94a3b8;">{len(display_df):,}</b> of '
            f'<b style="color:#94a3b8;">{total_rows:,}</b> rows · '
            f'<b style="color:#94a3b8;">{total_cols}</b> columns'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_dl:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "⬇ CSV",
            data=buf.getvalue(),
            file_name="dataset_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=380,
    )
