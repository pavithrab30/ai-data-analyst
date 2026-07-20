"""
Dataset profile report — premium dark theme.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from models.analysis_models import DataProfile
from utils.formatters import format_bytes, format_number

# Base layout — NO legend key here to avoid duplicate keyword argument errors
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"),
    margin=dict(t=36, b=28, l=44, r=16),
    height=260,
    xaxis=dict(gridcolor="#1e2640", linecolor="#1e2640", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1e2640", linecolor="#1e2640", tickfont=dict(color="#64748b")),
)

_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]


def _layout(**extra):
    """Return base layout merged with extra kwargs — prevents duplicate keys."""
    return {**_BASE_LAYOUT, **extra}


def render_profile_report(profile: DataProfile) -> None:
    qr = profile.quality_report
    score = qr.overall_quality_score
    score_color = (
        "#10b981" if score >= 80 else
        "#f59e0b" if score >= 60 else
        "#ef4444"
    )

    # ── Summary strip ──────────────────────────────────────────────────────
    cols = st.columns(7)
    metrics = [
        ("Rows",        f"{profile.row_count:,}"),
        ("Columns",     str(profile.column_count)),
        ("Memory",      format_bytes(profile.memory_usage_bytes)),
        ("Numeric",     str(len(profile.numeric_columns))),
        ("Categorical", str(len(profile.categorical_columns))),
        ("Dates",       str(len(profile.date_columns))),
        ("Quality",     f"{score:.0f}/100"),
    ]
    for col, (label, val) in zip(cols, metrics):
        is_quality = label == "Quality"
        col.markdown(
            f"""<div style="background:#141828;border:1px solid #1e2640;
                border-radius:10px;padding:12px 10px;text-align:center;">
                <div style="font-size:1.25rem;font-weight:700;
                    color:{score_color if is_quality else '#e2e8f0'};">{val}</div>
                <div style="font-size:0.65rem;text-transform:uppercase;
                    letter-spacing:0.09em;color:#475569;margin-top:3px;">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Quality bar
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:12px;margin:14px 0 4px;">
            <div style="flex:1;height:4px;background:#1e2640;border-radius:2px;overflow:hidden;">
                <div style="width:{score:.0f}%;height:100%;background:{score_color};
                    border-radius:2px;transition:width .5s;"></div>
            </div>
            <div style="font-size:0.78rem;font-weight:700;color:{score_color};min-width:52px;">
                {score:.0f} / 100
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab_overview, tab_columns, tab_quality, tab_correlations = st.tabs(
        ["📋 Overview", "🔎 Column Detail", "🛡 Quality", "📐 Correlations"]
    )

    # ── Overview ───────────────────────────────────────────────────────────
    with tab_overview:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            _section("Column Summary")
            summary = []
            for cp in profile.column_profiles:
                summary.append({
                    "Column": cp.name,
                    "Type":   cp.dtype,
                    "Null %": f"{cp.null_percentage:.1f}%",
                    "Unique": f"{cp.unique_count:,}",
                    "Sample": ", ".join(str(v) for v in cp.sample_values[:3]),
                })
            if summary:
                st.dataframe(
                    pd.DataFrame(summary),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

        with col_right:
            _section("Type Distribution")
            dtype_data = {k: v for k, v in profile.dtypes_summary.items() if v > 0}
            if dtype_data:
                fig = go.Figure(go.Pie(
                    labels=list(dtype_data.keys()),
                    values=list(dtype_data.values()),
                    hole=0.55,
                    marker_colors=_COLORS,
                    textfont=dict(color="#94a3b8"),
                    hovertemplate="<b>%{label}</b><br>%{value} columns<extra></extra>",
                ))
                fig.update_layout(**_layout(
                    height=260,
                    showlegend=True,
                    legend=dict(
                        orientation="v", x=1.02, y=0.5,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", size=10),
                    ),
                ))
                fig.update_traces(textposition="inside", textinfo="percent")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Column Detail ──────────────────────────────────────────────────────
    with tab_columns:
        col_names = [cp.name for cp in profile.column_profiles]
        selected = st.selectbox(
            "Select column",
            col_names,
            label_visibility="collapsed",
        )
        cp = next((c for c in profile.column_profiles if c.name == selected), None)
        if not cp:
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Type",    cp.dtype)
        m2.metric("Missing", f"{cp.null_count:,} ({cp.null_percentage:.1f}%)")
        m3.metric("Unique",  f"{cp.unique_count:,} ({cp.unique_percentage:.1f}%)")

        col_a, col_b = st.columns(2)

        with col_a:
            if cp.mean is not None:
                _section("Statistics")
                stats_rows = [
                    ("Mean",   cp.mean),  ("Std Dev", cp.std),
                    ("Min",    cp.min_value), ("Q25",   cp.q25),
                    ("Median", cp.median),    ("Q75",   cp.q75),
                    ("Max",    cp.max_value),
                ]
                stats_df = pd.DataFrame(
                    [{"Statistic": k, "Value": format_number(v)}
                     for k, v in stats_rows if v is not None]
                )
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            if cp.min_date:
                _section("Date Range")
                st.markdown(
                    f"""<div style="background:#141828;border:1px solid #1e2640;
                        border-radius:8px;padding:12px 14px;font-size:0.82rem;color:#94a3b8;">
                        <b style="color:#e2e8f0;">Min:</b> {cp.min_date}<br>
                        <b style="color:#e2e8f0;">Max:</b> {cp.max_date}<br>
                        <b style="color:#e2e8f0;">Range:</b> {cp.date_range_days} days
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col_b:
            if cp.value_counts:
                _section("Top Values")
                vc_df = pd.DataFrame(
                    list(cp.value_counts.items())[:12],
                    columns=["Value", "Count"],
                )
                fig = px.bar(
                    vc_df, x="Count", y="Value",
                    orientation="h",
                    color_discrete_sequence=["#6366f1"],
                )
                fig.update_layout(**_layout(height=240, showlegend=False))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if cp.sample_values:
                _section("Sample Values")
                st.code(", ".join(str(v) for v in cp.sample_values), language=None)

    # ── Quality ────────────────────────────────────────────────────────────
    with tab_quality:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Duplicate Rows", f"{qr.duplicate_rows:,}")
        q2.metric("Cols w/ Nulls",  str(len(qr.columns_with_nulls)))
        q3.metric("Constant Cols",  str(len(qr.constant_columns)))
        q4.metric("Type Issues",    str(len(qr.type_inconsistencies)))

        st.markdown("---")
        if qr.quality_issues:
            _section("⚠ Issues Detected")
            for issue in qr.quality_issues:
                st.warning(issue)
        else:
            st.success("✅ No significant data quality issues detected.")

        if qr.columns_with_nulls:
            _section("Null Distribution")
            null_df = pd.DataFrame(
                sorted(qr.columns_with_nulls.items(), key=lambda x: x[1], reverse=True),
                columns=["Column", "Null %"],
            )
            fig = px.bar(
                null_df, x="Column", y="Null %",
                color="Null %",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                range_color=[0, 100],
            )
            fig.update_layout(**_layout(
                height=220,
                coloraxis_showscale=False,
                xaxis=dict(tickangle=-35, gridcolor="#1e2640"),
                yaxis=dict(gridcolor="#1e2640"),
            ))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Correlations ───────────────────────────────────────────────────────
    with tab_correlations:
        if not profile.top_correlations:
            st.info("Not enough numeric columns for correlation analysis.")
            return

        _section("Top Column Correlations")
        corr_data = []
        for c in profile.top_correlations:
            abs_val = abs(c["correlation"])
            corr_data.append({
                "Column A":    c["col1"],
                "Column B":    c["col2"],
                "Correlation": c["correlation"],
                "Strength":    "Strong" if abs_val > 0.7 else "Moderate" if abs_val > 0.4 else "Weak",
            })
        corr_df = pd.DataFrame(corr_data)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            corr_df.head(10),
            x="Correlation", y="Column A",
            orientation="h",
            color="Correlation",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            text="Correlation",
        )
        fig.update_layout(**_layout(height=280, coloraxis_showscale=False))
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _section(title: str) -> None:
    """Render a styled section title."""
    st.markdown(
        f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#475569;margin:14px 0 8px;padding-bottom:6px;'
        f'border-bottom:1px solid #1e2640;">{title}</div>',
        unsafe_allow_html=True,
    )
