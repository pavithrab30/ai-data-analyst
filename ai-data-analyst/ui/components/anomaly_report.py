"""Anomaly detection report component — deep navy purple theme."""

from __future__ import annotations
import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from models.analysis_models import AnomalyReport

# ── Shared Plotly dark layout ──────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"),
    margin=dict(t=36, b=28, l=44, r=16),
    height=240,
    xaxis=dict(gridcolor="#1a2540", linecolor="#1a2540", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1a2540", linecolor="#1a2540", tickfont=dict(color="#64748b")),
)


def _L(**extra) -> dict:
    """Merge base layout with extras — no duplicate key risk."""
    return {**_BASE, **extra}


def render_anomaly_report(report: AnomalyReport) -> None:
    """Render AnomalyReport: summary metrics, score histogram, records table."""

    pct = report.anomaly_percentage
    pct_color = "#ef4444" if pct > 10 else "#f59e0b" if pct > 5 else "#10b981"

    # ── Summary metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records Analyzed", f"{report.total_records_analyzed:,}")
    c2.metric("Anomalies Found",  str(report.anomalies_detected))
    c3.metric("Anomaly Rate",     f"{pct:.1f}%")
    c4.metric("Method", report.detection_method.replace("_", " ").title())

    # Rate bar
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;margin:10px 0 14px;">
            <div style="flex:1;height:4px;background:#141d35;
                        border-radius:2px;overflow:hidden;">
              <div style="width:{min(pct * 5, 100):.1f}%;height:100%;
                          background:{pct_color};border-radius:2px;"></div>
            </div>
            <div style="font-size:0.75rem;font-weight:700;
                        color:{pct_color};min-width:44px;text-align:right;">
              {pct:.1f}%
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── LLM narrative summary ──────────────────────────────────────────────
    if report.summary:
        st.markdown(
            f"""<div style="background:rgba(124,58,237,0.06);
                    border:1px solid rgba(124,58,237,0.18);
                    border-radius:8px;padding:14px 16px;
                    font-size:0.84rem;color:#94a3b8;
                    line-height:1.75;margin-bottom:16px;">
                {report.summary}
            </div>""",
            unsafe_allow_html=True,
        )

    if report.anomalies_detected == 0:
        st.success("✅ No anomalies detected. All records are within expected statistical ranges.")
        return

    # ── Score distribution chart ───────────────────────────────────────────
    if report.anomaly_records:
        scores = [r.anomaly_score for r in report.anomaly_records]
        fig = go.Figure(go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color="#7c3aed",
            marker_line_color="#1a2540",
            marker_line_width=1,
            opacity=0.9,
            name="Anomaly Scores",
        ))
        fig.update_layout(**_L(
            showlegend=False,
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            title=dict(
                text="Anomaly Score Distribution",
                font=dict(color="#e2e8f0", size=13),
                x=0.5, xanchor="center",
            ),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Records table ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">🔍 Anomalous Records</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for rec in report.anomaly_records[:25]:
        flags = rec.flagged_columns[:3]
        z_info = ", ".join(f"{c} (z={rec.z_scores.get(c, 0):.1f})" for c in flags)
        row: dict = {
            "Row #":    rec.row_index,
            "Score":    f"{rec.anomaly_score:.4f}",
            "Flagged":  z_info or "—",
            "Why":      (rec.explanation[:110] + "…") if len(rec.explanation) > 110 else rec.explanation,
        }
        for col in flags[:2]:
            val = rec.record_data.get(col, "—")
            row[col] = str(val)[:28]
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320)

        # CSV export
        buf = io.StringIO()
        full = []
        for rec in report.anomaly_records:
            r = {"row_index": rec.row_index, "anomaly_score": rec.anomaly_score,
                 "explanation": rec.explanation}
            r.update(rec.record_data)
            full.append(r)
        pd.DataFrame(full).to_csv(buf, index=False)
        st.download_button(
            "⬇ Download Anomaly Records (CSV)",
            data=buf.getvalue(),
            file_name="anomaly_records.csv",
            mime="text/csv",
        )
