"""
Main dashboard component.

Renders the post-upload dashboard: KPI row, suggested questions,
profile/data/insights/anomaly tabs, and merge banner.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from models.analysis_models import DataProfile, BusinessInsight
from services.session_service import SessionService
from tools.insight_generator import insight_generator
from ui.components.data_preview import render_data_preview
from ui.components.kpi_cards import render_kpi_cards
from ui.components.profile_report import render_profile_report
from ui.components.insight_cards import render_insight_cards
from ui.components.anomaly_report import render_anomaly_report
from utils.logger import get_logger

logger = get_logger(__name__)

# Suggested questions shown as quick chips
_SUGGESTED_QUESTIONS = [
    "Which region has the highest total revenue?",
    "Show monthly revenue trend as a line chart",
    "Who are the top 10 customers by revenue?",
    "Find underperforming products by profit",
    "Show a bar chart of revenue by region",
    "Detect anomalies in this dataset",
    "Generate business insights",
    "What is the average discount by sales rep?",
]


def render_dashboard(df: pd.DataFrame, profile: DataProfile) -> None:
    """
    Render the full dashboard for a loaded dataset.

    Args:
        df: Active DataFrame.
        profile: Pre-computed DataProfile.
    """
    # ── Merge recommendation banner ────────────────────────────────────────
    _render_merge_banner()

    # ── KPI row ────────────────────────────────────────────────────────────
    kpis = st.session_state.get("kpis")
    if kpis is None:
        with st.spinner("Computing KPIs…"):
            kpis = insight_generator.generate_kpis(df)
            st.session_state["kpis"] = kpis

    if kpis:
        render_kpi_cards(kpis)
        st.markdown("---")

    # ── Suggested questions ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">⚡ Quick Questions</div>',
        unsafe_allow_html=True,
    )
    # Render chips as buttons in a horizontal wrap
    chips_html = "".join(
        f'<span class="suggest-chip" '
        f'onclick="void(0)">{q}</span>'
        for q in _SUGGESTED_QUESTIONS
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px;">'
        f'{chips_html}</div>',
        unsafe_allow_html=True,
    )

    # Actual clickable buttons (Streamlit doesn't support HTML onclick)
    num_cols = 4
    q_rows = [
        _SUGGESTED_QUESTIONS[i : i + num_cols]
        for i in range(0, len(_SUGGESTED_QUESTIONS), num_cols)
    ]
    for q_row in q_rows:
        btn_cols = st.columns(len(q_row))
        for bcol, question in zip(btn_cols, q_row):
            with bcol:
                if st.button(
                    question,
                    key=f"suggest_{hash(question)}",
                    use_container_width=True,
                ):
                    st.session_state["pending_question"] = question
                    st.rerun()

    st.markdown("---")

    # ── Main tabs ──────────────────────────────────────────────────────────
    tab_profile, tab_data, tab_insights, tab_anomalies = st.tabs([
        "📊 Profile",
        "🗃 Data Preview",
        "💡 Insights",
        "🔍 Anomaly Detection",
    ])

    with tab_profile:
        render_profile_report(profile)

    with tab_data:
        render_data_preview(df, title=profile.dataset_name)

    with tab_insights:
        _render_insights_tab(df, profile.dataset_name)

    with tab_anomalies:
        _render_anomaly_tab(df)


def render_welcome_screen() -> None:
    """Welcome screen shown when no dataset is loaded."""
    st.markdown(
        """
        <div style="text-align:center;padding:48px 24px 32px;">
          <div style="font-size:3.5rem;margin-bottom:16px;">📊</div>
          <div style="font-size:1.8rem;font-weight:700;color:#e2e8f0;
                      letter-spacing:-0.5px;margin-bottom:12px;">
            AI Data Analyst
          </div>
          <div style="font-size:0.95rem;color:#64748b;max-width:500px;
                      margin:0 auto;line-height:1.7;">
            Upload a CSV file from the sidebar to begin.<br>
            The AI will profile your data and answer questions in plain English.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature grid
    features = [
        ("🤖", "Natural Language Q&A",
         "Ask questions in plain English, get instant data-backed answers."),
        ("📊", "Interactive Charts",
         "Choose chart type from a dropdown — bar, line, pie, scatter, and more."),
        ("🔍", "Anomaly Detection",
         "Isolation Forest identifies statistical outliers with explanations."),
        ("💡", "Business Insights",
         "AI surfaces top performers, trends, and concentration risks automatically."),
        ("🗄", "SQL Generation",
         "Get production-ready SQL with step-by-step explanations."),
        ("🐍", "Pandas Code",
         "Every analysis includes equivalent Python/Pandas code."),
        ("🔗", "Multi-CSV Joins",
         "Upload multiple CSVs — the AI recommends and executes the right join."),
        ("📄", "Downloadable Reports",
         "Export full analysis as Markdown or PDF in one click."),
        ("🛡", "Secure Execution",
         "All generated code is AST-validated — no eval(), no exec() risks."),
    ]

    for i in range(0, len(features), 3):
        row = features[i : i + 3]
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="card" style="text-align:left;min-height:110px;">
                      <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
                      <div style="font-size:0.85rem;font-weight:600;
                                  color:#e2e8f0;margin-bottom:5px;">{title}</div>
                      <div style="font-size:0.78rem;color:#64748b;
                                  line-height:1.6;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div style="text-align:center;color:#334155;'
        'font-size:0.8rem;margin-top:24px;padding:10px 20px;'
        'background:#141828;border:1px solid #252d47;'
        'border-radius:20px;width:fit-content;margin-left:auto;margin-right:auto;">'
        '← Upload a CSV in the sidebar to begin'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_insights_tab(df: pd.DataFrame, dataset_name: str) -> None:
    """Render the Insights tab with lazy loading."""
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate = st.button(
            "✨ Generate Insights",
            key="gen_insights",
            use_container_width=True,
        )

    cached = st.session_state.get("insights")

    if generate or cached is None:
        with st.spinner("Generating business insights…"):
            try:
                from tools.insight_generator import insight_generator
                insights = insight_generator.generate(df, dataset_name=dataset_name)
                st.session_state["insights"] = insights
                render_insight_cards(insights)
            except Exception as exc:
                st.error(f"Insight generation failed: {exc}")
    elif cached:
        render_insight_cards(cached)
    else:
        st.info("Click **Generate Insights** to analyse your dataset.")


def _render_anomaly_tab(df: pd.DataFrame) -> None:
    """Render the Anomaly Detection tab with controls."""
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        contamination = st.slider(
            "Expected anomaly rate",
            min_value=0.01, max_value=0.20,
            value=0.05, step=0.01,
            format="%.0f%%",
            help="Expected proportion of anomalous records in the dataset.",
        )
    with col_b:
        run_btn = st.button(
            "🔍 Detect Anomalies",
            key="run_anomaly",
            use_container_width=True,
        )

    cached = st.session_state.get("anomaly_report")

    if run_btn or cached is None:
        with st.spinner("Running anomaly detection…"):
            try:
                from tools.anomaly_detector import anomaly_detector
                report = anomaly_detector.detect(df, contamination=contamination)
                st.session_state["anomaly_report"] = report
                render_anomaly_report(report)
            except Exception as exc:
                st.error(f"Anomaly detection failed: {exc}")
    elif cached:
        render_anomaly_report(cached)
    else:
        st.info("Click **Detect Anomalies** to find outliers in your data.")


def _render_merge_banner() -> None:
    """Show merge recommendation if multiple datasets loaded."""
    if st.session_state.get("merge_dismissed"):
        return

    rec = st.session_state.get("merge_recommendation")
    if not rec:
        return

    with st.container():
        st.markdown(
            f"""
            <div style="background:rgba(99,102,241,0.07);
                        border:1px solid rgba(99,102,241,0.25);
                        border-radius:12px;padding:14px 18px;
                        margin-bottom:16px;">
              <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;color:#818cf8;margin-bottom:6px;">
                🔗 Join Recommendation
              </div>
              <div style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                <b style="color:#e2e8f0;">{rec['left_dataset']}</b>
                &nbsp;⟷&nbsp;
                <b style="color:#e2e8f0;">{rec['right_dataset']}</b>
                &nbsp;·&nbsp;
                Key: <code style="background:#1c2137;padding:1px 6px;
                              border-radius:4px;color:#818cf8;font-size:0.8rem;">
                  {', '.join(rec['join_keys'])}
                </code>
                &nbsp;·&nbsp;
                {rec['join_type']} join
                &nbsp;·&nbsp;
                <span style="color:#10b981;">{rec['confidence']*100:.0f}% confidence</span>
              </div>
              <div style="font-size:0.78rem;color:#475569;margin-top:4px;">
                {rec['reasoning']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Apply Merge", key="apply_merge", use_container_width=True):
                _execute_merge(rec)
        with col2:
            if st.button("✖ Skip", key="skip_merge", use_container_width=True):
                st.session_state["merge_dismissed"] = True
                st.rerun()


def _execute_merge(rec: dict) -> None:
    """Execute the recommended dataset merge."""
    from tools.schema_merger import schema_merger, MergeRecommendation
    from tools.data_profiler import data_profiler

    with st.spinner("Merging datasets…"):
        try:
            datasets = {
                rec["left_dataset"]:  SessionService.get_dataframe(rec["left_dataset"]),
                rec["right_dataset"]: SessionService.get_dataframe(rec["right_dataset"]),
            }
            if None in datasets.values():
                st.error("One or more datasets not found in session.")
                return

            merge_rec = MergeRecommendation(
                left_dataset=rec["left_dataset"],
                right_dataset=rec["right_dataset"],
                join_keys=rec["join_keys"],
                join_type=rec["join_type"],
                confidence=rec["confidence"],
                reasoning=rec["reasoning"],
            )
            merged_df = schema_merger.execute_merge(datasets, merge_rec)
            merged_name = f"{rec['left_dataset']}_x_{rec['right_dataset']}"

            profile = data_profiler.profile(merged_df, dataset_name=merged_name)
            SessionService.register_dataset(
                name=merged_name, df=merged_df,
                filename=f"{merged_name}.csv",
                is_merged=True,
                source_files=list(datasets.keys()),
            )
            st.session_state[f"profile_{merged_name}"] = profile
            st.session_state["active_profile"] = profile
            st.session_state["merge_dismissed"] = True
            st.session_state["kpis"] = None
            st.session_state["insights"] = None
            st.session_state["anomaly_report"] = None

            st.success(f"✅ Merged dataset created: **{merged_name}** ({len(merged_df):,} rows)")
            st.rerun()
        except Exception as exc:
            st.error(f"Merge failed: {exc}")
