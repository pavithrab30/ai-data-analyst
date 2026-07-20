"""
Conversational chat interface — dark corporate theme.

Key features:
- Mode toggle: Ask AI vs Chart-only
- Chart type dropdown (bar, line, pie, scatter, histogram, box, area, heatmap)
- Displays answer, supporting table, SQL, Pandas code, anomaly report,
  reasoning trace, and follow-up suggestions
"""

from __future__ import annotations
from typing import Optional, Callable
import streamlit as st

from models.analysis_models import AnalysisResult
from models.session_models import SessionState
from tools.reasoning_engine import reasoning_engine
from ui.components.anomaly_report import render_anomaly_report
from ui.components.chart_viewer import render_chart, render_chart_selector
from ui.components.insight_cards import render_insight_cards
from utils.logger import get_logger

logger = get_logger(__name__)


def render_chat_interface(
    session: SessionState,
    on_submit: Callable[[str, Optional[str]], Optional[AnalysisResult]],
) -> None:
    """
    Render the full chat UI.

    Args:
        session: Current SessionState.
        on_submit: Callback(question, chart_type_override) → AnalysisResult | None
    """
    # ── Chat history ───────────────────────────────────────────────────────
    if session.conversation_history:
        st.markdown(
            '<div class="section-title">💬 Conversation</div>',
            unsafe_allow_html=True,
        )
        _render_history(session)

    # ── Mode selector + chart type dropdown ────────────────────────────────
    st.markdown("---")

    col_mode, col_chart = st.columns([2, 3])
    with col_mode:
        mode = st.radio(
            "Analysis mode",
            options=["🤖 Ask AI", "📊 Chart Only"],
            horizontal=True,
            label_visibility="collapsed",
            key="chat_mode",
        )
    chart_type_override: Optional[str] = None
    with col_chart:
        if mode == "📊 Chart Only":
            chart_type_override = render_chart_selector(key="chat_chart_type")

    # ── Input ──────────────────────────────────────────────────────────────
    placeholder = (
        f"Describe what to visualize…"
        if mode == "📊 Chart Only"
        else "Ask anything about your data…"
    )

    user_input = st.chat_input(placeholder, key="main_chat_input")

    # Handle pending question from suggested chips
    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")

    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(
                f'<div class="chat-msg-user"><p>{user_input}</p></div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("Analysing…"):
                result = on_submit(user_input, chart_type_override)

            if result:
                _render_result(result)
            else:
                st.error("Analysis failed. Please try again.")


def _render_history(session: SessionState) -> None:
    """Render existing conversation turns."""
    for turn in session.conversation_history:
        if turn.is_user:
            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    f'<div class="chat-msg-user"><p>{turn.content}</p></div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant", avatar="📊"):
                st.markdown(
                    f'<div class="chat-msg-assistant"><p>{turn.content}</p></div>',
                    unsafe_allow_html=True,
                )


def _render_result(result: AnalysisResult) -> None:
    """Render a full AnalysisResult inside a chat message."""
    # ── Primary answer ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="chat-msg-assistant"><p>{result.answer}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Supporting data table ──────────────────────────────────────────────
    _render_supporting_data(result)

    # ── Charts ─────────────────────────────────────────────────────────────
    if result.chart_configs:
        for i, chart_data in enumerate(result.chart_configs):
            try:
                from models.chart_models import ChartResult, ChartConfig, ChartType
                config_dict = chart_data.get("config", {})
                chart_type_str = config_dict.get("chart_type", "bar")
                try:
                    ct = ChartType(chart_type_str)
                except ValueError:
                    ct = ChartType.BAR
                config = ChartConfig(
                    chart_type=ct,
                    title=config_dict.get("title", "Chart"),
                    reasoning=config_dict.get("reasoning", ""),
                )
                cr = ChartResult(
                    config=config,
                    figure_dict=chart_data.get("figure_dict", {}),
                    interpretation=chart_data.get("interpretation", ""),
                    success=True,
                )
                render_chart(cr, key=f"result_{i}_{id(result)}")
            except Exception as exc:
                logger.warning("Chart render failed in chat", error=str(exc))

    # ── SQL ────────────────────────────────────────────────────────────────
    if result.sql_result and result.sql_result.sql_query:
        with st.expander("🗄 Generated SQL", expanded=False):
            st.code(result.sql_result.sql_query, language="sql")
            if result.sql_result.explanation:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#64748b;'
                    f'line-height:1.6;padding:6px 0;">'
                    f'{result.sql_result.explanation}</div>',
                    unsafe_allow_html=True,
                )
            st.download_button(
                "⬇ Download SQL",
                data=result.sql_result.sql_query,
                file_name="query.sql",
                mime="text/plain",
                key=f"sql_dl_{id(result)}",
            )

    # ── Pandas code ────────────────────────────────────────────────────────
    if result.pandas_result and result.pandas_result.pandas_code:
        with st.expander("🐍 Generated Pandas Code", expanded=False):
            st.code(result.pandas_result.pandas_code, language="python")
            if result.pandas_result.explanation:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#64748b;'
                    f'line-height:1.6;padding:6px 0;">'
                    f'{result.pandas_result.explanation}</div>',
                    unsafe_allow_html=True,
                )
            st.download_button(
                "⬇ Download Code",
                data=result.pandas_result.pandas_code,
                file_name="analysis.py",
                mime="text/plain",
                key=f"pd_dl_{id(result)}",
            )

    # ── Anomaly report ─────────────────────────────────────────────────────
    if result.anomaly_report:
        n = result.anomaly_report.anomalies_detected
        with st.expander(
            f"🔍 Anomaly Report — {n} anomalies found",
            expanded=n > 0,
        ):
            render_anomaly_report(result.anomaly_report)

    # ── Business insights ──────────────────────────────────────────────────
    if result.business_insights:
        with st.expander(
            f"💡 Business Insights ({len(result.business_insights)})",
            expanded=False,
        ):
            render_insight_cards(result.business_insights)

    # ── Reasoning trace ────────────────────────────────────────────────────
    if result.reasoning_trace:
        trace_md = reasoning_engine.format_trace_for_display(result.reasoning_trace)
        with st.expander("🔬 Reasoning Trace", expanded=False):
            st.markdown(trace_md)

    # ── Follow-up suggestions ──────────────────────────────────────────────
    if result.suggested_followups:
        st.markdown(
            '<div style="font-size:0.75rem;color:#475569;'
            'font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.08em;margin:8px 0 6px;">Suggested follow-ups</div>',
            unsafe_allow_html=True,
        )
        fu_cols = st.columns(min(3, len(result.suggested_followups)))
        for fcol, q in zip(fu_cols, result.suggested_followups):
            with fcol:
                if st.button(
                    f"↩ {q}",
                    key=f"followup_{hash(q)}_{id(result)}",
                    use_container_width=True,
                ):
                    st.session_state["pending_question"] = q
                    st.rerun()


def _render_supporting_data(result: AnalysisResult) -> None:
    """Render the tabular data table if any results are available."""
    import pandas as pd

    data = None
    columns = None

    if result.query_result and result.query_result.supporting_data:
        data = result.query_result.supporting_data
        columns = result.query_result.columns
    elif result.sql_result and result.sql_result.execution_result:
        er = result.sql_result.execution_result
        if er.success and er.dataframe_result:
            data = er.dataframe_result
            columns = er.columns
    elif result.pandas_result and result.pandas_result.execution_result:
        er = result.pandas_result.execution_result
        if er.success:
            if er.dataframe_result:
                data = er.dataframe_result
                columns = er.columns
            elif er.scalar_result is not None:
                st.markdown(
                    f'<div style="font-size:1.2rem;font-weight:700;'
                    f'color:#e2e8f0;padding:8px 0;">Result: {er.scalar_result}</div>',
                    unsafe_allow_html=True,
                )
                return

    if data and len(data) > 0:
        df = pd.DataFrame(data, columns=columns)
        row_count = len(df)
        with st.expander(
            f"📋 Results table — {row_count:,} row{'s' if row_count != 1 else ''}",
            expanded=row_count <= 20,
        ):
            st.dataframe(df, use_container_width=True, hide_index=True)

            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇ Download Results (CSV)",
                data=buf.getvalue(),
                file_name="query_results.csv",
                mime="text/csv",
                key=f"results_dl_{id(result)}",
            )
