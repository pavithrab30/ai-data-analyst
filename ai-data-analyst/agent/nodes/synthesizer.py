"""
Synthesizer node — combines tool outputs into a final answer.

This node collects all tool outputs from the executor,
builds a results summary, calls Gemini to generate a
well-structured response, builds the ReasoningTrace,
and assembles the final AnalysisResult.

Reads from state:  all tool outputs, intent, question
Writes to state:   final_answer, reasoning_trace, analysis_result
"""

from __future__ import annotations

import time
from typing import Any

from agent.graph.state import AgentState
from agent.prompts.synthesizer_prompts import (
    SYNTHESIZER_FOLLOWUP_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from models.analysis_models import AnalysisResult, ConfidenceLevel, ReasoningTrace
from services.llm_service import LLMService
from tools.reasoning_engine import reasoning_engine
from utils.logger import get_logger

logger = get_logger(__name__)


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesize tool outputs into a final response.

    Args:
        state: AgentState with all tool outputs populated.

    Returns:
        Updated AgentState with final_answer and analysis_result.
    """
    start_time = time.perf_counter()
    question = state.get("question", "")
    dataset_name = state.get("dataset_name", "dataset")

    logger.info("Synthesizer node executing", question=question[:100])

    # ── Build results summary for Gemini ──────────────────────────────────────
    results_summary = _build_results_summary(state)

    # ── Generate final answer ─────────────────────────────────────────────────
    final_answer = _generate_answer(question, dataset_name, results_summary)

    # ── Generate follow-up suggestions ────────────────────────────────────────
    suggested_followups = _generate_followups(question, dataset_name)

    # ── Build reasoning trace ─────────────────────────────────────────────────
    intent = state.get("intent")
    tools_invoked = intent.tools_required if intent else []
    tool_times = state.get("tool_execution_times", {})

    sql_code = None
    pandas_code = None

    if state.get("sql_result"):
        sql_code = state["sql_result"].sql_query

    if state.get("pandas_result"):
        pandas_code = state["pandas_result"].pandas_code

    reasoning_trace = reasoning_engine.build_trace(
        question=question,
        intent=intent,
        tools_invoked=tools_invoked,
        tool_execution_times=tool_times,
        sql_generated=sql_code,
        pandas_code_generated=pandas_code,
        confidence=_determine_confidence(state),
        start_time=start_time,
    )

    # ── Assemble AnalysisResult ────────────────────────────────────────────────
    chart_configs = []
    for cr in (state.get("chart_results") or []):
        if cr.success:
            chart_configs.append({
                "config": cr.config.model_dump(),
                "figure_dict": cr.figure_dict,
                "interpretation": cr.interpretation,
            })

    analysis_result = AnalysisResult(
        question=question,
        answer=final_answer,
        reasoning_trace=reasoning_trace,
        query_result=state.get("query_result"),
        sql_result=state.get("sql_result"),
        pandas_result=state.get("pandas_result"),
        anomaly_report=state.get("anomaly_report"),
        business_insights=state.get("business_insights") or [],
        chart_configs=chart_configs,
        suggested_followups=suggested_followups,
        error=state.get("error"),
    )

    logger.info(
        "Synthesizer complete",
        answer_length=len(final_answer),
        charts=len(chart_configs),
        insights=len(analysis_result.business_insights),
    )

    return {
        **state,
        "final_answer": final_answer,
        "reasoning_trace": reasoning_trace,
        "analysis_result": analysis_result,
    }


def _build_results_summary(state: AgentState) -> str:
    """Build a structured summary of all tool outputs for the synthesis prompt."""
    parts: list[str] = []

    # Query result
    if state.get("query_result"):
        qr = state["query_result"]
        row_count = len(qr.supporting_data) if qr.supporting_data else 0
        parts.append(f"Query Result: {qr.answer} ({row_count} rows returned)")

    # SQL result
    if state.get("sql_result"):
        sr = state["sql_result"]
        if sr.is_valid and sr.execution_result and sr.execution_result.success:
            rows = sr.execution_result.row_count or 0
            parts.append(f"SQL Query executed successfully: {rows} rows returned")
            parts.append(f"SQL: {sr.sql_query}")

    # Pandas result
    if state.get("pandas_result"):
        pr = state["pandas_result"]
        if pr.is_safe and pr.execution_result and pr.execution_result.success:
            if pr.execution_result.scalar_result is not None:
                parts.append(f"Pandas computation result: {pr.execution_result.scalar_result}")
            else:
                parts.append(f"Pandas code executed: {pr.execution_result.row_count} rows")

    # Charts
    charts = state.get("chart_results") or []
    if charts:
        chart_descriptions = [
            f"  - {cr.config.chart_type.value} chart: {cr.config.title}"
            for cr in charts if cr.success
        ]
        if chart_descriptions:
            parts.append("Charts generated:\n" + "\n".join(chart_descriptions))

    # Anomaly report
    if state.get("anomaly_report"):
        ar = state["anomaly_report"]
        parts.append(
            f"Anomaly Detection: {ar.anomalies_detected} anomalies found "
            f"({ar.anomaly_percentage:.1f}%) using {ar.detection_method}"
        )

    # Business insights
    insights = state.get("business_insights") or []
    if insights:
        insight_lines = [f"  - {ins.title}" for ins in insights[:5]]
        parts.append("Business Insights:\n" + "\n".join(insight_lines))

    return "\n\n".join(parts) if parts else "No analytical results available."


def _generate_answer(question: str, dataset_name: str, results_summary: str) -> str:
    """Call LLM to synthesize results into a final answer."""
    prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
        dataset_name=dataset_name,
        question=question,
        results_summary=results_summary,
    )
    try:
        llm = LLMService()
        return llm.generate(results_summary, system_instruction=prompt)
    except Exception as exc:
        logger.error("Synthesizer LLM call failed", error=str(exc))
        return results_summary if results_summary else f"I analyzed your question: '{question}'. However, I encountered an error generating the full response. Please try again."


def _generate_followups(question: str, dataset_name: str) -> list[str]:
    """Generate follow-up question suggestions."""
    prompt = SYNTHESIZER_FOLLOWUP_PROMPT.format(
        dataset_name=dataset_name,
        question=question,
    )
    try:
        llm = LLMService()
        raw = llm.generate(prompt, temperature_override=0.5)
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in raw.strip().split("\n")
            if line.strip() and any(c.isalpha() for c in line)
        ]
        return [l for l in lines if len(l) > 10][:3]
    except Exception:
        return []


def _determine_confidence(state: AgentState) -> ConfidenceLevel:
    """Determine overall response confidence from tool outputs."""
    if state.get("error"):
        return ConfidenceLevel.LOW

    successful_tools = 0
    total_tools = 0

    if state.get("query_result"):
        total_tools += 1
        if state["query_result"].confidence != ConfidenceLevel.LOW:
            successful_tools += 1

    if state.get("sql_result"):
        total_tools += 1
        if state["sql_result"].is_valid:
            successful_tools += 1

    if state.get("pandas_result"):
        total_tools += 1
        if state["pandas_result"].is_safe:
            successful_tools += 1

    if state.get("anomaly_report"):
        total_tools += 1
        successful_tools += 1

    if state.get("business_insights"):
        total_tools += 1
        successful_tools += 1

    if total_tools == 0:
        return ConfidenceLevel.LOW

    ratio = successful_tools / total_tools
    if ratio >= 0.8:
        return ConfidenceLevel.HIGH
    if ratio >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


