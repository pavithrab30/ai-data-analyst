"""
Executor node — dispatches to analytical tools based on planner intent.

Reads from state: intent, question, dataframe, table_name, dataset_name
Writes to state:  query_result, sql_result, pandas_result,
                  chart_results, anomaly_report, business_insights,
                  tool_execution_times
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from agent.graph.state import AgentState
from models.analysis_models import IntentType
from tools.anomaly_detector import anomaly_detector
from tools.insight_generator import insight_generator
from tools.pandas_generator import pandas_generator
from tools.query_engine import query_engine
from tools.sql_generator import sql_generator
from tools.visualizer import visualizer
from utils.logger import get_logger

logger = get_logger(__name__)


def executor_node(state: AgentState) -> AgentState:
    """
    Execute the tools specified by the planner's intent classification.

    Each tool invocation is timed individually for the reasoning trace.
    Failures are caught and logged without stopping the workflow —
    partial results are better than total failures.

    Args:
        state: AgentState with populated `intent` field.

    Returns:
        Updated AgentState with tool outputs.
    """
    intent = state.get("intent")
    if not intent:
        logger.warning("Executor called without intent, using query_engine as default")
        from models.analysis_models import IntentClassification, IntentType, ConfidenceLevel
        intent = IntentClassification(
            primary_intent=IntentType.QUESTION_ANSWERING,
            tools_required=["query_engine"],
            reasoning="No intent provided",
            confidence=ConfidenceLevel.LOW,
            original_query=state.get("question", ""),
        )

    question = state.get("question", "")
    df: pd.DataFrame = state.get("dataframe")
    table_name = state.get("table_name", "dataset")

    tools_to_invoke = intent.tools_required
    logger.info("Executor dispatching tools", tools=tools_to_invoke)

    times: dict[str, float] = dict(state.get("tool_execution_times", {}))

    # Accumulators
    query_result = state.get("query_result")
    sql_result = state.get("sql_result")
    pandas_result = state.get("pandas_result")
    chart_results = list(state.get("chart_results") or [])
    anomaly_report = state.get("anomaly_report")
    business_insights = list(state.get("business_insights") or [])

    if df is None:
        logger.error("No DataFrame available in executor")
        return {**state, "error": "No dataset loaded. Please upload a CSV file first."}

    # ── Dispatch ───────────────────────────────────────────────────────────────
    for tool_name in tools_to_invoke:

        if tool_name == "query_engine":
            start = time.perf_counter()
            try:
                query_result = query_engine.answer(question, df, table_name)
                logger.info("query_engine completed")
            except Exception as exc:
                logger.error("query_engine failed", error=str(exc))
            times["query_engine"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "sql_generator":
            start = time.perf_counter()
            try:
                sql_result = sql_generator.generate_and_execute(question, df, table_name)
                logger.info("sql_generator completed")
            except Exception as exc:
                logger.error("sql_generator failed", error=str(exc))
            times["sql_generator"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "pandas_generator":
            start = time.perf_counter()
            try:
                pandas_result = pandas_generator.generate_and_execute(question, df)
                logger.info("pandas_generator completed")
            except Exception as exc:
                logger.error("pandas_generator failed", error=str(exc))
            times["pandas_generator"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "visualizer":
            start = time.perf_counter()
            try:
                # Read chart type hint from Streamlit session state if available
                chart_type_hint = None
                try:
                    import streamlit as _st
                    hint = _st.session_state.get("chart_type_hint")
                    if hint is not None:
                        chart_type_hint = hint
                except Exception:
                    pass

                chart = visualizer.create_chart(question, df, chart_type_hint=chart_type_hint)
                chart_results.append(chart)
                logger.info("visualizer completed", chart_type=str(chart_type_hint))
            except Exception as exc:
                logger.error("visualizer failed", error=str(exc))
            times["visualizer"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "anomaly_detector":
            start = time.perf_counter()
            try:
                anomaly_report = anomaly_detector.detect(df)
                logger.info(
                    "anomaly_detector completed",
                    anomalies=anomaly_report.anomalies_detected,
                )
            except Exception as exc:
                logger.error("anomaly_detector failed", error=str(exc))
            times["anomaly_detector"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "insight_generator":
            start = time.perf_counter()
            try:
                new_insights = insight_generator.generate(
                    df, dataset_name=state.get("dataset_name", "dataset")
                )
                business_insights.extend(new_insights)
                logger.info("insight_generator completed", insights=len(new_insights))
            except Exception as exc:
                logger.error("insight_generator failed", error=str(exc))
            times["insight_generator"] = round((time.perf_counter() - start) * 1000, 2)

        elif tool_name == "reasoning_engine":
            # Reasoning engine is invoked in the synthesizer, skip here
            pass

        elif tool_name == "data_profiler":
            # Data profiler result is already in state from upload
            pass

    return {
        **state,
        "query_result": query_result,
        "sql_result": sql_result,
        "pandas_result": pandas_result,
        "chart_results": chart_results,
        "anomaly_report": anomaly_report,
        "business_insights": business_insights,
        "tool_execution_times": times,
    }
