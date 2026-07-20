"""
Planner node — intent classification and tool selection.

This is the first node executed in the LangGraph workflow.
It reads the user question and dataset context, calls Gemini
to classify the intent, and populates the AgentState with
the IntentClassification that drives all downstream routing.

Reads from state: question, session, data_profile, table_name
Writes to state:  intent
"""

from __future__ import annotations

import json
import re
import time

from agent.graph.state import AgentState
from agent.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from models.analysis_models import ConfidenceLevel, IntentClassification, IntentType
from services.llm_service import LLMService
from tools.data_profiler import data_profiler
from utils.logger import get_logger

logger = get_logger(__name__)

# Default tool mapping by intent (fallback when LLM classification fails)
_INTENT_DEFAULT_TOOLS: dict[str, list[str]] = {
    IntentType.QUESTION_ANSWERING: ["query_engine", "reasoning_engine"],
    IntentType.SQL_GENERATION: ["sql_generator", "reasoning_engine"],
    IntentType.PANDAS_GENERATION: ["pandas_generator", "reasoning_engine"],
    IntentType.VISUALIZATION: ["visualizer", "reasoning_engine"],
    IntentType.ANOMALY_DETECTION: ["anomaly_detector", "reasoning_engine"],
    IntentType.BUSINESS_INSIGHT: ["insight_generator", "reasoning_engine"],
    IntentType.DASHBOARD: [
        "query_engine", "insight_generator", "visualizer",
        "anomaly_detector", "reasoning_engine",
    ],
    IntentType.DATA_PROFILE: ["data_profiler", "reasoning_engine"],
    IntentType.MULTI_INTENT: ["query_engine", "visualizer", "reasoning_engine"],
    IntentType.UNKNOWN: ["query_engine", "reasoning_engine"],
}


def planner_node(state: AgentState) -> AgentState:
    """
    Classify user intent and select tools.

    Uses Gemini to parse the question against the dataset schema
    and return a structured JSON classification. Falls back to
    keyword-based heuristics if the LLM call fails.

    Args:
        state: Current AgentState.

    Returns:
        Updated AgentState with `intent` populated.
    """
    start_time = time.perf_counter()
    question = state.get("question", "")

    logger.info("Planner node executing", question=question[:100])

    # Build schema summary for context
    data_profile = state.get("data_profile")
    if data_profile:
        schema_summary = data_profiler.generate_schema_summary(data_profile)
    else:
        df = state.get("dataframe")
        if df is not None:
            cols = ", ".join(f"{c} ({str(df[c].dtype)})" for c in df.columns[:20])
            schema_summary = f"Columns: {cols}\nRows: {len(df):,}"
        else:
            schema_summary = "No dataset loaded."

    # Build conversation history context
    session = state.get("session")
    if session:
        history = session.get_history_for_prompt(max_turns=4)
        history_str = "\n".join(f"{m['role'].title()}: {m['content'][:200]}" for m in history[:-1])
    else:
        history_str = "No prior conversation."

    # Build prompt
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        schema_summary=schema_summary,
        conversation_history=history_str,
    )
    user_prompt = PLANNER_USER_TEMPLATE.format(question=question)

    # Call LLM
    try:
        llm = LLMService()
        raw_response = llm.generate(
            user_prompt,
            system_instruction=system_prompt,
            temperature_override=0.0,
        )
        intent = _parse_intent_response(raw_response, question)
    except Exception as exc:
        logger.warning("Planner LLM call failed, using keyword fallback", error=str(exc))
        intent = _keyword_fallback_intent(question)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Planner complete",
        intent=intent.primary_intent.value,
        tools=intent.tools_required,
        confidence=intent.confidence.value,
        elapsed_ms=round(elapsed_ms, 2),
    )

    times = state.get("tool_execution_times", {})
    times["planner"] = round(elapsed_ms, 2)

    return {**state, "intent": intent, "tool_execution_times": times}


def _parse_intent_response(raw: str, question: str) -> IntentClassification:
    """
    Parse the Gemini response JSON into an IntentClassification.

    Falls back to keyword heuristics if JSON parsing fails.

    Args:
        raw: Raw LLM response string.
        question: Original question for fallback classification.

    Returns:
        IntentClassification model.
    """
    try:
        # Extract JSON block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in planner response")

        data = json.loads(json_match.group())

        primary_str = data.get("primary_intent", "question_answering")
        try:
            primary = IntentType(primary_str)
        except ValueError:
            primary = IntentType.QUESTION_ANSWERING

        secondary_intents = []
        for intent_str in data.get("secondary_intents", []):
            try:
                secondary_intents.append(IntentType(intent_str))
            except ValueError:
                pass

        tools = data.get("tools_required", _INTENT_DEFAULT_TOOLS.get(primary, ["query_engine"]))
        reasoning = data.get("reasoning", "Classification based on question analysis.")

        confidence_str = data.get("confidence", "high").lower()
        try:
            confidence = ConfidenceLevel(confidence_str)
        except ValueError:
            confidence = ConfidenceLevel.HIGH

        return IntentClassification(
            primary_intent=primary,
            secondary_intents=secondary_intents,
            tools_required=tools,
            reasoning=reasoning,
            confidence=confidence,
            original_query=question,
        )

    except Exception as exc:
        logger.warning("Intent JSON parsing failed, using heuristic fallback", error=str(exc))
        return _keyword_fallback_intent(question)


def _keyword_fallback_intent(question: str) -> IntentClassification:
    """
    Keyword-based intent classification as a fallback.

    Args:
        question: User question string.

    Returns:
        IntentClassification based on keyword matching.
    """
    q = question.lower()

    if any(kw in q for kw in ["sql", "query", "select"]):
        intent = IntentType.SQL_GENERATION
    elif any(kw in q for kw in ["python", "pandas", "code", "dataframe"]):
        intent = IntentType.PANDAS_GENERATION
    elif any(kw in q for kw in ["chart", "plot", "graph", "visualize", "show me"]):
        intent = IntentType.VISUALIZATION
    elif any(kw in q for kw in ["anomaly", "outlier", "unusual", "weird", "strange"]):
        intent = IntentType.ANOMALY_DETECTION
    elif any(kw in q for kw in ["insight", "summary", "overview", "executive", "report"]):
        intent = IntentType.BUSINESS_INSIGHT
    elif any(kw in q for kw in ["schema", "columns", "profile", "describe", "data types"]):
        intent = IntentType.DATA_PROFILE
    elif any(kw in q for kw in ["dashboard", "full analysis", "everything"]):
        intent = IntentType.DASHBOARD
    else:
        intent = IntentType.QUESTION_ANSWERING

    return IntentClassification(
        primary_intent=intent,
        secondary_intents=[],
        tools_required=_INTENT_DEFAULT_TOOLS[intent],
        reasoning=f"Keyword-based fallback classification for question: {question[:50]}",
        confidence=ConfidenceLevel.MEDIUM,
        original_query=question,
    )


