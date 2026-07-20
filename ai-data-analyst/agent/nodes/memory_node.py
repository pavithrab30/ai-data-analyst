"""
Memory node — saves the final response to conversation history.

Runs as the last node in the workflow to persist
the Q&A pair to the session so subsequent queries
have access to prior context.

Reads from state: question, final_answer, session
Writes to state: session (updated)
"""

from __future__ import annotations

from agent.graph.state import AgentState
from tools.memory_tool import memory_tool
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def memory_save_node(state: AgentState) -> AgentState:
    """
    Persist the current Q&A turn to session memory.

    Args:
        state: AgentState after synthesis.

    Returns:
        Updated AgentState with session history updated.
    """
    session = state.get("session")
    if not session:
        logger.debug("No session in state, skipping memory save")
        return state

    question = state.get("question", "")
    final_answer = state.get("final_answer", "")

    if question and final_answer:
        memory_tool.add_assistant_turn(
            session=session,
            content=final_answer,
            metadata={
                "tools_used": state.get("intent", {}).tools_required if state.get("intent") else [],
                "has_chart": bool(state.get("chart_results")),
                "has_anomalies": state.get("anomaly_report") is not None,
            },
        )

        # Prune if over limit
        memory_tool.prune_history(session, max_turns=settings.session_history_limit)

    logger.debug("Memory save complete", history_length=len(session.conversation_history))
    return {**state, "session": session}
