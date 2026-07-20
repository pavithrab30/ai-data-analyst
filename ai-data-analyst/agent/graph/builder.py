"""
LangGraph StateGraph builder.

Constructs the agent workflow graph by connecting nodes
with edges and conditional routing. The graph is compiled
once and reused for all queries within a session.

Workflow:
  planner → executor → synthesizer → memory_save

The graph is intentionally linear for reliability.
All tool dispatch decisions happen in the executor node
based on the planner's intent classification.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from agent.graph.state import AgentState
from agent.nodes.executor import executor_node
from agent.nodes.memory_node import memory_save_node
from agent.nodes.planner import planner_node
from agent.nodes.synthesizer import synthesizer_node
from utils.logger import get_logger

logger = get_logger(__name__)


def build_agent_graph():
    """
    Build and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph runnable ready for .invoke() calls.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("memory_save", memory_save_node)

    # ── Wire edges ─────────────────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "synthesizer")
    graph.add_edge("synthesizer", "memory_save")
    graph.add_edge("memory_save", END)

    compiled = graph.compile()

    logger.info("LangGraph agent compiled successfully")
    return compiled


def run_agent(
    question: str,
    dataframe,
    table_name: str = "dataset",
    dataset_name: str = "dataset",
    session=None,
    data_profile=None,
) -> AgentState:
    """
    Run the compiled agent graph for a single user query.

    Args:
        question: Sanitized user question.
        dataframe: Active pandas DataFrame.
        table_name: SQLite table name for the dataset.
        dataset_name: Display name of the dataset.
        session: SessionState for conversation memory.
        data_profile: Pre-computed DataProfile for schema context.

    Returns:
        Final AgentState after all nodes have executed.
    """
    from tools.memory_tool import memory_tool

    # Add user turn to memory before running the graph
    if session:
        memory_tool.add_user_turn(session, question)

    initial_state: AgentState = {
        "question": question,
        "dataframe": dataframe,
        "table_name": table_name,
        "dataset_name": dataset_name,
        "session": session,
        "data_profile": data_profile,
        "chart_results": [],
        "business_insights": [],
        "tool_execution_times": {},
        "intent": None,
        "query_result": None,
        "sql_result": None,
        "pandas_result": None,
        "anomaly_report": None,
        "reasoning_trace": None,
        "final_answer": "",
        "analysis_result": None,
        "error": None,
    }

    graph = build_agent_graph()

    try:
        final_state = graph.invoke(initial_state)
        logger.info(
            "Agent graph execution complete",
            question=question[:80],
            answer_length=len(final_state.get("final_answer", "")),
        )
        return final_state
    except Exception as exc:
        logger.error("Agent graph execution failed", error=str(exc))
        initial_state["error"] = str(exc)
        initial_state["final_answer"] = (
            f"I encountered an error while processing your question: {exc}. "
            "Please try rephrasing or check the dataset for issues."
        )
        return initial_state
