"""
Explainability and reasoning engine.

Every AI response includes a ReasoningTrace that explains:
- What intent was classified
- Which tools were invoked
- What SQL or Pandas code was generated
- What assumptions were made
- The confidence level

This module builds and formats these traces so responses
are fully transparent and auditable.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from models.analysis_models import (
    AnalysisResult,
    ConfidenceLevel,
    IntentClassification,
    ReasoningTrace,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine:
    """
    Builds explainability traces for every analytical response.

    The reasoning engine doesn't compute results — it annotates
    results with the context needed to understand how they were
    produced, making the system auditable and trustworthy.
    """

    def build_trace(
        self,
        question: str,
        intent: Optional[IntentClassification] = None,
        tools_invoked: Optional[list[str]] = None,
        tool_execution_times: Optional[dict[str, float]] = None,
        sql_generated: Optional[str] = None,
        pandas_code_generated: Optional[str] = None,
        assumptions: Optional[list[str]] = None,
        data_limitations: Optional[list[str]] = None,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
        start_time: Optional[float] = None,
    ) -> ReasoningTrace:
        """
        Build a complete ReasoningTrace for an analysis result.

        Args:
            question: Original user question.
            intent: Classified intent from the planner.
            tools_invoked: List of tool names that were called.
            tool_execution_times: Per-tool timing in milliseconds.
            sql_generated: SQL query if generated.
            pandas_code_generated: Pandas code if generated.
            assumptions: List of analytical assumptions.
            data_limitations: Known limitations of this analysis.
            confidence: Overall confidence in the result.
            start_time: perf_counter start time for total elapsed calculation.

        Returns:
            Populated ReasoningTrace model.
        """
        total_elapsed = 0.0
        if start_time is not None:
            total_elapsed = (time.perf_counter() - start_time) * 1000

        trace = ReasoningTrace(
            question=question,
            intent_classification=intent,
            tools_invoked=tools_invoked or [],
            tool_execution_times=tool_execution_times or {},
            sql_generated=sql_generated,
            pandas_code_generated=pandas_code_generated,
            assumptions=assumptions or self._default_assumptions(question),
            data_limitations=data_limitations or [],
            confidence=confidence,
            total_execution_time_ms=round(total_elapsed, 2),
        )

        logger.debug(
            "Reasoning trace built",
            tools=len(trace.tools_invoked),
            elapsed_ms=trace.total_execution_time_ms,
            confidence=confidence.value,
        )

        return trace

    def format_trace_for_display(self, trace: ReasoningTrace) -> str:
        """
        Format a ReasoningTrace into a human-readable markdown string.

        Used by the UI to show the "How was this answer generated?" section.

        Args:
            trace: The ReasoningTrace to format.

        Returns:
            Markdown-formatted string.
        """
        lines = [
            "### 🔍 How This Answer Was Generated",
            "",
        ]

        # Intent
        if trace.intent_classification:
            intent = trace.intent_classification
            lines.append(
                f"**Intent Detected:** {intent.primary_intent.value.replace('_', ' ').title()}"
            )
            if intent.secondary_intents:
                secondary = ", ".join(
                    i.value.replace("_", " ").title() for i in intent.secondary_intents
                )
                lines.append(f"**Additional Intents:** {secondary}")
            lines.append(f"**Confidence:** {intent.confidence.value.title()}")
            lines.append(f"**Planner Reasoning:** {intent.reasoning}")
            lines.append("")

        # Tools invoked
        if trace.tools_invoked:
            lines.append("**Tools Invoked:**")
            for tool in trace.tools_invoked:
                elapsed = trace.tool_execution_times.get(tool, 0)
                lines.append(f"  - `{tool}` ({elapsed:.0f}ms)")
            lines.append("")

        # SQL
        if trace.sql_generated:
            lines.append("**Generated SQL:**")
            lines.append(f"```sql\n{trace.sql_generated}\n```")
            lines.append("")

        # Pandas code
        if trace.pandas_code_generated:
            lines.append("**Generated Pandas Code:**")
            lines.append(f"```python\n{trace.pandas_code_generated}\n```")
            lines.append("")

        # Assumptions
        if trace.assumptions:
            lines.append("**Assumptions Made:**")
            for assumption in trace.assumptions:
                lines.append(f"  - {assumption}")
            lines.append("")

        # Limitations
        if trace.data_limitations:
            lines.append("**Known Limitations:**")
            for limitation in trace.data_limitations:
                lines.append(f"  - {limitation}")
            lines.append("")

        # Performance
        confidence_emoji = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴",
        }
        emoji = confidence_emoji.get(trace.confidence, "⚪")
        lines.append(
            f"**Result Confidence:** {emoji} {trace.confidence.value.title()}"
        )
        lines.append(
            f"**Total Processing Time:** {trace.total_execution_time_ms:.0f}ms"
        )

        return "\n".join(lines)

    def _default_assumptions(self, question: str) -> list[str]:
        """
        Return standard assumptions based on the question type.

        These are appended to all traces to set appropriate expectations.
        """
        assumptions = [
            "Analysis is based on the currently uploaded dataset only.",
            "Null/missing values are excluded from aggregations unless otherwise specified.",
        ]

        question_lower = question.lower()
        if any(kw in question_lower for kw in ["top", "best", "highest", "most"]):
            assumptions.append(
                "Rankings are computed on the full dataset and may change if filtered."
            )
        if any(kw in question_lower for kw in ["trend", "over time", "monthly", "weekly"]):
            assumptions.append(
                "Temporal trends assume the date column reflects event time, not processing time."
            )
        if any(kw in question_lower for kw in ["average", "mean"]):
            assumptions.append(
                "Mean calculations are arithmetic means and sensitive to outliers."
            )

        return assumptions


# Module-level singleton
reasoning_engine = ReasoningEngine()
