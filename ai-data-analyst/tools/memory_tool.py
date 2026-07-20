"""
Conversation memory tool.

Manages conversation history for the LangGraph workflow.
The memory tool provides:
1. History retrieval formatted for LLM context windows.
2. History pruning to stay within token limits.
3. Context summarization when history grows too long.
4. Session-scoped persistence via Streamlit session state.

This tool is invoked at the start (load) and end (save)
of every LangGraph workflow execution.
"""

from __future__ import annotations

from typing import Optional

from models.session_models import ConversationTurn, SessionState
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of tokens to budget for conversation history in prompts.
# This is approximate (1 token ≈ 4 chars).
MAX_HISTORY_CHARS = 8000


class MemoryTool:
    """
    Manages conversational context for the agent workflow.

    Operates on the SessionState model which is stored in
    Streamlit session state. Does not depend directly on st
    so it remains testable.
    """

    def get_context_for_prompt(
        self,
        session: SessionState,
        max_turns: int = 10,
        max_chars: int = MAX_HISTORY_CHARS,
    ) -> str:
        """
        Build a conversation history string for inclusion in LLM prompts.

        Truncates history to stay within character budget while
        preserving the most recent turns.

        Args:
            session: Current SessionState.
            max_turns: Maximum number of recent turns to include.
            max_chars: Maximum total character count.

        Returns:
            Formatted conversation history string.
        """
        history = session.conversation_history[-max_turns:]
        if not history:
            return ""

        lines: list[str] = ["Previous conversation:"]
        total_chars = 0

        for turn in reversed(history):
            role = "User" if turn.is_user else "Assistant"
            entry = f"{role}: {turn.content}"

            # Truncate individual turns that are very long
            if len(entry) > 1000:
                entry = entry[:1000] + "..."

            if total_chars + len(entry) > max_chars:
                lines.insert(1, "[Earlier history truncated for context window]")
                break

            lines.insert(1, entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def get_messages_for_llm(
        self,
        session: SessionState,
        max_turns: int = 6,
    ) -> list[dict[str, str]]:
        """
        Return conversation history formatted for the LLM API.

        Args:
            session: Current SessionState.
            max_turns: Maximum number of turns to include.

        Returns:
            List of {"role": "user"|"assistant", "content": str} dicts.
        """
        history = session.conversation_history[-max_turns:]
        messages = []

        for turn in history:
            role = "user" if turn.is_user else "assistant"
            messages.append({"role": role, "content": turn.content})

        return messages

    def add_user_turn(self, session: SessionState, content: str) -> ConversationTurn:
        """
        Add a user message to the session history.

        Args:
            session: SessionState to update.
            content: User message content.

        Returns:
            Created ConversationTurn.
        """
        turn = session.add_turn(role="user", content=content)
        logger.debug("User turn added to memory", turn_id=turn.turn_id)
        return turn

    def add_assistant_turn(
        self,
        session: SessionState,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ConversationTurn:
        """
        Add an assistant response to the session history.

        Args:
            session: SessionState to update.
            content: Assistant response content.
            metadata: Optional context dict (tools used, charts, etc.).

        Returns:
            Created ConversationTurn.
        """
        turn = session.add_turn(
            role="assistant", content=content, metadata=metadata or {}
        )
        logger.debug("Assistant turn added to memory", turn_id=turn.turn_id)
        return turn

    def prune_history(
        self,
        session: SessionState,
        max_turns: int,
    ) -> int:
        """
        Remove oldest turns to stay within the session history limit.

        Args:
            session: SessionState to prune.
            max_turns: Maximum number of turns to retain.

        Returns:
            Number of turns removed.
        """
        excess = len(session.conversation_history) - max_turns
        if excess <= 0:
            return 0

        session.conversation_history = session.conversation_history[excess:]
        logger.info("History pruned", removed_turns=excess, remaining=len(session.conversation_history))
        return excess

    def clear_history(self, session: SessionState) -> None:
        """
        Clear all conversation history from the session.

        Args:
            session: SessionState to clear.
        """
        count = len(session.conversation_history)
        session.conversation_history = []
        logger.info("Conversation history cleared", turns_removed=count)


# Module-level singleton
memory_tool = MemoryTool()
