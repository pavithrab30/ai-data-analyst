"""
Pydantic models for session and conversation state management.

These models define the shape of conversation history maintained
in Streamlit Session State and passed through the LangGraph workflow.
Clean session modelling enables proper memory, history export,
and session reset functionality.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """
    A single exchange in the conversation between user and AI.

    Stored in session history and passed to the LangGraph memory node
    so the agent can reference prior questions when answering follow-ups.
    """

    turn_id: str = Field(..., description="Unique ID for this conversation turn")
    role: str = Field(
        ..., description="Message author: 'user' or 'assistant'"
    )
    content: str = Field(..., description="Text content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: tools used, charts generated, etc.",
    )

    @property
    def is_user(self) -> bool:
        """True if this turn is from the user."""
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        """True if this turn is from the assistant."""
        return self.role == "assistant"


class DatasetInfo(BaseModel):
    """Lightweight descriptor for an uploaded dataset held in session."""

    name: str = Field(..., description="Display name (usually filename without extension)")
    filename: str = Field(..., description="Original uploaded filename")
    row_count: int
    column_count: int
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = Field(default=0)
    is_merged: bool = Field(
        default=False,
        description="True if this dataset is the result of merging multiple uploads",
    )
    source_files: list[str] = Field(
        default_factory=list,
        description="Source filenames when this is a merged dataset",
    )


class SessionState(BaseModel):
    """
    Complete application session state.

    Persisted in Streamlit's st.session_state under a single key.
    Encapsulates uploaded datasets, conversation history, and UI state
    so that session reset is a single clear operation.
    """

    session_id: str = Field(..., description="Unique session identifier")
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    datasets: dict[str, DatasetInfo] = Field(
        default_factory=dict,
        description="Mapping of dataset name to DatasetInfo",
    )
    active_dataset_name: Optional[str] = Field(
        default=None, description="Name of the currently selected dataset"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    total_queries: int = Field(default=0)
    api_key_override: Optional[str] = Field(
        default=None,
        description="User-provided NVIDIA API key overriding the .env value",
    )

    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> ConversationTurn:
        """
        Append a new conversation turn and return it.

        Args:
            role: 'user' or 'assistant'
            content: Message text
            metadata: Optional additional context dict

        Returns:
            The newly created ConversationTurn.
        """
        import uuid
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.conversation_history.append(turn)
        self.last_activity_at = datetime.utcnow()
        if role == "user":
            self.total_queries += 1
        return turn

    def get_history_for_prompt(self, max_turns: int = 10) -> list[dict[str, str]]:
        """
        Return the most recent conversation turns formatted for LLM context.

        Args:
            max_turns: Maximum number of turns to include.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        recent = self.conversation_history[-max_turns:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def clear(self) -> None:
        """Reset session to initial state, preserving session_id and creation time."""
        self.conversation_history = []
        self.datasets = {}
        self.active_dataset_name = None
        self.total_queries = 0
        self.last_activity_at = datetime.utcnow()
