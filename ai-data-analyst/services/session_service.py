"""
Session state management service.

Provides a clean interface for creating, reading, and mutating
Streamlit session state. By going through this service rather than
accessing st.session_state directly in UI components, we:

1. Keep session shape validated by Pydantic models.
2. Centralize all session key names (no magic string literals in UI).
3. Make session logic testable without a Streamlit runtime.
4. Enable clean session reset from a single method call.
"""

from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd
import streamlit as st

from models.session_models import DatasetInfo, SessionState
from utils.logger import get_logger

logger = get_logger(__name__)

# Single key under which the entire SessionState model is stored.
_SESSION_KEY = "app_session_state"
# DataFrames are stored separately because Pydantic cannot serialize them.
_DATAFRAMES_KEY = "app_dataframes"


class SessionService:
    """
    Facade for Streamlit session state management.

    All interactions with st.session_state go through this class.
    The session state model (SessionState) and raw DataFrames
    are stored under separate keys to keep serialization clean.
    """

    # ── Initialization ─────────────────────────────────────────────────────────

    @staticmethod
    def initialize() -> SessionState:
        """
        Initialize session state if it does not exist yet.

        Called at the top of app.py on every Streamlit rerun.
        Idempotent: if the session already exists, returns it unchanged.

        Returns:
            The current (possibly newly created) SessionState.
        """
        if _SESSION_KEY not in st.session_state:
            session = SessionState(session_id=str(uuid.uuid4()))
            st.session_state[_SESSION_KEY] = session
            st.session_state[_DATAFRAMES_KEY] = {}
            logger.info("New session initialized", session_id=session.session_id)

        return st.session_state[_SESSION_KEY]

    @staticmethod
    def get() -> SessionState:
        """
        Retrieve the current SessionState.

        Returns:
            Current SessionState from Streamlit session.

        Raises:
            RuntimeError: If called before initialize().
        """
        if _SESSION_KEY not in st.session_state:
            raise RuntimeError(
                "SessionService.initialize() must be called before SessionService.get()"
            )
        return st.session_state[_SESSION_KEY]

    @staticmethod
    def reset() -> SessionState:
        """
        Clear all session data and start a fresh session.

        Preserves the session_id for logging continuity.

        Returns:
            The new empty SessionState.
        """
        old_id = st.session_state.get(_SESSION_KEY, SessionState(session_id="unknown")).session_id
        session = SessionState(session_id=str(uuid.uuid4()))
        st.session_state[_SESSION_KEY] = session
        st.session_state[_DATAFRAMES_KEY] = {}
        logger.info("Session reset", old_session_id=old_id, new_session_id=session.session_id)
        return session

    # ── DataFrame Management ───────────────────────────────────────────────────

    @staticmethod
    def store_dataframe(name: str, df: pd.DataFrame) -> None:
        """
        Store a DataFrame in session state under the given name.

        Args:
            name: Logical dataset name (used as lookup key).
            df: DataFrame to store.
        """
        if _DATAFRAMES_KEY not in st.session_state:
            st.session_state[_DATAFRAMES_KEY] = {}
        st.session_state[_DATAFRAMES_KEY][name] = df
        logger.debug("DataFrame stored in session", name=name, shape=df.shape)

    @staticmethod
    def get_dataframe(name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a DataFrame by name from session state.

        Args:
            name: Dataset name used when storing.

        Returns:
            DataFrame if found, None otherwise.
        """
        frames = st.session_state.get(_DATAFRAMES_KEY, {})
        return frames.get(name)

    @staticmethod
    def get_active_dataframe() -> Optional[pd.DataFrame]:
        """
        Retrieve the currently active / selected DataFrame.

        Returns:
            Active DataFrame or None if no dataset is selected.
        """
        session = SessionService.get()
        if not session.active_dataset_name:
            return None
        return SessionService.get_dataframe(session.active_dataset_name)

    @staticmethod
    def list_dataframe_names() -> list[str]:
        """Return names of all stored DataFrames."""
        return list(st.session_state.get(_DATAFRAMES_KEY, {}).keys())

    @staticmethod
    def remove_dataframe(name: str) -> None:
        """
        Remove a DataFrame and its metadata from session state.

        Args:
            name: Dataset name to remove.
        """
        frames = st.session_state.get(_DATAFRAMES_KEY, {})
        if name in frames:
            del frames[name]

        session = SessionService.get()
        if name in session.datasets:
            del session.datasets[name]

        if session.active_dataset_name == name:
            remaining = list(session.datasets.keys())
            session.active_dataset_name = remaining[0] if remaining else None

        logger.info("DataFrame removed from session", name=name)

    # ── Dataset Info Management ────────────────────────────────────────────────

    @staticmethod
    def register_dataset(
        name: str,
        df: pd.DataFrame,
        filename: str,
        file_size_bytes: int = 0,
        is_merged: bool = False,
        source_files: Optional[list[str]] = None,
    ) -> DatasetInfo:
        """
        Register an uploaded dataset in session state.

        Stores both the raw DataFrame and the DatasetInfo metadata model.
        Sets this dataset as the active dataset.

        Args:
            name: Logical dataset name.
            df: The loaded DataFrame.
            filename: Original uploaded filename.
            file_size_bytes: File size for display.
            is_merged: True if this is a joined dataset.
            source_files: Source files for merged datasets.

        Returns:
            The created DatasetInfo model.
        """
        dataset_info = DatasetInfo(
            name=name,
            filename=filename,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            dtypes={col: str(df[col].dtype) for col in df.columns},
            file_size_bytes=file_size_bytes,
            is_merged=is_merged,
            source_files=source_files or [],
        )

        session = SessionService.get()
        session.datasets[name] = dataset_info
        session.active_dataset_name = name

        SessionService.store_dataframe(name, df)

        logger.info(
            "Dataset registered",
            name=name,
            rows=len(df),
            columns=len(df.columns),
        )
        return dataset_info

    @staticmethod
    def set_active_dataset(name: str) -> None:
        """
        Switch the active dataset to the given name.

        Args:
            name: Dataset name that must already be registered.

        Raises:
            ValueError: If the named dataset is not registered.
        """
        session = SessionService.get()
        if name not in session.datasets:
            raise ValueError(
                f"Dataset '{name}' not found in session. "
                f"Available: {list(session.datasets.keys())}"
            )
        session.active_dataset_name = name
        logger.info("Active dataset changed", name=name)

    # ── Conversation ───────────────────────────────────────────────────────────

    @staticmethod
    def add_user_message(content: str) -> None:
        """Append a user message to conversation history."""
        session = SessionService.get()
        session.add_turn(role="user", content=content)

    @staticmethod
    def add_assistant_message(content: str, metadata: Optional[dict] = None) -> None:
        """Append an assistant message to conversation history."""
        session = SessionService.get()
        session.add_turn(role="assistant", content=content, metadata=metadata or {})

    @staticmethod
    def get_conversation_history(max_turns: int = 10) -> list[dict[str, str]]:
        """Return recent conversation history formatted for LLM context."""
        session = SessionService.get()
        return session.get_history_for_prompt(max_turns=max_turns)


# Module-level singleton
session_service = SessionService()
