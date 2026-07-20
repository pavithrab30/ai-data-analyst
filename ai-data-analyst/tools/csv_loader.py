"""
CSV file loading and parsing tool.

Responsible for accepting uploaded file objects (from Streamlit's
file uploader), validating the file at the binary level, parsing
into a DataFrame, and applying initial column name sanitization.

This tool intentionally does NOT perform deep data validation —
that responsibility belongs to data_validator.py. The separation
keeps each module focused on a single concern.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config.settings import settings
from utils.logger import get_logger
from utils.sanitizers import sanitize_filename, validate_column_names

logger = get_logger(__name__)


class CSVLoadError(Exception):
    """Raised when a CSV file cannot be loaded."""


class CSVLoader:
    """
    Loads and parses CSV files from various sources.

    Supports loading from:
    - Streamlit UploadedFile objects
    - File paths on disk
    - Raw bytes
    - String content

    Applies consistent column name sanitization so downstream tools
    always receive clean, predictable column names.
    """

    def __init__(
        self,
        max_file_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[list[str]] = None,
    ) -> None:
        self._max_file_size_bytes = max_file_size_bytes or settings.max_file_size_bytes
        self._allowed_extensions = allowed_extensions or settings.allowed_extensions_list

    def load_from_upload(self, uploaded_file: object) -> tuple[pd.DataFrame, str]:
        """
        Load a CSV from a Streamlit UploadedFile object.

        Args:
            uploaded_file: Streamlit UploadedFile with .name, .size, .getvalue()

        Returns:
            Tuple of (DataFrame, sanitized_filename)

        Raises:
            CSVLoadError: On validation or parsing failures.
        """
        filename: str = getattr(uploaded_file, "name", "unknown.csv")
        file_size: int = getattr(uploaded_file, "size", 0)

        logger.info("Loading CSV from upload", filename=filename, size_bytes=file_size)

        # Validate extension
        self._validate_extension(filename)

        # Validate file size
        if file_size > self._max_file_size_bytes:
            raise CSVLoadError(
                f"File '{filename}' is {file_size / 1024 / 1024:.1f} MB, "
                f"which exceeds the {settings.max_file_size_mb} MB limit."
            )

        # Read bytes
        try:
            content: bytes = uploaded_file.getvalue()
        except Exception as exc:
            raise CSVLoadError(f"Failed to read file content: {exc}") from exc

        safe_name = sanitize_filename(filename)
        df = self._parse_csv_bytes(content, safe_name)

        logger.info(
            "CSV loaded successfully",
            filename=safe_name,
            rows=len(df),
            columns=len(df.columns),
        )
        return df, safe_name

    def load_from_path(self, file_path: Union[str, Path]) -> tuple[pd.DataFrame, str]:
        """
        Load a CSV from a filesystem path.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Tuple of (DataFrame, filename)

        Raises:
            CSVLoadError: If file doesn't exist or cannot be parsed.
        """
        path = Path(file_path)

        if not path.exists():
            raise CSVLoadError(f"File not found: {path}")

        if not path.is_file():
            raise CSVLoadError(f"Path is not a file: {path}")

        self._validate_extension(path.name)

        file_size = path.stat().st_size
        if file_size > self._max_file_size_bytes:
            raise CSVLoadError(
                f"File '{path.name}' is {file_size / 1024 / 1024:.1f} MB, "
                f"exceeds the {settings.max_file_size_mb} MB limit."
            )

        logger.info("Loading CSV from path", path=str(path), size_bytes=file_size)

        content = path.read_bytes()
        safe_name = sanitize_filename(path.name)
        df = self._parse_csv_bytes(content, safe_name)

        logger.info(
            "CSV loaded from path",
            filename=safe_name,
            rows=len(df),
            columns=len(df.columns),
        )
        return df, safe_name

    def load_from_bytes(self, content: bytes, filename: str = "data.csv") -> tuple[pd.DataFrame, str]:
        """
        Load a CSV from raw bytes.

        Args:
            content: Raw CSV bytes.
            filename: Display filename for error messages.

        Returns:
            Tuple of (DataFrame, sanitized_filename)
        """
        if len(content) > self._max_file_size_bytes:
            raise CSVLoadError(
                f"File '{filename}' is {len(content) / 1024 / 1024:.1f} MB, "
                f"which exceeds the {self._max_file_size_bytes // 1024 // 1024} MB limit."
            )
        safe_name = sanitize_filename(filename)
        df = self._parse_csv_bytes(content, safe_name)
        return df, safe_name

    # ── Private Methods ────────────────────────────────────────────────────────

    def _validate_extension(self, filename: str) -> None:
        """Ensure the file has an allowed extension."""
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in self._allowed_extensions:
            raise CSVLoadError(
                f"File type '.{ext}' is not supported. "
                f"Allowed types: {self._allowed_extensions}"
            )

    def _parse_csv_bytes(self, content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse raw CSV bytes into a DataFrame with robust encoding detection.

        Tries UTF-8 first, then falls back to latin-1 which can handle
        almost any byte sequence. Applies column name sanitization after
        successful parsing.

        Args:
            content: Raw bytes of the CSV file.
            filename: Used only for error messages.

        Returns:
            Parsed and column-sanitized DataFrame.

        Raises:
            CSVLoadError: If the file cannot be parsed after all attempts.
        """
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_error: Optional[Exception] = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    low_memory=False,
                    on_bad_lines="warn",
                )

                if df.empty:
                    raise CSVLoadError(
                        f"File '{filename}' parsed to an empty DataFrame. "
                        "Ensure the file contains data rows."
                    )

                if len(df.columns) == 0:
                    raise CSVLoadError(
                        f"File '{filename}' has no columns. Check the CSV format."
                    )

                # Sanitize column names
                df = self._sanitize_columns(df)

                # Infer better dtypes
                df = self._infer_dtypes(df)

                logger.debug(
                    "CSV parsed successfully",
                    encoding=encoding,
                    rows=len(df),
                    columns=len(df.columns),
                )
                return df

            except CSVLoadError:
                raise
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "CSV parsing failed with encoding, trying next",
                    encoding=encoding,
                    error=str(exc),
                )
                continue

        raise CSVLoadError(
            f"Failed to parse '{filename}' with any supported encoding. "
            f"Last error: {last_error}"
        )

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to safe Python identifiers.

        - Strip leading/trailing whitespace
        - Replace spaces with underscores
        - Remove special characters
        - Handle duplicates by appending index suffix
        """
        import re

        new_columns: list[str] = []
        seen: set[str] = set()

        for col in df.columns:
            # Convert to string and strip
            clean = str(col).strip()
            # Replace whitespace runs with underscore
            clean = re.sub(r"\s+", "_", clean)
            # Remove characters that are not alphanumeric or underscore
            clean = re.sub(r"[^a-zA-Z0-9_]", "", clean)
            # Ensure it doesn't start with a digit
            if clean and clean[0].isdigit():
                clean = f"col_{clean}"
            # Fallback for empty names
            if not clean:
                clean = f"column_{len(new_columns)}"
            # Handle duplicates
            original_clean = clean
            counter = 1
            while clean in seen:
                clean = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean)
            new_columns.append(clean)

        df.columns = new_columns
        return df

    def _infer_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to infer better data types for object columns.

        Tries to convert object columns to numeric or datetime types.
        Failures are silently ignored — the column retains its original type.
        """
        for col in df.select_dtypes(include=["object"]).columns:
            # Try numeric conversion
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() / max(len(df), 1) > 0.8:
                df[col] = numeric_series
                continue

            # Try datetime conversion
            try:
                date_series = pd.to_datetime(df[col], infer_format=True, errors="coerce")
                if date_series.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = date_series
            except Exception:
                pass

        return df


# Module-level singleton
csv_loader = CSVLoader()
