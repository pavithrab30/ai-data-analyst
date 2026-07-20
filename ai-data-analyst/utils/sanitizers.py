"""
Input sanitization and security utilities.

Provides functions to clean and validate user-supplied inputs before
they are passed to downstream tools, LLM prompts, or execution engines.
This is a critical security layer for an application that accepts
arbitrary user text and uploaded files.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# Characters / patterns that must never appear in an LLM prompt
# built from user input.
_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now",
    r"act\s+as\s+",
    r"jailbreak",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[SYS\]",
]

_COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS
]


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Clean and validate a user-supplied natural language query.

    Steps:
    1. Normalize unicode to NFC form.
    2. Strip leading/trailing whitespace.
    3. Truncate to max_length.
    4. Remove null bytes and other control characters.
    5. Detect and neutralize common prompt injection patterns.

    Args:
        text: Raw user input string.
        max_length: Maximum allowed character count.

    Returns:
        Sanitized string safe for inclusion in LLM prompts.

    Raises:
        ValueError: If prompt injection is detected.
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Strip whitespace
    text = text.strip()

    # Remove null bytes and non-printable control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Truncate
    if len(text) > max_length:
        text = text[:max_length]

    # Detect prompt injection
    for pattern in _COMPILED_INJECTION_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Potentially unsafe input detected. "
                f"Query contains a disallowed pattern: '{pattern.pattern}'"
            )

    return text


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded file name to prevent path traversal attacks.

    Args:
        filename: Original filename from the upload.

    Returns:
        Safe filename string with only alphanumeric characters,
        underscores, hyphens, and a single extension.

    Raises:
        ValueError: If the sanitized filename is empty or has no valid stem.
    """
    # Extract just the filename (no path components)
    safe = Path(filename).name

    # Remove any remaining path separators
    safe = safe.replace("/", "_").replace("\\", "_").replace("..", "_")

    # Keep only safe characters in the stem
    stem = Path(safe).stem
    suffix = Path(safe).suffix.lower()

    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

    if not safe_stem:
        raise ValueError(f"Filename '{filename}' produced an empty stem after sanitization.")

    return f"{safe_stem}{suffix}"


def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitize a string intended for use as a SQL table or column name.

    Only allows alphanumeric characters and underscores.
    Wraps the result in double quotes for safe use in SQL statements.

    Args:
        identifier: Raw identifier string.

    Returns:
        Quoted, sanitized SQL identifier.

    Raises:
        ValueError: If the identifier is empty after sanitization.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", identifier.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")

    if not clean:
        raise ValueError(
            f"Identifier '{identifier}' is empty after sanitization."
        )

    return f'"{clean}"'


def validate_column_names(columns: list[str]) -> list[str]:
    """
    Validate and normalize a list of DataFrame column names.

    Ensures all columns are non-empty strings and free of
    dangerous characters that could cause issues in SQL or code generation.

    Args:
        columns: List of raw column name strings.

    Returns:
        List of cleaned column name strings.

    Raises:
        ValueError: If duplicate column names are detected after sanitization.
    """
    cleaned = []
    for col in columns:
        clean_col = re.sub(r"[^a-zA-Z0-9_\s\-\.]", "", str(col)).strip()
        if not clean_col:
            clean_col = f"unnamed_column_{len(cleaned)}"
        cleaned.append(clean_col)

    if len(cleaned) != len(set(cleaned)):
        raise ValueError(
            "Duplicate column names detected after sanitization. "
            "Please ensure all column names are unique."
        )

    return cleaned
