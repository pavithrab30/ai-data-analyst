"""
Response and data formatting utilities.

Centralizes all display formatting logic so that UI components
and tool outputs share consistent formatting without duplication.
"""

from __future__ import annotations

import re
from typing import Any
import pandas as pd
import numpy as np


def format_number(value: float | int, decimals: int = 2) -> str:
    """
    Format a numeric value with thousands separator and specified decimal places.

    Args:
        value: Numeric value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string, e.g. "1,234,567.89"
    """
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Value between 0 and 100 (or 0 and 1).
        decimals: Decimal places.

    Returns:
        Formatted percentage string, e.g. "42.5%"
    """
    if pd.isna(value):
        return "N/A"
    # Normalise fractions to percentages
    if abs(value) <= 1.0:
        value = value * 100
    return f"{value:.{decimals}f}%"


def format_bytes(size_bytes: int) -> str:
    """
    Convert byte count into a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable string, e.g. "2.34 MB"
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_dataframe_for_display(
    df: pd.DataFrame,
    max_rows: int = 100,
    max_col_width: int = 50,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for clean display in Streamlit.

    Truncates long string values, limits rows, and resets the index.

    Args:
        df: Source DataFrame.
        max_rows: Maximum number of rows to include.
        max_col_width: Maximum character width for string columns.

    Returns:
        Display-ready DataFrame.
    """
    display_df = df.head(max_rows).copy()

    for col in display_df.select_dtypes(include=["object"]).columns:
        display_df[col] = display_df[col].astype(str).apply(
            lambda x: x[:max_col_width] + "..." if len(x) > max_col_width else x
        )

    return display_df.reset_index(drop=True)


def sanitize_column_name(name: str) -> str:
    """
    Convert a column name into a safe Python identifier.

    Replaces spaces and special characters with underscores and
    ensures the name starts with a letter.

    Args:
        name: Raw column name.

    Returns:
        Safe column name string.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip())
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    return sanitized.lower()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate a text string to a maximum length.

    Args:
        text: Input text.
        max_length: Maximum characters before truncation.
        suffix: String appended when truncated.

    Returns:
        Possibly truncated string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Convert a DataFrame to a Markdown table string.

    Args:
        df: Source DataFrame.
        max_rows: Maximum rows to include.

    Returns:
        Markdown-formatted table as string.
    """
    return df.head(max_rows).to_markdown(index=False)


def format_dict_as_markdown(data: dict[str, Any], title: str = "") -> str:
    """
    Render a dictionary as a Markdown bullet list.

    Args:
        data: Dictionary to format.
        title: Optional section heading.

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    for key, value in data.items():
        if isinstance(value, float):
            lines.append(f"- **{key}**: {format_number(value)}")
        else:
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)
