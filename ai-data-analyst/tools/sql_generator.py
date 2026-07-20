"""
SQL generation and execution tool.

Translates natural language questions into SQLite-compatible SQL queries,
validates them for safety, executes them against an in-memory SQLite database,
and returns both the query and results with full explanations.

Architecture:
1. Load DataFrame into a temporary SQLite in-memory database.
2. Ask Gemini to generate a SQL query given the schema.
3. Validate the generated SQL using whitelist of allowed operations.
4. Execute the validated SQL.
5. Return results alongside the query and explanation.

Safety:
- Only SELECT statements are allowed.
- DDL (CREATE, DROP, ALTER) and DML (INSERT, UPDATE, DELETE) are blocked.
- Table and column names are validated against the actual schema.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, Optional

import pandas as pd

from models.analysis_models import ExecutionResult, SQLGenerationResult
from services.llm_service import LLMService
from utils.logger import get_logger

logger = get_logger(__name__)

# SQL keywords that are always forbidden (DML + DDL)
_FORBIDDEN_SQL_KEYWORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
        "MERGE",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "REVOKE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
    }
)

_SQL_GENERATION_PROMPT = """You are a SQL expert. Generate a SQLite-compatible SQL query to answer the user's question.

Dataset: {table_name}
Schema:
{schema}

Sample data (first 3 rows):
{sample_data}

User question: {question}

RULES:
1. Generate ONLY a SELECT statement — no INSERT, UPDATE, DELETE, DROP, CREATE.
2. Use the exact table name: {table_name}
3. Use only the column names listed in the schema.
4. Prefer readable aliases (e.g. AS total_revenue).
5. Use SQLite syntax (STRFTIME for dates, CAST for type conversion).
6. If the question requires aggregation, use GROUP BY.
7. Limit results to 100 rows unless the user asks for all data.
8. Return ONLY the SQL query, no markdown fences, no explanation.

SQL Query:"""

_SQL_EXPLANATION_PROMPT = """Explain the following SQL query in clear, non-technical language.
Describe what each clause does and how the query answers the question.

Question: {question}
SQL Query:
{sql_query}

Provide a step-by-step explanation in 3-5 bullet points."""


class SQLGenerator:
    """
    Generates and executes SQL queries from natural language.

    Uses Gemini to generate SQLite-compatible queries, validates
    them for safety, executes them against an in-memory SQLite
    database, and returns structured results.
    """

    def __init__(self, llm_service_arg: Optional[LLMService] = None) -> None:
        self._llm = llm_service_arg or LLMService()

    def generate_and_execute(
        self,
        question: str,
        df: pd.DataFrame,
        table_name: str = "dataset",
    ) -> SQLGenerationResult:
        """
        Full pipeline: generate SQL → validate → execute → explain.

        Args:
            question: Natural language question from the user.
            df: DataFrame to query.
            table_name: Name to use for the SQLite table.

        Returns:
            SQLGenerationResult with query, explanation, and execution output.
        """
        logger.info("SQL generation started", question=question[:100], table=table_name)

        # ── Step 1: Build schema description ──────────────────────────────────
        schema_str = self._build_schema_description(df)
        sample_str = df.head(3).to_string(index=False)

        # ── Step 2: Generate SQL with Gemini ──────────────────────────────────
        prompt = _SQL_GENERATION_PROMPT.format(
            table_name=table_name,
            schema=schema_str,
            sample_data=sample_str,
            question=question,
        )

        try:
            raw_sql = self._llm.generate(prompt, temperature_override=0.0)
            sql_query = self._clean_sql(raw_sql)
        except Exception as exc:
            logger.error("SQL generation LLM call failed", error=str(exc))
            return SQLGenerationResult(
                question=question,
                sql_query="",
                explanation=f"Failed to generate SQL: {exc}",
                table_name=table_name,
                is_valid=False,
                validation_errors=[str(exc)],
            )

        # ── Step 3: Validate SQL ──────────────────────────────────────────────
        validation_errors = self._validate_sql(sql_query, df, table_name)
        if validation_errors:
            logger.warning("SQL validation failed", errors=validation_errors)
            return SQLGenerationResult(
                question=question,
                sql_query=sql_query,
                explanation="Generated SQL failed safety validation.",
                table_name=table_name,
                is_valid=False,
                validation_errors=validation_errors,
            )

        # ── Step 4: Execute SQL ───────────────────────────────────────────────
        execution_result = self._execute_sql(sql_query, df, table_name)

        # ── Step 5: Generate explanation ──────────────────────────────────────
        explanation = self._explain_sql(question, sql_query)

        result = SQLGenerationResult(
            question=question,
            sql_query=sql_query,
            explanation=explanation,
            table_name=table_name,
            execution_result=execution_result,
            is_valid=True,
        )

        logger.info(
            "SQL generation complete",
            success=execution_result.success,
            rows=execution_result.row_count,
        )

        return result

    def _build_schema_description(self, df: pd.DataFrame) -> str:
        """Build a human-readable schema description for the LLM prompt."""
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample)
            lines.append(
                f"  {col} ({dtype}) — nulls: {null_count} — samples: [{sample_str}]"
            )
        return "\n".join(lines)

    def _clean_sql(self, raw: str) -> str:
        """
        Strip markdown fences and extra whitespace from LLM output.

        Args:
            raw: Raw LLM response.

        Returns:
            Clean SQL string.
        """
        # Remove code fences
        raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```", "", raw)
        # Remove leading/trailing whitespace
        raw = raw.strip()
        # Collapse multiple semicolons to one (keep only first statement)
        parts = raw.split(";")
        sql = parts[0].strip() + ";"
        return sql

    def _validate_sql(
        self,
        sql: str,
        df: pd.DataFrame,
        table_name: str,
    ) -> list[str]:
        """
        Validate SQL against safety rules.

        Checks:
        1. Statement must be a SELECT.
        2. No forbidden keywords.
        3. Referenced table name matches expected table.

        Args:
            sql: SQL string to validate.
            df: Source DataFrame (used for column validation).
            table_name: Expected table name.

        Returns:
            List of violation strings. Empty = safe.
        """
        errors: list[str] = []
        sql_upper = sql.upper().strip()

        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            errors.append("Only SELECT statements are allowed.")

        # Check forbidden keywords
        for keyword in _FORBIDDEN_SQL_KEYWORDS:
            # Word boundary check
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, sql_upper):
                errors.append(f"Forbidden SQL keyword detected: {keyword}")

        # Table name reference check
        table_pattern = rf"\b{re.escape(table_name.upper())}\b"
        if not re.search(table_pattern, sql_upper):
            errors.append(
                f"Generated SQL does not reference the expected table '{table_name}'."
            )

        return errors

    def _execute_sql(
        self,
        sql: str,
        df: pd.DataFrame,
        table_name: str,
    ) -> ExecutionResult:
        """
        Execute a validated SQL query against an in-memory SQLite database.

        Args:
            sql: Validated SQL SELECT statement.
            df: DataFrame to load into SQLite.
            table_name: Name for the SQLite table.

        Returns:
            ExecutionResult with the query results.
        """
        start_time = time.perf_counter()

        try:
            # Create in-memory SQLite database and load DataFrame
            conn = sqlite3.connect(":memory:")
            df.to_sql(table_name, conn, index=False, if_exists="replace")

            # Execute query
            result_df = pd.read_sql_query(sql, conn)
            conn.close()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                success=True,
                result_type="dataframe",
                dataframe_result=result_df.to_dict(orient="records"),
                columns=list(result_df.columns),
                row_count=len(result_df),
                execution_time_ms=round(elapsed_ms, 2),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("SQL execution failed", error=str(exc), sql=sql[:200])

            try:
                conn.close()
            except Exception:
                pass

            return ExecutionResult(
                success=False,
                result_type="error",
                execution_time_ms=round(elapsed_ms, 2),
                error_message=str(exc),
            )

    def _explain_sql(self, question: str, sql_query: str) -> str:
        """
        Generate a natural language explanation of the SQL query.

        Args:
            question: Original user question.
            sql_query: The generated SQL.

        Returns:
            Explanation string.
        """
        prompt = _SQL_EXPLANATION_PROMPT.format(
            question=question,
            sql_query=sql_query,
        )

        try:
            return self._llm.generate(prompt, temperature_override=0.0)
        except Exception as exc:
            logger.warning("SQL explanation generation failed", error=str(exc))
            return f"SQL query generated to answer: {question}"


# Module-level singleton
sql_generator = SQLGenerator()


