"""
Natural language query engine.

Routes user questions to the most appropriate execution method
(SQL or Pandas) based on question complexity and returns
a unified QueryResult.

This is the primary entry point for question-answering queries.
The query engine:
1. Analyzes the question to pick the best execution method.
2. Delegates to SQLGenerator or PandasGenerator.
3. Wraps results in a unified QueryResult model.
4. Handles fallback if one method fails.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from models.analysis_models import ConfidenceLevel, QueryResult
from tools.pandas_generator import PandasGenerator, pandas_generator
from tools.sql_generator import SQLGenerator, sql_generator
from utils.logger import get_logger

logger = get_logger(__name__)

# Keywords that suggest SQL is better suited
_SQL_PREFERRED_KEYWORDS = [
    "join", "where", "group by", "order by", "having",
    "count", "distinct", "between", "like",
    "sql", "query",
]

# Keywords that suggest Pandas is better suited
_PANDAS_PREFERRED_KEYWORDS = [
    "calculate", "compute", "rolling", "cumulative",
    "apply", "transform", "reshape", "pivot",
    "resample", "pandas",
]


class QueryEngine:
    """
    Dispatches natural language questions to the right execution backend.

    Decision logic:
    - SQL: simpler aggregation, filtering, joins
    - Pandas: complex transforms, rolling windows, multi-step computation
    - Default: SQL (more deterministic, easier to explain)
    """

    def __init__(
        self,
        sql_gen: Optional[SQLGenerator] = None,
        pd_gen: Optional[PandasGenerator] = None,
    ) -> None:
        self._sql = sql_gen or sql_generator
        self._pandas = pd_gen or pandas_generator

    def answer(
        self,
        question: str,
        df: pd.DataFrame,
        table_name: str = "dataset",
        prefer_pandas: bool = False,
    ) -> QueryResult:
        """
        Answer a natural language question about the dataset.

        Args:
            question: User question.
            df: DataFrame to query.
            table_name: SQLite table name.
            prefer_pandas: Force Pandas execution if True.

        Returns:
            QueryResult with answer, supporting data, and execution details.
        """
        logger.info("Query engine processing question", question=question[:100])

        method = self._select_method(question, prefer_pandas)

        if method == "sql":
            return self._answer_with_sql(question, df, table_name)
        else:
            return self._answer_with_pandas(question, df)

    # ── Private Methods ────────────────────────────────────────────────────────

    def _select_method(self, question: str, prefer_pandas: bool) -> str:
        """Choose between SQL and Pandas based on question keywords."""
        if prefer_pandas:
            return "pandas"

        q_lower = question.lower()

        pandas_score = sum(1 for kw in _PANDAS_PREFERRED_KEYWORDS if kw in q_lower)
        sql_score = sum(1 for kw in _SQL_PREFERRED_KEYWORDS if kw in q_lower)

        if pandas_score > sql_score:
            return "pandas"
        return "sql"  # Default to SQL

    def _answer_with_sql(
        self, question: str, df: pd.DataFrame, table_name: str
    ) -> QueryResult:
        """Execute question via SQL generator."""
        sql_result = self._sql.generate_and_execute(question, df, table_name)

        supporting_data = None
        columns = None
        confidence = ConfidenceLevel.HIGH

        if sql_result.execution_result and sql_result.execution_result.success:
            supporting_data = sql_result.execution_result.dataframe_result
            columns = sql_result.execution_result.columns
        elif not sql_result.is_valid:
            confidence = ConfidenceLevel.LOW

        # Build a natural language answer from the result
        answer = self._format_sql_answer(question, sql_result)

        return QueryResult(
            question=question,
            answer=answer,
            supporting_data=supporting_data,
            columns=columns,
            execution_result=sql_result.execution_result,
            confidence=confidence,
        )

    def _answer_with_pandas(self, question: str, df: pd.DataFrame) -> QueryResult:
        """Execute question via Pandas generator."""
        pd_result = self._pandas.generate_and_execute(question, df)

        supporting_data = None
        columns = None
        confidence = ConfidenceLevel.HIGH

        if pd_result.execution_result and pd_result.execution_result.success:
            supporting_data = pd_result.execution_result.dataframe_result
            columns = pd_result.execution_result.columns
        elif not pd_result.is_safe:
            confidence = ConfidenceLevel.LOW

        answer = self._format_pandas_answer(question, pd_result)

        return QueryResult(
            question=question,
            answer=answer,
            supporting_data=supporting_data,
            columns=columns,
            execution_result=pd_result.execution_result,
            confidence=confidence,
        )

    def _format_sql_answer(self, question: str, sql_result) -> str:
        """Format the SQL execution result into a readable answer."""
        exec_r = sql_result.execution_result

        if not sql_result.is_valid:
            return f"I was unable to generate a valid SQL query for this question. Issues: {', '.join(sql_result.validation_errors)}"

        if not exec_r or not exec_r.success:
            error = exec_r.error_message if exec_r else "Unknown error"
            return f"The query was generated but failed to execute: {error}"

        row_count = exec_r.row_count or 0
        if row_count == 0:
            return "The query returned no results. This may mean no data matches the specified criteria."

        if exec_r.result_type == "dataframe":
            return (
                f"Query executed successfully. Found {row_count:,} result(s). "
                f"See the table below for details.\n\nSQL: `{sql_result.sql_query}`"
            )
        elif exec_r.scalar_result is not None:
            return f"Result: **{exec_r.scalar_result}**\n\nSQL: `{sql_result.sql_query}`"

        return f"Query completed. SQL: `{sql_result.sql_query}`"

    def _format_pandas_answer(self, question: str, pd_result) -> str:
        """Format the Pandas execution result into a readable answer."""
        exec_r = pd_result.execution_result

        if not pd_result.is_safe:
            return f"Code generation was blocked for safety reasons: {', '.join(pd_result.safety_violations)}"

        if not exec_r or not exec_r.success:
            error = exec_r.error_message if exec_r else "Unknown error"
            return f"The code was generated but failed to execute: {error}"

        if exec_r.scalar_result is not None:
            return f"Result: **{exec_r.scalar_result}**"

        row_count = exec_r.row_count or 0
        return (
            f"Analysis complete. Found {row_count:,} result(s). "
            f"See the table below for details."
        )


# Module-level singleton
query_engine = QueryEngine()
