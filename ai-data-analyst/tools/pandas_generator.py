"""
Pandas code generation tool.

Translates natural language questions into safe, executable Pandas code
snippets. The generated code is:
1. Validated by the AST security layer before execution.
2. Executed in a sandboxed namespace via ExecutionService.
3. Accompanied by a plain-English explanation.

The code always assigns its final result to a variable named `result`
so the execution service can extract it consistently.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from models.analysis_models import ExecutionResult, PandasGenerationResult
from services.execution_service import ExecutionService, SecurityViolationError, execution_service
from services.llm_service import LLMService
from utils.logger import get_logger

logger = get_logger(__name__)

_PANDAS_GENERATION_PROMPT = """You are a Python/Pandas expert. Generate safe Pandas code to answer the user's question.

DataFrame variable name: df
Schema:
{schema}

Sample data (first 3 rows):
{sample_data}

User question: {question}

STRICT RULES:
1. Use ONLY the variable `df` — it is already loaded.
2. Assign the final result to a variable named `result`.
3. Do NOT import any modules (pandas and numpy are already available as pd and np).
4. Do NOT use eval(), exec(), open(), or any file/system operations.
5. Do NOT print() anything.
6. The code must be a complete, runnable snippet (no function definitions needed).
7. For aggregations, use df.groupby() and .agg() or .sum()/.mean() etc.
8. For filtering, use boolean indexing: df[df['col'] > value]
9. Keep the code concise — avoid unnecessary intermediate variables.
10. result must be a DataFrame, Series, or scalar value.

Return ONLY the Python code, no markdown fences, no explanation.

Python Code:"""

_PANDAS_EXPLANATION_PROMPT = """Explain the following Pandas code in plain English.
Describe what each line does and how it answers the question.

Question: {question}
Code:
{code}

Provide a step-by-step explanation in 3-5 bullet points. Be concise and clear."""


class PandasGenerator:
    """
    Generates executable Pandas code from natural language questions.

    Integrates with LLMService for code generation and
    ExecutionService for safe AST-validated execution.
    """

    def __init__(
        self,
        llm_service_arg: Optional[LLMService] = None,
        exec_service: Optional[ExecutionService] = None,
    ) -> None:
        self._llm = llm_service_arg or LLMService()
        self._exec = exec_service or execution_service

    def generate_and_execute(
        self,
        question: str,
        df: pd.DataFrame,
    ) -> PandasGenerationResult:
        """
        Full pipeline: generate Pandas code → validate → execute → explain.

        Args:
            question: Natural language question from the user.
            df: DataFrame to analyze.

        Returns:
            PandasGenerationResult with code, explanation, and execution output.
        """
        logger.info("Pandas code generation started", question=question[:100])

        # ── Step 1: Build schema description ──────────────────────────────────
        schema_str = self._build_schema_description(df)
        sample_str = df.head(3).to_string(index=False)

        # ── Step 2: Generate code with Gemini ─────────────────────────────────
        prompt = _PANDAS_GENERATION_PROMPT.format(
            schema=schema_str,
            sample_data=sample_str,
            question=question,
        )

        try:
            raw_code = self._llm.generate(prompt, temperature_override=0.0)
            pandas_code = self._clean_code(raw_code)
        except Exception as exc:
            logger.error("Pandas code generation LLM call failed", error=str(exc))
            return PandasGenerationResult(
                question=question,
                pandas_code="",
                explanation=f"Failed to generate code: {exc}",
                is_safe=False,
                safety_violations=[str(exc)],
            )

        # ── Step 3: Validate and execute ──────────────────────────────────────
        try:
            raw_result = self._exec.execute_pandas_code(
                code=pandas_code,
                dataframe=df,
                variable_name="df",
            )
        except SecurityViolationError as sec_exc:
            logger.warning(
                "Pandas code blocked by security validator",
                error=str(sec_exc),
            )
            return PandasGenerationResult(
                question=question,
                pandas_code=pandas_code,
                explanation="Code was blocked by the security validator.",
                is_safe=False,
                safety_violations=[str(sec_exc)],
            )
        except SyntaxError as syn_exc:
            return PandasGenerationResult(
                question=question,
                pandas_code=pandas_code,
                explanation=f"Generated code has a syntax error: {syn_exc}",
                is_safe=False,
                safety_violations=[str(syn_exc)],
            )

        # ── Step 4: Build ExecutionResult ─────────────────────────────────────
        execution_result = self._build_execution_result(raw_result)

        # ── Step 5: Generate explanation ──────────────────────────────────────
        explanation = self._explain_code(question, pandas_code)

        result = PandasGenerationResult(
            question=question,
            pandas_code=pandas_code,
            explanation=explanation,
            execution_result=execution_result,
            is_safe=True,
        )

        logger.info(
            "Pandas code generation complete",
            success=execution_result.success,
            result_type=execution_result.result_type,
        )

        return result

    def _build_schema_description(self, df: pd.DataFrame) -> str:
        """Build a concise schema description for the prompt."""
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v) for v in sample)
            lines.append(
                f"  df['{col}'] — dtype: {dtype}, nulls: {null_count}, "
                f"samples: [{sample_str}]"
            )
        return "\n".join(lines)

    def _clean_code(self, raw: str) -> str:
        """
        Strip markdown fences and normalize the generated code.

        Args:
            raw: Raw LLM response.

        Returns:
            Clean Python code string.
        """
        # Remove code fences
        raw = re.sub(r"```(?:python)?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```", "", raw)
        # Strip surrounding whitespace
        return raw.strip()

    def _build_execution_result(self, raw_result: dict) -> ExecutionResult:
        """
        Convert the raw dict from ExecutionService into an ExecutionResult model.

        Args:
            raw_result: Dict returned by ExecutionService.execute_pandas_code.

        Returns:
            Typed ExecutionResult model.
        """
        success = raw_result.get("success", False)
        result = raw_result.get("result")
        result_type = raw_result.get("result_type", "none")
        error = raw_result.get("error")
        elapsed = raw_result.get("execution_time_ms", 0.0)

        exec_result = ExecutionResult(
            success=success,
            result_type=result_type,
            execution_time_ms=elapsed,
            error_message=error,
        )

        if success and result is not None:
            if isinstance(result, pd.DataFrame):
                exec_result.dataframe_result = result.head(500).to_dict(orient="records")
                exec_result.columns = list(result.columns)
                exec_result.row_count = len(result)
            elif isinstance(result, pd.Series):
                exec_result.dataframe_result = result.head(500).reset_index().to_dict(orient="records")
                exec_result.columns = ["index", result.name or "value"]
                exec_result.row_count = len(result)
            elif isinstance(result, (int, float, bool, str)):
                exec_result.scalar_result = result
            elif isinstance(result, dict):
                exec_result.dataframe_result = [result]
                exec_result.columns = list(result.keys())
                exec_result.row_count = 1
            elif isinstance(result, (list, tuple)):
                exec_result.scalar_result = result

        return exec_result

    def _explain_code(self, question: str, code: str) -> str:
        """Generate a plain-English explanation of the Pandas code."""
        prompt = _PANDAS_EXPLANATION_PROMPT.format(
            question=question,
            code=code,
        )
        try:
            return self._llm.generate(prompt, temperature_override=0.0)
        except Exception as exc:
            logger.warning("Code explanation generation failed", error=str(exc))
            return f"Pandas code generated to answer: {question}"


# Module-level singleton
pandas_generator = PandasGenerator()


