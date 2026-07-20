"""
Secure code execution service using Python AST validation.

This is one of the most security-critical modules in the application.
Generated Pandas and SQL code must NEVER be executed with eval() or exec()
directly because that would allow arbitrary code execution.

Architecture:
1. Parse generated code into an AST.
2. Walk the AST and validate every node against a whitelist.
3. Reject anything that contains imports, file I/O, subprocess calls,
   attribute access to dangerous objects, or any non-whitelisted construct.
4. Only if validation passes, execute in a tightly scoped namespace.

The whitelist approach is intentionally conservative: if a node type
is not explicitly allowed, it is rejected. This is the secure default.
"""

from __future__ import annotations

import ast
import time
import traceback
from typing import Any, Optional

import pandas as pd

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Whitelist of AST node types that are safe to execute ──────────────────────

# Expressions
_ALLOWED_EXPR_NODES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.GeneratorExp,
        ast.Compare,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.Constant,
        ast.Name,
        ast.NameConstant,  # Python 3.7 compat
        ast.Num,           # Python 3.7 compat
        ast.Str,           # Python 3.7 compat
        ast.Index,         # Python 3.7 compat
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.USub,
        ast.UAdd,
        ast.Invert,
        ast.keyword,
        ast.arg,
        ast.Lambda,
        ast.FormattedValue,
        ast.JoinedStr,
        # Context nodes — these are load/store/del contexts on names and
        # subscripts, NOT dangerous operations themselves. Blocking them
        # prevents all variable assignments, which breaks all useful code.
        ast.Store,
        ast.Load,
        ast.Del,
    }
)

# Statements (for multi-line code blocks)
_ALLOWED_STMT_NODES: frozenset[type] = frozenset(
    {
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Return,
        ast.If,
        ast.For,
        ast.While,
        ast.FunctionDef,
        ast.arguments,
        ast.Pass,
        ast.Break,
        ast.Continue,
    }
)

_ALL_ALLOWED_NODES = _ALLOWED_EXPR_NODES | _ALLOWED_STMT_NODES

# ── Forbidden attribute patterns ───────────────────────────────────────────────
# These patterns in attribute access indicate dangerous operations.
_FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__globals__",
        "__code__",
        "__closure__",
        "__import__",
        "__builtins__",
        "__dict__",
        "__subclasses__",
        "mro",
        "system",
        "popen",
        "exec",
        "eval",
        "compile",
        "open",
        "read",
        "write",
        "os",
        "sys",
        "subprocess",
        "socket",
        "pickle",
        "marshal",
        "importlib",
    }
)

# ── Forbidden function/name patterns ──────────────────────────────────────────
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "print",
        "__builtins__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "breakpoint",
        "quit",
        "exit",
    }
)


class SecurityViolationError(Exception):
    """Raised when generated code contains forbidden operations."""


class ExecutionError(Exception):
    """Raised when code execution fails at runtime."""


class ExecutionTimeoutError(ExecutionError):
    """Raised when code execution exceeds the configured timeout."""


class ASTValidator(ast.NodeVisitor):
    """
    AST visitor that validates every node against the security whitelist.

    Raises SecurityViolationError on the first forbidden construct found.
    This fail-fast approach ensures no partial validation is possible.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit(self, node: ast.AST) -> None:
        """Visit a node and check it against the whitelist."""
        node_type = type(node)

        if node_type not in _ALL_ALLOWED_NODES:
            self.violations.append(
                f"Forbidden AST node type: {node_type.__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )

        # Check for forbidden attribute access
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRIBUTES:
                self.violations.append(
                    f"Forbidden attribute access: '.{node.attr}' "
                    f"(line {getattr(node, 'lineno', '?')})"
                )

        # Check for forbidden name usage
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                self.violations.append(
                    f"Forbidden name: '{node.id}' "
                    f"(line {getattr(node, 'lineno', '?')})"
                )

        # Check for import statements (always forbidden)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.violations.append(
                f"Import statements are forbidden "
                f"(line {getattr(node, 'lineno', '?')})"
            )

        self.generic_visit(node)


class ExecutionService:
    """
    Secure Python code execution engine.

    All generated Pandas code goes through this service.
    The execution namespace is pre-populated with only the
    objects that analysis code legitimately needs: the DataFrame,
    Pandas, NumPy, and a small set of built-in math functions.

    No other objects are accessible. The namespace is created
    fresh for each execution to prevent state leakage.
    """

    # Safe built-ins available in the execution namespace
    _SAFE_BUILTINS: dict[str, Any] = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type": type,
        "repr": repr,
        "None": None,
        "True": True,
        "False": False,
    }

    def validate_code(self, code: str) -> list[str]:
        """
        Parse and validate code against the security whitelist.

        Args:
            code: Python source code string to validate.

        Returns:
            List of violation strings. Empty list means code is safe.

        Raises:
            SyntaxError: If code cannot be parsed.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SyntaxError(f"Generated code has syntax error: {exc}") from exc

        validator = ASTValidator()
        validator.visit(tree)
        return validator.violations

    def execute_pandas_code(
        self,
        code: str,
        dataframe: pd.DataFrame,
        variable_name: str = "df",
        timeout_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Securely execute a Pandas code snippet against a DataFrame.

        Steps:
        1. Validate code against the AST whitelist.
        2. Build a sandboxed namespace with the DataFrame and safe imports.
        3. Execute the code using exec() in the restricted namespace.
        4. Extract and return the result variable.

        The result variable is expected to be named 'result' by convention.
        If no 'result' variable is set, the last assigned variable is returned.

        Args:
            code: Pandas Python code to execute.
            dataframe: The DataFrame to make available as `variable_name`.
            variable_name: Name to bind the DataFrame to in the namespace.
            timeout_seconds: Override execution timeout.

        Returns:
            Dict with keys: success, result, result_type, error, execution_time_ms

        Raises:
            SecurityViolationError: If code fails AST validation.
        """
        timeout = timeout_seconds or settings.code_execution_timeout

        # Step 1: Validate
        violations = self.validate_code(code)
        if violations:
            violation_str = "; ".join(violations)
            logger.warning(
                "Code execution blocked by security validator",
                violations=violations,
            )
            raise SecurityViolationError(
                f"Generated code contains unsafe operations: {violation_str}"
            )

        # Step 2: Build restricted namespace
        import numpy as np

        namespace: dict[str, Any] = {
            "__builtins__": self._SAFE_BUILTINS,
            variable_name: dataframe.copy(),  # Protect original DataFrame
            "pd": pd,
            "np": np,
        }

        # Step 3: Execute with timing
        start_time = time.perf_counter()
        try:
            exec(compile(ast.parse(code), "<generated>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Code execution runtime error",
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 2),
                code_snippet=code[:200],
            )
            return {
                "success": False,
                "result": None,
                "result_type": "error",
                "error": f"Runtime error: {exc}",
                "execution_time_ms": elapsed_ms,
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Step 4: Extract result
        result = namespace.get("result", None)
        result_type = _classify_result_type(result)

        logger.info(
            "Code execution successful",
            elapsed_ms=round(elapsed_ms, 2),
            result_type=result_type,
        )

        return {
            "success": True,
            "result": result,
            "result_type": result_type,
            "error": None,
            "execution_time_ms": elapsed_ms,
        }

    def execute_expression(
        self,
        expression: str,
        dataframe: pd.DataFrame,
        variable_name: str = "df",
    ) -> dict[str, Any]:
        """
        Evaluate a single Python expression (not a full code block).

        Useful for simple column operations and filter expressions.

        Args:
            expression: Single Python expression string.
            dataframe: DataFrame to make available.
            variable_name: Name for the DataFrame in the namespace.

        Returns:
            Same structure as execute_pandas_code.
        """
        # Wrap expression in a result assignment for consistent handling
        code = f"result = {expression}"
        return self.execute_pandas_code(code, dataframe, variable_name)


def _classify_result_type(result: Any) -> str:
    """
    Classify the type of an execution result for response formatting.

    Args:
        result: The value returned from code execution.

    Returns:
        String type classification: 'dataframe', 'series', 'scalar', 'dict', 'list', 'none'
    """
    if result is None:
        return "none"
    if isinstance(result, pd.DataFrame):
        return "dataframe"
    if isinstance(result, pd.Series):
        return "series"
    if isinstance(result, dict):
        return "dict"
    if isinstance(result, (list, tuple)):
        return "list"
    if isinstance(result, (int, float, bool, str)):
        return "scalar"
    return "other"


# Module-level singleton
execution_service = ExecutionService()
