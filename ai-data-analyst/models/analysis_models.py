"""
Core Pydantic models for analytical tool inputs and outputs.

Every tool in the system returns a typed Pydantic model rather than
raw dictionaries. This enforces contracts between tools, the agent
orchestration layer, and the UI layer, catching data shape errors early
and making the codebase self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class IntentType(str, Enum):
    """Supported analytical intent types recognized by the planner."""

    QUESTION_ANSWERING = "question_answering"
    SQL_GENERATION = "sql_generation"
    PANDAS_GENERATION = "pandas_generation"
    VISUALIZATION = "visualization"
    ANOMALY_DETECTION = "anomaly_detection"
    BUSINESS_INSIGHT = "business_insight"
    DASHBOARD = "dashboard"
    DATA_PROFILE = "data_profile"
    MULTI_INTENT = "multi_intent"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels for AI-generated responses."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IntentClassification(BaseModel):
    """
    Output of the planner node after classifying user intent.

    The planner uses this model to communicate routing decisions
    to the executor node in the LangGraph workflow.
    """

    primary_intent: IntentType = Field(
        ..., description="Primary intent identified from user query"
    )
    secondary_intents: list[IntentType] = Field(
        default_factory=list,
        description="Additional intents when query spans multiple analytical tasks",
    )
    tools_required: list[str] = Field(
        ..., description="Ordered list of tool names to invoke"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this classification was chosen"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Confidence in the intent classification",
    )
    original_query: str = Field(..., description="The user query that was classified")


class ExecutionResult(BaseModel):
    """
    Result of executing generated Pandas or SQL code.

    Wraps both the computed data and metadata about the execution
    so the synthesizer node can include full context in responses.
    """

    success: bool = Field(..., description="Whether execution completed without error")
    result_type: str = Field(
        ..., description="Type of result: 'dataframe', 'scalar', 'series', 'dict'"
    )
    scalar_result: Optional[Any] = Field(
        default=None, description="Result when output is a single value"
    )
    dataframe_result: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Result serialized as list of records when output is a DataFrame",
    )
    columns: Optional[list[str]] = Field(
        default=None, description="Column names when result is tabular"
    )
    row_count: Optional[int] = Field(
        default=None, description="Number of rows in result"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Code execution time in milliseconds"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error description if execution failed"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings during execution"
    )


class QueryResult(BaseModel):
    """Result from the natural language query engine."""

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Natural language answer")
    supporting_data: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Tabular data supporting the answer"
    )
    columns: Optional[list[str]] = Field(
        default=None, description="Column names of supporting data"
    )
    execution_result: Optional[ExecutionResult] = Field(
        default=None, description="Raw execution result if code was run"
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)


class SQLGenerationResult(BaseModel):
    """Result from the SQL generator tool."""

    question: str = Field(..., description="Original user question")
    sql_query: str = Field(..., description="Generated SQL query")
    explanation: str = Field(
        ..., description="Step-by-step explanation of the SQL query"
    )
    table_name: str = Field(..., description="SQLite table the query runs against")
    execution_result: Optional[ExecutionResult] = Field(
        default=None, description="Result of executing the SQL query"
    )
    is_valid: bool = Field(
        default=True, description="Whether the SQL passed safety validation"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors if invalid"
    )


class PandasGenerationResult(BaseModel):
    """Result from the Pandas code generator tool."""

    question: str = Field(..., description="Original user question")
    pandas_code: str = Field(..., description="Generated Pandas code snippet")
    explanation: str = Field(
        ..., description="Step-by-step explanation of the Pandas operations"
    )
    execution_result: Optional[ExecutionResult] = Field(
        default=None, description="Result of executing the Pandas code"
    )
    is_safe: bool = Field(
        default=True, description="Whether AST validation passed"
    )
    safety_violations: list[str] = Field(
        default_factory=list, description="List of detected unsafe operations"
    )


class ColumnProfile(BaseModel):
    """Profile statistics for a single DataFrame column."""

    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: list[Any] = Field(default_factory=list)
    # Numeric-specific
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    # Categorical-specific
    top_value: Optional[str] = None
    top_frequency: Optional[int] = None
    value_counts: Optional[dict[str, int]] = None
    # Date-specific
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None


class DataQualityReport(BaseModel):
    """Summary of data quality checks for an uploaded dataset."""

    total_rows: int
    total_columns: int
    duplicate_rows: int
    duplicate_row_percentage: float
    duplicate_columns: list[str] = Field(default_factory=list)
    columns_with_nulls: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of column name to null percentage",
    )
    constant_columns: list[str] = Field(
        default_factory=list, description="Columns with only one unique value"
    )
    high_cardinality_columns: list[str] = Field(
        default_factory=list,
        description="String columns where unique values > 50% of rows",
    )
    type_inconsistencies: dict[str, str] = Field(
        default_factory=dict,
        description="Columns where inferred type differs from stored dtype",
    )
    memory_usage_bytes: int = Field(default=0)
    overall_quality_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="0–100 quality score based on completeness and consistency",
    )
    quality_issues: list[str] = Field(
        default_factory=list, description="Human-readable list of quality issues"
    )


class DataProfile(BaseModel):
    """
    Comprehensive dataset profile generated after CSV upload.

    Contains structural info, column-level statistics, quality report,
    and correlation data. Displayed on the main dashboard immediately
    after file upload.
    """

    dataset_name: str
    file_path: Optional[str] = None
    row_count: int
    column_count: int
    memory_usage_bytes: int
    dtypes_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of columns per dtype category",
    )
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    date_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    quality_report: DataQualityReport = Field(
        default_factory=DataQualityReport.model_construct
    )
    top_correlations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 10 numeric column pairs by absolute correlation",
    )
    profiled_at: datetime = Field(default_factory=datetime.utcnow)


class AnomalyRecord(BaseModel):
    """A single record identified as anomalous."""

    row_index: int = Field(..., description="Original DataFrame row index")
    anomaly_score: float = Field(
        ..., description="Isolation Forest anomaly score (lower = more anomalous)"
    )
    z_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Z-scores for each numeric column in this row",
    )
    flagged_columns: list[str] = Field(
        default_factory=list,
        description="Columns contributing most to the anomaly",
    )
    record_data: dict[str, Any] = Field(
        default_factory=dict, description="Actual field values for this record"
    )
    explanation: str = Field(
        default="", description="Human-readable explanation of why this is anomalous"
    )


class AnomalyReport(BaseModel):
    """Complete anomaly detection report for a dataset."""

    total_records_analyzed: int
    anomalies_detected: int
    anomaly_percentage: float
    detection_method: str = Field(
        default="isolation_forest",
        description="Primary method used: 'isolation_forest' or 'z_score'",
    )
    contamination_rate: float = Field(
        default=0.05, description="Assumed contamination rate used in Isolation Forest"
    )
    anomaly_records: list[AnomalyRecord] = Field(default_factory=list)
    summary: str = Field(
        default="", description="LLM-generated natural language summary of anomalies"
    )
    columns_analyzed: list[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class BusinessInsight(BaseModel):
    """A single business insight extracted from the dataset."""

    title: str = Field(..., description="Short headline for the insight")
    description: str = Field(
        ..., description="Detailed explanation of the insight"
    )
    metric_name: Optional[str] = Field(
        default=None, description="Primary metric this insight relates to"
    )
    metric_value: Optional[Any] = Field(
        default=None, description="Value of the primary metric"
    )
    insight_type: str = Field(
        default="general",
        description="Category: 'trend', 'anomaly', 'top_performer', 'underperformer', 'general'",
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    supporting_data: Optional[list[dict[str, Any]]] = Field(default=None)


class ReasoningTrace(BaseModel):
    """
    Explainability trace attached to every AI response.

    This is the transparency layer that makes the system auditable.
    Every response includes a full trace of what the agent did,
    why it did it, and what assumptions it made.
    """

    question: str
    intent_classification: Optional[IntentClassification] = None
    tools_invoked: list[str] = Field(default_factory=list)
    tool_execution_times: dict[str, float] = Field(
        default_factory=dict,
        description="Execution time in milliseconds per tool",
    )
    sql_generated: Optional[str] = None
    pandas_code_generated: Optional[str] = None
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made during analysis",
    )
    data_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of the analysis",
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    total_execution_time_ms: float = Field(default=0.0)


class AnalysisResult(BaseModel):
    """
    Top-level container returned by the agent for every user query.

    This is the final output that the UI layer receives and renders.
    It bundles the answer, all supporting artifacts (charts, code,
    data), and the reasoning trace for full explainability.
    """

    question: str
    answer: str = Field(..., description="Primary natural language answer")
    reasoning_trace: ReasoningTrace
    query_result: Optional[QueryResult] = None
    sql_result: Optional[SQLGenerationResult] = None
    pandas_result: Optional[PandasGenerationResult] = None
    anomaly_report: Optional[AnomalyReport] = None
    business_insights: list[BusinessInsight] = Field(default_factory=list)
    chart_configs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized ChartConfig objects to render as Plotly figures",
    )
    suggested_followups: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions based on the analysis",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = Field(
        default=None, description="Error message if analysis partially failed"
    )
