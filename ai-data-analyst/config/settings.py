"""
Application configuration management using Pydantic Settings.

All configuration is loaded from environment variables via the .env file.
The LLM provider is NVIDIA NIM exclusively.
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="nvidia", description="LLM provider (nvidia)")

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA NIM API key from build.nvidia.com",
    )
    nvidia_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="NVIDIA NIM model identifier",
    )

    # ── LLM generation settings (shared) ──────────────────────────────────────
    llm_temperature: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Sampling temperature"
    )
    llm_max_tokens: int = Field(
        default=8192, gt=0, description="Maximum output tokens"
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_title: str = Field(default="AI Data Analyst")
    app_version: str = Field(default="1.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── Data ───────────────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50, gt=0)
    max_rows_display: int = Field(default=1000, gt=0)
    max_columns: int = Field(default=100, gt=0)
    allowed_extensions: str = Field(default="csv")

    # ── Execution ──────────────────────────────────────────────────────────────
    code_execution_timeout: int = Field(default=30, gt=0)
    max_anomaly_samples: int = Field(default=10000, gt=0)

    # ── Session ────────────────────────────────────────────────────────────────
    session_history_limit: int = Field(default=50, gt=0)

    # ── Reports ────────────────────────────────────────────────────────────────
    report_output_dir: str = Field(default="reports")

    # ── Computed properties ────────────────────────────────────────────────────
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def report_output_path(self) -> Path:
        path = Path(self.report_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        valid = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"app_env must be one of {valid}")
        return lower


settings = Settings()
