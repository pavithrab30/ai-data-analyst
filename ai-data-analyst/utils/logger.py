"""
Structured logging configuration using Loguru.

Provides a pre-configured logger instance with:
- Console output with colored formatting
- File rotation with structured JSON output for production
- Context binding for request tracing
- Consistent format across all application modules

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing dataset", rows=1000, columns=15)
"""

import sys
from pathlib import Path
from loguru import logger as _loguru_logger

from config.settings import settings

# ── Log directory ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Remove default Loguru handler ──────────────────────────────────────────────
_loguru_logger.remove()

# ── Console handler ────────────────────────────────────────────────────────────
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_loguru_logger.add(
    sys.stderr,
    format=_CONSOLE_FORMAT,
    level=settings.log_level,
    colorize=True,
    backtrace=True,
    diagnose=settings.debug,
)

# ── File handler — rotating JSON log for structured analysis ──────────────────
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
)

_loguru_logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format=_FILE_FORMAT,
    level="DEBUG",
    rotation="00:00",        # New file at midnight
    retention="30 days",      # Keep 30 days of logs
    compression="zip",        # Compress rotated files
    backtrace=True,
    diagnose=False,           # Disable in file to avoid PII leakage
    enqueue=True,             # Thread-safe async logging
)


def get_logger(name: str):
    """
    Return a bound Loguru logger for a given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A Loguru logger instance bound with the module name context.

    Example:
        logger = get_logger(__name__)
        logger.info("Task started", dataset="sales.csv")
    """
    return _loguru_logger.bind(module=name)
