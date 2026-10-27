"""
Centralised Logging Configuration for StudentAI.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loading dataset...")

Logging levels (in order of increasing severity):
    DEBUG    → detailed developer information
    INFO     → normal operation milestones (file loaded, model trained)
    WARNING  → something unexpected but recoverable happened
    ERROR    → a serious failure that prevented an operation

Logs are written to TWO places simultaneously:
    1. The terminal (console) — so you see them while the app is running.
    2. outputs/logs/studentai.log — a persistent file you can review later.

The log file uses a RotatingFileHandler, meaning:
    - Each log file grows up to 1 MB maximum.
    - When it reaches 1 MB, Python automatically starts a new file.
    - Up to 3 old log files are kept (studentai.log.1, .log.2, .log.3).
    - This prevents the log file from filling your disk.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


# Paths
_LOG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "outputs",
    "logs",
)
_LOG_FILE = os.path.join(_LOG_DIR, "studentai.log")

# Shared formatter: timestamp | level | module | message
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Module-level flag so we configure handlers exactly once
_configured: bool = False


def _setup_root_logger() -> None:
    """
    Attaches one console handler and one rotating file handler to the
    root logger. Idempotent: subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger("studentai")
    root.setLevel(logging.DEBUG)

    # 1. Console handler — shows INFO and above in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)

    # 2. File handler — saves DEBUG and above to disk
    os.makedirs(_LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=1_048_576,   # 1 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.propagate = False  # Don't bubble up to the default root logger

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named child logger under the 'studentai' hierarchy.

    Parameters:
        name (str): Module name, typically passed as __name__.

    Returns:
        logging.Logger: Configured logger ready to use.

    Example:
        logger = get_logger(__name__)
        logger.info("Dataset loaded successfully.")
    """
    _setup_root_logger()
    return logging.getLogger(f"studentai.{name}")
