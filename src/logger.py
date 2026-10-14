"""Lightweight file logging for the Streamlit app.

Privacy note
------------
Only the tweet text the user explicitly submitted for classification, the
prediction, and the confidence are recorded. No IP address, no session
identifier, no browser metadata, no account information. The log exists so
the student can demonstrate the app's behaviour during a mentor review and
diagnose errors - not to build a user profile.

``logs/`` is git-ignored, so nothing logged locally is ever committed.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str = "digo") -> logging.Logger:
    """Return the project logger, configuring handlers exactly once.

    Falls back to console-only logging if the log directory is not writable
    (which is the case on some hosted environments). A logging failure must
    never take down the app, so the filesystem error is swallowed
    deliberately and noted on the console handler instead.
    """
    global _configured
    logger = logging.getLogger(name)

    if _configured:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.WARNING)
    logger.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / "digo_app.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except OSError as error:  # read-only filesystem, permissions, etc.
        logger.warning("File logging disabled (%s); using console only.", error)

    _configured = True
    return logger


def log_prediction(text: str, label: str, confidence: float | None) -> None:
    """Record one prediction.

    The tweet is truncated to 200 characters to keep the log readable.
    """
    logger = get_logger()
    snippet = text[:200].replace("\n", " ")
    confidence_text = f"{confidence:.4f}" if confidence is not None else "n/a"
    logger.info("PREDICTION | label=%s | confidence=%s | text=%r", label, confidence_text, snippet)


def log_error(message: str, error: Exception) -> None:
    """Record an application error with its type and message."""
    get_logger().error("%s | %s: %s", message, type(error).__name__, error)
