"""Logging configuration for RAG application."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to log to

    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("rag_langchain")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"rag_langchain.{name}")


def extract_json(text: str):
    """Best-effort extraction of a JSON array/object from an LLM response.

    LLMs often wrap JSON in markdown fences or add stray commentary before/after
    the JSON payload. This pulls out the first well-formed JSON value it can find.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON (list or dict)

    Raises:
        ValueError: If no valid JSON could be extracted
    """
    import json
    import re

    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first [...] or {...} block
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not extract valid JSON from LLM response")
