"""Structured logging configuration for solana-mcp."""

import logging
import sys
from typing import Any


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (default: INFO)
        json_format: If True, output JSON-formatted logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("solana_mcp")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include any extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'solana_mcp.')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"solana_mcp.{name}")
    return logging.getLogger("solana_mcp")


# Module-level logger for convenience
logger = get_logger()
