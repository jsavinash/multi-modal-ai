"""JSON structured logging configuration (no print statements anywhere in the service)."""

import json
import logging
import sys
from typing import Any

from app.config import settings


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.service_name,
            "environment": settings.environment,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge structured extras attached via `extra={...}` (skip stdlib keys).
        std_keys = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
            "message", "asctime", "taskName",
        }
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in std_keys}
        )
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    # Quiet noisy third-party loggers.
    for noisy in ("pdfminer", "pdfminer.pdffont", "urllib3", "kombu"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
