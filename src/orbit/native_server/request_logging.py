"""Optional structured request logging for the native server."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

LOG_FILENAME = "requests.jsonl"
MAX_LOG_VALUE_CHARS = 1_048_576


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= MAX_LOG_VALUE_CHARS:
            return value
        return value[:MAX_LOG_VALUE_CHARS] + "...[truncated]"
    if isinstance(value, list):
        return [_bounded(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _bounded(item) for key, item in value.items()}
    return value


class RequestLogger:
    """Append one JSON object per event to a server-owned log file."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError(f"--log must point to a directory: {directory}")
        self.directory = directory
        self.path = directory / LOG_FILENAME
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp": time.time(),
            "event": event,
            **{key: _bounded(value) for key, value in fields.items()},
        }
        with self._lock:
            self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self._handle.flush()
