"""Optional structured request logging for the native server."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

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
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir():
            raise ValueError(f"--log must point to a directory: {directory}")
        self.directory = directory
        self.path = directory / LOG_FILENAME
        # Prompts are logged verbatim: the file is owner-only, never world-readable.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        self._handle = os.fdopen(descriptor, "a", encoding="utf-8")
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


AUDIT_HASH_MODEL_ENV = "ORBIT_AUDIT_HASH_MODEL"
_SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def _sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _meminfo_kib(field: str) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def run_manifest(
    argv: list[str], *, model_path: Path | None, alias: str | None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Facts an auditor needs to reproduce or explain this server run.

    Records the resolved model file (size, mtime, and its SHA-256 when
    ORBIT_AUDIT_HASH_MODEL=1, because hashing a multi-GB file delays startup),
    the interpreter and host, and every ORBIT_* variable in effect. Values of
    variables whose names look like secrets are replaced, never logged.
    """
    env = os.environ if environ is None else environ
    orbit_env = {
        name: ("[redacted]" if _SECRET_NAME.search(name) else value)
        for name, value in sorted(env.items()) if name.startswith("ORBIT_")
    }
    model: dict[str, Any] | None = None
    if model_path is not None:
        model = {"path": str(model_path)}
        try:
            stat = model_path.stat()
            model.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if env.get(AUDIT_HASH_MODEL_ENV, "").strip() == "1":
                model["sha256"] = _sha256_file(model_path)
        except OSError as exc:
            model["error"] = type(exc).__name__
    try:
        from orbit.build_identity import PROCESS_BUILD_IDENTITY as build
        build_info: Any = {"commit": build.commit, "version": build.version,
                           "description": build.description}
    except Exception:  # identity is best-effort evidence, never a start blocker
        build_info = None
    return {
        "argv": list(argv),
        "alias": alias,
        "model": model,
        "build": build_info,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "mem_total_kib": _meminfo_kib("MemTotal"),
        "mem_available_kib": _meminfo_kib("MemAvailable"),
        "orbit_env": orbit_env,
    }
