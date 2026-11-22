"""Optional structured request logging for the native server."""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
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

LOG_CONTENT_ENV = "ORBIT_LOG_CONTENT"        # full (default) | redact | none
LOG_MAX_BYTES_ENV = "ORBIT_LOG_MAX_BYTES"    # rotate at this size; 0 disables rotation
LOG_BACKUPS_ENV = "ORBIT_LOG_BACKUPS"        # rotated files kept
DEFAULT_LOG_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_LOG_BACKUPS = 7
GENESIS = "0" * 64

# Fields that carry user/model text; the log-content mode decides what is kept.
_CONTENT_KEYS = frozenset({"content", "prompt", "input", "text", "system", "instructions"})
# Credentials that must never reach a log file, whatever the content mode.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def scrub_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) + " " if m.lastindex else "") + "[redacted-secret]", text)
    return text


def _summarize(text: str) -> dict[str, Any]:
    return {"chars": len(text), "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()}


def _bounded(value: Any, mode: str = "full", key: str = "") -> Any:
    if isinstance(value, str):
        if key in _CONTENT_KEYS and mode != "full":
            return _summarize(value)
        value = scrub_secrets(value)
        if len(value) <= MAX_LOG_VALUE_CHARS:
            return value
        return value[:MAX_LOG_VALUE_CHARS] + "...[truncated]"
    if isinstance(value, list):
        return [_bounded(item, mode, key) for item in value]
    if isinstance(value, dict):
        return {str(k): _bounded(item, mode, str(k)) for k, item in value.items()}
    return value


def _env_int(name: str, default: int, environ: Mapping[str, str]) -> int:
    try:
        value = int(environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


class _PrivateRotatingHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler whose files are always created owner-only (0600)."""

    def _open(self):  # type: ignore[override]
        descriptor = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        return os.fdopen(descriptor, self.mode, encoding=self.encoding)


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _last_chain(path: Path) -> tuple[int, str]:
    """(seq, hash) of the last well-formed row, to continue a chain after restart."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            lines = handle.read().splitlines()
    except OSError:
        return 0, GENESIS
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return int(row["seq"]), str(row["hash"])
        except (ValueError, KeyError, TypeError):
            continue
    return 0, GENESIS


class RequestLogger:
    """Structured, rotating, tamper-evident request log built on stdlib ``logging``.

    * one JSON object per line, written through a ``RotatingFileHandler``
      (``ORBIT_LOG_MAX_BYTES`` / ``ORBIT_LOG_BACKUPS``), files are 0600;
    * ``ORBIT_LOG_CONTENT`` = ``full`` (default, prompts verbatim), ``redact``
      (prompt/message text replaced by length + SHA-256) or ``none`` (same, the
      mode name documents intent: no text is retained);
    * credentials (bearer tokens, hf_/sk-/gh*_ keys, private-key headers) are
      scrubbed in every mode;
    * every row carries ``seq`` and a SHA-256 ``hash`` over its content and the
      previous row's hash, so removal or edits of rows are detectable with
      ``verify_chain`` (the chain continues across rotated files and restarts).
    """

    def __init__(self, directory: Path, *, environ: Mapping[str, str] | None = None) -> None:
        env = os.environ if environ is None else environ
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir():
            raise ValueError(f"--log must point to a directory: {directory}")
        self.directory = directory
        self.path = directory / LOG_FILENAME
        mode = env.get(LOG_CONTENT_ENV, "full").strip().lower()
        self.content_mode = mode if mode in {"full", "redact", "none"} else "full"
        max_bytes = _env_int(LOG_MAX_BYTES_ENV, DEFAULT_LOG_MAX_BYTES, env)
        backups = _env_int(LOG_BACKUPS_ENV, DEFAULT_LOG_BACKUPS, env)
        self._logger = logging.getLogger(f"orbit.requests.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler = _PrivateRotatingHandler(
            self.path, maxBytes=max_bytes if backups else 0, backupCount=backups, encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)
        self._lock = threading.Lock()
        self._seq, self._prev = _last_chain(self.path)

    def close(self) -> None:
        with self._lock:
            self._logger.removeHandler(self._handler)
            self._handler.close()

    def write(self, event: str, **fields: Any) -> None:
        with self._lock:
            row = {
                "timestamp": time.time(),
                "event": event,
                **{key: _bounded(value, self.content_mode, key) for key, value in fields.items()},
                "seq": self._seq + 1,
                "prev": self._prev,
            }
            digest = hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()
            row["hash"] = digest
            self._logger.info(json.dumps(row, ensure_ascii=False, sort_keys=True))
            self._seq, self._prev = row["seq"], digest


def verify_chain(paths: "list[Path]") -> tuple[bool, str]:
    """Check hash links over ``paths`` (oldest first). Returns (ok, reason)."""
    prev: str | None = None
    expected_seq: int | None = None
    count = 0
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                claimed = row.pop("hash")
            except (ValueError, KeyError):
                return False, f"{path.name}:{number}: malformed row"
            if hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest() != claimed:
                return False, f"{path.name}:{number}: row content was altered"
            if prev is not None and row.get("prev") != prev:
                return False, f"{path.name}:{number}: chain broken (a row was removed or reordered)"
            if expected_seq is not None and row.get("seq") != expected_seq:
                return False, f"{path.name}:{number}: sequence gap"
            prev, expected_seq, count = claimed, int(row["seq"]) + 1, count + 1
    return True, f"{count} rows verified"


def chain_files(directory: Path) -> "list[Path]":
    """Log files of a directory, oldest first (``requests.jsonl.N`` ... ``requests.jsonl``)."""
    rotated = sorted(
        (p for p in directory.glob(LOG_FILENAME + ".*") if p.suffix[1:].isdigit()),
        key=lambda p: int(p.suffix[1:]), reverse=True,
    )
    current = directory / LOG_FILENAME
    return rotated + ([current] if current.exists() else [])


def main(argv: "list[str] | None" = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "verify":
        print("usage: python -m orbit.native_server.request_logging verify <log-directory>", file=sys.stderr)
        return 2
    ok, reason = verify_chain(chain_files(Path(args[1])))
    print(("OK: " if ok else "FAILED: ") + reason)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


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
