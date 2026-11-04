#!/usr/bin/env python3
"""Qualify every locally available registry model against the matrix fixture set.

For each model in the registry: if its target GGUF is not available locally the
row is SKIPPED; otherwise an `orbit server --model-id <id>` is launched with the
qualified analysis profile, health is awaited, the qualification runner executes,
and the server is always stopped.

Row statuses: PASS (every fixture applicable and passing), PARTIAL (passing, but
some fixtures not applicable to the profile), NOT_COVERED (no fixture applies to
the profile: the model is not covered by the fixture set, which is not a model
defect), FAIL, SKIPPED (model not on disk). The runner's identity gate compares
the server's compatibility_profile with the registry profile_id, which is what
this driver passes.

Exit status: 1 any FAIL; 2 unknown --model-id; 3 nothing was qualified (every
row SKIPPED/NOT_COVERED) unless --allow-empty; otherwise 0.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"
PARTIAL, NOT_COVERED = "PARTIAL", "NOT_COVERED"
STATUSES = (PASS, PARTIAL, NOT_COVERED, FAIL, SKIPPED)
DEFAULT_FIXTURES = ROOT / "qualification/fixtures/matrix-v1.json"


@dataclass
class MatrixConfig:
    ctx: int = 8192
    threads: int = 6
    threads_batch: int = 6
    batch: int = 256
    ubatch: int = 128
    startup_timeout: float = 600.0
    request_timeout: float = 900.0
    models_dir: Path | None = None
    hf_cache: Path | None = None
    fixtures: Path = DEFAULT_FIXTURES
    output_dir: Path = field(default_factory=lambda: Path("workdir/model-matrix"))


def server_command(model_id: str, port: int, cfg: MatrixConfig) -> list[str]:
    """Same flags as the qualified profile used by orbit_qualify_compare/lifecycle."""
    command = [
        sys.executable, "-m", "orbit.terminal.cli", "server",
        "--host", "127.0.0.1", "--port", str(port),
        "--model-id", model_id,
        "--ctx", str(cfg.ctx), "--threads", str(cfg.threads),
        "--threads-batch", str(cfg.threads_batch), "--batch", str(cfg.batch),
        "--ubatch", str(cfg.ubatch), "--think", "off",
    ]
    if cfg.models_dir is not None:
        command += ["--models-dir", str(cfg.models_dir)]
    return command


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def launch_server(model_id: str, port: int, cfg: MatrixConfig, log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(SRC), "ORBIT_KV_PREFIX_PREWARM": "off"}
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            server_command(model_id, port, cfg), cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )


def wait_for_health(base_url: str, process, timeout: float, log_path: Path) -> None:
    from orbit.backend.llama_server import LlamaServerBackend

    backend = LlamaServerBackend(base_url=base_url, timeout=5)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:] if log_path.exists() else ""
            raise RuntimeError(f"server exited during startup: {tail}")
        if backend.health():
            return
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout:.0f}s")


def stop_server(process) -> None:
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=5)


def _signal_group(process, sig) -> None:
    """Signal the server's whole process group (it is its own session leader)."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()


def run_qualification(base_url: str, profile_id: str, server_pid: int | None,
                      cfg: MatrixConfig, output_path: Path) -> dict[str, Any]:
    """Run QualificationRunner (as scripts/orbit_qualify.py does) against base_url."""
    from orbit.backend.llama_server import LlamaServerBackend
    from orbit.qualification.fixtures import load_fixture_set
    from orbit.qualification.reporting import format_summary, write_result
    from orbit.qualification.runner import QualificationRunner, RuntimeFixtureExecutor
    from orbit.qualification.schema import Status
    from scripts.orbit_qualify import build_provenance

    backend = LlamaServerBackend(base_url=base_url, timeout=cfg.request_timeout)
    props = backend.backend_props()
    profile = props.get("model_compatibility")
    if not isinstance(profile, dict):
        raise RuntimeError("server did not expose model compatibility metadata")
    reported = profile.get("compatibility_profile")
    if reported != profile_id:
        raise RuntimeError(
            f"server reports compatibility_profile {reported!r}, registry expects {profile_id!r}"
        )
    fixture_set = load_fixture_set(str(cfg.fixtures))
    with tempfile.TemporaryDirectory(prefix="orbit-matrix-qualification-") as workdir:
        run = QualificationRunner(
            fixture_set=fixture_set,
            profile=profile,
            provenance=build_provenance(profile_id, fixture_set.content_hash, profile, props, server_pid),
            executor=RuntimeFixtureExecutor(backend, process_pid=server_pid),
            workdir=Path(workdir),
        ).run(None)
    write_result(run, str(output_path))
    total = len(run.fixtures)
    applicable = sum(1 for item in run.fixtures if getattr(item, "applicable", True))
    if run.overall_status is Status.PASS:
        status = PASS if applicable == total else PARTIAL
    elif run.overall_status is Status.NOT_APPLICABLE:
        status = NOT_COVERED
    else:
        status = FAIL
    return {"status": status, "summary": format_summary(run),
            "coverage": f"{applicable}/{total}"}


def model_available(manifest, cfg: MatrixConfig) -> Path | None:
    """Return the resolved target path, or None when the model is not usable locally."""
    from orbit.native_llama.model_registry import resolve_model

    try:
        return resolve_model(
            manifest, models_dir=cfg.models_dir, hf_cache=cfg.hf_cache
        ).target_path
    except FileNotFoundError:
        # Missing and incomplete (split) models are both "not usable here";
        # any other error is a real problem and surfaces as FAIL for the row.
        return None


def qualify_model(manifest, cfg: MatrixConfig, *, available: Callable, launch: Callable,
                  wait_health: Callable, run: Callable, stop: Callable,
                  port_factory: Callable[[], int] = free_port) -> dict[str, Any]:
    row: dict[str, Any] = {"model_id": manifest.id, "display_name": manifest.display_name,
                           "status": SKIPPED, "detail": "", "wall_seconds": 0.0}
    started = time.monotonic()
    process = None
    log_path = None
    try:
        path = available(manifest, cfg)
        if path is None:
            row["detail"] = "model not available locally"
            return row
        row["path"] = str(path)
        log_fd, log_name = tempfile.mkstemp(prefix=f"orbit-matrix-{manifest.id}-", suffix=".log")
        os.close(log_fd)
        log_path = Path(log_name)
        port = port_factory()
        base_url = f"http://127.0.0.1:{port}"
        process = launch(manifest.id, port, cfg, log_path)
        wait_health(base_url, process, cfg.startup_timeout, log_path)
        out = Path(cfg.output_dir) / f"{manifest.id}.qualification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run(base_url, manifest.profile_id, getattr(process, "pid", None), cfg, out)
        status = result.get("status")
        row["status"] = status if status in (PASS, PARTIAL, NOT_COVERED) else FAIL
        row["coverage"] = str(result.get("coverage", ""))
        row["detail"] = str(result.get("summary", ""))
        row["result_path"] = str(out)
    except Exception as error:  # noqa: BLE001 - recorded as FAIL for this model
        row["status"] = FAIL
        row["detail"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            try:
                stop(process)
            except Exception as stop_error:  # noqa: BLE001
                row["status"] = FAIL
                row["detail"] = (row["detail"] + f" | stop failed: {stop_error}").strip(" |")
        if log_path is not None:
            log_path.unlink(missing_ok=True)
        row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def build_report(cfg: MatrixConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    qualified = counts[PASS] + counts[PARTIAL]
    overall = FAIL if counts[FAIL] else (PASS if qualified else NOT_COVERED)
    return {"schema_version": 2, "fixtures": str(cfg.fixtures), "counts": counts,
            "overall": overall, "models": rows}


def run_matrix(manifests: Sequence, cfg: MatrixConfig, *, on_row: Callable | None = None,
               **hooks) -> dict[str, Any]:
    """Run every model; `on_row(report)` is called after each so an interrupt keeps results."""
    rows: list[dict[str, Any]] = []
    try:
        for manifest in manifests:
            rows.append(qualify_model(manifest, cfg, **hooks))
            if on_row is not None:
                on_row(build_report(cfg, rows))
    except KeyboardInterrupt:
        rows.append({"model_id": "(interrupted)", "display_name": "", "status": FAIL,
                     "detail": "interrupted", "wall_seconds": 0.0})
        if on_row is not None:
            on_row(build_report(cfg, rows))
        raise
    return build_report(cfg, rows)


def format_table(report: dict[str, Any]) -> str:
    rows = report["models"]
    width = max([len("MODEL")] + [len(r["model_id"]) for r in rows])
    lines = [f"{'MODEL'.ljust(width)}  {'STATUS':<8}  {'WALL_S':>8}  DETAIL"]
    for r in rows:
        detail = (r["detail"].strip().splitlines() or [""])[0][:100]
        lines.append(f"{r['model_id'].ljust(width)}  {r['status']:<8}  {r['wall_seconds']:>8.1f}  {detail}")
    c = report["counts"]
    lines.append(" ".join(f"{s}={c[s]}" for s in STATUSES) + f" overall={report['overall']}")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, text_path = output_dir / "matrix.json", output_dir / "matrix.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(format_table(report) + "\n", encoding="utf-8")
    return json_path, text_path


def _raise_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("workdir/model-matrix"))
    p.add_argument("--models-dir", type=Path)
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument("--model-id", action="append", dest="model_ids", help="restrict to these registry ids")
    for name, default in (("ctx", 8192), ("threads", 6), ("threads-batch", 6), ("batch", 256), ("ubatch", 128)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--allow-empty", action="store_true",
                   help="exit 0 even when no model was qualified (all SKIPPED/NOT_COVERED)")
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    return p


def main(argv: Sequence[str] | None = None, *, manifests: Sequence | None = None, **hooks) -> int:
    a = _parser().parse_args(argv)
    cfg = MatrixConfig(ctx=a.ctx, threads=a.threads, threads_batch=a.threads_batch, batch=a.batch,
                       ubatch=a.ubatch, startup_timeout=a.startup_timeout,
                       request_timeout=a.request_timeout, models_dir=a.models_dir,
                       fixtures=a.fixtures, output_dir=a.output_dir)
    if manifests is None:
        from orbit.native_llama.model_registry import load_registry
        manifests = load_registry()
    if a.model_ids:
        unknown = sorted(set(a.model_ids) - {m.id for m in manifests})
        if unknown:
            print(f"unknown --model-id: {', '.join(unknown)}", file=sys.stderr)
            return 2
        manifests = [m for m in manifests if m.id in set(a.model_ids)]
    hooks.setdefault("available", model_available)
    hooks.setdefault("launch", launch_server)
    hooks.setdefault("wait_health", wait_for_health)
    hooks.setdefault("run", run_qualification)
    hooks.setdefault("stop", stop_server)
    # SIGTERM must unwind through the per-model `finally` so no server is orphaned.
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        report = run_matrix(manifests, cfg, on_row=lambda r: write_reports(r, cfg.output_dir), **hooks)
    finally:
        signal.signal(signal.SIGTERM, previous)
    write_reports(report, cfg.output_dir)
    print(format_table(report))
    if report["counts"][FAIL]:
        return 1
    if report["counts"][PASS] + report["counts"][PARTIAL] == 0 and not a.allow_empty:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
