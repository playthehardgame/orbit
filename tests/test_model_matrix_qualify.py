from __future__ import annotations

import importlib.util
import json
import sys
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location("orbit_model_matrix_qualify", ROOT / "scripts/orbit_model_matrix_qualify.py")
mq = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mq
_SPEC.loader.exec_module(mq)


class FakeProcess:
    pid = 4242


def manifest(model_id):
    return SimpleNamespace(id=model_id, display_name=model_id.upper(), profile_id=f"profile-{model_id}")


class Harness:
    def __init__(self, present=(), run_status="PASS", health_error=None, run_error=None):
        self.present, self.run_status = set(present), run_status
        self.health_error, self.run_error = health_error, run_error
        self.launched, self.stopped, self.ran = [], [], []

    def hooks(self):
        return dict(
            available=lambda m, cfg: Path(f"/x/{m.id}.gguf") if m.id in self.present else None,
            launch=self.launch, wait_health=self.wait, run=self.run, stop=self.stopped.append,
            port_factory=lambda: 1234,
        )

    def launch(self, model_id, port, cfg, log):
        self.launched.append(model_id)
        return FakeProcess()

    def wait(self, url, process, timeout, log):
        if self.health_error:
            raise self.health_error

    def run(self, url, profile_id, pid, cfg, out):
        self.ran.append((url, profile_id, pid))
        if self.run_error:
            raise self.run_error
        return {"status": self.run_status, "summary": "summary line", "coverage": "4/4"}


class MatrixQualifyTests(unittest.TestCase):
    def go(self, harness, ids=("a", "b")):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        code = mq.main(["--output-dir", self.tmp.name], manifests=[manifest(i) for i in ids], **harness.hooks())
        return code, json.loads((Path(self.tmp.name) / "matrix.json").read_text())

    def test_missing_models_skipped_not_fail(self):
        h = Harness(present={"a"})
        code, report = self.go(h)
        self.assertEqual(code, 0)
        self.assertEqual({r["model_id"]: r["status"] for r in report["models"]}, {"a": "PASS", "b": "SKIPPED"})
        self.assertEqual(h.launched, ["a"])
        self.assertEqual(report["overall"], "PASS")

    def test_all_missing_is_not_a_green_run(self):
        code, report = self.go(Harness())
        self.assertEqual(code, 3)
        self.assertEqual(report["overall"], "NOT_COVERED")
        self.assertEqual(report["counts"]["SKIPPED"], 2)

    def test_all_missing_allowed_with_flag(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        code = mq.main(["--output-dir", self.tmp.name, "--allow-empty"],
                       manifests=[manifest("a")], **Harness().hooks())
        self.assertEqual(code, 0)

    def test_unknown_model_id_is_rejected(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        h = Harness(present={"a"})
        code = mq.main(["--output-dir", self.tmp.name, "--model-id", "zzz"],
                       manifests=[manifest("a")], **h.hooks())
        self.assertEqual(code, 2)
        self.assertEqual(h.launched, [])

    def test_runner_receives_registry_profile_id_not_model_id(self):
        h = Harness(present={"a"})
        self.go(h, ids=("a",))
        self.assertEqual(h.ran[0][1], "profile-a")

    def test_partial_and_not_covered_statuses_are_reported_not_failed(self):
        for status, code in (("PARTIAL", 0), ("NOT_COVERED", 3)):
            with self.subTest(status=status):
                h = Harness(present={"a"}, run_status=status)
                c, report = self.go(h, ids=("a",))
                self.assertEqual(c, code)
                self.assertEqual(report["models"][0]["status"], status)
                self.assertEqual(report["models"][0]["coverage"], "4/4")

    def test_unknown_runner_status_is_fail(self):
        c, report = self.go(Harness(present={"a"}, run_status="TECHNICAL_STOP"), ids=("a",))
        self.assertEqual((c, report["models"][0]["status"]), (1, "FAIL"))

    def test_available_exception_fails_the_row_and_continues(self):
        h = Harness(present={"b"})
        hooks = h.hooks()
        original = hooks["available"]
        def flaky(m, cfg):
            if m.id == "a":
                raise ValueError("bad manifest")
            return original(m, cfg)
        hooks["available"] = flaky
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        code = mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a"), manifest("b")], **hooks)
        report = json.loads((Path(self.tmp.name) / "matrix.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual([r["status"] for r in report["models"]], ["FAIL", "PASS"])

    def test_interrupt_keeps_finished_rows_and_stops_server(self):
        h = Harness(present={"a", "b"})
        hooks = h.hooks()
        calls = []
        def run(url, profile_id, pid, cfg, out):
            calls.append(profile_id)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return {"status": "PASS", "summary": "", "coverage": "4/4"}
        hooks["run"] = run
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with self.assertRaises(KeyboardInterrupt):
            mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a"), manifest("b")], **hooks)
        report = json.loads((Path(self.tmp.name) / "matrix.json").read_text())
        self.assertEqual(report["models"][0]["status"], "PASS")
        self.assertEqual(len(h.stopped), 2)  # the interrupted model's server was stopped too

    def test_failing_runner_exit_1(self):
        h = Harness(present={"a", "b"}, run_status="FAIL")
        code, report = self.go(h)
        self.assertEqual(code, 1)
        self.assertEqual(report["overall"], "FAIL")
        self.assertEqual(len(h.stopped), 2)

    def test_runner_exception_stops_server_and_fails(self):
        h = Harness(present={"a"}, run_error=RuntimeError("boom"))
        code, report = self.go(h, ids=("a",))
        self.assertEqual(code, 1)
        self.assertEqual(len(h.stopped), 1)
        self.assertIn("boom", report["models"][0]["detail"])

    def test_health_timeout_stops_server_and_continues(self):
        h = Harness(present={"a", "b"}, health_error=TimeoutError("slow"))
        code, report = self.go(h)
        self.assertEqual(code, 1)
        self.assertEqual(len(h.stopped), 2)
        self.assertEqual(h.ran, [])
        self.assertTrue(all(r["status"] == "FAIL" for r in report["models"]))

    def test_json_and_text_shape(self):
        h = Harness(present={"a"})
        code, report = self.go(h)
        self.assertEqual(set(report), {"schema_version", "fixtures", "counts", "overall", "models"})
        self.assertEqual(report["schema_version"], 2)
        row = report["models"][0]
        for key in ("model_id", "display_name", "status", "detail", "wall_seconds", "path",
                    "result_path", "coverage"):
            self.assertIn(key, row)
        self.assertEqual(row["path"], "/x/a.gguf")
        self.assertTrue(row["result_path"].endswith("a.qualification.json"))
        text = (Path(self.tmp.name) / "matrix.txt").read_text()
        self.assertIn("PASS=1 PARTIAL=0 NOT_COVERED=0 FAIL=0 SKIPPED=1", text)
        self.assertEqual(h.ran[0], ("http://127.0.0.1:1234", "profile-a", 4242))
        self.assertEqual(len(h.stopped), 1)  # the server is stopped on success too

    def test_server_command_uses_real_flags(self):
        cmd = mq.server_command("m1", 99, mq.MatrixConfig(models_dir=Path("/md")))
        self.assertEqual(cmd[1:], [
            "-m", "orbit.terminal.cli", "server", "--host", "127.0.0.1", "--port", "99",
            "--model-id", "m1", "--ctx", "8192", "--threads", "6", "--threads-batch", "6",
            "--batch", "256", "--ubatch", "128", "--think", "off", "--models-dir", "/md",
        ])
        self.assertNotIn("--models-dir", mq.server_command("m1", 99, mq.MatrixConfig()))

    def test_stop_signals_the_process_group_then_kills(self):
        class Stubborn:
            pid = 77
            def __init__(self): self.waits = 0
            def poll(self): return None
            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise mq.subprocess.TimeoutExpired("x", timeout)
        with mock.patch.object(mq.os, "killpg") as killpg:
            mq.stop_server(Stubborn())
        self.assertEqual([c.args for c in killpg.call_args_list],
                         [(77, signal.SIGTERM), (77, signal.SIGKILL)])

    def test_run_qualification_rejects_a_profile_mismatch(self):
        class Backend:
            def __init__(self, **_): pass
            def backend_props(self):
                return {"model_compatibility": {"compatibility_profile": "orbit-other", "verified": True}}
        with mock.patch("orbit.backend.llama_server.LlamaServerBackend", Backend):
            with self.assertRaisesRegex(RuntimeError, "compatibility_profile"):
                mq.run_qualification("http://x", "orbit-expected", None, mq.MatrixConfig(),
                                     Path(tempfile.gettempdir()) / "unused.json")

    def test_default_availability_skips_without_models(self):
        from orbit.native_llama.model_registry import load_registry
        with tempfile.TemporaryDirectory() as d:
            cfg = mq.MatrixConfig(models_dir=Path(d), hf_cache=Path(d) / "hf")
            for manifest_ in load_registry():
                with self.subTest(model=manifest_.id):
                    self.assertIsNone(mq.model_available(manifest_, cfg))

    def test_default_availability_returns_the_path_when_present(self):
        target = Path("/m/x.gguf")
        resolved = SimpleNamespace(target_path=target)
        with mock.patch("orbit.native_llama.model_registry.resolve_model", return_value=resolved):
            self.assertEqual(mq.model_available(manifest("a"), mq.MatrixConfig()), target)

    # ---- gap tests -------------------------------------------------------

    def run_main(self, argv, ids=("a",), harness=None, **override):
        h = harness or Harness(present=set(ids))
        hooks = h.hooks()
        hooks.update(override)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        code = mq.main(["--output-dir", self.tmp.name, *argv], manifests=[manifest(i) for i in ids], **hooks)
        path = Path(self.tmp.name) / "matrix.json"
        return code, (json.loads(path.read_text()) if path.exists() else None), h

    def test_launch_failure_fails_row_without_stop_and_continues(self):
        h = Harness(present={"a", "b"})
        def launch(model_id, port, cfg, log):
            if model_id == "a":
                raise FileNotFoundError("no python")
            return FakeProcess()
        code, report, _ = self.run_main([], ids=("a", "b"), harness=h, launch=launch)
        self.assertEqual(code, 1)
        self.assertEqual([r["status"] for r in report["models"]], ["FAIL", "PASS"])
        self.assertIn("FileNotFoundError", report["models"][0]["detail"])
        self.assertEqual(len(h.stopped), 1)

    def test_stop_failure_marks_row_fail_and_keeps_detail(self):
        def stop(process):
            raise OSError("cannot kill")
        code, report, _ = self.run_main([], stop=stop)
        row = report["models"][0]
        self.assertEqual((code, row["status"]), (1, "FAIL"))
        self.assertIn("stop failed: cannot kill", row["detail"])

    def test_interrupt_during_health_wait_stops_server_and_writes_report(self):
        h = Harness(present={"a", "b"}, health_error=KeyboardInterrupt())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with self.assertRaises(KeyboardInterrupt):
            mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a"), manifest("b")], **h.hooks())
        self.assertEqual(h.launched, ["a"])
        self.assertEqual(len(h.stopped), 1)
        report = json.loads((Path(self.tmp.name) / "matrix.json").read_text())
        self.assertEqual(report["models"][-1]["detail"], "interrupted")
        self.assertEqual(report["overall"], "FAIL")

    def test_sigterm_handler_is_installed_during_run_and_restored(self):
        before = signal.getsignal(signal.SIGTERM)
        seen = []
        h = Harness(present={"a"})
        hooks = h.hooks()
        def wait(url, process, timeout, log):
            seen.append(signal.getsignal(signal.SIGTERM))
        hooks["wait_health"] = wait
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a")], **hooks)
        self.assertIs(seen[0], mq._raise_interrupt)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        with self.assertRaises(KeyboardInterrupt):
            mq._raise_interrupt(signal.SIGTERM, None)

    def test_sigterm_handler_restored_after_interrupt(self):
        before = signal.getsignal(signal.SIGTERM)
        h = Harness(present={"a"}, run_error=KeyboardInterrupt())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with self.assertRaises(KeyboardInterrupt):
            mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a")], **h.hooks())
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_cli_overrides_reach_launch_config(self):
        captured = []
        h = Harness(present={"a"})
        def launch(model_id, port, cfg, log):
            captured.append(cfg)
            return FakeProcess()
        self.run_main(["--ctx", "4096", "--threads", "2", "--threads-batch", "3", "--batch", "64",
                       "--ubatch", "32", "--startup-timeout", "7", "--request-timeout", "9",
                       "--models-dir", "/md", "--fixtures", str(mq.DEFAULT_FIXTURES)],
                      harness=h, launch=launch)
        cfg = captured[0]
        self.assertEqual((cfg.ctx, cfg.threads, cfg.threads_batch, cfg.batch, cfg.ubatch),
                         (4096, 2, 3, 64, 32))
        self.assertEqual((cfg.startup_timeout, cfg.request_timeout), (7.0, 9.0))
        self.assertEqual(cfg.models_dir, Path("/md"))
        cmd = mq.server_command("a", 1, cfg)
        self.assertEqual(cmd[cmd.index("--ctx") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--models-dir") + 1], "/md")

    def test_nonpositive_numeric_overrides_are_rejected_before_any_launch(self):
        for flag in ("--ctx", "--threads", "--batch", "--ubatch", "--startup-timeout", "--request-timeout"):
            with self.subTest(flag=flag):
                h = Harness(present={"a"})
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as ctx:
                    self.run_main([flag, "0"], harness=h)
                self.assertEqual(ctx.exception.code, 2)
                self.assertEqual(h.launched, [])

    def test_missing_fixture_file_exits_2_before_launching_servers(self):
        h = Harness(present={"a"})
        with mock.patch("sys.stderr"):
            code, report, _ = self.run_main(["--fixtures", "/nonexistent/fixtures.json"], harness=h)
        self.assertEqual(code, 2)
        self.assertEqual(h.launched, [])
        self.assertIsNone(report)

    def test_model_id_filter_dedups_and_preserves_registry_order(self):
        h = Harness(present={"a", "b", "c"})
        code, report, _ = self.run_main(["--model-id", "c", "--model-id", "a", "--model-id", "c"],
                                        ids=("a", "b", "c"), harness=h)
        self.assertEqual(h.launched, ["a", "c"])
        self.assertEqual([r["model_id"] for r in report["models"]], ["a", "c"])

    def test_unknown_model_id_among_valid_ones_exits_2(self):
        h = Harness(present={"a"})
        with mock.patch("sys.stderr"):
            code, _, _ = self.run_main(["--model-id", "a", "--model-id", "zzz"], harness=h)
        self.assertEqual((code, h.launched), (2, []))

    def test_empty_registry_is_not_a_green_run(self):
        code, report, _ = self.run_main([], ids=())
        self.assertEqual((code, report["overall"], report["models"]), (3, "NOT_COVERED", []))
        code, _, _ = self.run_main(["--allow-empty"], ids=())
        self.assertEqual(code, 0)

    def test_status_matrix_exit_codes(self):
        # PARTIAL alone qualifies a model: exit 0; PARTIAL + SKIPPED too.
        for ids, present in (("a", {"a"}), (("a", "b"), {"a"})):
            with self.subTest(ids=ids):
                ids = (ids,) if isinstance(ids, str) else ids
                code, report, _ = self.run_main([], ids=ids, harness=Harness(present=present, run_status="PARTIAL"))
                self.assertEqual((code, report["overall"]), (0, "PASS"))
        # NOT_COVERED + SKIPPED only: exit 3, but allowed with the flag.
        code, report, _ = self.run_main([], ids=("a", "b"), harness=Harness(present={"a"}, run_status="NOT_COVERED"))
        self.assertEqual((code, report["overall"]), (3, "NOT_COVERED"))
        code, _, _ = self.run_main(["--allow-empty"], ids=("a", "b"),
                                   harness=Harness(present={"a"}, run_status="NOT_COVERED"))
        self.assertEqual(code, 0)

    def test_fail_beats_empty_and_allow_empty(self):
        code, report, _ = self.run_main(["--allow-empty"], harness=Harness(present={"a"}, run_status="FAIL"))
        self.assertEqual((code, report["overall"]), (1, "FAIL"))

    def test_fail_row_alongside_pass_is_overall_fail(self):
        h = Harness(present={"a", "b"})
        def run(url, profile_id, pid, cfg, out):
            return {"status": "FAIL" if profile_id == "profile-b" else "PASS", "summary": "", "coverage": "1/1"}
        code, report, _ = self.run_main([], ids=("a", "b"), harness=h, run=run)
        self.assertEqual((code, report["overall"], report["counts"]["PASS"], report["counts"]["FAIL"]),
                         (1, "FAIL", 1, 1))

    def test_result_without_status_or_non_dict_is_fail(self):
        for result in ({}, {"status": None}):
            with self.subTest(result=result):
                code, report, _ = self.run_main([], run=lambda *a, **k: result)
                self.assertEqual((code, report["models"][0]["status"]), (1, "FAIL"))
        code, report, _ = self.run_main([], run=lambda *a, **k: None)
        self.assertEqual((code, report["models"][0]["status"]), (1, "FAIL"))
        self.assertIn("AttributeError", report["models"][0]["detail"])

    def test_skipped_row_has_no_path_or_result_path_and_no_server(self):
        h = Harness()
        _, report, _ = self.run_main([], ids=("a",), harness=h)
        row = report["models"][0]
        self.assertNotIn("result_path", row)
        self.assertNotIn("path", row)
        self.assertEqual((h.launched, h.stopped), ([], []))
        self.assertEqual(row["detail"], "model not available locally")

    def test_report_written_after_each_row(self):
        snapshots = []
        h = Harness(present={"a", "b"})
        hooks = h.hooks()
        orig = hooks["run"]
        def run(*args):
            path = Path(self.tmp.name) / "matrix.json"
            snapshots.append(len(json.loads(path.read_text())["models"]) if path.exists() else 0)
            return orig(*args)
        hooks["run"] = run
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a"), manifest("b")], **hooks)
        self.assertEqual(snapshots, [0, 1])

    def test_format_table_uses_first_detail_line_and_tolerates_empty(self):
        report = mq.build_report(mq.MatrixConfig(), [
            {"model_id": "a", "display_name": "", "status": "FAIL", "detail": "line1\nline2", "wall_seconds": 1.0},
            {"model_id": "b", "display_name": "", "status": "SKIPPED", "detail": "", "wall_seconds": 0.0},
        ])
        text = mq.format_table(report)
        self.assertIn("line1", text)
        self.assertNotIn("line2", text)
        self.assertIn("overall=NOT_COVERED", mq.format_table(mq.build_report(mq.MatrixConfig(), [])))

    def test_model_available_propagates_non_missing_errors_to_a_fail_row(self):
        with mock.patch("orbit.native_llama.model_registry.resolve_model", side_effect=FileNotFoundError("x")):
            self.assertIsNone(mq.model_available(manifest("a"), mq.MatrixConfig()))
        with mock.patch("orbit.native_llama.model_registry.resolve_model", side_effect=ValueError("corrupt")):
            with self.assertRaises(ValueError):
                mq.model_available(manifest("a"), mq.MatrixConfig())
            row = mq.qualify_model(manifest("a"), mq.MatrixConfig(), available=mq.model_available,
                                   launch=None, wait_health=None, run=None, stop=None)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("ValueError", row["detail"])

    def test_classify_run_status_mapping(self):
        from orbit.qualification.schema import Status
        def run(overall, flags):
            return SimpleNamespace(overall_status=overall, fixtures=[SimpleNamespace(applicable=f) for f in flags])
        self.assertEqual(mq.classify_run(run(Status.PASS, [True, True])), ("PASS", "2/2"))
        self.assertEqual(mq.classify_run(run(Status.PASS, [True, False])), ("PARTIAL", "1/2"))
        self.assertEqual(mq.classify_run(run(Status.PASS, [False, False])), ("NOT_COVERED", "0/2"))
        self.assertEqual(mq.classify_run(run(Status.PASS, [])), ("NOT_COVERED", "0/0"))
        self.assertEqual(mq.classify_run(run(Status.NOT_APPLICABLE, [False])), ("NOT_COVERED", "0/1"))
        self.assertEqual(mq.classify_run(run(Status.FAIL, [True])), ("FAIL", "1/1"))
        self.assertEqual(mq.classify_run(run(Status.TECHNICAL_STOP, [True])), ("FAIL", "1/1"))

    def test_wait_for_health_detects_dead_server_and_times_out(self):
        class Backend:
            def __init__(self, **_): pass
            def health(self): return False
        log = Path(tempfile.gettempdir()) / "orbit-matrix-test-health.log"
        log.write_text("boom: bad gguf", encoding="utf-8")
        self.addCleanup(lambda: log.unlink(missing_ok=True))
        dead = SimpleNamespace(poll=lambda: 1)
        alive = SimpleNamespace(poll=lambda: None)
        with mock.patch("orbit.backend.llama_server.LlamaServerBackend", Backend):
            with self.assertRaisesRegex(RuntimeError, "exited during startup: boom"):
                mq.wait_for_health("http://x", dead, 5, log)
            with mock.patch.object(mq.time, "sleep"):
                with self.assertRaises(TimeoutError):
                    mq.wait_for_health("http://x", alive, 0.01, log)
        class Ready(Backend):
            def health(self): return True
        with mock.patch("orbit.backend.llama_server.LlamaServerBackend", Ready):
            mq.wait_for_health("http://x", alive, 5, log)

    def test_launch_server_isolates_session_and_disables_prewarm(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(mq.subprocess, "Popen") as popen:
            mq.launch_server("a", 99, mq.MatrixConfig(), Path(d) / "s.log")
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["env"]["ORBIT_KV_PREFIX_PREWARM"], "off")
        self.assertEqual(kwargs["stderr"], mq.subprocess.STDOUT)

    def test_stop_server_noop_when_exited_and_falls_back_when_killpg_fails(self):
        with mock.patch.object(mq.os, "killpg") as killpg:
            mq.stop_server(SimpleNamespace(poll=lambda: 0))
        killpg.assert_not_called()
        proc = mock.Mock(pid=5)
        proc.poll.return_value = None
        with mock.patch.object(mq.os, "killpg", side_effect=ProcessLookupError):
            mq.stop_server(proc)
        proc.terminate.assert_called_once()

    def test_log_file_is_removed_even_on_failure(self):
        logs = []
        h = Harness(present={"a"}, health_error=RuntimeError("x"))
        hooks = h.hooks()
        orig = hooks["launch"]
        def launch(model_id, port, cfg, log):
            logs.append(log)
            return orig(model_id, port, cfg, log)
        hooks["launch"] = launch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        mq.main(["--output-dir", self.tmp.name], manifests=[manifest("a")], **hooks)
        self.assertFalse(logs[0].exists())


class FakeSpan:
    def __init__(self, name, attrs):
        self.name, self.attrs = name, dict(attrs)

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_attributes(self, attrs):
        self.attrs.update(attrs)


class TelemetryRecorder:
    """Stand-in for orbit.telemetry.span/count that records instead of exporting."""

    def __init__(self):
        self.spans, self.counts = [], []

    def span(self, name, **attrs):
        import contextlib

        @contextlib.contextmanager
        def cm():
            sp = FakeSpan(name, attrs)
            self.spans.append(sp)
            yield sp
        return cm()

    def count(self, name, value=1, **attrs):
        self.counts.append((name, attrs))


class MatrixTelemetryTests(unittest.TestCase):
    def run_driver(self, ids=("a", "b"), harness=None):
        h = harness or Harness(present=set(ids))
        with tempfile.TemporaryDirectory() as tmp:
            code = mq.main(["--output-dir", tmp], manifests=[manifest(i) for i in ids], **h.hooks())
            out = ((Path(tmp) / "matrix.json").read_text(), (Path(tmp) / "matrix.txt").read_text())
        return code, out

    def test_disabled_telemetry_changes_neither_exit_code_nor_reports(self):
        import os
        with mock.patch.dict(os.environ, {"ORBIT_OTEL": "0"}):
            plain = self.run_driver(ids=("a", "b"), harness=Harness(present={"a"}))
        rec = TelemetryRecorder()
        with mock.patch.object(mq.telemetry, "span", rec.span), mock.patch.object(mq.telemetry, "count", rec.count):
            traced = self.run_driver(ids=("a", "b"), harness=Harness(present={"a"}))
        import re
        norm = lambda t: re.sub(r"/tmp/tmp\w+", "TMP", re.sub(r'"wall_seconds": [0-9.]+', '"wall_seconds": 0', t))
        self.assertEqual(plain[0], traced[0])
        self.assertEqual(norm(plain[1][0]), norm(traced[1][0]))
        self.assertEqual([l.split()[:2] for l in plain[1][1].splitlines()[1:-1]],
                         [l.split()[:2] for l in traced[1][1].splitlines()[1:-1]])

    def test_disabled_span_is_the_noop(self):
        import os
        with mock.patch.dict(os.environ, {"ORBIT_OTEL": "0"}):
            self.assertFalse(mq.telemetry.enabled())
            with mq.telemetry.span("x", **{"orbit.model": "a"}) as sp:
                sp.set_attribute("orbit.status", "PASS")

    def test_one_span_per_model_row_with_structural_attributes_only(self):
        rec = TelemetryRecorder()
        with mock.patch.object(mq.telemetry, "span", rec.span), mock.patch.object(mq.telemetry, "count", rec.count):
            code, _ = self.run_driver(ids=("a", "b"), harness=Harness(present={"a"}))
        rows = [s for s in rec.spans if s.name == "orbit.matrix.model"]
        self.assertEqual(code, 0)
        self.assertEqual([s.attrs["orbit.model"] for s in rows], ["a", "b"])
        self.assertEqual([s.attrs["orbit.status"] for s in rows], ["PASS", "SKIPPED"])
        for s in rows:
            self.assertLessEqual(set(s.attrs), mq.telemetry.ALLOWED_ATTRIBUTES)
            self.assertNotIn("detail", " ".join(s.attrs))
        self.assertEqual(len(rec.counts), 2)

    def test_failing_row_still_exits_1_with_telemetry_attached(self):
        rec = TelemetryRecorder()
        with mock.patch.object(mq.telemetry, "span", rec.span), mock.patch.object(mq.telemetry, "count", rec.count):
            code, _ = self.run_driver(ids=("a",), harness=Harness(present={"a"}, run_error=RuntimeError("secret /path")))
        self.assertEqual(code, 1)
        span = rec.spans[0]
        self.assertEqual(span.attrs["orbit.status"], "FAIL")
        self.assertNotIn("secret", repr(span.attrs))

    def test_traced_executor_wraps_each_fixture_and_passes_through(self):
        rec = TelemetryRecorder()

        class Inner:
            def execute(self, fixture, workdir):
                if fixture.name == "boom":
                    raise ValueError("x")
                return ("obs", fixture.name)

        ex = mq._TracedExecutor(Inner(), "profile-a")
        with mock.patch.object(mq.telemetry, "span", rec.span):
            self.assertEqual(ex.execute(SimpleNamespace(name="f1"), Path("/w")), ("obs", "f1"))
            with self.assertRaises(ValueError):
                ex.execute(SimpleNamespace(name="boom"), Path("/w"))
        self.assertEqual([s.name for s in rec.spans], ["orbit.matrix.fixture"] * 2)
        self.assertEqual(rec.spans[0].attrs["orbit.matrix.case"], "f1")
        self.assertEqual(rec.spans[0].attrs["orbit.matrix.profile"], "profile-a")
        self.assertEqual(rec.spans[0].attrs["orbit.outcome"], "completed")
        self.assertNotIn("orbit.outcome", rec.spans[1].attrs)

    def test_real_span_filters_unsafe_values_when_disabled_or_enabled(self):
        self.assertEqual(mq.telemetry.safe_attributes({"orbit.matrix.case": "/etc/passwd", "orbit.matrix.profile": "p-1"}),
                         {"orbit.matrix.profile": "p-1"})


if __name__ == "__main__":
    unittest.main()
