from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import ChatResult
from orbit.qualification.fixtures import load_fixture_text
from orbit.qualification.reporting import result_json
from orbit.qualification.runner import QualificationRunner, RuntimeFixtureExecutor
from orbit.qualification.schema import CallMetric, FixtureObservation, LifecycleOutcome, RunProvenance, Status
from orbit.qualification.validators import validate_observation

PROFILE = {"compatibility_profile": "profile-a", "verified": True,
           "capabilities": {"chat": True, "tools": True}}


def fixture_set(route: str = "FILESYSTEM"):
    payload = {"schema_version": 1, "fixtures": [{
        "name": "route_fx", "capability": "tools", "profiles": ["profile-a"],
        "request": {"prompt": "list", "tools": True},
        "expect": {"route": route, "tool_calls": [{"name": "exec_shell_full_command", "arguments": {"command": "pwd"}}],
                   "finish_reason": "stop", "max_model_calls": 3},
        "parity": {"mode": "structural"}}]}
    return load_fixture_text(json.dumps(payload))


def provenance(fixture_hash: str) -> RunProvenance:
    from tests.test_qualification_core import provenance as core_provenance
    return core_provenance(fixture_hash)


class Boom:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, fixture, workdir):
        raise self.error


class FakeBackend:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, messages, **_kwargs):
        return ChatResult(content=self.content, model=None, finish_reason="stop", tool_calls=[],
                          prompt_tokens=10, completion_tokens=3, cached_tokens=0,
                          prompt_tokens_per_second=1.0, generation_tokens_per_second=1.0)


def run_with(executor, fixtures):
    with tempfile.TemporaryDirectory() as directory:
        return QualificationRunner(fixtures, PROFILE, provenance(fixtures.content_hash), executor,
                                   Path(directory)).run(None)


class ErrorDetailTests(unittest.TestCase):
    def test_message_and_traceback_captured_and_bounded(self) -> None:
        fixtures = fixture_set()
        run = run_with(Boom(RuntimeError("bad\nthing\x00 " + "x" * 5000)), fixtures)
        item = run.fixtures[0]
        self.assertEqual(item.status, Status.TECHNICAL_STOP)
        self.assertEqual(item.reason.code, "execution_error")
        self.assertEqual(item.reason.detail, "fixture execution failed (RuntimeError)")
        detail = item.error_detail
        self.assertLessEqual(len(detail["message"]), 500)
        self.assertTrue(detail["message"].startswith("RuntimeError: bad thing"))
        self.assertNotIn("\n", detail["message"])
        self.assertNotIn("\x00", detail["message"])
        self.assertLessEqual(len(detail["traceback_tail"]), 2000)
        self.assertIn("RuntimeError", detail["traceback_tail"])
        self.assertIn("error_detail", json.loads(result_json(run))["fixtures"][0])

    def test_fixture_hash_and_pass_results_unchanged(self) -> None:
        fixtures = fixture_set()
        before = fixtures.content_hash
        run_with(Boom(ValueError("x")), fixtures)
        self.assertEqual(before, fixture_set().content_hash)


class RouteExcerptTests(unittest.TestCase):
    def execute(self, content: str):
        fixtures = fixture_set("CHAT")  # backend answers a filesystem-ish text -> mismatch or parse failure
        executor = RuntimeFixtureExecutor(FakeBackend(content))
        with tempfile.TemporaryDirectory() as directory:
            observation = executor.execute(fixtures.fixtures[0], Path(directory))
        return fixtures.fixtures[0], observation

    def test_excerpt_on_failing_route_only_and_bounded(self) -> None:
        fx, observation = self.execute("not a route " + "y" * 5000)
        self.assertIsNotNone(observation.route_output)
        failing = validate_observation(fx, observation)
        self.assertNotEqual(failing.status, Status.PASS)
        route_calls = [c for c in failing.calls if c.phase == "route"]
        self.assertEqual(len(route_calls), 1)
        self.assertLessEqual(len(route_calls[0].output_excerpt), 2000)
        self.assertTrue(route_calls[0].output_excerpt.startswith("not a route"))

    def test_no_excerpt_when_passing_or_not_route_phase(self) -> None:
        from tests.test_qualification_core import fixture as core_fixture, observation as core_observation
        result = validate_observation(core_fixture(), core_observation(route_output="secret-ish text"))
        self.assertEqual(result.status, Status.PASS)
        for call in result.calls:
            self.assertIsNone(call.output_excerpt)
        # failing but with a non-route phase only: nothing to attach to
        other = FixtureObservation(
            route="CHAT", tool_calls=(), executed_tools=(), final_output="", finish_reason="stop",
            model_call_count=1, retry_count=0,
            calls=(CallMetric("final", 1, 1, 0, 1, 1.0, 1.0, 1.0, "stop"),),
            artifact=None, lifecycle=LifecycleOutcome(True, "clean"), peak_rss_bytes=None,
            route_output="text")
        failed = validate_observation(fixture_set().fixtures[0], other)
        self.assertNotEqual(failed.status, Status.PASS)
        self.assertTrue(all(c.output_excerpt is None for c in failed.calls))


def full_request_fixture():
    payload = {"schema_version": 1, "fixtures": [{
        "name": "full_fx", "capability": "tools", "profiles": ["profile-a"],
        "request": {"prompt": "hello there", "tools": True, "full_request": True},
        "expect": {"route": "CHAT", "tool_calls": [], "finish_reason": "stop", "max_model_calls": 4},
        "parity": {"mode": "structural"}}]}
    return load_fixture_text(json.dumps(payload)).fixtures[0]


def artifact_fixture():
    payload = {"schema_version": 1, "fixtures": [{
        "name": "art_fx", "capability": "artifacts", "profiles": ["profile-a"],
        "request": {"prompt": "write it", "tools": True},
        "expect": {"tool_calls": [{"name": "write_artifact"}, {"name": "verify_artifact"}],
                   "finish_reason": "stop", "max_model_calls": 5,
                   "artifact": {"path": "qualification.json", "json_equals": {"a": 1},
                                "publication": True, "verification": True}},
        "parity": {"mode": "structural"}}]}
    return load_fixture_text(json.dumps(payload)).fixtures[0]


class ScriptedBackend:
    """Fake backend answering every call from a fixed script; records call phases."""

    def __init__(self, contents, tool_calls=None) -> None:
        self.contents = list(contents)
        self.tool_calls = tool_calls or []
        self.thinking = False
        self.phases = []

    def chat(self, messages, **kwargs):
        from orbit.runtime.kv_diag import current_phase
        self.phases.append(current_phase())
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return ChatResult(content=content, model=None, finish_reason="stop", tool_calls=list(self.tool_calls),
                          prompt_tokens=10, completion_tokens=3, cached_tokens=0,
                          prompt_tokens_per_second=1.0, generation_tokens_per_second=1.0)


class AskAutoRouteCaptureTests(unittest.TestCase):
    def execute(self, fx, backend):
        with tempfile.TemporaryDirectory() as directory:
            return RuntimeFixtureExecutor(backend).execute(fx, Path(directory))

    def test_failing_full_request_fixture_gets_bounded_excerpt(self) -> None:
        fx = full_request_fixture()
        raw = "garbled \x00route " + "z" * 5000
        observation = self.execute(fx, ScriptedBackend([raw]))
        self.assertIsNotNone(observation.route_output)
        result = validate_observation(fx, observation)
        self.assertNotEqual(result.status, Status.PASS)
        routes = [c for c in result.calls if c.phase == "route"]
        self.assertEqual(len(routes), 1)
        self.assertLessEqual(len(routes[0].output_excerpt), 2000)
        self.assertTrue(routes[0].output_excerpt.startswith("garbled route"))
        self.assertNotIn("\x00", routes[0].output_excerpt)
        self.assertTrue(all(c.output_excerpt is None for c in result.calls if c.phase != "route"))

    def test_failing_artifact_fixture_gets_excerpt_from_tool_call_json(self) -> None:
        fx = artifact_fixture()
        observation = self.execute(fx, ScriptedBackend(["nothing useful"]))
        self.assertTrue(observation.route_output)
        result = validate_observation(fx, observation)
        self.assertNotEqual(result.status, Status.PASS)
        self.assertTrue(any(c.phase == "route" and c.output_excerpt for c in result.calls))

    def test_passing_observation_has_no_excerpt(self) -> None:
        from tests.test_qualification_core import fixture as core_fixture, observation as core_observation
        result = validate_observation(core_fixture(), core_observation(route_output="x" * 50))
        self.assertEqual(result.status, Status.PASS)
        self.assertTrue(all(c.output_excerpt is None for c in result.calls))

    def test_ask_auto_result_unchanged_with_and_without_capture(self) -> None:
        from orbit.runtime.chat import ChatRuntime
        from orbit.qualification.runner import _RouteCaptureBackend
        outcomes = []
        for wrap in (False, True):
            backend = ScriptedBackend(["not a route", "final answer"])
            runtime = ChatRuntime(backend=_RouteCaptureBackend(backend) if wrap else backend)
            with tempfile.TemporaryDirectory() as directory:
                result = runtime.ask_auto("hello there", temperature=0, max_tokens=64, workdir=Path(directory),
                                          max_loops=4, allowed_tool_names=())
            outcomes.append((result.content, result.finish_reason, result.tool_calls,
                             result.prompt_tokens, result.completion_tokens, backend.phases))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][5][0], "route")

    def test_proxy_forwards_attribute_writes_and_first_route_only(self) -> None:
        from orbit.qualification.runner import _RouteCaptureBackend
        from orbit.runtime.kv_diag import model_call_context
        backend = ScriptedBackend(["first", "second"])
        proxy = _RouteCaptureBackend(backend)
        proxy.thinking = True
        self.assertTrue(backend.thinking)
        with model_call_context(phase="chat_final"):
            proxy.chat([], temperature=0, max_tokens=1)
        self.assertIsNone(proxy.route_output)
        with model_call_context(phase="route"):
            proxy.chat([], temperature=0, max_tokens=1)
            proxy.chat([], temperature=0, max_tokens=1)
        self.assertEqual(proxy.route_output, "second")


if __name__ == "__main__":
    unittest.main()
