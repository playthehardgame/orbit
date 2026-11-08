from __future__ import annotations

import os
import unittest
from unittest import mock

from orbit import telemetry

try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAVE_SDK = True
except Exception:  # pragma: no cover
    HAVE_SDK = False


class SafeAttributesTests(unittest.TestCase):
    def test_allowlist_and_value_shape(self) -> None:
        out = telemetry.safe_attributes(
            {
                "orbit.tokens.prompt": 12,
                "orbit.status": "ok",
                "prompt": "secret prompt",  # key not allowlisted
                "orbit.model": "/home/user/samples/evil.exe",  # path-shaped
                "orbit.outcome": "some free text with spaces",
                "orbit.tool": "x" * 200,  # size cap
                "orbit.input_sha256": "a" * 64,
                "orbit.count": {"nested": "dict"},
                "orbit.bytes": None,
            }
        )
        self.assertEqual(out, {"orbit.tokens.prompt": 12, "orbit.status": "ok", "orbit.input_sha256": "a" * 64})

    def test_no_allowed_key_names_content(self) -> None:
        for key in telemetry.ALLOWED_ATTRIBUTES:
            for bad in ("prompt_text", "content", "output", "path", "env", "secret", "filename", "filepath"):
                self.assertNotIn(bad, key.split(".")[-1])


class NoopTests(unittest.TestCase):
    def test_disabled_is_noop(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORBIT_OTEL", None)
            self.assertFalse(telemetry.enabled())
            with telemetry.span("x", **{"orbit.status": "ok"}) as sp:
                sp.set_attribute("orbit.count", 1)
            telemetry.count("c")
            telemetry.record("h", 1.0)

            @telemetry.traced("t")
            def f(a: int) -> int:
                return a + 1

            self.assertEqual(f(1), 2)

    def test_enabled_without_sdk_state_is_safe(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_OTEL": "1"}):
            telemetry._install_providers_for_tests(None, None)
            with telemetry.span("x") as sp:
                sp.set_attribute("orbit.count", 1)
            telemetry.count("c")
        telemetry._reset_for_tests()

    def test_exception_propagates(self) -> None:
        with mock.patch.dict(os.environ, {"ORBIT_OTEL": "1"}):
            telemetry._install_providers_for_tests(None, None)
            with self.assertRaises(ValueError):
                with telemetry.span("x"):
                    raise ValueError("boom")
        telemetry._reset_for_tests()


@unittest.skipUnless(HAVE_SDK, "opentelemetry-sdk not installed")
class SdkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = InMemorySpanExporter()
        tp = TracerProvider()
        tp.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        mp = MeterProvider(metric_readers=[self.reader])
        patcher = mock.patch.dict(os.environ, {"ORBIT_OTEL": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        telemetry._install_providers_for_tests(tp, mp)
        self.addCleanup(telemetry._reset_for_tests)

    def test_span_attributes_filtered_and_error_type_only(self) -> None:
        with self.assertRaises(RuntimeError):
            with telemetry.span("s", **{"orbit.status": "ok", "orbit.model": "/etc/passwd"}) as sp:
                sp.set_attribute("leak", "data")
                raise RuntimeError("secret message")
        (finished,) = self.exporter.get_finished_spans()
        attrs = dict(finished.attributes)
        self.assertEqual(attrs, {"orbit.status": "ok", "orbit.error_type": "RuntimeError"})
        self.assertNotIn("secret message", repr(finished.events) + repr(finished.status))

    def test_metrics_recorded(self) -> None:
        telemetry.count("orbit.test.count", 3, **{"orbit.status": "ok", "bad": "x"})
        data = self.reader.get_metrics_data()
        points = data.resource_metrics[0].scope_metrics[0].metrics[0].data.data_points
        self.assertEqual(points[0].value, 3)
        self.assertEqual(dict(points[0].attributes), {"orbit.status": "ok"})


if __name__ == "__main__":
    unittest.main()
