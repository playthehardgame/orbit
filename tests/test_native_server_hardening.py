"""Request-size validation and clean termination of the native server."""
from __future__ import annotations

import os
import signal
import threading
import time
import unittest
from unittest import mock

from orbit.native_server import app


class ContentLengthTests(unittest.TestCase):
    def test_missing_or_blank_is_empty(self):
        self.assertEqual(app.parse_content_length(None), 0)
        self.assertEqual(app.parse_content_length("  "), 0)

    def test_valid_value_within_limit(self):
        self.assertEqual(app.parse_content_length("1234", limit=2000), 1234)

    def test_malformed_is_a_value_error_not_a_crash(self):
        for raw in ("abc", "1.5", "-1", "0x10", "1e3"):
            with self.assertRaises(ValueError, msg=raw):
                app.parse_content_length(raw, limit=100)

    def test_over_limit_is_refused_before_reading(self):
        with self.assertRaises(ValueError) as ctx:
            app.parse_content_length("101", limit=100)
        self.assertIn("exceeds", str(ctx.exception))
        self.assertEqual(app.parse_content_length("100", limit=100), 100)

    def test_environment_override_and_fallback(self):
        self.assertEqual(app.max_request_body_bytes({}), app.DEFAULT_MAX_REQUEST_BODY_BYTES)
        self.assertEqual(app.max_request_body_bytes({app.MAX_REQUEST_BODY_ENV: "4096"}), 4096)
        for bad in ("0", "-5", "many", ""):
            self.assertEqual(app.max_request_body_bytes({app.MAX_REQUEST_BODY_ENV: bad}),
                             app.DEFAULT_MAX_REQUEST_BODY_BYTES, bad)


class _BlockingServer:
    def serve_forever(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            time.sleep(0.01)


@unittest.skipUnless(hasattr(signal, "SIGTERM"), "POSIX signals required")
class TerminationTests(unittest.TestCase):
    def setUp(self):
        # Never sweep the developer's real temporary directory from a test.
        patcher = mock.patch.object(app, "_sweep_stale_workspaces")
        self.sweep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_sigterm_stops_serving_and_restores_handlers(self):
        before = signal.getsignal(signal.SIGTERM)
        timer = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(KeyboardInterrupt):
                app.serve_until_terminated(_BlockingServer())
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        self.sweep.assert_called_once_with()

    def test_off_main_thread_installs_nothing(self):
        before = signal.getsignal(signal.SIGTERM)
        seen = {}

        class Quick:
            def serve_forever(self):
                seen["during"] = signal.getsignal(signal.SIGTERM)

        worker = threading.Thread(target=app.serve_until_terminated, args=(Quick(),))
        worker.start()
        worker.join(5)
        self.assertEqual(seen["during"], before)



class RequestGateTests(unittest.TestCase):
    def test_refuses_past_the_limit_and_recovers(self):
        gate = app.RequestGate(limit=2)
        self.assertTrue(gate.try_enter())
        self.assertTrue(gate.try_enter())
        self.assertFalse(gate.try_enter())
        gate.leave()
        self.assertTrue(gate.try_enter())
        self.assertEqual(gate.in_flight, 2)

    def test_leave_never_goes_negative(self):
        gate = app.RequestGate(limit=1)
        gate.leave()
        self.assertEqual(gate.in_flight, 0)

    def test_limit_environment_override_and_fallback(self):
        self.assertEqual(app.max_inflight_requests({}), app.DEFAULT_MAX_INFLIGHT_REQUESTS)
        self.assertEqual(app.max_inflight_requests({app.MAX_INFLIGHT_ENV: "3"}), 3)
        for bad in ("0", "-1", "x", ""):
            self.assertEqual(app.max_inflight_requests({app.MAX_INFLIGHT_ENV: bad}),
                             app.DEFAULT_MAX_INFLIGHT_REQUESTS, bad)

    def test_http_503_when_saturated_and_health_stays_up(self):
        import http.client
        from http.server import ThreadingHTTPServer
        saturated = app.RequestGate(limit=1)
        saturated.try_enter()
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.OrbitNativeHandler)
        server.request_logger = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.object(app, "REQUEST_GATE", saturated):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                conn.request("POST", "/v1/chat/completions", body="{}",
                             headers={"Content-Type": "application/json"})
                self.assertEqual(conn.getresponse().status, 503)
                conn.close()
                conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                conn.request("GET", "/health")
                self.assertEqual(conn.getresponse().status, 200)
                conn.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
