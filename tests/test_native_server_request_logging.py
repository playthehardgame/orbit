from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server import app as app_module  # noqa: E402
from orbit.native_server.request_logging import LOG_FILENAME, RequestLogger  # noqa: E402


class RequestLoggingTests(unittest.TestCase):
    def test_log_argument_is_optional_and_keeps_path_as_a_directory(self) -> None:
        parser = app_module.build_parser()

        self.assertIsNone(parser.parse_args([]).log)
        log_dir = pathlib.Path("/tmp/orbit-request-log")
        self.assertEqual(parser.parse_args(["--log", str(log_dir)]).log, log_dir)

    def test_logger_creates_directory_and_writes_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = pathlib.Path(temp) / "nested" / "logs"
            logger = RequestLogger(directory)
            try:
                logger.write(
                    "request",
                    method="POST",
                    path="/chat",
                    payload={"messages": [{"role": "user", "content": "hello"}]},
                )
            finally:
                logger.close()

            rows = [
                json.loads(line)
                for line in (directory / LOG_FILENAME).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "request")
        self.assertEqual(rows[0]["method"], "POST")
        self.assertEqual(rows[0]["path"], "/chat")
        self.assertEqual(rows[0]["payload"]["messages"][0]["content"], "hello")
        self.assertIsInstance(rows[0]["timestamp"], float)

    def test_handler_logs_received_parameters_and_response_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            logger = RequestLogger(pathlib.Path(temp))
            handler = object.__new__(app_module.OrbitNativeHandler)
            handler.server = types.SimpleNamespace(request_logger=logger)
            handler.path = "/chat"
            handler.command = "POST"
            handler.client_address = ("127.0.0.1", 1234)
            handler._request_started = 0.0
            try:
                handler._log_request(
                    "POST",
                    payload={"max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]},
                )
                handler._log_response(200, handler._request_started)
            finally:
                logger.close()

            rows = [
                json.loads(line)
                for line in (pathlib.Path(temp) / LOG_FILENAME).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rows[0]["event"], "request")
        self.assertEqual(rows[0]["payload"]["max_tokens"], 32)
        self.assertEqual(rows[1]["event"], "response")
        self.assertEqual(rows[1]["status"], 200)
        self.assertGreaterEqual(rows[1]["duration_ms"], 0)

    def test_log_path_must_not_be_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "not-a-directory"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises((OSError, ValueError)):
                RequestLogger(path)


if __name__ == "__main__":
    unittest.main()
