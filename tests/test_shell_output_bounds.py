"""A runaway shell command must not be able to exhaust host memory."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime import shell_guardrails as sg


def _run(command: str, timeout: int = 10):
    with tempfile.TemporaryDirectory() as tmp:
        return sg._run_shell_command(command, cwd=Path(tmp), env=dict(os.environ), timeout=timeout)


class ShellOutputBoundsTests(unittest.TestCase):
    def test_small_output_is_returned_unchanged(self):
        done = _run("printf 'out'; printf 'err' >&2")
        self.assertEqual((done.stdout, done.stderr, done.returncode), ("out", "err", 0))

    def test_output_over_the_cap_is_truncated_not_buffered(self):
        with mock.patch.object(sg, "SHELL_STREAM_CAPTURE_CHARS", 1000):
            done = _run(f"{sys.executable} -c \"print('x' * 200000)\"")
        self.assertEqual(done.returncode, 0)
        self.assertTrue(done.stdout.startswith("x" * 1000))
        self.assertIn("output truncated", done.stdout)
        self.assertLess(len(done.stdout), 1200)

    def test_stderr_is_bounded_independently(self):
        with mock.patch.object(sg, "SHELL_STREAM_CAPTURE_CHARS", 500):
            done = _run(f"{sys.executable} -c \"import sys; sys.stderr.write('e' * 100000)\"")
        self.assertIn("output truncated", done.stderr)
        self.assertLess(len(done.stderr), 700)
        self.assertEqual(done.stdout, "")

    def test_endless_producer_is_killed_at_the_timeout(self):
        with mock.patch.object(sg, "SHELL_STREAM_CAPTURE_CHARS", 4096):
            with self.assertRaises(subprocess.TimeoutExpired) as ctx:
                _run("yes", timeout=1)
        self.assertLess(len(ctx.exception.stdout), 5000)

    def test_invalid_utf8_does_not_raise(self):
        done = _run("printf '\\377\\376ok'")
        self.assertIn("ok", done.stdout)


if __name__ == "__main__":
    unittest.main()
