"""Rotation, redaction, permissions and tamper-evidence of the request log."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from orbit.native_server.request_logging import (
    LOG_FILENAME, RequestLogger, chain_files, scrub_secrets, verify_chain,
)


class RequestLogFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def rows(self):
        return [json.loads(l) for l in (self.dir / LOG_FILENAME).read_text().splitlines()]

    def test_secrets_are_scrubbed_in_every_mode(self):
        self.assertNotIn("hf_", scrub_secrets("token hf_" + "a" * 30))
        self.assertIn("Bearer [redacted-secret]", scrub_secrets("Authorization: Bearer abcdefghijklmnop12345"))
        logger = RequestLogger(self.dir, environ={})
        logger.write("request", payload={"messages": [{"content": "key sk-" + "x" * 30}]})
        logger.close()
        self.assertNotIn("sk-xxxx", (self.dir / LOG_FILENAME).read_text())

    def test_redact_mode_keeps_length_and_hash_not_text(self):
        logger = RequestLogger(self.dir, environ={"ORBIT_LOG_CONTENT": "redact"})
        logger.write("request", payload={"messages": [{"role": "user", "content": "secret plan"}]})
        logger.close()
        msg = self.rows()[0]["payload"]["messages"][0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"]["chars"], 11)
        self.assertNotIn("secret plan", (self.dir / LOG_FILENAME).read_text())

    def test_rotation_keeps_private_files_and_a_valid_chain(self):
        logger = RequestLogger(self.dir, environ={"ORBIT_LOG_MAX_BYTES": "600", "ORBIT_LOG_BACKUPS": "50"})
        for i in range(30):
            logger.write("request", n=i, pad="x" * 80)
        logger.close()
        files = chain_files(self.dir)
        self.assertGreater(len(files), 2)
        for path in files:
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        ok, reason = verify_chain(files)
        self.assertTrue(ok, reason)

    def test_chain_survives_restart_and_detects_tampering(self):
        for _ in range(2):
            logger = RequestLogger(self.dir, environ={})
            logger.write("a"); logger.write("b")
            logger.close()
        self.assertTrue(verify_chain(chain_files(self.dir))[0])
        path = self.dir / LOG_FILENAME
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")   # remove a row
        self.assertFalse(verify_chain([path])[0])
        edited = [json.loads(l) for l in lines]
        edited[1]["event"] = "forged"
        path.write_text("\n".join(json.dumps(r) for r in edited) + "\n")
        self.assertIn("altered", verify_chain([path])[1])

    def test_rotation_can_be_disabled(self):
        logger = RequestLogger(self.dir, environ={"ORBIT_LOG_BACKUPS": "0", "ORBIT_LOG_MAX_BYTES": "100"})
        for i in range(10):
            logger.write("x", pad="y" * 100)
        logger.close()
        self.assertEqual(len(chain_files(self.dir)), 1)


if __name__ == "__main__":
    unittest.main()
