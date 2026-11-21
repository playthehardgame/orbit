"""The server_start manifest and the log file's permissions are audit evidence."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from orbit.native_server.request_logging import RequestLogger, run_manifest


class RunManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "m.gguf"
        self.model.write_bytes(b"GGUF" + b"x" * 1000)

    def test_records_model_size_build_and_orbit_env_only(self):
        env = {"ORBIT_CPU_REPACK": "off", "HOME": "/h", "PATH": "/bin"}
        m = run_manifest(["--port", "1"], model_path=self.model, alias="a", environ=env)
        self.assertEqual(m["argv"], ["--port", "1"])
        self.assertEqual(m["model"]["size_bytes"], 1004)
        self.assertNotIn("sha256", m["model"])
        self.assertEqual(m["orbit_env"], {"ORBIT_CPU_REPACK": "off"})
        for key in ("python", "platform", "cpu_count", "build"):
            self.assertIn(key, m)

    def test_secret_looking_variables_are_redacted(self):
        env = {"ORBIT_API_TOKEN": "s3cr3t", "ORBIT_HF_KEY": "k", "ORBIT_MODE": "x"}
        m = run_manifest([], model_path=None, alias=None, environ=env)
        self.assertEqual(m["orbit_env"]["ORBIT_API_TOKEN"], "[redacted]")
        self.assertEqual(m["orbit_env"]["ORBIT_HF_KEY"], "[redacted]")
        self.assertEqual(m["orbit_env"]["ORBIT_MODE"], "x")
        self.assertNotIn("s3cr3t", json.dumps(m))

    def test_model_hash_is_opt_in_and_correct(self):
        m = run_manifest([], model_path=self.model, alias=None,
                         environ={"ORBIT_AUDIT_HASH_MODEL": "1"})
        self.assertEqual(m["model"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())

    def test_missing_model_is_reported_not_raised(self):
        m = run_manifest([], model_path=Path(self.tmp.name) / "none.gguf", alias=None, environ={})
        self.assertEqual(m["model"]["error"], "FileNotFoundError")

    def test_log_file_and_directory_are_owner_only(self):
        directory = Path(self.tmp.name) / "logs"
        logger = RequestLogger(directory)
        self.addCleanup(logger.close)
        logger.write("server_start", **run_manifest([], model_path=None, alias=None, environ={}))
        self.assertEqual(stat.S_IMODE(os.stat(directory).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(logger.path).st_mode), 0o600)
        row = json.loads(logger.path.read_text().splitlines()[0])
        self.assertEqual(row["event"], "server_start")


if __name__ == "__main__":
    unittest.main()
