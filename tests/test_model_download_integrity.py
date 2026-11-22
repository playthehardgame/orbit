"""Registry pins are enforced when a model is downloaded."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from orbit.native_llama.model_download import IntegrityError, verify_pinned_download
from orbit.native_llama.model_registry import registry_pins


class VerifyPinnedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "m.gguf"
        self.path.write_bytes(b"orbit-model-bytes")
        self.digest = hashlib.sha256(b"orbit-model-bytes").hexdigest()

    def test_matching_pin_passes(self):
        pins = {("r", "f"): (len(b"orbit-model-bytes"), self.digest)}
        self.assertTrue(verify_pinned_download(self.path, "r", "f", pins=pins))
        self.assertTrue(self.path.exists())

    def test_unpinned_file_is_not_checked(self):
        self.assertFalse(verify_pinned_download(self.path, "r", "f", pins={}))

    def test_size_mismatch_removes_file(self):
        with self.assertRaises(IntegrityError):
            verify_pinned_download(self.path, "r", "f", pins={("r", "f"): (3, None)})
        self.assertFalse(self.path.exists())

    def test_hash_mismatch_removes_file(self):
        with self.assertRaises(IntegrityError):
            verify_pinned_download(self.path, "r", "f", pins={("r", "f"): (None, "0" * 64)})
        self.assertFalse(self.path.exists())


class PackagedRegistryTests(unittest.TestCase):
    def test_packaged_registry_has_pins_for_most_files(self):
        pins = registry_pins()
        self.assertGreaterEqual(len(pins), 19)
        for (repo, file), (size, digest) in pins.items():
            self.assertTrue(repo and file)
            self.assertTrue(size is None or size > 0)
            self.assertTrue(digest is None or len(digest) == 64)


if __name__ == "__main__":
    unittest.main()
