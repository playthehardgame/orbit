from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "generate_sbom", Path(__file__).resolve().parents[1] / "scripts" / "generate_sbom.py")
sbom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sbom)


class SbomTests(unittest.TestCase):
    def test_structure_and_pins(self):
        doc = sbom.build_sbom()
        self.assertEqual((doc["bomFormat"], doc["specVersion"]), ("CycloneDX", "1.5"))
        models = [c for c in doc["components"] if c["type"] == "machine-learning-model"]
        self.assertGreaterEqual(len(models), 19)
        for component in models:
            for h in component.get("hashes", []):
                self.assertEqual((h["alg"], len(h["content"])), ("SHA-256", 64))

    def test_output_is_deterministic(self):
        self.assertEqual(sbom.build_sbom(), sbom.build_sbom())


if __name__ == "__main__":
    unittest.main()
