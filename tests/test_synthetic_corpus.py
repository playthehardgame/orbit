"""The committed synthetic corpus: harmless, deterministic, oracle-independent.

Unlike the six frozen malware samples (machine-local, SHA-pinned, never in git),
these files are generated from plain text by `tests/synthetic_corpus/generate.py`
and live in the repository, so this suite runs on every host. The expected
decoded layers are the generator's inputs, never the output of the code under
test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "synthetic_corpus"
sys.path.insert(0, str(CORPUS))

import generate  # noqa: E402

from orbit.runtime.analysis_deobfuscate import COMPLETE, deobfuscate_with_status  # noqa: E402
from orbit.runtime.analysis_ole import extract_office_vba, is_ole_container  # noqa: E402

MANIFEST = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
URL = re.compile(r"https?://([A-Za-z0-9.\-]+)")


def _text(name: str) -> str:
    """The text the analyzers see: the file itself, or for the OLE container the
    VBA module source extracted by the production OLE reader."""
    if name.endswith(".doc"):
        ex = extract_office_vba((CORPUS / name).read_bytes())
        assert ex is not None and len(ex.modules) == 1, "macro sample must yield one module"
        return ex.modules[0].source
    return (CORPUS / name).read_text(encoding="utf-8")


class SyntheticCorpusTests(unittest.TestCase):
    def test_manifest_lists_exactly_the_generated_samples(self) -> None:
        self.assertEqual(set(MANIFEST["samples"]), set(generate.SAMPLES))
        on_disk = {p.name for p in CORPUS.iterdir() if p.name.startswith("syn_")}
        self.assertEqual(on_disk, set(generate.SAMPLES))

    def test_files_match_the_generator_byte_for_byte(self) -> None:
        for name, (source, entry) in generate.build().items():
            with self.subTest(sample=name):
                data = (CORPUS / name).read_bytes()
                expected = source if isinstance(source, bytes) else source.encode("utf-8")
                self.assertEqual(data, expected, "regenerate the corpus")
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(MANIFEST["samples"][name], entry)

    def test_every_sample_decodes_to_its_independent_oracle(self) -> None:
        for name, entry in MANIFEST["samples"].items():
            with self.subTest(sample=name):
                result = deobfuscate_with_status(_text(name))
                self.assertEqual(result.status, COMPLETE)
                self.assertEqual(
                    [s.kind for s in result.stages], entry["expected_stage_kinds"]
                )
                self.assertEqual(
                    sorted(s.output for s in result.stages),
                    sorted(entry["expected_stage_outputs"]),
                )
                self.assertEqual(
                    sorted(hashlib.sha256(s.output.encode()).hexdigest() for s in result.stages),
                    sorted(entry["expected_stage_output_sha256"]),
                )

    def test_a_tampered_sample_no_longer_matches_its_oracle(self) -> None:
        """Negative control: the oracle comparison can fail."""
        source = _text("syn_jscript_xor.js")
        tampered = source.replace(', 23, "-");', ', 24, "-");')
        self.assertNotEqual(tampered, source)
        result = deobfuscate_with_status(tampered)
        outputs = sorted(s.output for s in result.stages)
        self.assertNotEqual(
            outputs, sorted(MANIFEST["samples"]["syn_jscript_xor.js"]["expected_stage_outputs"])
        )

    def test_the_layered_sample_reaches_its_innermost_layer(self) -> None:
        entry = MANIFEST["samples"]["syn_layered.vbs"]
        innermost = entry["expected_stage_outputs"][-1]
        result = deobfuscate_with_status(_text("syn_layered.vbs"))
        self.assertIn(innermost, [s.output for s in result.stages])

    def test_the_macro_container_yields_its_independent_module_oracle(self) -> None:
        entry = MANIFEST["samples"]["syn_macro.doc"]
        raw = (CORPUS / "syn_macro.doc").read_bytes()
        self.assertTrue(is_ole_container(raw))
        ex = extract_office_vba(raw)
        self.assertIsNotNone(ex)
        self.assertEqual([m.name for m in ex.modules], [entry["expected_module_name"]])
        module = ex.modules[0]
        self.assertEqual(module.source, entry["expected_module_source"])
        self.assertEqual(module.source_sha256, entry["expected_module_source_sha256"])
        self.assertEqual(module.stream_path, "VBA/" + entry["expected_module_name"])

    def test_the_stringarray_sample_folds_the_joined_chunks(self) -> None:
        entry = MANIFEST["samples"]["syn_stringarray.js"]
        self.assertEqual(entry["expected_stage_kinds"], ["js_stringarray_fold"])
        result = deobfuscate_with_status(_text("syn_stringarray.js"))
        folds = [s.output for s in result.stages if s.kind == "js_stringarray_fold"]
        self.assertEqual(folds, entry["expected_stage_outputs"])

    def test_corpus_is_harmless_only_invalid_hosts_anywhere(self) -> None:
        """Samples and every decoded layer reference only the `.invalid` TLD."""
        for name, entry in MANIFEST["samples"].items():
            texts = [_text(name), *entry["expected_stage_outputs"]]
            for text in texts:
                for host in URL.findall(text):
                    with self.subTest(sample=name, host=host):
                        self.assertTrue(host.endswith(".invalid"), host)
            with self.subTest(sample=name, check="no process execution"):
                for text in texts:
                    for needle in ("Shell", "Run(", "Exec(", "Invoke-Expression", "iex ", "DownloadFile"):
                        self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
