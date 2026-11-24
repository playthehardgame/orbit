"""Checks that need a real GGUF on this host; skipped elsewhere.

Kept out of scripts/check_core.py on purpose: the core gate forbids skips, and CI
has no model files.
"""
from __future__ import annotations

import glob
import unittest
from pathlib import Path

from orbit.native_llama.gguf_split import read_gguf_header
from orbit.native_llama.model_profiles import (
    GEMMA4_E4B_HERETIC_PROFILE_ID,
    INKLING2_MINI_PROFILE_ID,
    detect_native_model_profile,
)


class RealGgufTests(unittest.TestCase):
    def test_real_gguf_template_matches_pinned_hash(self) -> None:
        matches = glob.glob(
            "/mnt/l/models/llmfan46--*/gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
        )
        if not matches:
            self.skipTest("E4B Heretic GGUF not present on this host")
        header = read_gguf_header(Path(matches[0]))
        metadata = {k: str(v) for k, v in header.kv.items() if not isinstance(v, list)}
        profile = detect_native_model_profile(metadata, header.kv["tokenizer.chat_template"])
        self.assertTrue(profile.verified, profile.failure_reason)
        self.assertEqual(profile.profile_id, GEMMA4_E4B_HERETIC_PROFILE_ID)

    def test_real_inkling_gguf_matches_pinned_identity(self) -> None:
        matches = glob.glob("/mnt/l/models/ggml-org--Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf")
        if not matches:
            self.skipTest("Inkling 2 Mini IQ4_NL GGUF not present on this host")
        header = read_gguf_header(Path(matches[0]))
        metadata = {k: str(v) for k, v in header.kv.items() if not isinstance(v, list)}
        profile = detect_native_model_profile(metadata, header.kv["tokenizer.chat_template"])
        self.assertTrue(profile.verified, profile.failure_reason)
        self.assertEqual(profile.profile_id, INKLING2_MINI_PROFILE_ID)


if __name__ == "__main__":
    unittest.main()
