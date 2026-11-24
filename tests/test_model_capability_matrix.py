"""Offline table-driven capability matrix for every verified native model profile.

Flags are pinned to what the current detection code produces; a change to any
profile capability must be an explicit edit of this table.
"""

from __future__ import annotations

import hashlib
import unittest
from contextlib import ExitStack
from unittest import mock

from orbit.native_llama import model_profiles as mp
from orbit.native_llama.model_profiles import (
    VERIFIED_NATIVE_MODEL_IDENTITIES,
    detect_native_model_profile,
)

try:
    from tests.test_native_model_profiles import (
        GEMMA4_E4B_HERETIC_METADATA,
        GRANITE42_8B_METADATA,
        GRANITE42_METADATA,
        INKLING2_MINI_METADATA,
        LFM25_METADATA,
        MINICPM5_METADATA,
        QWEN3_CODER_METADATA,
        QWEN38_9B_DISTILL_METADATA,
        QWEN38_METADATA,
        QWEN38_ARETE_METADATA,
        QWEN38_TURBO_FC_METADATA,
        SCOPE_GUARD_METADATA,
        QWEN38_SALUKI_METADATA,
        QWEN_METADATA,
    )
except ImportError:  # run with the tests directory itself on sys.path
    from test_native_model_profiles import (  # type: ignore[no-redef]
        GEMMA4_E4B_HERETIC_METADATA,
        GRANITE42_8B_METADATA,
        GRANITE42_METADATA,
        INKLING2_MINI_METADATA,
        LFM25_METADATA,
        MINICPM5_METADATA,
        QWEN3_CODER_METADATA,
        QWEN38_9B_DISTILL_METADATA,
        QWEN38_METADATA,
        QWEN38_ARETE_METADATA,
        QWEN38_TURBO_FC_METADATA,
        SCOPE_GUARD_METADATA,
        QWEN38_SALUKI_METADATA,
        QWEN_METADATA,
    )

# No fixtures exist in test_native_model_profiles.py for these; built from the
# detection code in model_profiles.py.
ORNITH15_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": mp.ORNITH15_VERIFIED_MODEL_NAME,
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen35moe.context_length": "262144",
    "qwen35moe.block_count": "41",
    "qwen35moe.expert_count": "256",
    "qwen35moe.expert_used_count": "8",
}

FLASH_NEXT_METADATA = {
    "general.architecture": "qwen4exp",
    "general.name": mp.QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME,
    "general.file_type": "31",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen4exp.context_length": "262144",
    "qwen4exp.block_count": "48",
    "qwen4exp.expert_count": "512",
    "qwen4exp.expert_used_count": "10",
}

GEMMA4_METADATA = {
    "general.architecture": "gemma4",
    "general.name": mp.GEMMA4_VERIFIED_MODEL_NAME,
    "tokenizer.ggml.model": "gemma4",
}

_DIGEST_CONSTANTS = (
    "QWEN36_OFFICIAL_TEMPLATE_SHA256",
    "QWEN38_OFFICIAL_TEMPLATE_SHA256",
    "QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256",
    "QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256",
    "QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256",
    "SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256",
    "ORNITH15_OFFICIAL_TEMPLATE_SHA256",
    "LFM25_OFFICIAL_TEMPLATE_SHA256",
    "INKLING2_MINI_OFFICIAL_TEMPLATE_SHA256",
    "QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256",
    "MINICPM5_OFFICIAL_TEMPLATE_SHA256",
    "GRANITE42_OFFICIAL_TEMPLATE_SHA256",
    "QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256",
    "GEMMA4_E4B_HERETIC_TEMPLATE_SHA256",
)

TEMPLATE = "official-template-for-capability-matrix"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(profile_id, metadata, quant, mtp, route, multimodal, thinking, drifts):
    return {
        "profile_id": profile_id,
        "metadata": metadata,
        "quant": quant,
        "mtp": mtp,
        "route": route,
        "multimodal": multimodal,
        "thinking": thinking,
        "drifts": drifts,
    }


_FT_NAME = (("general.file_type", "2"), ("general.name", "Some Other Model"))

# profile_id, metadata, verified_quantization, mtp, route_prefix, multimodal, thinking, drifts
MATRIX = (
    _row(mp.GEMMA4_PROFILE_ID, GEMMA4_METADATA, None, True, True, True, True,
         (("general.architecture", "llama"), ("tokenizer.ggml.model", "gpt2"))),
    _row(mp.GEMMA4_E4B_HERETIC_PROFILE_ID, GEMMA4_E4B_HERETIC_METADATA, "Q4_K_M", False, False, True, True, _FT_NAME),
    _row(mp.QWEN38_9B_DISTILL_PROFILE_ID, QWEN38_9B_DISTILL_METADATA, "Q4_K_M", False, False, False, True, _FT_NAME),
    _row(mp.QWEN36_PROFILE_ID, QWEN_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
    _row(mp.ORNITH15_PROFILE_ID, ORNITH15_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
    _row(mp.QWEN38_PROFILE_ID, QWEN38_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
    _row(mp.QWEN38_SALUKI_PROFILE_ID, QWEN38_SALUKI_METADATA, "IQ2_XS", False, False, False, True, _FT_NAME),
    _row(mp.QWEN38_ARETE_PROFILE_ID, QWEN38_ARETE_METADATA, "Q4_K_M", False, True, True, True, _FT_NAME),
    _row(mp.QWEN38_TURBO_FC_PROFILE_ID, QWEN38_TURBO_FC_METADATA, "Q4_K_M", False, False, False, True, _FT_NAME),
    _row(mp.SCOPE_GUARD_PROFILE_ID, SCOPE_GUARD_METADATA, "Q8_0", False, False, False, True, _FT_NAME),
    _row(mp.QWEN38_FLASH_NEXT_PROFILE_ID, FLASH_NEXT_METADATA, "UD-IQ1_M", False, True, False, True, _FT_NAME),
    _row(mp.LFM25_PROFILE_ID, LFM25_METADATA, "Q4_K_M", False, False, False, True, _FT_NAME),
    _row(mp.LFM25_Q8_PROFILE_ID, {**LFM25_METADATA, "general.file_type": "7"}, "Q8_0",
         False, False, False, True, _FT_NAME),
    _row(mp.LFM25_Q6_PROFILE_ID, {**LFM25_METADATA, "general.file_type": "18"}, "Q6_K",
         False, False, False, True, _FT_NAME),
    _row(mp.INKLING2_MINI_PROFILE_ID, INKLING2_MINI_METADATA, "IQ4_NL", False, False, False, True, _FT_NAME),
    _row(mp.QWEN3_CODER_PROFILE_ID, QWEN3_CODER_METADATA, "Q4_K_M", False, True, False, False, _FT_NAME),
    _row(mp.MINICPM5_PROFILE_ID, MINICPM5_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
    _row(mp.GRANITE42_PROFILE_ID, GRANITE42_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
    _row(mp.GRANITE42_8B_PROFILE_ID, GRANITE42_8B_METADATA, "Q4_K_M", False, True, False, True, _FT_NAME),
)


def _detect(metadata, *, template=TEMPLATE, pinned=TEMPLATE):
    """Detect with every pinned template digest patched to hash(`pinned`)."""
    with ExitStack() as stack:
        for name in _DIGEST_CONSTANTS:
            stack.enter_context(mock.patch.object(mp, name, _digest(pinned)))
        return detect_native_model_profile(metadata, template)


class ModelCapabilityMatrixTests(unittest.TestCase):
    def test_matrix_covers_every_verified_identity(self) -> None:
        self.assertEqual(
            sorted(row["profile_id"] for row in MATRIX),
            sorted(identity.profile_id for identity in VERIFIED_NATIVE_MODEL_IDENTITIES),
        )

    def test_pinned_identity_and_capabilities(self) -> None:
        for row in MATRIX:
            with self.subTest(profile=row["profile_id"]):
                profile = _detect(row["metadata"])
                self.assertTrue(profile.verified)
                self.assertIsNone(profile.failure_reason)
                self.assertEqual(profile.profile_id, row["profile_id"])
                self.assertEqual(profile.verified_quantization, row["quant"])
                self.assertIs(profile.mtp_supported, row["mtp"])
                self.assertIs(profile.route_prefix_reuse_supported, row["route"])
                self.assertIs(profile.multimodal_supported, row["multimodal"])
                self.assertIs(profile.thinking_supported, row["thinking"])

    def test_identity_architecture_matches_detected_profile(self) -> None:
        for row in MATRIX:
            with self.subTest(profile=row["profile_id"]):
                identity = mp.verified_native_model_identity(row["profile_id"])
                self.assertIsNotNone(identity)
                self.assertEqual(identity.architecture, _detect(row["metadata"]).architecture)

    def test_drift_fails_closed(self) -> None:
        for row in MATRIX:
            for key, value in row["drifts"]:
                with self.subTest(profile=row["profile_id"], drift=key):
                    profile = _detect({**row["metadata"], key: value})
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, row["profile_id"])
                    self.assertFalse(profile.mtp_supported)
                    self.assertFalse(profile.route_prefix_reuse_supported)
                    self.assertFalse(profile.multimodal_supported)
            if row["profile_id"] == mp.GEMMA4_PROFILE_ID:
                continue  # Gemma 4 identity is architecture + tokenizer only; no template pin.
            with self.subTest(profile=row["profile_id"], drift="template"):
                self.assertFalse(_detect(row["metadata"], template=TEMPLATE + "-drift").verified)

    def test_diagnostics_capabilities_agree_with_flags(self) -> None:
        for row in MATRIX:
            for thinking_enabled in (True, False):
                with self.subTest(profile=row["profile_id"], thinking_enabled=thinking_enabled):
                    diag = _detect(row["metadata"]).diagnostics(thinking_enabled=thinking_enabled)
                    caps = diag["capabilities"]
                    self.assertEqual(diag["compatibility_profile"], row["profile_id"])
                    self.assertIs(diag["verified"], True)
                    self.assertIs(diag["mtp_supported"], row["mtp"])
                    self.assertIs(caps["mtp"], row["mtp"])
                    self.assertIs(caps["route_prefix_reuse"], row["route"])
                    self.assertIs(caps["multimodal"], row["multimodal"])
                    self.assertIs(diag["thinking_supported"], row["thinking"])
                    self.assertEqual(diag["verified_quantization"], row["quant"])
                    for name in ("chat", "tools", "tool_history_results", "write_artifact",
                                 "verify_artifact", "full_document_analysis"):
                        self.assertTrue(caps[name], name)
                    self.assertFalse(caps["arbitrary_exact_copy_artifact"])
                    self.assertFalse(caps["empty_artifact"])

    def test_unverified_diagnostics_report_no_capabilities(self) -> None:
        profile = _detect({**QWEN_METADATA, "general.name": "Nope"})
        caps = profile.diagnostics(thinking_enabled=False)["capabilities"]
        self.assertFalse(profile.verified)
        for name in ("chat", "tools", "mtp", "route_prefix_reuse", "multimodal"):
            self.assertFalse(caps[name], name)


if __name__ == "__main__":
    unittest.main()
