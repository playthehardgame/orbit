from __future__ import annotations

import hashlib
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.capabilities import LlamaCppBuildInfo, safe_native_capability_manifest
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    GRANITE42_PROFILE_ID,
    GRANITE42_8B_PROFILE_ID,
    LFM25_PROFILE_ID,
    LFM25_Q6_PROFILE_ID,
    LFM25_Q8_PROFILE_ID,
    MINICPM5_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
    QWEN38_PROFILE_ID,
    GEMMA4_E4B_HERETIC_PROFILE_ID,
    GEMMA4_E4B_HERETIC_TEMPLATE_SHA256,
    QWEN38_9B_DISTILL_PROFILE_ID,
    SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256,
    SCOPE_GUARD_PROFILE_ID,
    QWEN38_ARETE_PROFILE_ID,
    QWEN38_TURBO_FC_PROFILE_ID,
    QWEN38_SALUKI_PROFILE_ID,
    QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256,
    NativeModelProfile,
    detect_native_model_profile,
    supports_low_memory_mode,
)
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT


QWEN_METADATA = {
    "general.architecture": "qwen35moe",
    "general.name": "Qwen3.6-35B-A3B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
}

QWEN3_CODER_METADATA = {
    "general.architecture": "qwen3moe",
    "general.name": "Qwen3-Coder-30B-A3B-Instruct",
    "general.file_type": "15",
    "general.quantization_version": "2",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen2",
    "tokenizer.ggml.add_bos_token": "false",
    "tokenizer.ggml.eos_token_id": "151645",
    "tokenizer.ggml.padding_token_id": "151654",
    "qwen3moe.context_length": "262144",
    "qwen3moe.expert_count": "128",
    "qwen3moe.expert_used_count": "8",
}


QWEN38_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.8-27B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "65",
}

QWEN38_SALUKI_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.8-27B GSQ-RCO",
    "general.file_type": "20",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "tokenizer.ggml.padding_token_id": "248044",
    "tokenizer.ggml.add_bos_token": "false",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "64",
}

QWEN38_TURBO_FC_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.8 27B Brainwaves NM HERETIC BR LOA1",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "tokenizer.ggml.padding_token_id": "248044",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "64",
}

QWEN38_ARETE_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Arete 27B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.bos_token_id": "248044",
    "tokenizer.ggml.eos_token_id": "248046",
    "tokenizer.ggml.padding_token_id": "248044",
    "tokenizer.ggml.add_bos_token": "false",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "64",
}

QWEN38_9B_DISTILL_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.8 9B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.eos_token_id": "248046",
    "tokenizer.ggml.padding_token_id": "248044",
    "tokenizer.ggml.add_bos_token": "false",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "33",
}

SCOPE_GUARD_METADATA = {
    "general.architecture": "qwen35",
    "general.name": "Scope Guard v2 4B Q 2609",
    "general.file_type": "7",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "qwen35",
    "tokenizer.ggml.eos_token_id": "248046",
    "tokenizer.ggml.padding_token_id": "248055",
    "tokenizer.ggml.add_bos_token": "false",
    "qwen35.context_length": "262144",
    "qwen35.block_count": "33",
}

GEMMA4_E4B_HERETIC_METADATA = {
    "general.architecture": "gemma4",
    "general.name": "Gemma 4 E4B It Ultra Uncensored Heretic",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gemma4",
    "tokenizer.ggml.bos_token_id": "2",
    "tokenizer.ggml.eos_token_id": "1",
    "tokenizer.ggml.padding_token_id": "0",
    "gemma4.context_length": "131072",
    "gemma4.block_count": "42",
    "gemma4.embedding_length_per_layer_input": "256",
}

MINICPM5_METADATA = {
    "general.architecture": "llama",
    "general.name": "MiniCPM5 2B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "minicpm5",
    "tokenizer.ggml.bos_token_id": "0",
    "tokenizer.ggml.eos_token_id": "1",
    "tokenizer.ggml.padding_token_id": "1",
    "tokenizer.ggml.add_bos_token": "false",
    "llama.context_length": "131072",
    "llama.block_count": "42",
}

GRANITE42_METADATA = {
    "general.architecture": "granite",
    "general.name": "Granite 4.2 3b",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "granite-docling",
    "tokenizer.ggml.bos_token_id": "100283",
    "tokenizer.ggml.eos_token_id": "100257",
    "tokenizer.ggml.padding_token_id": "100257",
    "granite.context_length": "131072",
    "granite.block_count": "40",
}

GRANITE42_8B_METADATA = {
    **GRANITE42_METADATA,
    "general.name": "Granite 4.2 8b",
}

LFM25_METADATA = {
    "general.architecture": "lfm2moe",
    "general.name": "Lfm2.5-8B-A1B",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "lfm2",
    "tokenizer.ggml.bos_token_id": "124894",
    "tokenizer.ggml.eos_token_id": "124900",
    "tokenizer.ggml.padding_token_id": "124893",
    "lfm2moe.context_length": "128000",
    "lfm2moe.block_count": "24",
    "lfm2moe.expert_count": "32",
    "lfm2moe.expert_used_count": "4",
}


class Qwen38ProfileTests(unittest.TestCase):
    """Qwen3.8 is a distinct dense `qwen35` identity, isolated from Qwen3.6."""

    def _detect(self, metadata=None, template="official-qwen38-template"):
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_OFFICIAL_TEMPLATE_SHA256", digest
        ):
            return detect_native_model_profile(metadata or QWEN38_METADATA, template)

    def test_exact_identity_is_verified_and_isolated(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, QWEN38_PROFILE_ID)
        self.assertNotEqual(profile.profile_id, QWEN36_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertIsNone(profile.failure_reason)
        self.assertEqual(profile.family, "qwen3.8")
        self.assertEqual(profile.architecture, "qwen35")
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.verified_quantization, "Q4_K_M")

    def test_tool_protocol_matches_verified_wire_format(self) -> None:
        # Envelope proven byte-identical to Qwen3.6 in the GGUF template.
        profile = self._detect()
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")

    def test_mtp_and_gemma_prefix_reuse_stay_disabled(self) -> None:
        profile = self._detect()
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)

    def test_thinking_supported(self) -> None:
        self.assertTrue(self._detect().thinking_supported)

    def test_template_drift_fails_closed(self) -> None:
        profile = detect_native_model_profile(QWEN38_METADATA, "unreviewed-template")
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_template_identity_mismatch")

    def test_model_name_drift_fails_closed(self) -> None:
        profile = self._detect({**QWEN38_METADATA, "general.name": "Qwen3.8-9B"})
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_model_identity_mismatch")

    def test_tokenizer_drift_fails_closed(self) -> None:
        profile = self._detect({**QWEN38_METADATA, "tokenizer.ggml.pre": "qwen2"})
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_tokenizer_identity_mismatch")

    def test_quantization_drift_fails_closed(self) -> None:
        profile = self._detect({**QWEN38_METADATA, "general.file_type": "7"})
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_quantization_identity_mismatch")

    def test_near_match_metadata_fails_closed(self) -> None:
        for key, value in (
            ("tokenizer.ggml.bos_token_id", "1"),
            ("tokenizer.ggml.eos_token_id", "2"),
            ("qwen35.context_length", "32768"),
            ("qwen35.block_count", "64"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertEqual(profile.failure_reason, "qwen38_metadata_identity_mismatch")

    def test_trailing_system_messages_are_demoted_like_qwen36(self) -> None:
        """The Qwen3.8 template rejects a non-leading system role, so the
        declared history contract must actually be enforced."""
        from orbit.native_llama.client import NativeLlamaClient

        messages = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "evidence card"},
        ]
        client = object.__new__(NativeLlamaClient)
        client.model_profile = self._detect()
        serialized = NativeLlamaClient._serialize_profile_messages(client, messages)
        self.assertEqual([m["role"] for m in serialized], ["system", "user", "user"])

    def test_capability_manifest_does_not_report_unsupported(self) -> None:
        """A verified profile must not self-report as unsupported."""
        profile = self._detect()
        client = SimpleNamespace(model_profile=profile)
        manifest = safe_native_capability_manifest(
            client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT
        )
        self.assertNotEqual(manifest["profile_id"], "unsupported")
        self.assertEqual(manifest["profile_id"], profile.profile_id)

    def test_qwen36_metadata_does_not_resolve_to_qwen38(self) -> None:
        digest = hashlib.sha256("official-qwen-template".encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(QWEN_METADATA, "official-qwen-template")
        self.assertEqual(profile.profile_id, QWEN36_PROFILE_ID)


class Qwen38SalukiProfileTests(unittest.TestCase):
    def _detect(self, metadata=None, template="official-saluki-template"):
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256",
            digest,
        ):
            return detect_native_model_profile(metadata or QWEN38_SALUKI_METADATA, template)

    def test_exact_gguf_and_template_identity_is_verified(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, QWEN38_SALUKI_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.family, "qwen3.8-saluki")
        self.assertEqual(profile.verified_quantization, "IQ2_XS")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")

    def test_saluki_isolated_from_full_qwen38_profile(self) -> None:
        profile = self._detect()
        self.assertNotEqual(profile.profile_id, QWEN38_PROFILE_ID)

    def test_cache_and_mtp_capabilities_are_not_inherited(self) -> None:
        profile = self._detect()
        self.assertFalse(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.mtp_supported)

    def test_template_or_metadata_drift_fails_closed(self) -> None:
        self.assertFalse(
            detect_native_model_profile(QWEN38_SALUKI_METADATA, "wrong-template").verified
        )
        for key, value in (
            ("general.architecture", "qwen35moe"),
            ("general.name", "Qwen3.8-27B"),
            ("general.file_type", "15"),
            ("tokenizer.ggml.model", "llama"),
            ("tokenizer.ggml.pre", "qwen2"),
            ("tokenizer.ggml.bos_token_id", "1"),
            ("tokenizer.ggml.eos_token_id", "2"),
            ("tokenizer.ggml.padding_token_id", "0"),
            ("tokenizer.ggml.add_bos_token", "true"),
            ("qwen35.context_length", "131072"),
            ("qwen35.block_count", "65"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_SALUKI_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertNotEqual(profile.profile_id, QWEN38_SALUKI_PROFILE_ID)

    def test_template_drift_fails_closed_under_the_patched_digest(self) -> None:
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256",
            hashlib.sha256(b"official-saluki-template").hexdigest(),
        ):
            profile = detect_native_model_profile(QWEN38_SALUKI_METADATA, "drifted-template")
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen38_saluki_template_identity_mismatch")

    def test_saluki_drift_reports_saluki_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.pre", "qwen2", "qwen38_saluki_tokenizer_identity_mismatch"),
            ("general.file_type", "15", "qwen38_saluki_quantization_identity_mismatch"),
            ("qwen35.block_count", "65", "qwen38_saluki_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_SALUKI_METADATA, key: value})
                self.assertEqual(profile.failure_reason, reason)


class Qwen38AreteProfileTests(unittest.TestCase):
    def _detect(self, metadata=None, template="official-arete-template"):
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256",
            digest,
        ):
            return detect_native_model_profile(metadata or QWEN38_ARETE_METADATA, template)

    def test_exact_gguf_and_template_identity_is_verified(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, QWEN38_ARETE_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.family, "qwen3.8-arete")
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")

    def test_pinned_template_is_the_reviewed_saluki_template(self) -> None:
        from orbit.native_llama import model_profiles as mp

        self.assertEqual(
            mp.QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256,
            mp.QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256,
        )

    def test_route_prefix_is_enabled_but_mtp_is_not_inherited(self) -> None:
        # Route-prefix reuse is a live-qualified capability of this exact
        # artifact (see AGENTS.md); MTP stays off.
        profile = self._detect()
        self.assertTrue(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.mtp_supported)

    def test_template_or_metadata_drift_fails_closed(self) -> None:
        self.assertFalse(
            detect_native_model_profile(QWEN38_ARETE_METADATA, "wrong-template").verified
        )
        for key, value in (
            ("general.architecture", "qwen35moe"),
            ("general.name", "Qwen3.8-27B GSQ-RCO"),
            ("general.file_type", "20"),
            ("tokenizer.ggml.pre", "qwen2"),
            ("tokenizer.ggml.bos_token_id", "1"),
            ("tokenizer.ggml.add_bos_token", "true"),
            ("qwen35.context_length", "131072"),
            ("qwen35.block_count", "65"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_ARETE_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertNotEqual(profile.profile_id, QWEN38_ARETE_PROFILE_ID)

    def test_arete_drift_reports_arete_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.pre", "qwen2", "qwen38_arete_tokenizer_identity_mismatch"),
            ("general.file_type", "20", "qwen38_arete_quantization_identity_mismatch"),
            ("qwen35.block_count", "65", "qwen38_arete_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_ARETE_METADATA, key: value})
                self.assertEqual(profile.failure_reason, reason)


class Qwen38NineBDistillProfileTests(unittest.TestCase):
    TEMPLATE = "official-9b-distill-template"

    def _detect(self, metadata=None, template=None):
        digest = hashlib.sha256(self.TEMPLATE.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256",
            digest,
        ):
            return detect_native_model_profile(
                metadata or QWEN38_9B_DISTILL_METADATA, template or self.TEMPLATE
            )

    def test_exact_identity_is_verified_and_isolated(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, QWEN38_9B_DISTILL_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertFalse(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.mtp_supported)

    def test_drift_fails_closed_with_specific_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.pre", "qwen2", "qwen38_9b_distill_tokenizer_identity_mismatch"),
            ("general.file_type", "17", "qwen38_9b_distill_quantization_identity_mismatch"),
            ("qwen35.block_count", "32", "qwen38_9b_distill_metadata_identity_mismatch"),
            ("tokenizer.ggml.eos_token_id", "1", "qwen38_9b_distill_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_9B_DISTILL_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertEqual(profile.failure_reason, reason)
        profile = self._detect(template="drifted")
        self.assertEqual(profile.failure_reason, "qwen38_9b_distill_template_identity_mismatch")


class ScopeGuardProfileTests(unittest.TestCase):
    TEMPLATE = "official-scope-guard-template"

    def _detect(self, metadata=None, template=None):
        digest = hashlib.sha256(self.TEMPLATE.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256", digest
        ):
            return detect_native_model_profile(
                metadata or SCOPE_GUARD_METADATA, template or self.TEMPLATE
            )

    def test_exact_identity_is_verified_and_conservative(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, SCOPE_GUARD_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.verified_quantization, "Q8_0")
        self.assertFalse(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.multimodal_supported)

    def test_drift_fails_closed_with_specific_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.pre", "qwen2", "scope_guard_tokenizer_identity_mismatch"),
            ("general.file_type", "15", "scope_guard_quantization_identity_mismatch"),
            ("qwen35.block_count", "32", "scope_guard_metadata_identity_mismatch"),
            ("tokenizer.ggml.padding_token_id", "248044", "scope_guard_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**SCOPE_GUARD_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertEqual(profile.failure_reason, reason)
        profile = self._detect(template="drifted")
        self.assertEqual(profile.failure_reason, "scope_guard_template_identity_mismatch")

    def test_real_embedded_template_hash_is_pinned(self) -> None:
        self.assertEqual(
            SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256,
            "8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8",
        )


class Gemma4E4BHereticProfileTests(unittest.TestCase):
    TEMPLATE = "official-e4b-heretic-template"

    def _detect(self, metadata=None, template=None):
        digest = hashlib.sha256(self.TEMPLATE.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.GEMMA4_E4B_HERETIC_TEMPLATE_SHA256", digest
        ):
            return detect_native_model_profile(
                metadata or GEMMA4_E4B_HERETIC_METADATA, template or self.TEMPLATE
            )

    def test_exact_identity_is_verified_and_isolated(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, GEMMA4_E4B_HERETIC_PROFILE_ID)
        self.assertNotEqual(profile.profile_id, GEMMA4_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertEqual(profile.tool_call_protocol, "gemma4-native")
        for flag in (
            profile.mtp_supported,
            profile.gemma_prefix_reuse_supported,
            profile.route_prefix_reuse_supported,
        ):
            self.assertFalse(flag)
        self.assertTrue(profile.multimodal_supported)

    def test_drift_fails_closed_with_specific_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.model", "gpt2", "gemma4_e4b_heretic_tokenizer_identity_mismatch"),
            ("general.file_type", "17", "gemma4_e4b_heretic_quantization_identity_mismatch"),
            ("gemma4.block_count", "35", "gemma4_e4b_heretic_metadata_identity_mismatch"),
            ("tokenizer.ggml.eos_token_id", "106", "gemma4_e4b_heretic_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**GEMMA4_E4B_HERETIC_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertEqual(profile.failure_reason, reason)
        profile = self._detect(template="drifted")
        self.assertEqual(profile.failure_reason, "gemma4_e4b_heretic_template_identity_mismatch")

    def test_capability_manifest_does_not_report_unsupported(self) -> None:
        profile = self._detect()
        manifest = safe_native_capability_manifest(
            SimpleNamespace(model_profile=profile),
            final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT,
        )
        self.assertEqual(manifest["profile_id"], GEMMA4_E4B_HERETIC_PROFILE_ID)
        self.assertEqual(manifest["status"], "verified")

    def test_real_gguf_template_matches_pinned_hash(self) -> None:
        import glob
        from pathlib import Path

        from orbit.native_llama.gguf_split import read_gguf_header

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

    def test_26b_with_zero_per_layer_input_keeps_26b_profile(self) -> None:
        for name in ("gemma-4-26B-A4B-it", "Some Other Gemma4 Build"):
            with self.subTest(name=name):
                profile = detect_native_model_profile(
                    {
                        **GEMMA4_E4B_HERETIC_METADATA,
                        "general.name": name,
                        "gemma4.embedding_length_per_layer_input": "0",
                        "gemma4.block_count": "30",
                    },
                    "x",
                )
                self.assertEqual(profile.profile_id, GEMMA4_PROFILE_ID)

    def test_heretic_requires_per_layer_input_256(self) -> None:
        profile = self._detect(
            {**GEMMA4_E4B_HERETIC_METADATA, "gemma4.embedding_length_per_layer_input": "0"}
        )
        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "gemma4_e4b_heretic_metadata_identity_mismatch")

    def test_real_template_hash_is_pinned(self) -> None:
        self.assertEqual(
            GEMMA4_E4B_HERETIC_TEMPLATE_SHA256,
            "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf",
        )

    def test_other_e_series_gemma4_models_fail_closed(self) -> None:
        for name in ("gemma-4-E4B-it", "Gemma 4 E2B It"):
            with self.subTest(name=name):
                profile = detect_native_model_profile(
                    {**GEMMA4_E4B_HERETIC_METADATA, "general.name": name}, "x"
                )
                self.assertFalse(profile.verified)
                self.assertEqual(profile.failure_reason, "gemma4_e_series_unqualified")

    def test_other_gemma4_models_keep_the_26b_profile(self) -> None:
        profile = detect_native_model_profile(
            {
                **{k: v for k, v in GEMMA4_E4B_HERETIC_METADATA.items() if "per_layer" not in k},
                "general.name": "gemma-4-26B-A4B-it",
            },
            "x",
        )
        self.assertEqual(profile.profile_id, GEMMA4_PROFILE_ID)


class NativeModelProfileTests(unittest.TestCase):
    def test_detects_verified_lfm25_only_from_exact_identity(self) -> None:
        template = "official-lfm25-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.LFM25_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(LFM25_METADATA, template)

        self.assertEqual(profile.profile_id, LFM25_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.family, "lfm2.5")
        self.assertEqual(profile.tool_call_protocol, "lfm2-python")
        self.assertFalse(profile.route_prefix_reuse_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")

    def test_detects_verified_lfm25_q8_artifact(self) -> None:
        template = "official-lfm25-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**LFM25_METADATA, "general.file_type": "7"}
        with mock.patch("orbit.native_llama.model_profiles.LFM25_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertEqual(profile.profile_id, LFM25_Q8_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.verified_quantization, "Q8_0")

    def test_detects_verified_lfm25_q6_artifact(self) -> None:
        template = "official-lfm25-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**LFM25_METADATA, "general.file_type": "18"}
        with mock.patch("orbit.native_llama.model_profiles.LFM25_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertEqual(profile.profile_id, LFM25_Q6_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.verified_quantization, "Q6_K")

    def test_lfm25_identity_drift_fails_closed(self) -> None:
        template = "official-lfm25-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.LFM25_OFFICIAL_TEMPLATE_SHA256", digest):
            for key, value in (
                ("general.name", "Lfm2.5-1B"),
                ("tokenizer.ggml.pre", "qwen35"),
                ("general.file_type", "12"),
                ("lfm2moe.block_count", "23"),
                ("tokenizer.ggml.eos_token_id", "2"),
            ):
                with self.subTest(key=key):
                    profile = detect_native_model_profile({**LFM25_METADATA, key: value}, template)
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, LFM25_PROFILE_ID)

    def test_detects_verified_granite42_only_from_exact_gguf_identity(self) -> None:
        template = "official-granite42-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.GRANITE42_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(GRANITE42_METADATA, template)

        self.assertEqual(profile.profile_id, GRANITE42_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.family, "granite4.2")
        self.assertEqual(profile.tool_call_protocol, "qwen3-coder-xml")
        self.assertTrue(profile.thinking_supported)
        self.assertTrue(profile.route_prefix_reuse_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")

    def test_granite42_identity_drift_fails_closed(self) -> None:
        template = "official-granite42-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.GRANITE42_OFFICIAL_TEMPLATE_SHA256", digest):
            for key, value in (
                ("general.name", "Granite 4.2 9b"),
                ("tokenizer.ggml.pre", "gpt2"),
                ("granite.block_count", "39"),
                ("tokenizer.ggml.eos_token_id", "1"),
            ):
                with self.subTest(key=key):
                    profile = detect_native_model_profile({**GRANITE42_METADATA, key: value}, template)
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, GRANITE42_PROFILE_ID)

    def test_detects_verified_granite42_8b_identity(self) -> None:
        template = "official-granite42-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.GRANITE42_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(GRANITE42_8B_METADATA, template)

        self.assertEqual(profile.profile_id, GRANITE42_8B_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertTrue(profile.route_prefix_reuse_supported)

    def test_accepts_qualified_granite42_q8_and_q6_variants(self) -> None:
        template = "official-granite42-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.GRANITE42_OFFICIAL_TEMPLATE_SHA256", digest):
            q8 = detect_native_model_profile(
                {**GRANITE42_METADATA, "general.file_type": "7"},
                template,
            )
            q6 = detect_native_model_profile(
                {**GRANITE42_8B_METADATA, "general.file_type": "18"},
                template,
            )

        self.assertEqual(q8.profile_id, GRANITE42_PROFILE_ID)
        self.assertEqual(q8.verified_quantization, "Q8_0")
        self.assertEqual(q6.profile_id, GRANITE42_8B_PROFILE_ID)
        self.assertEqual(q6.verified_quantization, "Q6_K")

    def test_granite42_cross_variant_quantization_fails_closed(self) -> None:
        template = "official-granite42-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.GRANITE42_OFFICIAL_TEMPLATE_SHA256", digest):
            q6_for_3b = detect_native_model_profile(
                {**GRANITE42_METADATA, "general.file_type": "18"},
                template,
            )
            q8_for_8b = detect_native_model_profile(
                {**GRANITE42_8B_METADATA, "general.file_type": "7"},
                template,
            )

        self.assertEqual(q6_for_3b.failure_reason, "granite42_quantization_identity_mismatch")
        self.assertEqual(q8_for_8b.failure_reason, "granite42_quantization_identity_mismatch")

    def test_detects_verified_minicpm5_only_from_exact_gguf_identity(self) -> None:
        template = "official-minicpm5-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.MINICPM5_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(MINICPM5_METADATA, template)

        self.assertEqual(profile.profile_id, MINICPM5_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertIsNone(profile.failure_reason)
        self.assertEqual(profile.family, "minicpm5")
        self.assertEqual(profile.architecture, "llama")
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.tool_call_protocol, "minicpm5-xml")
        self.assertTrue(profile.thinking_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertTrue(profile.route_prefix_reuse_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")

    def test_minicpm5_identity_drift_fails_closed(self) -> None:
        template = "official-minicpm5-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.MINICPM5_OFFICIAL_TEMPLATE_SHA256", digest):
            for key, value in (
                ("general.name", "MiniCPM5 1B"),
                ("tokenizer.ggml.pre", "llama"),
                ("general.file_type", "2"),
                ("llama.block_count", "41"),
            ):
                with self.subTest(key=key):
                    profile = detect_native_model_profile({**MINICPM5_METADATA, key: value}, template)
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, MINICPM5_PROFILE_ID)

    def test_detects_verified_qwen_only_from_complete_identity(self) -> None:
        template = "official-qwen-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(QWEN_METADATA, template)

        self.assertEqual(profile.profile_id, QWEN36_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")
        self.assertEqual(profile.history_serialization, "qwen-leading-system-only")
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertFalse(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

    def test_qwen_template_drift_fails_closed(self) -> None:
        profile = detect_native_model_profile(QWEN_METADATA, "unreviewed-template")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_template_identity_mismatch")

    def test_qwen_model_size_drift_fails_closed(self) -> None:
        metadata = {**QWEN_METADATA, "general.name": "Qwen3.6-8B"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_model_identity_mismatch")

    def test_qwen_quantization_drift_fails_closed(self) -> None:
        template = "official-qwen-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**QWEN_METADATA, "general.file_type": "2"}

        with mock.patch("orbit.native_llama.model_profiles.QWEN36_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen36_quantization_identity_mismatch")

    def test_existing_gemma_family_keeps_orbit_profile(self) -> None:
        profile = detect_native_model_profile(
            {
                "general.architecture": "gemma4",
                "general.name": "Gemma 4 26B-A4B",
                "tokenizer.ggml.model": "gemma4",
            },
            "embedded-template-not-used-by-orbit-renderer",
        )

        self.assertEqual(profile.profile_id, GEMMA4_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "orbit-gemma4")
        self.assertTrue(profile.mtp_supported)
        self.assertTrue(profile.gemma_prefix_reuse_supported)
        self.assertFalse(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

    def test_detects_verified_qwen3_coder_only_from_complete_identity(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(QWEN3_CODER_METADATA, template)

        self.assertEqual(profile.profile_id, QWEN3_CODER_PROFILE_ID)
        self.assertEqual(profile.family, "qwen3-coder")
        self.assertTrue(profile.verified)
        self.assertEqual(profile.renderer, "llama.cpp-jinja")
        self.assertEqual(profile.tool_call_protocol, "qwen3-coder-xml")
        self.assertEqual(profile.history_serialization, "qwen3-coder-chatml")
        self.assertEqual(profile.artifact_content_protocol, "qwen3-coder-json-string-v1")
        self.assertFalse(profile.thinking_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.gemma_prefix_reuse_supported)
        self.assertTrue(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.multimodal_supported)
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertTrue(supports_low_memory_mode(profile))
        self.assertTrue(profile.diagnostics(thinking_enabled=False)["capabilities"]["full_document_analysis"])

    def test_qwen3_coder_template_drift_fails_closed(self) -> None:
        profile = detect_native_model_profile(QWEN3_CODER_METADATA, "unreviewed-template")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_template_identity_mismatch")
        self.assertFalse(supports_low_memory_mode(profile))

    def test_qwen3_coder_architecture_drift_fails_closed(self) -> None:
        metadata = {**QWEN3_CODER_METADATA, "general.architecture": "qwen3"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "unsupported_model_profile")

    def test_unknown_qwen3_variant_fails_closed(self) -> None:
        metadata = {**QWEN3_CODER_METADATA, "general.name": "Qwen3-Coder-Next"}

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_model_identity_mismatch")

    def test_qwen3_coder_filename_like_metadata_cannot_authorize_profile(self) -> None:
        metadata = {
            "general.filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            "general.architecture": "unknown",
        }

        profile = detect_native_model_profile(metadata, "anything")

        self.assertFalse(profile.verified)
        self.assertEqual(profile.profile_id, "unsupported")

    def test_qwen3_coder_metadata_drift_fails_closed(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        metadata = {**QWEN3_CODER_METADATA, "qwen3moe.expert_used_count": "4"}

        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            profile = detect_native_model_profile(metadata, template)

        self.assertFalse(profile.verified)
        self.assertEqual(profile.failure_reason, "qwen3_coder_metadata_identity_mismatch")

    def test_each_qwen3_coder_authorizing_metadata_field_fails_closed_on_drift(self) -> None:
        template = "official-qwen3-coder-template"
        digest = hashlib.sha256(template.encode()).hexdigest()
        drifted_values = {
            "general.file_type": "14",
            "general.quantization_version": "3",
            "tokenizer.ggml.model": "qwen",
            "tokenizer.ggml.pre": "qwen3",
            "tokenizer.ggml.add_bos_token": "true",
            "tokenizer.ggml.eos_token_id": "151643",
            "tokenizer.ggml.padding_token_id": "151645",
            "qwen3moe.context_length": "131072",
            "qwen3moe.expert_count": "64",
            "qwen3moe.expert_used_count": "4",
        }

        with mock.patch("orbit.native_llama.model_profiles.QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256", digest):
            for key, value in drifted_values.items():
                with self.subTest(key=key):
                    profile = detect_native_model_profile({**QWEN3_CODER_METADATA, key: value}, template)
                    self.assertFalse(profile.verified)
                    self.assertNotEqual(profile.profile_id, QWEN3_CODER_PROFILE_ID)

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_qwen_capability_manifest_is_profile_aware(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(9551, "6f79e02", "x86_64", "GNU", "a" * 64, "runtime_symbols")
        profile = NativeModelProfile(
            profile_id=QWEN36_PROFILE_ID,
            family="qwen3.6",
            model_name="Qwen3.6-35B-A3B",
            architecture="qwen35moe",
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256="b" * 64,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
        )
        client = SimpleNamespace(model_profile=profile, paths=SimpleNamespace(build_bin="/native"))

        manifest = safe_native_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["profile_id"], QWEN36_PROFILE_ID)
        self.assertEqual(manifest["status"], "verified")
        self.assertTrue(manifest["behavior_enforced"])
        self.assertEqual(manifest["renderer"]["template_hash"], "b" * 64)
        self.assertFalse(manifest["requirements"]["mtp_supported"])

    @mock.patch("orbit.native_llama.capabilities.read_llama_cpp_build_info")
    def test_qwen3_coder_capability_manifest_declares_profile_limits(self, build_info) -> None:
        build_info.return_value = LlamaCppBuildInfo(9551, "6f79e02", "x86_64", "GNU", "a" * 64, "runtime_symbols")
        profile = NativeModelProfile(
            profile_id=QWEN3_CODER_PROFILE_ID,
            family="qwen3-coder",
            model_name="Qwen3-Coder-30B-A3B-Instruct",
            architecture="qwen3moe",
            renderer="llama.cpp-jinja",
            reasoning_protocol="none",
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="qwen3-coder-chatml",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256="c" * 64,
            thinking_supported=False,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization="Q4_K_M",
            artifact_content_protocol="qwen3-coder-json-string-v1",
        )
        client = SimpleNamespace(model_profile=profile, paths=SimpleNamespace(build_bin="/native"))

        manifest = safe_native_capability_manifest(client, final_system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)

        self.assertEqual(manifest["profile_id"], QWEN3_CODER_PROFILE_ID)
        self.assertEqual(manifest["renderer"]["artifact_content_protocol"], "qwen3-coder-json-string-v1")
        self.assertFalse(manifest["requirements"]["thinking_supported"])
        self.assertFalse(manifest["requirements"]["route_prefix_reuse_supported"])
        self.assertFalse(manifest["requirements"]["multimodal_supported"])



class Qwen38TurboFcProfileTests(unittest.TestCase):
    def _detect(self, metadata=None, template="official-turbo-fc-template"):
        digest = hashlib.sha256(template.encode()).hexdigest()
        with mock.patch(
            "orbit.native_llama.model_profiles.QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256",
            digest,
        ):
            return detect_native_model_profile(metadata or QWEN38_TURBO_FC_METADATA, template)

    def test_exact_gguf_and_template_identity_is_verified(self) -> None:
        profile = self._detect()
        self.assertEqual(profile.profile_id, QWEN38_TURBO_FC_PROFILE_ID)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.family, "qwen3.8-turbo-fcfusion")
        self.assertEqual(profile.verified_quantization, "Q4_K_M")
        self.assertEqual(profile.tool_call_protocol, "qwen3.6-xml")

    def test_pinned_template_is_distinct_from_the_saluki_arete_one(self) -> None:
        from orbit.native_llama import model_profiles as mp

        self.assertNotEqual(
            mp.QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256,
            mp.QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256,
        )

    def test_cache_mtp_and_multimodal_capabilities_are_not_inherited(self) -> None:
        profile = self._detect()
        self.assertFalse(profile.route_prefix_reuse_supported)
        self.assertFalse(profile.mtp_supported)
        self.assertFalse(profile.multimodal_supported)

    def test_template_or_metadata_drift_fails_closed(self) -> None:
        self.assertFalse(
            detect_native_model_profile(QWEN38_TURBO_FC_METADATA, "wrong-template").verified
        )
        for key, value in (
            ("general.architecture", "qwen35moe"),
            ("general.name", "Arete 27B"),
            ("general.file_type", "20"),
            ("tokenizer.ggml.pre", "qwen2"),
            ("tokenizer.ggml.bos_token_id", "1"),
            ("tokenizer.ggml.add_bos_token", "true"),
            ("qwen35.context_length", "131072"),
            ("qwen35.block_count", "65"),
            ("qwen35.nextn_predict_layers", "1"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_TURBO_FC_METADATA, key: value})
                self.assertFalse(profile.verified)
                self.assertNotEqual(profile.profile_id, QWEN38_TURBO_FC_PROFILE_ID)

    def test_drift_reports_turbo_fc_reasons(self) -> None:
        for key, value, reason in (
            ("tokenizer.ggml.pre", "qwen2", "qwen38_turbo_fc_tokenizer_identity_mismatch"),
            ("general.file_type", "20", "qwen38_turbo_fc_quantization_identity_mismatch"),
            ("qwen35.block_count", "65", "qwen38_turbo_fc_metadata_identity_mismatch"),
        ):
            with self.subTest(key=key):
                profile = self._detect({**QWEN38_TURBO_FC_METADATA, key: value})
                self.assertEqual(profile.failure_reason, reason)


if __name__ == "__main__":
    unittest.main()
