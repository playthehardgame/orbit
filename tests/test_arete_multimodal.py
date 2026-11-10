from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama import model_profiles as mp
from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.model_registry import (
    get_manifest,
    load_registry,
    local_model_path,
    resolve_model,
)
from orbit.native_llama.paths import NativeLlamaPaths

ARETE_ID = "nous-arete-27b-q4-k-m"
MMPROJ_SIZE = 927607104
MMPROJ_SHA256 = "0c9844f4193e2d25642dc62c5921aef23fd71855b8283cb2b2543a77934966f8"


class AreteMmprojRegistryTests(unittest.TestCase):
    def test_registry_pins_f16_mmproj_identity(self) -> None:
        spec = get_manifest(ARETE_ID).mmproj
        self.assertIsNotNone(spec)
        self.assertEqual(spec.repo, "bartowski/NousResearch_Arete-27B-GGUF")
        self.assertEqual(spec.file, "mmproj-NousResearch_Arete-27B-f16.gguf")
        self.assertEqual(spec.size, MMPROJ_SIZE)
        self.assertEqual(spec.sha256, MMPROJ_SHA256)

    def test_only_arete_among_qwen_entries_declares_mmproj(self) -> None:
        for manifest in load_registry():
            if manifest.architecture.startswith(("qwen", "qwen35")) and manifest.id != ARETE_ID:
                with self.subTest(model=manifest.id):
                    self.assertIsNone(manifest.mmproj)

    def test_gemma_mmproj_unpinned_and_unchanged(self) -> None:
        spec = get_manifest("gemma4-26b-a4b-it-q40").mmproj
        self.assertEqual(spec.file, "mmproj-gemma-4-26B-A4B-it-Q8_0.gguf")
        self.assertIsNone(spec.size)
        self.assertIsNone(spec.sha256)

    def _resolve(self, root: Path, size: int | None):
        manifest = get_manifest(ARETE_ID)
        target = local_model_path(manifest.target, models_dir=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("t", encoding="utf-8")
        if size is not None:
            mm = local_model_path(manifest.mmproj, models_dir=root)
            with open(mm, "wb") as handle:
                handle.truncate(size)
        return resolve_model(manifest, models_dir=root, hf_cache=root / "hf")

    def test_missing_mmproj_falls_back_to_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = self._resolve(Path(tmp), None)
        self.assertIsNone(resolved.mmproj_path)
        self.assertFalse(resolved.multimodal_available)
        self.assertEqual(resolved.multimodal_fallback_reason, "mmproj-missing")

    def test_wrong_size_mmproj_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = self._resolve(Path(tmp), MMPROJ_SIZE - 1)
        self.assertIsNone(resolved.mmproj_path)
        self.assertFalse(resolved.multimodal_available)
        self.assertEqual(resolved.multimodal_fallback_reason, "mmproj-size-mismatch")

    def test_exact_size_mmproj_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = self._resolve(Path(tmp), MMPROJ_SIZE)
        self.assertIsNotNone(resolved.mmproj_path)
        self.assertTrue(resolved.multimodal_available)

    def test_registry_rejects_malformed_pins(self) -> None:
        text = json.loads(
            (Path(mp.__file__).parent / "model_registry.json").read_text(encoding="utf-8")
        )
        for key, bad in (("size", 0), ("size", "12"), ("sha256", "ABC"), ("sha256", "0" * 63)):
            data = json.loads(json.dumps(text))
            for model in data["models"]:
                if model["id"] == ARETE_ID:
                    model["mmproj"][key] = bad
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "r.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    load_registry(path)


class AreteMmprojClientGateTests(unittest.TestCase):
    def _client(self, profile) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/m.gguf"),
            mmproj_model=Path("/models/mmproj.gguf"),
            model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, NativeClientConfig())
        client.model_profile = profile
        client._model = 1  # type: ignore[assignment]
        return client

    def _profile(self, profile_id: str, **overrides):
        base = dict(
            profile_id=profile_id, family="f", model_name="n", architecture="qwen35",
            renderer="llama.cpp-jinja", reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml", history_serialization="x", verified=True,
            failure_reason=None, template_source="t", template_sha256="h",
            thinking_supported=True, mtp_supported=False, gemma_prefix_reuse_supported=False,
        )
        base.update(overrides)
        return mp.NativeModelProfile(**base)

    def test_verified_non_multimodal_profile_rejects_projector(self) -> None:
        client = self._client(self._profile(mp.QWEN38_PROFILE_ID))
        client.mtmd = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "multimodal input is unsupported"):
            client._initialize_multimodal_context()
        client.mtmd.lib.orbit_mtmd_context_create.assert_not_called()

    def test_arete_profile_passes_the_gate_and_reaches_native_init(self) -> None:
        client = self._client(self._profile(mp.QWEN38_ARETE_PROFILE_ID, multimodal_supported=True))
        client.mtmd = mock.Mock()
        client.mtmd.lib.orbit_mtmd_context_create.return_value = 0
        client.mtmd.last_error.return_value = "boom"
        with self.assertRaisesRegex(RuntimeError, "failed to load multimodal projector: boom"):
            client._initialize_multimodal_context()
        client.mtmd.lib.orbit_mtmd_context_create.assert_called_once()

    def test_projector_without_bridge_fails_closed(self) -> None:
        client = self._client(self._profile(mp.QWEN38_ARETE_PROFILE_ID, multimodal_supported=True))
        client.mtmd = None
        with self.assertRaisesRegex(RuntimeError, "mtmd ABI bridge is required"):
            client._initialize_multimodal_context()


if __name__ == "__main__":
    unittest.main()
