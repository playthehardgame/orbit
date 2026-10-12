from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest import mock

from orbit.native_llama.client import (
    LLAMA_DEFAULT_SEED,
    NativeClientConfig,
    NativeLlamaClient,
    _add_default_sampler_stages,
    _canonical_sampler_hash,
)
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_server.app import OrbitNativeServer


class NativeRuntimePropsTests(unittest.TestCase):
    def _paths(self) -> NativeLlamaPaths:
        return NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/target.gguf"),
            mmproj_model=Path("/models/mmproj.gguf"),
            draft_mtp_model=None,
            multimodal_available=True,
            multimodal_fallback_reason=None,
            mtp_available=False,
            fallback_reason="draft-mtp-missing",
            model_id="gemma4-12b-it-q4km",
        )

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_runtime_info_matches_native_config_baseline(self, _mocked_lib) -> None:
        client = NativeLlamaClient(
            self._paths(),
            NativeClientConfig(
                context_tokens=8192,
                threads=6,
                threads_batch=6,
                batch_size=256,
                ubatch_size=128,
                temperature=0.7,
            ),
        )
        server = OrbitNativeServer(client=client, model_alias="m")

        runtime = server.runtime_info()

        self.assertEqual(runtime["threads"], 6)
        self.assertEqual(runtime["threads_batch"], 6)
        self.assertEqual(runtime["ctx_size"], 8192)
        self.assertEqual(runtime["batch_size"], 256)
        self.assertEqual(runtime["ubatch_size"], 128)
        self.assertEqual(runtime["parallel_slots"], 1)
        self.assertEqual(runtime["thinking_mode"], "off")
        self.assertEqual(runtime["temperature"], 0.7)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_server_props_include_multimodal_paths(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig())
        client.supports_vision = True
        client.supports_audio = True
        server = OrbitNativeServer(client=client, model_alias="m")

        self.assertEqual(server.client.paths.mmproj_model, Path("/models/mmproj.gguf"))
        self.assertTrue(server.client.paths.multimodal_available)
        self.assertTrue(server.client.supports_vision)
        self.assertTrue(server.client.supports_audio)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_token_counts_do_not_touch_completion_state(self, _mocked_lib) -> None:
        client = NativeLlamaClient(self._paths(), NativeClientConfig(context_tokens=16384))
        client.count_text_tokens = mock.Mock(return_value=17)
        client.inspect_chat_tokens = mock.Mock(return_value=(29, "a" * 64, "b" * 64))
        client.inspect_artifact_content_tokens = mock.Mock(return_value=(31, "c" * 64, "d" * 64))
        server = OrbitNativeServer(client=client, model_alias="m")

        text = server.count_text_tokens("hello")
        chat = server.count_chat_tokens(
            [{"role": "user", "content": "hello"}],
            tools=[],
            thinking=False,
        )
        artifact = server.count_artifact_content_tokens(
            [{"role": "user", "content": "artifact"}],
        )

        self.assertEqual(text, {"tokens": 17, "context_tokens": 16384})
        self.assertEqual(
            chat,
            {
                "tokens": 29,
                "context_tokens": 16384,
                "rendered_hash": "a" * 64,
                "token_hash": "b" * 64,
            },
        )
        self.assertEqual(
            artifact,
            {
                "tokens": 31,
                "context_tokens": 16384,
                "rendered_hash": "c" * 64,
                "token_hash": "d" * 64,
            },
        )
        client.count_text_tokens.assert_called_once_with("hello")
        client.inspect_chat_tokens.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tools=[],
            thinking=False,
        )
        client.inspect_artifact_content_tokens.assert_called_once_with(
            [{"role": "user", "content": "artifact"}],
        )

    @mock.patch("orbit.native_server.app.safe_native_capability_manifest")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_server_caches_bounded_native_capability_manifest(self, _mocked_lib, build_manifest) -> None:
        build_manifest.return_value = {
            "schema_version": 1,
            "profile_id": "orbit-gemma4-native-v1",
            "status": "verified",
            "behavior_enforced": False,
        }
        client = NativeLlamaClient(self._paths(), NativeClientConfig())

        server = OrbitNativeServer(client=client, model_alias="m")

        self.assertEqual(server.native_backend_capabilities["status"], "verified")
        build_manifest.assert_called_once_with(client, final_system_prompt=mock.ANY)


class DefaultSamplerStagesTests(unittest.TestCase):
    def test_zero_temperature_keeps_the_greedy_chain(self) -> None:
        lib = mock.Mock()
        lib.llama_sampler_init_greedy.return_value = 11
        _add_default_sampler_stages(lib, 7, 0.0)
        lib.llama_sampler_chain_add.assert_called_once_with(7, 11)
        lib.llama_sampler_init_temp.assert_not_called()
        lib.llama_sampler_init_dist.assert_not_called()

    def test_positive_temperature_adds_a_selecting_dist_stage(self) -> None:
        lib = mock.Mock()
        lib.llama_sampler_init_temp.return_value = 21
        lib.llama_sampler_init_dist.return_value = 22
        _add_default_sampler_stages(lib, 7, 0.7)
        lib.llama_sampler_init_temp.assert_called_once_with(0.7)
        lib.llama_sampler_init_dist.assert_called_once_with(LLAMA_DEFAULT_SEED)
        lib.llama_sampler_chain_add.assert_has_calls([mock.call(7, 21), mock.call(7, 22)])
        lib.llama_sampler_init_greedy.assert_not_called()

    def test_missing_stage_fails_loudly_without_a_half_built_chain(self) -> None:
        lib = mock.Mock()
        lib.llama_sampler_init_temp.return_value = 21
        lib.llama_sampler_init_dist.return_value = None
        with self.assertRaises(RuntimeError):
            _add_default_sampler_stages(lib, 7, 0.7)
        lib.llama_sampler_chain_add.assert_not_called()
        lib.llama_sampler_free.assert_called_once_with(21)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_client_rejects_positive_temperature_with_mtp(self, _mocked_lib) -> None:
        with self.assertRaises(ValueError):
            NativeLlamaClient(
                NativeRuntimePropsTests._paths(None),
                NativeClientConfig(temperature=0.7, use_mtp_experimental=True),
            )

    def test_greedy_sampler_hash_is_unchanged(self) -> None:
        self.assertEqual(
            _canonical_sampler_hash(0.0),
            hashlib.sha256(
                b'{"min_p":0.0,"repeat_penalty":1.0,"sampler":"greedy","temperature":0.0,"top_k":1,"top_p":1.0}'
            ).hexdigest()[:16],
        )
        self.assertNotEqual(_canonical_sampler_hash(0.7), _canonical_sampler_hash(0.0))


if __name__ == "__main__":
    unittest.main()
