from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import unittest

from orbit.native_llama.model_registry import (
    default_models_dir,
    get_manifest,
    local_model_path,
    load_registry,
    resolve_model,
)


def _registry_data() -> dict:
    return {
        "version": 1,
        "models": [
            {
                "id": "test-gemma",
                "display_name": "Test Gemma",
                "profile_id": "orbit-gemma4-native-v1",
                "backend": "native-llama",
                "architecture": "gemma4",
                "target": {
                    "repo": "target/repo",
                    "file": "target.gguf",
                    "cache_glob": "target/snapshots/*/target.gguf",
                },
                "mmproj": {
                    "repo": "target/repo",
                    "file": "mmproj.gguf",
                    "cache_glob": "target/snapshots/*/mmproj.gguf",
                },
                "mtp": {
                    "enabled_by_default": True,
                    "required": False,
                    "spec_type": "draft-mtp",
                    "repo": "draft/repo",
                    "file": "MTP/draft.gguf",
                    "cache_glob": "draft/snapshots/*/MTP/draft.gguf",
                },
            }
        ],
    }


def _write_registry(path: Path, data: dict | None = None) -> None:
    path.write_text(json.dumps(data or _registry_data()), encoding="utf-8")


class NativeModelRegistryTests(unittest.TestCase):
    def test_packaged_registry_contains_only_current_verified_models(self) -> None:
        manifests = load_registry()

        self.assertEqual(
            [(item.display_name, item.profile_id) for item in manifests],
            [
                ("Gemma 4 26B-A4B", "orbit-gemma4-native-v1"),
                ("Qwen 3.6 35B-A3B", "orbit-qwen36-native-v1"),
                ("Qwen 3.8 27B", "orbit-qwen38-native-v1"),
                ("Ornith 1.5 35B-A3B", "orbit-ornith15-native-v1"),
                ("Qwen3-Coder 30B-A3B", "orbit-qwen3-coder-native-v1"),
                ("Qwen 3.8 Flash Next", "orbit-qwen38-flash-next-native-v1"),
                ("LFM2.5 8B-A1B", "orbit-lfm25-native-v1"),
                ("LFM2.5 8B-A1B Q8_0", "orbit-lfm25-q8-native-v1"),
                ("LFM2.5 8B-A1B UD-Q6_K", "orbit-lfm25-q6-native-v1"),
                ("MiniCPM5 2B", "orbit-minicpm5-native-v1"),
                ("Granite 4.2 3B", "orbit-granite42-native-v1"),
                ("Granite 4.2 8B", "orbit-granite42-8b-native-v1"),
                ("Underdog Saluki 27B 1.0", "orbit-qwen38-saluki-native-v1"),
                ("Qwen3.8 9B Distill", "orbit-qwen38-9b-distill-native-v1"),
                ("ScopeGuard-2 4B", "orbit-scope-guard-v2-4b-native-v1"),
                ("NousResearch Arete 27B", "orbit-qwen38-arete-native-v1"),
                ("Qwen3.8 27B TurboFCFusion Heretic NEO-CODER", "orbit-qwen38-turbo-fcfusion-native-v1"),
                ("Gemma 4 E4B Heretic", "orbit-gemma4-e4b-heretic-native-v1"),
            ],
        )
        self.assertEqual(
            [(item.profile_id, item.target.repo, item.target.file) for item in manifests],
            [
                (
                    "orbit-gemma4-native-v1",
                    "ggml-org/gemma-4-26B-A4B-it-GGUF",
                    "gemma-4-26B-A4B-it-Q4_0.gguf",
                ),
                (
                    "orbit-qwen36-native-v1",
                    "ggml-org/Qwen3.6-35B-A3B-GGUF",
                    "Qwen3.6-35B-A3B-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen38-native-v1",
                    "unsloth/Qwen3.8-27B-GGUF",
                    "Qwen3.8-27B-Q4_K_M.gguf",
                ),
                (
                    "orbit-ornith15-native-v1",
                    "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
                    "Ornith-1.5-35B-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen3-coder-native-v1",
                    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen38-flash-next-native-v1",
                    "unsloth/Qwen3.8-Flash-Next-GGUF",
                    "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf",
                ),
                (
                    "orbit-lfm25-native-v1",
                    "unsloth/LFM2.5-8B-A1B-GGUF",
                    "LFM2.5-8B-A1B-UD-Q4_K_M.gguf",
                ),
                (
                    "orbit-lfm25-q8-native-v1",
                    "unsloth/LFM2.5-8B-A1B-GGUF",
                    "LFM2.5-8B-A1B-Q8_0.gguf",
                ),
                (
                    "orbit-lfm25-q6-native-v1",
                    "unsloth/LFM2.5-8B-A1B-GGUF",
                    "LFM2.5-8B-A1B-UD-Q6_K.gguf",
                ),
                (
                    "orbit-minicpm5-native-v1",
                    "bartowski/MiniCPM5-2B-GGUF",
                    "MiniCPM5-2B-Q4_K_M.gguf",
                ),
                (
                    "orbit-granite42-native-v1",
                    "ibm-granite/granite-4.2-3b-GGUF",
                    "granite-4.2-3b-Q4_K_M.gguf",
                ),
                (
                    "orbit-granite42-8b-native-v1",
                    "ibm-granite/granite-4.2-8b-GGUF",
                    "granite-4.2-8b-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen38-saluki-native-v1",
                    "ConwayResearch/Underdog-Saluki-27B-1.0",
                    "Underdog-Saluki-27B-1.0-IQ2-mix.gguf",
                ),
                (
                    "orbit-qwen38-9b-distill-native-v1",
                    "empero-ai/Qwen3.8-9B-Distill-GGUF",
                    "Qwen3.8-9B-Q4_K_M.gguf",
                ),
                (
                    "orbit-scope-guard-v2-4b-native-v1",
                    "mradermacher/scope-guard-v2-4B-q-2609-GGUF",
                    "scope-guard-v2-4B-q-2609.Q8_0.gguf",
                ),
                (
                    "orbit-qwen38-arete-native-v1",
                    "bartowski/NousResearch_Arete-27B-GGUF",
                    "NousResearch_Arete-27B-Q4_K_M.gguf",
                ),
                (
                    "orbit-qwen38-turbo-fcfusion-native-v1",
                    "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF",
                    "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q4_K_M.gguf",
                ),
                (
                    "orbit-gemma4-e4b-heretic-native-v1",
                    "llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF",
                    "gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
                ),
            ],
        )

    def test_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            _write_registry(registry_path)

            manifests = load_registry(registry_path)

        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].id, "test-gemma")
        self.assertEqual(manifests[0].display_name, "Test Gemma")
        self.assertEqual(manifests[0].profile_id, "orbit-gemma4-native-v1")
        self.assertEqual(manifests[0].backend, "native-llama")
        self.assertEqual(manifests[0].architecture, "gemma4")
        self.assertIsNotNone(manifests[0].mtp)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            registry_path.write_text('{"version":1,"version":1,"models":[]}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate model registry key"):
                load_registry(registry_path)

    def test_unknown_registry_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            data = _registry_data()
            data["models"][0]["unexpected"] = True
            _write_registry(registry_path, data)

            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_registry(registry_path)

    def test_bad_version_missing_fields_and_wrong_types_are_rejected(self) -> None:
        cases = []
        bad_version = _registry_data()
        bad_version["version"] = True
        cases.append(bad_version)
        missing_field = _registry_data()
        del missing_field["models"][0]["target"]
        cases.append(missing_field)
        wrong_type = _registry_data()
        wrong_type["models"] = {}
        cases.append(wrong_type)
        for data in cases:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                _write_registry(registry_path, data)

                with self.assertRaises(ValueError):
                    load_registry(registry_path)

    def test_duplicate_ids_names_profiles_and_downloads_are_rejected(self) -> None:
        mutations = {
            "model id": {},
            "model display name": {"id": "other"},
            "model profile": {"id": "other", "display_name": "Other"},
            "download": {
                "id": "other",
                "display_name": "Other",
                "profile_id": "orbit-qwen36-native-v1",
                "architecture": "qwen35moe",
            },
        }
        for reason, changes in mutations.items():
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                data = _registry_data()
                duplicate = json.loads(json.dumps(data["models"][0]))
                duplicate.update(changes)
                data["models"].append(duplicate)
                _write_registry(registry_path, data)

                with self.assertRaisesRegex(ValueError, f"duplicate (model registry )?{reason}"):
                    load_registry(registry_path)

    def test_incorrect_profile_architecture_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.json"
            data = _registry_data()
            data["models"][0]["architecture"] = "qwen35moe"
            _write_registry(registry_path, data)

            with self.assertRaisesRegex(ValueError, "invalid profile/architecture mapping"):
                load_registry(registry_path)

    def test_unsafe_download_paths_and_globs_are_rejected(self) -> None:
        for key, value in (("file", "../model.gguf"), ("cache_glob", "../../**/model.gguf")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                registry_path = Path(tmp) / "registry.json"
                data = _registry_data()
                data["models"][0]["target"][key] = value
                _write_registry(registry_path, data)

                with self.assertRaises(ValueError):
                    load_registry(registry_path)

    def test_falls_back_to_no_mtp_when_optional_draft_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            models_dir = root / "models"
            manifest = get_manifest("test-gemma", registry_path=registry_path)
            target = local_model_path(manifest.target, models_dir=models_dir)
            mmproj = local_model_path(manifest.mmproj, models_dir=models_dir)
            target.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            resolved = resolve_model(manifest, models_dir=models_dir, hf_cache=root)

        self.assertEqual(resolved.target_path, target)
        self.assertEqual(resolved.mmproj_path, mmproj)
        self.assertTrue(resolved.multimodal_available)
        self.assertIsNone(resolved.draft_mtp_path)
        self.assertFalse(resolved.mtp_available)
        self.assertEqual(resolved.fallback_reason, "draft-mtp-missing")

    def test_mtp_available_when_target_and_draft_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            models_dir = root / "models"
            manifest = get_manifest("test-gemma", registry_path=registry_path)
            target = local_model_path(manifest.target, models_dir=models_dir)
            mmproj = local_model_path(manifest.mmproj, models_dir=models_dir)
            draft = local_model_path(manifest.mtp, models_dir=models_dir)
            target.parent.mkdir(parents=True)
            draft.parent.mkdir(parents=True)
            target.write_text("target", encoding="utf-8")
            mmproj.write_text("mmproj", encoding="utf-8")
            draft.write_text("draft", encoding="utf-8")
            resolved = resolve_model(manifest, models_dir=models_dir, hf_cache=root)

        self.assertEqual(resolved.target_path, target)
        self.assertEqual(resolved.mmproj_path, mmproj)
        self.assertEqual(resolved.draft_mtp_path, draft)
        self.assertTrue(resolved.multimodal_available)
        self.assertTrue(resolved.mtp_available)
        self.assertIsNone(resolved.fallback_reason)

    def test_default_models_dir_uses_project_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/orbit").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='orbit'\n", encoding="utf-8")

            self.assertEqual(default_models_dir(root / "src"), root / "models")

    def test_missing_target_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            _write_registry(registry_path)
            manifest = get_manifest("test-gemma", registry_path=registry_path)

            with self.assertRaises(FileNotFoundError):
                resolve_model(manifest, hf_cache=root)


if __name__ == "__main__":
    unittest.main()


class Qwen38RegistryTests(unittest.TestCase):
    def test_ornith15_entry_resolves_with_isolated_profile(self) -> None:
        manifest = get_manifest("ornith15-35b-a3b-q4-k-m")
        self.assertEqual(manifest.profile_id, "orbit-ornith15-native-v1")
        self.assertEqual(manifest.architecture, "qwen35moe")
        # Shares Qwen3.6's architecture but must never share its identity.
        self.assertNotEqual(manifest.profile_id, "orbit-qwen36-native-v1")

    def test_ornith15_entry_declares_no_mtp_or_mmproj(self) -> None:
        manifest = get_manifest("ornith15-35b-a3b-q4-k-m")
        self.assertIsNone(manifest.mtp)
        self.assertIsNone(manifest.mmproj)

    def test_qwen38_entry_resolves_with_isolated_profile(self) -> None:
        manifest = get_manifest("qwen38-27b-q4-k-m")
        self.assertEqual(manifest.profile_id, "orbit-qwen38-native-v1")
        self.assertEqual(manifest.architecture, "qwen35")

    def test_qwen38_entry_declares_no_mtp_or_mmproj(self) -> None:
        manifest = get_manifest("qwen38-27b-q4-k-m")
        self.assertIsNone(getattr(manifest, "mtp", None))
        self.assertIsNone(getattr(manifest, "mmproj", None))

    def test_existing_model_ids_still_resolve(self) -> None:
        for model_id, profile_id in (
            ("qwen36-35b-a3b-q4-k-m", "orbit-qwen36-native-v1"),
            ("qwen3-coder-30b-a3b-instruct-q4-k-m", "orbit-qwen3-coder-native-v1"),
            ("gemma4-26b-a4b-it-q40", "orbit-gemma4-native-v1"),
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(get_manifest(model_id).profile_id, profile_id)


class RegistryIntegrityPinTests(unittest.TestCase):
    _HEX = set("0123456789abcdef")
    _PACKAGED = Path(__file__).resolve().parents[1] / "src/orbit/native_llama/model_registry.json"

    def _specs(self):
        data = json.loads(self._PACKAGED.read_text(encoding="utf-8"))
        for model in data["models"]:
            for kind in ("target", "mmproj", "mtp"):
                if kind in model:
                    yield f"{model['id']}.{kind}", model[kind]

    def test_packaged_pins_are_well_formed_and_registry_loads(self) -> None:
        self.assertTrue(load_registry())
        pinned = 0
        for label, spec in self._specs():
            with self.subTest(spec=label):
                # size and sha256 are pinned together or not at all.
                self.assertEqual("size" in spec, "sha256" in spec)
                if "size" in spec:
                    pinned += 1
                    self.assertIs(type(spec["size"]), int)
                    self.assertGreater(spec["size"], 0)
                    self.assertEqual(len(spec["sha256"]), 64)
                    self.assertLessEqual(set(spec["sha256"]), self._HEX)
        self.assertGreater(pinned, 0)

    def _write_registry(self, data: dict) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = tmp / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_target_and_mtp_accept_valid_pins(self) -> None:
        data = _registry_data()
        data["models"][0]["target"].update(size=10, sha256="a" * 64)
        data["models"][0]["mtp"].update(size=10, sha256="b" * 64)
        self.assertEqual(len(load_registry(self._write_registry(data))), 1)

    def test_malformed_pins_are_rejected(self) -> None:
        for pins in ({"size": 0}, {"size": "5"}, {"size": True}, {"sha256": "A" * 64}, {"sha256": "ab"}):
            for kind in ("target", "mtp"):
                with self.subTest(pins=pins, kind=kind):
                    data = _registry_data()
                    data["models"][0][kind].update(pins)
                    with self.assertRaises(ValueError):
                        load_registry(self._write_registry(data))
