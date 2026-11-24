"""Offline consistency matrix across every verified native model.

The registry (`model_registry.json`), the verified identities in
`model_profiles.py`, the rolling-route sets in `client.py` and the README table
are four hand-maintained lists. Adding a model touches all of them, and a miss
is silent at runtime. These tests pin that the lists agree, per model, so a new
or changed profile fails here instead of in the field.

Nothing here loads a model. The live counterpart, which inspects real GGUF
files, is `scripts/orbit_model_matrix.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from orbit.native_llama import client as native_client
from orbit.native_llama import model_profiles
from orbit.native_llama.model_profiles import (
    VERIFIED_NATIVE_MODEL_IDENTITIES,
    verified_native_model_identity,
)
from orbit.native_llama.model_registry import load_registry
from orbit.qualification.fixtures import load_fixture_set

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _squash(text: str) -> str:
    """Compare display names ignoring case, spaces and hyphens."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == ".")


def _profile_id_constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(model_profiles).items()
        if name.endswith("_PROFILE_ID") and isinstance(value, str)
    }


class ModelMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifests = load_registry()
        self.identities = {i.profile_id: i for i in VERIFIED_NATIVE_MODEL_IDENTITIES}

    def test_matrix_is_not_empty(self) -> None:
        self.assertGreaterEqual(len(self.manifests), 15)

    def test_registry_ids_and_profiles_are_unique_per_entry(self) -> None:
        ids = [m.id for m in self.manifests]
        self.assertEqual(len(ids), len(set(ids)), "duplicate registry id")
        profiles = [m.profile_id for m in self.manifests]
        # Granite and LFM keep one profile per quantization family, so a
        # profile may back several files only if the registry says so; today
        # every entry owns its own profile.
        self.assertEqual(len(profiles), len(set(profiles)), "duplicate registry profile_id")
        self.assertEqual(len(self.identities), len(VERIFIED_NATIVE_MODEL_IDENTITIES))

    def test_every_registry_entry_has_a_verified_identity(self) -> None:
        for manifest in self.manifests:
            with self.subTest(model=manifest.id):
                identity = verified_native_model_identity(manifest.profile_id)
                self.assertIsNotNone(identity, f"{manifest.profile_id} has no verified identity")
                self.assertEqual(identity.architecture, manifest.architecture)

    def test_every_verified_identity_is_in_the_registry(self) -> None:
        registry_profiles = {m.profile_id for m in self.manifests}
        self.assertEqual(set(self.identities), registry_profiles)

    def test_every_profile_constant_has_an_identity(self) -> None:
        for name, value in _profile_id_constants().items():
            with self.subTest(constant=name):
                self.assertIn(value, self.identities)

    def test_registry_entries_declare_a_gguf_target(self) -> None:
        for manifest in self.manifests:
            with self.subTest(model=manifest.id):
                self.assertTrue(manifest.target.file.endswith(".gguf"))
                self.assertEqual(manifest.backend, "native-llama")
                if manifest.mmproj is not None:
                    self.assertTrue(manifest.mmproj.file.endswith(".gguf"))

    def test_readme_lists_every_registry_model(self) -> None:
        for manifest in self.manifests:
            with self.subTest(model=manifest.id):
                rows = [
                    line.split("|")[1]
                    for line in README.splitlines()
                    if line.startswith("|") and line.count("|") >= 3
                ]
                self.assertTrue(
                    any(
                        _squash(row) in _squash(manifest.display_name)
                        or _squash(manifest.display_name) in _squash(row)
                        for row in rows
                        if _squash(row) not in ("model", "")
                    ),
                    f"README support table has no row for {manifest.display_name!r}",
                )

    def test_rolling_route_sets_are_nested_and_known(self) -> None:
        rolling = native_client.ROLLING_ROUTE_PROFILE_IDS
        boundary = native_client.ROLLING_ROUTE_BOUNDARY_PROFILE_IDS
        post_final = native_client.POST_FINAL_ROUTE_SHADOW_PROFILE_IDS
        self.assertTrue(rolling <= set(self.identities), "rolling profile without identity")
        self.assertTrue(boundary <= rolling, "boundary capture outside the rolling set")
        self.assertTrue(post_final <= rolling, "post-final shadow outside the rolling set")
        self.assertFalse(
            boundary & post_final,
            "boundary capture and post-final shadow are mutually exclusive",
        )

    def test_boundary_profiles_have_a_checkpoint_document(self) -> None:
        docs = [
            p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "checkpoints").glob("*.md")
        ]
        for profile_id in native_client.ROLLING_ROUTE_BOUNDARY_PROFILE_IDS:
            with self.subTest(profile=profile_id):
                identity = self.identities[profile_id]
                # One document must carry both the model's full name and its
                # rolling-route qualification section, not just a stray word.
                self.assertTrue(
                    any(
                        identity.model_name in doc
                        and "rolling route checkpoint qualification" in doc.lower()
                        for doc in docs
                    ),
                    f"no checkpoint document qualifies {identity.model_name!r}",
                )

    def test_template_hash_constants_are_sha256_hex(self) -> None:
        for name, value in vars(model_profiles).items():
            if name.endswith("_TEMPLATE_SHA256") and isinstance(value, str):
                with self.subTest(constant=name):
                    self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_file_type_constants_match_quantization_names(self) -> None:
        names = {
            "15": "Q4_K_M",
            "7": "Q8_0",
            "18": "Q6_K",
            "20": "IQ2_XS",
            "31": "UD-IQ1_M",
            "25": "IQ4_NL",
        }
        for name, value in vars(model_profiles).items():
            if not (name.endswith("_VERIFIED_FILE_TYPE") and isinstance(value, str)):
                continue
            prefix = name[: -len("_VERIFIED_FILE_TYPE")]
            quant = getattr(model_profiles, f"{prefix}_VERIFIED_QUANTIZATION", None)
            if quant is None:
                continue
            with self.subTest(constant=name):
                self.assertEqual(names.get(value), quant)


class MatrixFixtureSetTests(unittest.TestCase):
    """qualification/fixtures/matrix-v1.json must track core-v1 and the registry."""

    def setUp(self) -> None:
        self.core = json.loads((ROOT / "qualification/fixtures/core-v1.json").read_text())
        self.matrix_path = ROOT / "qualification/fixtures/matrix-v1.json"
        self.matrix = json.loads(self.matrix_path.read_text())

    def test_matrix_fixture_set_loads(self) -> None:
        load_fixture_set(str(self.matrix_path))

    def test_matrix_covers_every_verified_profile_on_every_fixture(self) -> None:
        expected = {i.profile_id for i in VERIFIED_NATIVE_MODEL_IDENTITIES}
        for fixture in self.matrix["fixtures"]:
            with self.subTest(fixture=fixture["name"]):
                self.assertEqual(set(fixture["profiles"]), expected)
                self.assertEqual(len(fixture["profiles"]), len(expected))

    def test_matrix_differs_from_core_only_in_profiles(self) -> None:
        def strip(data):
            return [{k: v for k, v in f.items() if k != "profiles"} for f in data["fixtures"]]

        self.assertEqual(strip(self.matrix), strip(self.core))
        self.assertEqual(self.matrix["schema_version"], self.core["schema_version"])

    def test_core_fixture_set_stays_frozen_to_its_four_profiles(self) -> None:
        for fixture in self.core["fixtures"]:
            self.assertEqual(len(fixture["profiles"]), 4)


if __name__ == "__main__":
    unittest.main()
