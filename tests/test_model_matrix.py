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

from pathlib import Path
import unittest

from orbit.native_llama import client as native_client
from orbit.native_llama import model_profiles
from orbit.native_llama.model_profiles import (
    VERIFIED_NATIVE_MODEL_IDENTITIES,
    verified_native_model_identity,
)
from orbit.native_llama.model_registry import load_registry

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
                self.assertTrue(
                    _squash(manifest.display_name) in _squash(README),
                    f"README support table is missing {manifest.display_name!r}",
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
        checkpoints = "\n".join(
            p.read_text(encoding="utf-8") for p in (ROOT / "docs" / "checkpoints").glob("*.md")
        )
        for profile_id in native_client.ROLLING_ROUTE_BOUNDARY_PROFILE_IDS:
            with self.subTest(profile=profile_id):
                self.assertIn("rolling route checkpoint", checkpoints.lower())
                identity = self.identities[profile_id]
                self.assertIn(identity.model_name.split()[0], checkpoints)

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


if __name__ == "__main__":
    unittest.main()
