"""Characterization golden for ``detect_native_model_profile``.

Records the exact output of the detector for a broad deterministic corpus of
(metadata, template) inputs: every verified identity, one near-miss per
condition of each identity (key removed, key corrupted, whitespace/case
variants), every identity metadata crossed with every identity template,
unknown architectures and empty metadata. The golden lives in
tests/fixtures/model_profile_characterization.json.

Pinned template digests are the sha256 of templates Orbit does not ship, so
the corpus patches every ``*_TEMPLATE_SHA256`` constant to the digest of a
synthetic per-constant template; the detector reads those module globals at
call time. To regenerate (only for an intentional behaviour change):
ORBIT_REGEN_PROFILE_GOLDEN=1 python -m unittest tests.test_model_profile_characterization
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from orbit.native_llama import model_profiles as mp
from orbit.native_llama.model_profiles import detect_native_model_profile

GOLDEN = Path(__file__).parent / "fixtures" / "model_profile_characterization.json"


def _plain(value):
    if isinstance(value, enum.Enum):
        return f"enum:{value.name}={value.value!r}"
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def _profile_json(profile) -> str:
    return json.dumps(
        {"type": type(profile).__name__, "fields": _plain(dataclasses.asdict(profile))},
        sort_keys=True,
        separators=(",", ":"),
    )


HASH_CONSTANTS = sorted(
    n for n in dir(mp) if n.endswith("_TEMPLATE_SHA256") and isinstance(getattr(mp, n), str)
)


def _template_for(constant: str) -> str:
    return f"synthetic-template::{constant}"


def _patched_hashes() -> dict[str, str]:
    return {n: hashlib.sha256(_template_for(n).encode("utf-8")).hexdigest() for n in HASH_CONSTANTS}


def _meta(arch, name, file_type, pre, **extra):
    meta = {
        "general.architecture": arch,
        "general.name": name,
        "general.file_type": file_type,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
    }
    meta.update(extra)
    return meta


_T = "tokenizer.ggml."
# (identity id, hash constant, metadata); one entry per matcher site.
IDENTITIES = [
    ("gemma4-e4b-heretic", "GEMMA4_E4B_HERETIC_TEMPLATE_SHA256", {
        "general.architecture": "gemma4", "general.name": mp.GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME,
        "general.file_type": mp.GEMMA4_E4B_HERETIC_VERIFIED_FILE_TYPE, _T + "model": "gemma4",
        _T + "bos_token_id": "2", _T + "eos_token_id": "1", _T + "padding_token_id": "0",
        "gemma4.context_length": "131072", "gemma4.block_count": "42",
        "gemma4.embedding_length_per_layer_input": "256"}),
    ("gemma4", None, {
        "general.architecture": "gemma4", "general.name": "gemma-4-26B-A4B-it",
        "general.file_type": "15", _T + "model": "gemma4"}),
    ("qwen36", "QWEN36_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35moe", mp.QWEN36_VERIFIED_MODEL_NAME, mp.QWEN36_VERIFIED_FILE_TYPE, "qwen35")),
    ("ornith15", "ORNITH15_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35moe", mp.ORNITH15_VERIFIED_MODEL_NAME, mp.ORNITH15_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046",
         "qwen35moe.context_length": "262144", "qwen35moe.block_count": "41",
         "qwen35moe.expert_count": "256", "qwen35moe.expert_used_count": "8"})),
    ("qwen38", "QWEN38_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.QWEN38_VERIFIED_MODEL_NAME, mp.QWEN38_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046",
         "qwen35.context_length": "262144", "qwen35.block_count": "65"})),
    ("qwen38-saluki", "QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.QWEN38_SALUKI_VERIFIED_MODEL_NAME, mp.QWEN38_SALUKI_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046", _T + "padding_token_id": "248044",
         _T + "add_bos_token": "false", "qwen35.context_length": "262144", "qwen35.block_count": "64"})),
    ("qwen38-arete", "QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.QWEN38_ARETE_VERIFIED_MODEL_NAME, mp.QWEN38_ARETE_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046", _T + "padding_token_id": "248044",
         _T + "add_bos_token": "false", "qwen35.context_length": "262144", "qwen35.block_count": "64"})),
    ("qwen38-turbo-fc", "QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.QWEN38_TURBO_FC_VERIFIED_MODEL_NAME, mp.QWEN38_TURBO_FC_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046", _T + "padding_token_id": "248044",
         _T + "add_bos_token": "false", "qwen35.context_length": "262144", "qwen35.block_count": "64"})),
    ("qwen38-9b-distill", "QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.QWEN38_9B_DISTILL_VERIFIED_MODEL_NAME, mp.QWEN38_9B_DISTILL_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "eos_token_id": "248046", _T + "padding_token_id": "248044",
         _T + "add_bos_token": "false", "qwen35.context_length": "262144", "qwen35.block_count": "33"})),
    ("scope-guard", "SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen35", mp.SCOPE_GUARD_VERIFIED_MODEL_NAME, mp.SCOPE_GUARD_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "eos_token_id": "248046", _T + "padding_token_id": "248055",
         _T + "add_bos_token": "false", "qwen35.context_length": "262144", "qwen35.block_count": "33"})),
    ("qwen38-flash-next", "QWEN38_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen4exp", mp.QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME, mp.QWEN38_FLASH_NEXT_VERIFIED_FILE_TYPE, "qwen35", **{
         _T + "bos_token_id": "248044", _T + "eos_token_id": "248046",
         "qwen4exp.context_length": "262144", "qwen4exp.block_count": "48",
         "qwen4exp.expert_count": "512", "qwen4exp.expert_used_count": "10"})),
    ("lfm25", "LFM25_OFFICIAL_TEMPLATE_SHA256",
     _meta("lfm2moe", mp.LFM25_VERIFIED_MODEL_NAME, mp.LFM25_VERIFIED_FILE_TYPE, "lfm2", **{
         _T + "bos_token_id": "124894", _T + "eos_token_id": "124900", _T + "padding_token_id": "124893",
         "lfm2moe.context_length": "128000", "lfm2moe.block_count": "24",
         "lfm2moe.expert_count": "32", "lfm2moe.expert_used_count": "4"})),
    ("inkling2-mini", "INKLING2_MINI_OFFICIAL_TEMPLATE_SHA256",
     _meta("inkling2_loop", mp.INKLING2_MINI_VERIFIED_MODEL_NAME, mp.INKLING2_MINI_VERIFIED_FILE_TYPE,
           "inkling2", **{
         _T + "eos_token_id": "200006", _T + "padding_token_id": "199999",
         "inkling2_loop.block_count": "24", "inkling2_loop.expert_count": "128",
         "inkling2_loop.expert_used_count": "8", "inkling2_loop.loop.count": "3"})),
    ("qwen3-coder", "QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256",
     _meta("qwen3moe", mp.QWEN3_CODER_VERIFIED_MODEL_NAME, mp.QWEN3_CODER_VERIFIED_FILE_TYPE, "qwen2", **{
         "general.quantization_version": "2", _T + "add_bos_token": "false",
         _T + "eos_token_id": "151645", _T + "padding_token_id": "151654",
         "qwen3moe.context_length": "262144", "qwen3moe.expert_count": "128",
         "qwen3moe.expert_used_count": "8"})),
    ("minicpm5", "MINICPM5_OFFICIAL_TEMPLATE_SHA256",
     _meta("llama", mp.MINICPM5_VERIFIED_MODEL_NAME, mp.MINICPM5_VERIFIED_FILE_TYPE, "minicpm5", **{
         _T + "bos_token_id": "0", _T + "eos_token_id": "1", _T + "padding_token_id": "1",
         _T + "add_bos_token": "false", "llama.context_length": "131072", "llama.block_count": "42"})),
    ("granite42", "GRANITE42_OFFICIAL_TEMPLATE_SHA256",
     _meta("granite", mp.GRANITE42_VERIFIED_MODEL_NAME, mp.GRANITE42_VERIFIED_FILE_TYPE, "granite-docling", **{
         _T + "bos_token_id": "100283", _T + "eos_token_id": "100257", _T + "padding_token_id": "100257",
         "granite.context_length": "131072", "granite.block_count": "40"})),
    ("granite42-8b", "GRANITE42_OFFICIAL_TEMPLATE_SHA256",
     _meta("granite", mp.GRANITE42_8B_VERIFIED_MODEL_NAME, "15", "granite-docling", **{
         _T + "bos_token_id": "100283", _T + "eos_token_id": "100257", _T + "padding_token_id": "100257",
         "granite.context_length": "131072", "granite.block_count": "40"})),
]


def _cases() -> list[tuple[str, dict, str]]:
    cases: list[tuple[str, dict, str]] = []

    def add(cid, meta, template):
        cases.append((cid, dict(meta), template))

    templates = {"none": "", "wrong": "some other template", "ws": " "}
    for const in HASH_CONSTANTS:
        templates["hash:" + const] = _template_for(const)

    for ident, const, meta in IDENTITIES:
        good = _template_for(const) if const else "any template"
        add(f"{ident}/base", meta, good)
        for tkey, tval in templates.items():
            add(f"{ident}/template={tkey}", meta, tval)
        for key, value in meta.items():
            without = {k: v for k, v in meta.items() if k != key}
            add(f"{ident}/missing={key}", without, good)
            add(f"{ident}/bogus={key}", {**meta, key: "bogus"}, good)
            add(f"{ident}/empty={key}", {**meta, key: ""}, good)
            add(f"{ident}/padded={key}", {**meta, key: f"  {value}\t"}, good)
            add(f"{ident}/upper={key}", {**meta, key: value.upper()}, good)
            add(f"{ident}/title={key}", {**meta, key: value.title()}, good)
            add(f"{ident}/lower={key}", {**meta, key: value.lower()}, good)
        for extra, val in (
            ("qwen35.nextn_predict_layers", "1"),
            ("qwen35.nextn_predict_layers", ""),
            ("gemma4.embedding_length_per_layer_input", "0"),
            ("gemma4.embedding_length_per_layer_input", "256"),
            ("gemma4.embedding_length_per_layer_input", " "),
            (_T + "add_bos_token", "False"),
            (_T + "add_bos_token", "TRUE"),
            (_T + "add_bos_token", ""),
            ("unrelated.key", "x"),
        ):
            add(f"{ident}/extra={extra}={val!r}", {**meta, extra: val}, good)
        for ft in ("7", "15", "18", "20", "31", "99", "", "Q4_K_M"):
            add(f"{ident}/file_type={ft!r}", {**meta, "general.file_type": ft}, good)
        for arch in ("gemma4", "qwen35", "qwen35moe", "qwen4exp", "lfm2moe", "qwen3moe", "llama",
                     "granite", "mamba", "", " GEMMA4 "):
            add(f"{ident}/arch={arch!r}", {**meta, "general.architecture": arch}, good)
        for other_ident, other_const, _ in IDENTITIES:
            add(f"{ident}/with-template-of={other_ident}", meta,
                _template_for(other_const) if other_const else "any template")

    for pre in ("qwen35", "qwen2", "lfm2", "minicpm5", "granite-docling", "", "other"):
        for tm in ("gpt2", "gemma4", "llama", "", "GPT2"):
            for arch in ("qwen35", "gemma4", "qwen35moe", "llama", "granite"):
                add(f"shape/{arch}/{tm}/{pre}", _meta(arch, "x", "15", pre, **{_T + "model": tm}), "t")

    for name in ("gemma-4-26B-A4B-it", "gemma-4-E4B", mp.GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME):
        for per_layer in (None, "", "0", "1", "256"):
            for tm in ("gemma4", "gpt2", ""):
                for arch in ("gemma4", "gemma3", ""):
                    meta = {"general.architecture": arch, "general.name": name,
                            "general.file_type": "15", _T + "model": tm}
                    if per_layer is not None:
                        meta["gemma4.embedding_length_per_layer_input"] = per_layer
                    add(f"gemma-shape/{name}/{per_layer}/{tm}/{arch}", meta, "g")

    lfm_meta = dict(IDENTITIES[11][2])
    for ft in sorted(mp.LFM25_VERIFIED_FILE_TYPES) + ["99", ""]:
        add(f"lfm25/ft={ft!r}", {**lfm_meta, "general.file_type": ft},
            _template_for("LFM25_OFFICIAL_TEMPLATE_SHA256"))
    for ft in ("7", "15", "18", "20", "99", ""):
        for idx in (14, 15):
            add(f"granite{idx}/ft={ft!r}", {**IDENTITIES[idx][2], "general.file_type": ft},
                _template_for("GRANITE42_OFFICIAL_TEMPLATE_SHA256"))

    for arch in ("", "unknown", "llama", "mamba", "qwen2", "gemma3", "GEMMA4", " Qwen35 "):
        for tmpl in ("", "template"):
            add(f"unknown/{arch!r}/{tmpl!r}", {"general.architecture": arch}, tmpl)
    add("empty/empty", {}, "")
    add("empty/template", {}, "template")
    add("empty/unicode-template", {}, "tëmplate ☃\n\t ")
    add("empty/name-only", {"general.name": "  Name With Spaces  "}, "x")
    add("empty/ws-only", {"general.architecture": "   ", "general.name": "\t", "general.file_type": " "}, "x")
    return cases


def _run_all() -> dict:
    profiles: dict[str, int] = {}
    cases: dict[str, int] = {}
    ordered: list[str] = []
    with mock.patch.multiple(mp, **_patched_hashes()):
        outputs = [
            (cid, _profile_json(detect_native_model_profile(meta, tmpl)))
            for cid, meta, tmpl in _cases()
        ]
    # real pinned digests: no synthetic template can match them
    for ident, _const, meta in IDENTITIES:
        outputs.append((f"real-digests/{ident}", _profile_json(detect_native_model_profile(meta, "t"))))
    for cid, text in outputs:
        if text not in profiles:
            profiles[text] = len(ordered)
            ordered.append(text)
        assert cid not in cases, cid
        cases[cid] = profiles[text]
    return {"profiles": [json.loads(t) for t in ordered], "cases": cases}


class ModelProfileCharacterizationTests(unittest.TestCase):
    def test_golden_matches_live_output(self) -> None:
        live = _run_all()
        if os.environ.get("ORBIT_REGEN_PROFILE_GOLDEN") == "1":
            GOLDEN.write_text(
                json.dumps(live, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(sorted(golden["cases"]), sorted(live["cases"]))
        mismatches = [
            cid
            for cid in golden["cases"]
            if golden["profiles"][golden["cases"][cid]] != live["profiles"][live["cases"][cid]]
        ]
        self.assertEqual(mismatches[:10], [], f"{len(mismatches)} golden cases differ")

    def test_corpus_reaches_every_verified_identity(self) -> None:
        live = _run_all()
        reached = {p["fields"]["profile_id"] for p in live["profiles"] if p["fields"]["verified"]}
        for pid in (
            mp.GEMMA4_E4B_HERETIC_PROFILE_ID, mp.GEMMA4_PROFILE_ID, mp.QWEN36_PROFILE_ID,
            mp.ORNITH15_PROFILE_ID, mp.QWEN38_PROFILE_ID, mp.QWEN38_SALUKI_PROFILE_ID,
            mp.QWEN38_ARETE_PROFILE_ID, mp.QWEN38_TURBO_FC_PROFILE_ID,
            mp.QWEN38_9B_DISTILL_PROFILE_ID, mp.SCOPE_GUARD_PROFILE_ID,
            mp.QWEN38_FLASH_NEXT_PROFILE_ID, mp.LFM25_PROFILE_ID, mp.LFM25_Q8_PROFILE_ID,
            mp.LFM25_Q6_PROFILE_ID, mp.QWEN3_CODER_PROFILE_ID, mp.MINICPM5_PROFILE_ID,
            mp.GRANITE42_PROFILE_ID, mp.GRANITE42_8B_PROFILE_ID,
            mp.INKLING2_MINI_PROFILE_ID,
        ):
            self.assertIn(pid, reached)


if __name__ == "__main__":
    unittest.main()
