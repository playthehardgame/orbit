#!/usr/bin/env python3
"""Deterministic generator for the synthetic, harmless analysis corpus.

Every sample is built here from plain text, so the expected decoded layers are
known independently of the production decoders under test (they are the inputs
to the encoders below, never outputs of `orbit.runtime`). Payloads only echo a
marker and mention a `.invalid` host (RFC 2606: guaranteed never to resolve).

Run `python3 tests/synthetic_corpus/generate.py` to rewrite the sample files and
`manifest.json`; `tests/test_synthetic_corpus.py` fails if they drift.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C2 = "http://synthetic.invalid/stage"


def _xor_tokens(text: str, key: int, delimiter: str) -> str:
    return delimiter.join(str(ord(c) ^ key) for c in text)


def _offset_tokens(text: str, constant: int) -> str:
    return ", ".join(str(ord(c) + constant) for c in text)


def jscript_xor() -> tuple[str, list[str], list[str]]:
    payload = f'WScript.Echo("orbit-synthetic-jscript " + "{C2}");'
    source = (
        "function dec(s, k, d) {\n"
        "    var out = '';\n"
        "    var parts = s.split(d);\n"
        "    for (var i = 0; i < parts.length; i++) {\n"
        "        out += String.fromCharCode(parts[i] ^ k);\n"
        "    }\n"
        "    return out;\n"
        "}\n"
        f'dec("{_xor_tokens(payload, 23, "-")}", 23, "-");\n'
    )
    return source, [payload], ["jscript_numeric_xor"]


def powershell_xor() -> tuple[str, list[str], list[str]]:
    payload = f"Write-Output 'orbit-synthetic-powershell {C2}'"
    tokens = ",".join(str(ord(c) ^ 41) for c in payload)
    source = (
        f"$data='{tokens}';$k=41;$out='';\n"
        "$parts=$data -split ',';\n"
        "foreach($t in $parts){$out=$out+[char]($t -bxor $k);}\n"
    )
    return source, [payload], ["powershell_numeric_xor"]


def fromcharcode() -> tuple[str, list[str], list[str]]:
    payload = f'WScript.Echo("orbit-synthetic-fromcharcode {C2}");'
    constant = 77
    args = ",".join(f"{ord(c) + constant}-K" for c in payload)
    source = f"var K={constant};\nvar x=String.fromCharCode({args});\n"
    return source, [payload], ["js_fromcharcode_offset"]


def vbscript_chr() -> tuple[str, list[str], list[str]]:
    payload = f'WScript.Echo "orbit-synthetic-vbscript {C2}"'
    source = (
        "Function dec(ByVal a)\n"
        "    Dim k\n"
        "    k = 311\n"
        "    For Each e In a\n"
        "        s = s & Chr(e - k)\n"
        "    Next\n"
        "    dec = s\n"
        "End Function\n"
        f"x = dec(Array({_offset_tokens(payload, 311)}))\n"
    )
    return source, [payload], ["vbscript_chr_offset"]


def layered_vbscript_powershell() -> tuple[str, list[str], list[str]]:
    inner = f"orbit-synthetic-inner {C2}"
    ps_text = (
        "function dd($z){$h=500;foreach($p in $z){$o+=[char]($p - $h)}}\n"
        f"x = dd @({_offset_tokens(inner, 500)})\n"
    )
    source = (
        "Function dec(ByVal a)\n"
        "    Dim k\n"
        "    k = 300\n"
        "    For Each e In a\n"
        "        s = s & Chr(e - k)\n"
        "    Next\n"
        "    dec = s\n"
        "End Function\n"
        f"c = dec(Array({_offset_tokens(ps_text, 300)}))\n"
    )
    return source, [ps_text, inner], ["vbscript_chr_offset", "powershell_chr_offset"]


SAMPLES = {
    "syn_jscript_xor.js": jscript_xor,
    "syn_powershell_xor.ps1": powershell_xor,
    "syn_fromcharcode.js": fromcharcode,
    "syn_vbscript_chr.vbs": vbscript_chr,
    "syn_layered.vbs": layered_vbscript_powershell,
}


def build() -> dict[str, tuple[str, dict]]:
    built = {}
    for name, factory in SAMPLES.items():
        source, outputs, kinds = factory()
        built[name] = (
            source,
            {
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "expected_stage_outputs": outputs,
                "expected_stage_kinds": kinds,
                "expected_stage_output_sha256": [
                    hashlib.sha256(o.encode("utf-8")).hexdigest() for o in outputs
                ],
            },
        )
    return built


def main() -> None:
    manifest = {"schema_version": 1, "c2_host": "synthetic.invalid", "samples": {}}
    for name, (source, entry) in build().items():
        (HERE / name).write_bytes(source.encode("utf-8"))
        manifest["samples"][name] = entry
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
