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
import struct
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


# --- string-array fold family (javascript-obfuscator shape) -----------------
#
# The oracle is the plain-text ProgID-position argument below; the encoders here
# (custom-order base64 over UTF-8 RC4 ciphertext) are written independently of
# `orbit.runtime`. The JS is a real, readable decoder, never executed by tests.

_B64_ORDER = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="


def _rc4(text: str, key: str) -> str:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + ord(key[i % len(key)])) % 256
        s[i], s[j] = s[j], s[i]
    i = j = 0
    out = []
    for ch in text:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(chr(ord(ch) ^ s[(s[i] + s[j]) % 256]))
    return "".join(out)


def _b64_custom(raw: bytes) -> str:
    out, val, bits = [], 0, 0
    for b in raw:
        val = (val << 8) | b
        bits += 8
        while bits >= 6:
            bits -= 6
            out.append(_B64_ORDER[(val >> bits) & 0x3F])
    if bits:
        out.append(_B64_ORDER[(val << (6 - bits)) & 0x3F])
    return "".join(out)


def _array_element(plain: str, key: str) -> str:
    return _b64_custom(_rc4(plain, key).encode("utf-8"))


def _js_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def stringarray_fold() -> tuple[str, list[str], list[str]]:
    chunks = [("orbit-synthetic-", "k3y!"), ("stringarray ", "P@ss"), (C2, "w0rd")]
    value = "".join(c for c, _ in chunks)
    offset = 0x1F4
    array = [_array_element(c, k) for c, k in chunks] + [_array_element("noise", "zz")]
    lits = ",".join(_js_str(e) for e in array)
    q = ",".join(f"dec({hex(i + offset)},{_js_str(k)})" for i, (_, k) in enumerate(chunks))
    source = (
        f"function arrfn(){{var data=[{lits}];"
        "arrfn=function(){return data;};return arrfn();}\n"
        f"function dec(a,q){{a=a-{hex(offset)};var X=arrfn();var V=X[a];\n"
        f"var m=function(C){{var S='{_B64_ORDER}';var w='';var n=0,v=0;\n"
        "for(var i=0;i<C.length;i++){var d=S.indexOf(C.charAt(i));if(d<0){break;}\n"
        "v=(v<<6)|d;n+=6;if(n>=8){n-=8;w+='%'+('00'+((v>>n)&255).toString(16)).slice(-2);}}\n"
        "return decodeURIComponent(w);};\n"
        "var p=function(C,S){var s=[];for(var i=0;i<256;i++){s[i]=i;}\n"
        "for(var i=0,j=0;i<256;i++){j=(j+s[i]+S.charCodeAt(i%S.length))%256;"
        "var t=s[i];s[i]=s[j];s[j]=t;}\n"
        "var i=0,j=0,r='';for(var k=0;k<C.length;k++){i=(i+1)%256;j=(j+s[i])%256;"
        "var t=s[i];s[i]=s[j];s[j]=t;r+=String.fromCharCode(C.charCodeAt(k)^s[(s[i]+s[j])%256]);}\n"
        "return r;};\n"
        "return p(m(V),q);}\n"
        "var d1=dec,d2=dec;\n"
        f"function selfdef(){{var q=[{q}];\n"
        "var g='ZZ';if(g===g){} return q;}\n"
        "function H(d,F){var C=selfdef(),S=C[d];return S;}\n"
        "var R=H,B=H;\n"
        "var a={'ap':function(x,y){return x(y);},'cc':function(x,y){return x+y;}};\n"
        "var obj=new ActiveXObject(a['cc'](a['cc'](a['ap'](R,0x0),H(0x1)),H(0x2)));\n"
    )
    return source, [value], ["js_stringarray_fold"]


# --- Office/OLE macro container ---------------------------------------------
#
# A minimal Compound File Binary (v3, 512-byte sectors) holding `VBA/dir` and
# one module stream `VBA/ThisDocument`, all in the mini stream. Built with
# struct only: no Office, no olefile. Streams are MS-OVBA CompressedContainers
# made of literal tokens only (a valid, if unambitious, compressor), so a module
# must fit a single chunk (<= 3640 source bytes). The oracle is the plain-text
# VBA source below. The macro only builds a marker string; it executes nothing.

_FREE, _END, _FATSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD
_MODULE = "ThisDocument"
_CACHE = b"ORBITSYN" * 2  # stand-in PerformanceCache preceding the source


def _ovba_literal_compress(data: bytes) -> bytes:
    if len(data) > 3640:
        raise ValueError("literal-only OVBA chunk limited to 3640 bytes")
    body = bytearray()
    for i in range(0, len(data), 8):
        body.append(0)
        body += data[i:i + 8]
    header = ((len(body) + 2 - 3) & 0x0FFF) | 0xB000
    return b"\x01" + struct.pack("<H", header) + bytes(body)


def _dir_stream() -> bytes:
    name = _MODULE.encode("ascii")
    uni = _MODULE.encode("utf-16-le")
    rec = lambda i, b: struct.pack("<HI", i, len(b)) + b  # noqa: E731
    return (
        rec(0x0003, struct.pack("<H", 1252))
        + rec(0x0019, name)
        + rec(0x001A, name)
        + rec(0x0032, uni)
        + rec(0x0031, struct.pack("<I", len(_CACHE)))
        + rec(0x0021, b"")
    )


def _dir_entry(name, typ, left=_FREE, right=_FREE, child=_FREE, start=0, size=0) -> bytes:
    raw = name.encode("utf-16-le")
    nlen = len(raw) + 2 if name else 0
    e = raw.ljust(64, b"\x00") + struct.pack("<H", nlen) + bytes([typ, 1])
    e += struct.pack("<III", left, right, child) + b"\x00" * 16 + b"\x00" * 4 + b"\x00" * 16
    e += struct.pack("<IQ", start, size)
    assert len(e) == 128
    return e


def _pad(b: bytes, unit: int) -> bytes:
    return b + b"\x00" * (-len(b) % unit)


def build_cfb(module_source: bytes) -> bytes:
    dir_c = _ovba_literal_compress(_dir_stream())
    mod = _CACHE + _ovba_literal_compress(module_source)
    # Mini stream: each stream padded to 64-byte mini sectors, placed in order.
    streams, mini, ms = [], b"", 0
    mini_fat: list[int] = []
    for data in (dir_c, mod):
        n = (len(data) + 63) // 64
        streams.append((ms, len(data)))
        mini_fat += [ms + k + 1 for k in range(n - 1)] + [_END]
        mini += _pad(data, 64)
        ms += n
    if len(mini_fat) > 128:
        raise ValueError("module too large for the single miniFAT sector")
    mini_fat += [_FREE] * (128 - len(mini_fat))
    n_mini = len(mini) // 512 + (1 if len(mini) % 512 else 0)
    mini_sec = _pad(mini, 512)
    # Sectors: 0 FAT, 1 directory, 2 miniFAT, 3.. mini stream container.
    fat = [_FATSECT, _END, _END] + [4 + k for k in range(n_mini - 1)] + [_END]
    fat += [_FREE] * (128 - len(fat))
    entries = [
        _dir_entry("Root Entry", 5, child=1, start=3, size=len(mini)),
        _dir_entry("VBA", 1, child=2),
        _dir_entry("dir", 2, right=3, start=streams[0][0], size=streams[0][1]),
        _dir_entry(_MODULE, 2, start=streams[1][0], size=streams[1][1]),
    ]
    header = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 16
        + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\x00" * 6
        + struct.pack("<IIIIIIIII", 0, 1, 1, 0, 4096, 2, 1, _END, 0)
        + struct.pack("<109I", 0, *([_FREE] * 108))
    )
    assert len(header) == 512
    return (
        header + struct.pack("<128I", *fat) + b"".join(entries)
        + struct.pack("<128I", *mini_fat) + mini_sec
    )


def office_macro() -> tuple[bytes, list[str], list[str], dict]:
    payload = f'Debug.Print "orbit-synthetic-macro {C2}"'
    module = (
        f'Attribute VB_Name = "{_MODULE}"\r\n'
        "Function dec(ByVal a)\r\n"
        "    Dim k\r\n"
        "    k = 311\r\n"
        "    For Each e In a\r\n"
        "        s = s & Chr(e - k)\r\n"
        "    Next\r\n"
        "    dec = s\r\n"
        "End Function\r\n"
        "Sub AutoOpen()\r\n"
        f"    x = dec(Array({_offset_tokens(payload, 311)}))\r\n"
        "End Sub\r\n"
    )
    extras = {
        "container": "ole2-cfb",
        "expected_module_name": _MODULE,
        "expected_module_source": module,
        "expected_module_source_sha256": hashlib.sha256(module.encode("utf-8")).hexdigest(),
    }
    return build_cfb(module.encode("ascii")), [payload], ["vbscript_chr_offset"], extras


SAMPLES = {
    "syn_jscript_xor.js": jscript_xor,
    "syn_powershell_xor.ps1": powershell_xor,
    "syn_fromcharcode.js": fromcharcode,
    "syn_vbscript_chr.vbs": vbscript_chr,
    "syn_layered.vbs": layered_vbscript_powershell,
    "syn_stringarray.js": stringarray_fold,
    "syn_macro.doc": office_macro,
}


def build() -> dict[str, tuple[str | bytes, dict]]:
    built = {}
    for name, factory in SAMPLES.items():
        source, outputs, kinds, *extra = factory()
        data = source if isinstance(source, bytes) else source.encode("utf-8")
        built[name] = (
            source,
            {
                **(extra[0] if extra else {}),
                "sha256": hashlib.sha256(data).hexdigest(),
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
        data = source if isinstance(source, bytes) else source.encode("utf-8")
        (HERE / name).write_bytes(data)
        manifest["samples"][name] = entry
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
