#!/usr/bin/env python3
"""Write a CycloneDX 1.5 SBOM for this checkout (stdlib only, deterministic).

Components: Orbit itself, its Python dependencies (none at runtime; the optional
telemetry extra is listed as optional), the vendored llama.cpp source (from
LLAMA_PROVENANCE.json when present) and every registry model file with its
pinned size and SHA-256. Usage: scripts/generate_sbom.py [output.json]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama.model_registry import registry_pins  # noqa: E402


def _pyproject() -> dict:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
    version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    extras = re.findall(r'"([A-Za-z0-9_.-]+)>=([0-9.]+)"', text)
    return {"name": name.group(1) if name else "orbit", "version": version.group(1) if version else "0",
            "extras": extras}


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_sbom() -> dict:
    meta = _pyproject()
    components: list[dict] = []
    for pkg, minimum in sorted(set(meta["extras"])):
        components.append({"type": "library", "name": pkg, "version": f">={minimum}",
                           "scope": "optional", "purl": f"pkg:pypi/{pkg}"})
    provenance = next(iter(ROOT.glob("**/LLAMA_PROVENANCE.json")), None)
    if provenance is not None and ".claude" not in provenance.parts:
        data = json.loads(provenance.read_text(encoding="utf-8"))
        components.append({
            "type": "library", "name": "llama.cpp", "version": str(data.get("upstream_tag") or data.get("upstream_commit", "")),
            "hashes": [{"alg": "SHA-256", "content": data["source_tree_sha256"]}] if data.get("source_tree_sha256") else [],
            "properties": [{"name": "upstream_commit", "value": str(data.get("upstream_commit", ""))},
                           {"name": "patchset_sha256", "value": str(data.get("patchset_sha256", ""))}],
        })
    for (repo, file), (size, digest) in sorted(registry_pins().items()):
        component = {"type": "machine-learning-model", "name": f"{repo}/{file}",
                     "purl": f"pkg:huggingface/{repo}"}
        if digest:
            component["hashes"] = [{"alg": "SHA-256", "content": digest}]
        if size:
            component["properties"] = [{"name": "size_bytes", "value": str(size)}]
        components.append(component)
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
        "metadata": {"component": {"type": "application", "name": meta["name"], "version": meta["version"],
                                   "properties": [{"name": "git_commit", "value": _commit()}]}},
        "components": components,
    }


def main(argv: list[str]) -> int:
    text = json.dumps(build_sbom(), indent=2, sort_keys=True) + "\n"
    if argv:
        Path(argv[0]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
