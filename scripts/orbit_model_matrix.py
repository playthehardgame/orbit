#!/usr/bin/env python3
"""Live model matrix: inspect every registry model that exists on this host.

For each registry entry the script finds the local GGUF, loads only its
vocabulary and metadata (no weights), and checks that the detected native
profile is verified and is the profile the registry declares. Models that are
not on disk are reported MISSING and do not fail the run unless --require-all
is given.

Behavioural qualification (chat, tools, route reuse) needs a running server
and stays with `scripts/orbit_qualify.py`:

    python3 scripts/orbit_model_matrix.py --build-bin <llama build bin>
    python3 scripts/orbit_qualify.py --base-url ... --profile <profile id> ...

Exit status: 0 all present models match, 1 a mismatch, 2 setup problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.model_discovery import discover_models  # noqa: E402
from orbit.native_llama.model_registry import load_registry  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-bin", type=Path, required=True, help="llama.cpp build bin directory")
    parser.add_argument("--models-dir", type=Path, help="model store (default: configured store)")
    parser.add_argument("--hf-cache", type=Path, help="Hugging Face cache root")
    parser.add_argument("--require-all", action="store_true", help="fail when a registry model is missing")
    parser.add_argument("--json", dest="json_out", type=Path, help="write the matrix as JSON")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.build_bin.is_dir():
        print(f"build bin not found: {args.build_bin}", file=sys.stderr)
        return 2
    manifests = load_registry()
    result = discover_models(
        models_dir=args.models_dir,
        hf_cache=args.hf_cache,
        build_bin=args.build_bin,
        manifests=manifests,
    )
    if result.metadata_inspections == 0 and any(r.local == "AVAILABLE" for r in result.rows):
        print("native inspector unavailable: no GGUF was inspected", file=sys.stderr)
        return 2

    by_id = {m.id: m for m in manifests}
    rows: list[dict[str, str]] = []
    failed = False
    for row in result.rows:
        manifest = by_id.get(row.model_id or "")
        if row.local != "AVAILABLE":
            status = "MISSING" if row.local == "MISSING" else row.local
            failed = failed or (args.require_all or row.local == "INCOMPLETE")
        elif row.support == "VERIFIED":
            status = "PASS"
        else:
            status = "FAIL"
            failed = True
        rows.append(
            {
                "model": row.model,
                "profile": manifest.profile_id if manifest else "",
                "local": row.local,
                "support": row.support,
                "status": status,
                "path": row.path_or_action,
            }
        )
    width = max((len(r["model"]) for r in rows), default=5)
    for r in rows:
        print(f"{r['status']:<8} {r['model']:<{width}}  {r['local']:<10} {r['support']}")
    passed = sum(r["status"] == "PASS" for r in rows)
    print(f"\n{passed}/{len(rows)} registry models present and verified")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
