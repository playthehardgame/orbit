"""Structural ratchets for the Orbit package.

These tests freeze the *current* architecture so regressions fail loudly while
existing debt does not block work. They are pure static analysis (``ast`` and
text search over ``src/orbit``): no native build, model or corpus is needed.

When a ratchet fails because you removed debt, shrink the allowlist. When it
fails because you added debt, either fix the dependency or document why the
allowlist must grow.
"""
from __future__ import annotations

import ast
import re
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "orbit"

# Known cycles that only exist through function-level imports.
KNOWN_LAZY_CYCLES = {
    frozenset({"orbit.native_llama.client", "orbit.native_llama.client_status"}),
    frozenset({"orbit.runtime.analysis_ioc_proof", "orbit.runtime.analysis_ioc_xor"}),
    frozenset({"orbit.runtime.chat", "orbit.runtime.environments"}),
}

# Package -> packages it must never import (module level or lazily), except the
# explicit edges below. ``runtime`` is the transport-agnostic core: it may use
# the backend interface but never a concrete native stack or a front end.
FORBIDDEN_LAYER_EDGES = {
    "runtime": {"terminal", "native_llama", "native_server", "qualification"},
    "backend": {"terminal", "native_server"},
    "native_llama": {"terminal", "runtime", "native_server", "qualification"},
    "native_server": {"terminal", "qualification"},
    "terminal": {"qualification"},
}
# Existing reverse edges that are accepted debt (importer, imported).
ALLOWED_REVERSE_EDGES = {
    ("orbit.native_llama.model_discovery", "orbit.terminal.theme"),
    ("orbit.native_server.app", "orbit.terminal.theme"),
    ("orbit.native_llama.client", "orbit.runtime.history_serialization"),
}

# Functions above this many lines are frozen; new ones must not appear and
# none of the frozen ones may grow.
LONG_FUNCTION_LINES = 300
FROZEN_LONG_FUNCTIONS = {
    "orbit.runtime.tool_loop.run_tool_loop": 1809,
    "orbit.runtime.analysis_runtime.AnalysisRuntime._run_autonomous_locked": 1199,
    "orbit.native_llama.model_profiles.detect_native_model_profile": 626,
    "orbit.runtime.analysis_runtime.AnalysisRuntime.step": 516,
    "orbit.runtime.chat.ChatRuntime.ask_auto": 402,
    "orbit.native_llama.client.NativeLlamaClient._complete_prompt_standard": 367,
    "orbit.runtime.environments.FinalFromToolEnvironment.answer": 314,
    "orbit.native_server.app.run_server": 312,
}

# ORBIT_* names read in src/ that no README/docs/AGENTS text mentions yet.
# Document a variable instead of adding it here.
KNOWN_UNDOCUMENTED_ENV = {
    "ORBIT_ANALYSIS_COMPLETION_SHADOW", "ORBIT_ANALYSIS_RETAIN_DIR", "ORBIT_BASE_URL",
    "ORBIT_INFERENCE_AUDIT_SHADOW", "ORBIT_LLAMA_BUILD_FLAGS", "ORBIT_LLAMA_PATCHSET_HASH",
    "ORBIT_LLAMA_SOURCE_TREE_HASH", "ORBIT_LLAMA_UPSTREAM_COMMIT", "ORBIT_LLAMA_UPSTREAM_TAG",
    "ORBIT_MINICPM5_ROUTE_PREFIX_REUSE", "ORBIT_MUTATIVE_TOOL_CALL_MAX_TOKENS",
    "ORBIT_ORNITH_ROUTE_PREFIX_REUSE", "ORBIT_SKIP_SERVER_READINESS", "ORBIT_TEST_MAX_TOKENS",
    "ORBIT_TEST_TIMEOUT", "ORBIT_TOOLS_SOURCE", "ORBIT_TOOLS_SRC",
}


def _modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in PKG.rglob("*.py"):
        if "vendor" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def _graphs() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    modules = _modules()
    top: dict[str, set[str]] = {}
    lazy: dict[str, set[str]] = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        top_level = {id(node) for node in tree.body}
        eager: set[str] = set()
        deferred: set[str] = set()
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    parts = parts[: len(parts) - (node.level - 1)]
                    base = ".".join(parts + ([node.module] if node.module else []))
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
            for target in targets:
                if target in modules and target != name:
                    (eager if id(node) in top_level else deferred).add(target)
        top[name], lazy[name] = eager, deferred
    return top, lazy


def _sccs(graph: dict[str, set[str]]) -> list[frozenset[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[frozenset[str]] = []
    counter = [0]

    def visit(node: str) -> None:
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for other in graph.get(node, ()):
            if other not in index:
                visit(other)
                low[node] = min(low[node], low[other])
            elif other in on_stack:
                low[node] = min(low[node], index[other])
        if low[node] == index[node]:
            component = []
            while True:
                other = stack.pop()
                on_stack.discard(other)
                component.append(other)
                if other == node:
                    break
            if len(component) > 1:
                result.append(frozenset(component))

    for node in graph:
        if node not in index:
            visit(node)
    return result


def _long_functions() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, path in _modules().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(body: list[ast.stmt], prefix: str) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    size = node.end_lineno - node.lineno + 1
                    if size > LONG_FUNCTION_LINES:
                        sizes[f"{name}.{prefix}{node.name}"] = size

        visit(tree.body, "")
    return sizes


def _layer(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


class ArchitectureRatchetTests(unittest.TestCase):
    def test_module_level_imports_have_no_cycles(self) -> None:
        top, _ = _graphs()
        self.assertEqual(_sccs(top), [], "module-level import cycle introduced")

    def test_only_known_cycles_hide_behind_lazy_imports(self) -> None:
        top, lazy = _graphs()
        merged = {name: top[name] | lazy[name] for name in top}
        unexpected = [c for c in _sccs(merged) if c not in KNOWN_LAZY_CYCLES]
        self.assertEqual(unexpected, [], "new import cycle (possibly via a lazy import)")
        stale = KNOWN_LAZY_CYCLES - set(_sccs(merged))
        self.assertEqual(stale, set(), "cycle fixed: remove it from KNOWN_LAZY_CYCLES")

    def test_layer_dependencies_do_not_point_the_wrong_way(self) -> None:
        top, lazy = _graphs()
        found: set[tuple[str, str]] = set()
        for importer in top:
            for imported in top[importer] | lazy[importer]:
                if _layer(imported) in FORBIDDEN_LAYER_EDGES.get(_layer(importer), set()):
                    found.add((importer, imported))
        self.assertEqual(found - ALLOWED_REVERSE_EDGES, set(), "new reverse layer dependency")
        self.assertEqual(ALLOWED_REVERSE_EDGES - found, set(), "debt paid: shrink ALLOWED_REVERSE_EDGES")

    def test_long_functions_are_frozen(self) -> None:
        sizes = _long_functions()
        new = set(sizes) - set(FROZEN_LONG_FUNCTIONS)
        self.assertEqual(new, set(), f"new function over {LONG_FUNCTION_LINES} lines")
        grown = {k: (sizes[k], FROZEN_LONG_FUNCTIONS[k]) for k in sizes if sizes[k] > FROZEN_LONG_FUNCTIONS[k]}
        self.assertEqual(grown, {}, "a frozen long function grew")
        gone = set(FROZEN_LONG_FUNCTIONS) - set(sizes)
        self.assertEqual(gone, set(), "function shrank below the limit: remove it from FROZEN_LONG_FUNCTIONS")

    def test_new_environment_variables_are_documented(self) -> None:
        used: set[str] = set()
        for path in _modules().values():
            used |= {m for m in re.findall(r"ORBIT_[A-Z0-9_]+", path.read_text(encoding="utf-8")) if not m.endswith("_")}
        documented: set[str] = set()
        # Audit records quote variable names as evidence; they are not user documentation.
        audit = ROOT / "docs" / "audit"
        for doc in [ROOT / "README.md", ROOT / "AGENTS.md",
                    *(d for d in (ROOT / "docs").rglob("*.md") if audit not in d.parents)]:
            documented |= set(re.findall(r"ORBIT_[A-Z0-9_]+", doc.read_text(encoding="utf-8")))
        undocumented = used - documented
        self.assertEqual(undocumented - KNOWN_UNDOCUMENTED_ENV, set(), "document the new ORBIT_* variable")
        self.assertEqual(KNOWN_UNDOCUMENTED_ENV - undocumented, set(), "now documented/removed: shrink the allowlist")


if __name__ == "__main__":
    unittest.main()
