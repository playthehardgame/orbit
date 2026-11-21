"""Only abandoned, own, old analysis workspaces are swept."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from orbit.runtime.analysis_runtime import AnalysisWorkspace

DAY = 86400


def _make(base: Path, name: str, age_days: float, *, inner_age_days: float | None = None) -> Path:
    root = base / name
    (root / "source").mkdir(parents=True)
    (root / "work").mkdir()
    (root / "work" / "f.txt").write_text("x")
    past = time.time() - age_days * DAY
    inner = past if inner_age_days is None else time.time() - inner_age_days * DAY
    for sub in ("source", "work"):
        os.utime(root / sub, (inner, inner))
    os.utime(root, (past, past))
    return root


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_old_workspace_is_removed_and_recent_is_kept(self):
        old = _make(self.base, "orbit-analysis-session-old", 10)
        new = _make(self.base, "orbit-analysis-session-new", 1)
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base), 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_active_inner_directory_protects_an_old_root(self):
        live = _make(self.base, "orbit-analysis-session-live", 30, inner_age_days=0.1)
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base), 0)
        self.assertTrue(live.exists())

    def test_other_names_files_and_symlinks_are_never_touched(self):
        other = _make(self.base, "unrelated-dir", 99)
        plain = self.base / "orbit-analysis-session-file"
        plain.write_text("x")
        os.utime(plain, (time.time() - 99 * DAY,) * 2)
        target = _make(self.base, "target-dir", 99)
        link = self.base / "orbit-analysis-session-link"
        link.symlink_to(target, target_is_directory=True)
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base), 0)
        self.assertTrue(other.exists() and plain.exists() and target.exists() and link.is_symlink())

    def test_zero_or_negative_age_disables_the_sweep(self):
        old = _make(self.base, "orbit-analysis-session-old", 365)
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base, max_age_days=0), 0)
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base, max_age_days=-3), 0)
        self.assertTrue(old.exists())

    def test_missing_base_does_not_raise(self):
        self.assertEqual(AnalysisWorkspace.sweep_stale(base=self.base / "nope"), 0)


if __name__ == "__main__":
    unittest.main()
