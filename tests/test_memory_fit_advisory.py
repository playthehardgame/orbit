"""The memory advisory warns about unfit models and never refuses or guesses."""
from __future__ import annotations

import unittest

from orbit.native_server import app
from orbit.native_server.server_profile import HostTopology, memory_fit_advisory

GIB = 1024 ** 3


def _topo(total_gib: int, available_gib: int) -> HostTopology:
    return HostTopology(total_ram_mib=total_gib * 1024, available_ram_mib=available_gib * 1024)


class MemoryFitAdvisoryTests(unittest.TestCase):
    def test_fitting_model_is_silent(self):
        self.assertIsNone(memory_fit_advisory(_topo(32, 28), model_bytes=8 * GIB, ctx_tokens=8192))

    def test_oversized_model_warns_with_remedies(self):
        text = memory_fit_advisory(_topo(16, 12), model_bytes=21 * GIB, ctx_tokens=8192)
        self.assertIn("memory advisory", text)
        self.assertIn("ORBIT_CPU_REPACK=off", text)
        self.assertIn("--ctx", text)

    def test_large_context_can_tip_a_borderline_model(self):
        topo = _topo(16, 11)
        self.assertIsNone(memory_fit_advisory(topo, model_bytes=8 * GIB, ctx_tokens=8192))
        self.assertIsNotNone(memory_fit_advisory(topo, model_bytes=8 * GIB, ctx_tokens=262144))

    def test_unknowns_never_warn(self):
        self.assertIsNone(memory_fit_advisory(HostTopology(), model_bytes=50 * GIB, ctx_tokens=8192))
        self.assertIsNone(memory_fit_advisory(_topo(8, 6), model_bytes=0, ctx_tokens=8192))

    def test_available_unknown_falls_back_to_total(self):
        topo = HostTopology(total_ram_mib=8 * 1024, available_ram_mib=0)
        self.assertIsNotNone(memory_fit_advisory(topo, model_bytes=20 * GIB, ctx_tokens=8192))

    def test_note_is_printed_once(self):
        app._NOTED.clear()
        seen = []
        original = app._startup_note
        app._startup_note = seen.append
        try:
            app._note_once("same")
            app._note_once("same")
            app._note_once(None)
        finally:
            app._startup_note = original
        self.assertEqual(seen, ["same"])


if __name__ == "__main__":
    unittest.main()
