from __future__ import annotations

import unittest

from orbit.native_server.server_profile import (
    HostTopology, MemoryGuardRefusal, apply_memory_guard, memory_guard_mode,
)

GIB = 1024 ** 3
SMALL = HostTopology(total_ram_mib=8 * 1024, available_ram_mib=6 * 1024)


class MemoryGuardTests(unittest.TestCase):
    def test_mode_parsing_defaults_to_warn(self):
        self.assertEqual(memory_guard_mode({}), "warn")
        for raw, want in (("refuse", "refuse"), (" OFF ", "off"), ("block", "warn"), ("", "warn")):
            self.assertEqual(memory_guard_mode({"ORBIT_MEMORY_GUARD": raw}), want, raw)

    def test_warn_returns_advisory_and_never_raises(self):
        self.assertIn("memory advisory", apply_memory_guard(SMALL, model_bytes=40 * GIB, ctx_tokens=8192, environ={}))

    def test_off_is_silent(self):
        self.assertIsNone(apply_memory_guard(SMALL, model_bytes=40 * GIB, ctx_tokens=8192,
                                             environ={"ORBIT_MEMORY_GUARD": "off"}))

    def test_refuse_blocks_only_when_it_does_not_fit(self):
        env = {"ORBIT_MEMORY_GUARD": "refuse"}
        with self.assertRaises(MemoryGuardRefusal) as ctx:
            apply_memory_guard(SMALL, model_bytes=40 * GIB, ctx_tokens=8192, environ=env)
        self.assertIn("ORBIT_MEMORY_GUARD=warn", str(ctx.exception))
        self.assertIsNone(apply_memory_guard(SMALL, model_bytes=1 * GIB, ctx_tokens=8192, environ=env))

    def test_unknowns_never_refuse(self):
        self.assertIsNone(apply_memory_guard(HostTopology(), model_bytes=40 * GIB, ctx_tokens=8192,
                                             environ={"ORBIT_MEMORY_GUARD": "refuse"}))


if __name__ == "__main__":
    unittest.main()
