"""Platform-class tuning rules: pure, injected inputs, no host access."""
from __future__ import annotations

import unittest

from orbit.native_server.host_class import (
    DECODE_CAP_CONSUMER,
    HostClass,
    detect_host_class,
    recommend,
)
from orbit.native_server.server_profile import (
    HostTopology,
    detect_topology,
    heuristic_profile,
    thread_candidates,
)

INTEL_FLAGS = "flags : fpu avx2 fma avx_vnni hypervisor\nvendor_id : GenuineIntel\n"
ULTRA = "Intel(R) Core(TM) Ultra 7 155H"


def classify(**kw):
    base = dict(cpuinfo_text=INTEL_FLAGS, osrelease_text="6.6.0", cgroup_cpu_max_text="max 100000",
                affinity=0, reader=lambda path: None, system="linux", machine="x86_64")
    base.update(kw)
    return detect_host_class(**base)


class DetectHostClassTests(unittest.TestCase):
    def test_wsl2_is_recognised_and_topology_untrusted(self):
        host = classify(osrelease_text="6.6.87.2-microsoft-standard-WSL2", cpu_model=ULTRA,
                        physical_cores=11, logical_cpus=22)
        self.assertEqual(host.environment, "wsl2")
        self.assertFalse(host.topology_trusted)
        self.assertEqual(host.hybrid, "suspected")
        self.assertIn("avx2", host.isa)

    def test_sysfs_hybrid_split_is_measured_not_guessed(self):
        files = {"/sys/devices/cpu_core/cpus": "0-11", "/sys/devices/cpu_atom/cpus": "12-21",
                 "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list": "0-1"}
        host = classify(reader=files.get, cpu_model=ULTRA, physical_cores=16, logical_cpus=22)
        self.assertEqual((host.hybrid, host.performance_cores), ("yes", 6))

    def test_cgroup_quota_and_affinity_become_cpu_limit(self):
        self.assertEqual(classify(cgroup_cpu_max_text="200000 100000").cpu_limit, 2.0)
        self.assertEqual(classify(cgroup_cpu_max_text="200000 100000").environment, "container")
        host = classify(affinity=4, logical_cpus=16, physical_cores=8)
        self.assertEqual(host.cpu_limit, 4.0)

    def test_non_hybrid_desktop_and_server(self):
        desk = classify(cpu_model="AMD Ryzen 7 7840U", physical_cores=8, logical_cpus=16,
                        cpuinfo_text="flags : avx2 avx512f\nvendor_id : AuthenticAMD\n")
        self.assertEqual((desk.vendor, desk.hybrid, desk.server_class), ("amd", "no", False))
        server = classify(cpu_model="AMD EPYC 9654", physical_cores=96, logical_cpus=192)
        self.assertTrue(server.server_class)

    def test_arm_big_little_uses_cpu_capacity(self):
        files = {f"/sys/devices/system/cpu/cpu{i}/cpu_capacity": str(1024 if i < 4 else 400)
                 for i in range(8)}
        host = classify(reader=files.get, machine="aarch64", cpuinfo_text="Features : fp asimd\n",
                        logical_cpus=8, physical_cores=8)
        self.assertEqual((host.arch, host.hybrid, host.performance_cores), ("arm64", "yes", 4))

    def test_macos_apple_silicon_uses_perflevel(self):
        host = classify(system="darwin", machine="arm64", cpuinfo_text="",
                        sysctl=lambda name: 8 if name == "hw.perflevel0.physicalcpu" else 0)
        self.assertEqual((host.vendor, host.hybrid, host.performance_cores), ("apple", "yes", 8))

    def test_failure_degrades_to_unknown(self):
        def boom(path):
            raise RuntimeError("x")
        self.assertFalse(detect_host_class(reader=boom, cpuinfo_text=None).known)


class RecommendTests(unittest.TestCase):
    def rec(self, host, physical, logical, ram_gib=32):
        return recommend(physical_cores=physical, logical_cpus=logical,
                         total_ram_mib=ram_gib * 1024, host=host)

    def test_decode_threads_are_capped_below_core_count(self):
        host = HostClass(known=True, hybrid="no")
        rec = self.rec(host, 16, 32)
        self.assertEqual(rec.decode_threads, DECODE_CAP_CONSUMER)
        self.assertEqual(rec.batch_threads, 16)

    def test_hybrid_uses_performance_cores_for_both_phases(self):
        host = HostClass(known=True, hybrid="yes", performance_cores=6)
        rec = self.rec(host, 16, 22)
        self.assertEqual((rec.decode_threads, rec.batch_threads), (6, 6))

    def test_container_limit_wins_over_core_count(self):
        host = HostClass(known=True, cpu_limit=3.5)
        self.assertEqual(self.rec(host, 16, 32).batch_threads, 3)

    def test_small_machines_stay_conservative(self):
        rec = self.rec(HostClass(known=True), 8, 16, ram_gib=8)
        self.assertEqual((rec.batch, rec.ubatch), (128, 64))
        self.assertLessEqual(rec.decode_threads, 4)

    def test_server_class_gets_larger_batch_and_decode_ceiling(self):
        host = HostClass(known=True, server_class=True)
        rec = self.rec(host, 64, 128, ram_gib=256)
        self.assertEqual((rec.batch, rec.ubatch, rec.decode_threads), (512, 256, 16))

    def test_candidates_are_bounded_and_contain_the_picks(self):
        host = HostClass(known=True, hybrid="yes", performance_cores=6)
        rec = self.rec(host, 16, 22)
        self.assertTrue({rec.decode_threads, rec.batch_threads} <= set(rec.candidates))
        self.assertLessEqual(len(rec.candidates), 8)


class WiringTests(unittest.TestCase):
    def test_unknown_host_keeps_the_legacy_heuristic_exactly(self):
        topo = HostTopology(physical_cores=8, logical_cpus=16, total_ram_mib=32 * 1024)
        profile = heuristic_profile(topo)
        self.assertEqual((profile.threads, profile.threads_batch), (8, 8))

    def test_known_host_drives_heuristic_and_candidates(self):
        host = HostClass(known=True, hybrid="yes", performance_cores=6)
        topo = HostTopology(physical_cores=16, logical_cpus=22, total_ram_mib=32 * 1024, host=host)
        profile = heuristic_profile(topo)
        self.assertEqual((profile.threads, profile.threads_batch), (6, 6))
        self.assertIn(6, thread_candidates(topo))

    def test_injected_texts_never_classify_the_live_host(self):
        topo = detect_topology(meminfo_text="MemTotal: 1024 kB\n", cpuinfo_text="")
        self.assertFalse(topo.host.known)


if __name__ == "__main__":
    unittest.main()
