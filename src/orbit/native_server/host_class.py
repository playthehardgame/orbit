"""Platform-aware tuning rules: from "what kind of PC is this" to thread/batch advice.

`server_profile.detect_topology` reports counts. Counts alone mislead on exactly
the machines people actually run Orbit on:

* WSL2 and VMs publish synthetic ``core id`` values, so "physical cores" is
  half of the logical count whatever the silicon is;
* hybrid packages (Intel Core Ultra / 12th gen and later, Apple Silicon, ARM
  big.LITTLE) mix fast and slow cores, and ggml waits for the slowest thread;
* containers and cgroup quotas cap the CPU time the process may use no matter
  how many cores ``/proc/cpuinfo`` lists;
* token generation is memory-bandwidth bound, so beyond a handful of threads
  extra cores add contention, while prompt processing is compute bound and does
  scale with cores.

This module turns those facts into a `HostClass` and then into a
`Recommendation`. It is **pure** (every input is injectable, nothing is read at
import time) and **advisory**: the recommendation feeds only the heuristic tier
and the candidate list of the bounded calibration sweep. Precedence is
unchanged -- CLI, environment, user profile, qualified profile, cache and
measurement all still win. When nothing is known (`HostClass.known` is False)
the legacy rules apply untouched, so a machine we cannot read is never
"improved" by guesswork.

The numbers encode widely documented llama.cpp behaviour, not a benchmark of
any one PC: decode saturates early, prefill scales with performance cores. They
are a better starting point than "physical cores", and the sweep (`orbit server
--show-profile --recalibrate`) remains the way to turn them into a measurement.
"""
from __future__ import annotations

import math
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

# Decode (token generation) stops scaling long before the core count does, on
# anything with consumer memory. Server parts with many memory channels do keep
# scaling, so they get a higher ceiling.
DECODE_CAP_CONSUMER = 8
DECODE_CAP_SERVER = 16
SERVER_MIN_PHYSICAL = 32

# Intel packages that mix performance and efficiency cores: Core Ultra
# (Meteor/Arrow/Lunar Lake) and the 12th-gen-and-later Core i-series.
_HYBRID_NAME = re.compile(r"Core\(TM\)\s+Ultra|Core\(TM\)\s+i[3579]-1[2-9]\d{3}", re.IGNORECASE)
_SERVER_NAME = re.compile(r"Xeon|EPYC|Threadripper|Neoverse|Graviton|Ampere", re.IGNORECASE)

ISA_FLAGS = (
    "avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_tile", "amx_int8", "fma",
    "neon", "asimd", "asimddp", "i8mm", "sve",
)


@dataclass(frozen=True)
class HostClass:
    """Facts about the platform, with `known` False when nothing was read."""

    known: bool = False
    system: str = ""            # linux / darwin / windows
    arch: str = ""              # x86_64 / arm64 / other
    vendor: str = ""            # intel / amd / apple / arm / other
    environment: str = "bare"   # bare / wsl2 / vm / container
    isa: "tuple[str, ...]" = ()
    smt: bool = False
    hybrid: str = "no"          # yes (measured by sysfs/sysctl) / suspected / no
    performance_cores: int = 0  # physical performance cores; 0 = unknown
    cpu_limit: float = 0.0      # cgroup/affinity CPU ceiling in cores; 0 = none
    server_class: bool = False
    topology_trusted: bool = True


@dataclass(frozen=True)
class Recommendation:
    decode_threads: int
    batch_threads: int
    batch: int
    ubatch: int
    candidates: "tuple[int, ...]"
    rationale: "tuple[str, ...]" = field(default_factory=tuple)


def _parse_cpu_list(text: str) -> int:
    """Count CPUs in a kernel list such as ``0-11,16,18-19``."""
    total = 0
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            total += (int(hi) - int(lo) + 1) if hi else 1
        except ValueError:
            return 0
    return total


def _cgroup_cpu_limit(cpu_max_text: str | None) -> float:
    if not cpu_max_text:
        return 0.0
    parts = cpu_max_text.split()
    if len(parts) != 2 or parts[0] == "max":
        return 0.0
    try:
        quota, period = int(parts[0]), int(parts[1])
    except ValueError:
        return 0.0
    return quota / period if quota > 0 and period > 0 else 0.0


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _sysctl(name: str) -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2, check=False
        ).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def detect_host_class(
    *,
    cpu_model: str = "",
    physical_cores: int = 1,
    logical_cpus: int = 1,
    cpuinfo_text: str | None = None,
    osrelease_text: str | None = None,
    cgroup_cpu_max_text: str | None = None,
    affinity: int | None = None,
    reader: Callable[[str], str | None] = _read,
    system: str | None = None,
    machine: str | None = None,
    sysctl: Callable[[str], int] = _sysctl,
) -> HostClass:
    """Classify the platform. Never raises; unreadable facts stay unknown."""
    try:
        system = (system or platform.system()).lower()
        machine = (machine or platform.machine()).lower()
        arch = "x86_64" if machine in {"x86_64", "amd64"} else "arm64" if machine in {"aarch64", "arm64"} else machine
        cpuinfo = cpuinfo_text if cpuinfo_text is not None else (reader("/proc/cpuinfo") or "")
        flags_line = ""
        vendor_id = ""
        for line in cpuinfo.splitlines():
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key in {"flags", "features"} and not flags_line:
                flags_line = value
            elif key == "vendor_id" and not vendor_id:
                vendor_id = value.strip()
        flags = set(flags_line.split())
        isa = tuple(f for f in ISA_FLAGS if f in flags)

        model = cpu_model or ""
        if system == "darwin":
            vendor = "apple" if arch == "arm64" else "intel"
        elif "intel" in vendor_id.lower() or "intel" in model.lower():
            vendor = "intel"
        elif "amd" in vendor_id.lower() or "amd" in model.lower():
            vendor = "amd"
        elif arch == "arm64":
            vendor = "arm"
        else:
            vendor = "other"

        osrelease = (osrelease_text if osrelease_text is not None
                     else (reader("/proc/sys/kernel/osrelease") or "")).lower()
        if "microsoft" in osrelease or "wsl" in osrelease:
            environment = "wsl2"
        elif cgroup_cpu_max_text is not None and _cgroup_cpu_limit(cgroup_cpu_max_text) > 0:
            environment = "container"
        elif "hypervisor" in flags:
            environment = "vm"
        else:
            environment = "bare"

        cpu_limit = _cgroup_cpu_limit(
            cgroup_cpu_max_text if cgroup_cpu_max_text is not None else reader("/sys/fs/cgroup/cpu.max")
        )
        if affinity is None:
            try:
                affinity = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                affinity = 0
        if affinity and logical_cpus and affinity < logical_cpus:
            cpu_limit = min(cpu_limit, float(affinity)) if cpu_limit else float(affinity)

        smt = logical_cpus > physical_cores

        # Hybrid: prefer what the kernel/OS states, fall back to the CPU name.
        performance = 0
        hybrid = "no"
        core_cpus = reader("/sys/devices/cpu_core/cpus")
        atom_cpus = reader("/sys/devices/cpu_atom/cpus")
        if core_cpus and atom_cpus:
            hybrid = "yes"
            performance = _parse_cpu_list(core_cpus)
            # P cores carry SMT; the sysfs list is in logical CPUs.
            sibling = reader("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
            if sibling and _parse_cpu_list(sibling) > 1:
                performance = max(1, performance // _parse_cpu_list(sibling))
        elif system == "darwin" and arch == "arm64":
            perf = sysctl("hw.perflevel0.physicalcpu")
            if perf:
                hybrid, performance = "yes", perf
        elif arch == "arm64" and system == "linux":
            capacities = []
            for index in range(max(1, min(logical_cpus, 256))):
                value = reader(f"/sys/devices/system/cpu/cpu{index}/cpu_capacity")
                if value is None:
                    capacities = []
                    break
                try:
                    capacities.append(int(value.strip()))
                except ValueError:
                    capacities = []
                    break
            if capacities and len(set(capacities)) > 1:
                hybrid = "yes"
                performance = sum(1 for c in capacities if c >= max(capacities) * 0.7)
        if hybrid == "no" and vendor == "intel" and _HYBRID_NAME.search(model):
            hybrid = "suspected"

        server = bool(_SERVER_NAME.search(model)) or physical_cores >= SERVER_MIN_PHYSICAL
        # WSL2 and VMs invent core ids, so the physical count is only a guess
        # unless sysfs gave us a real sibling list.
        trusted = environment not in {"wsl2", "vm"}
        return HostClass(
            known=True, system=system, arch=arch, vendor=vendor, environment=environment,
            isa=isa, smt=smt, hybrid=hybrid, performance_cores=performance,
            cpu_limit=cpu_limit, server_class=server, topology_trusted=trusted,
        )
    except Exception:  # classification is advisory; never block startup
        return HostClass()


def _usable_cores(physical: int, logical: int, host: HostClass) -> "tuple[int, int]":
    if host.cpu_limit:
        limit = max(1, math.floor(host.cpu_limit))
        return min(physical, limit), min(logical, limit)
    return physical, logical


def recommend(
    *,
    physical_cores: int,
    logical_cpus: int,
    total_ram_mib: int,
    host: HostClass,
    max_threads: int = 32,
    fallback_threads: int = 6,
    fallback_batch: int = 256,
    fallback_ubatch: int = 128,
) -> Recommendation:
    """The rule table. Requires ``host.known``; callers keep the legacy rules otherwise."""
    physical = max(1, physical_cores)
    logical = max(physical, logical_cpus)
    why: list[str] = []
    physical, logical = _usable_cores(physical, logical, host)
    if host.cpu_limit:
        why.append(f"cpu limit {host.cpu_limit:.1f} cores (cgroup/affinity)")

    # Real performance-core count when known; an informed estimate when the
    # package is hybrid but the platform hides the split (WSL2 on Core Ultra).
    if host.hybrid == "yes" and host.performance_cores:
        fast = min(physical, host.performance_cores)
        why.append(f"hybrid CPU: {fast} performance cores")
    elif host.hybrid == "suspected":
        fast = max(4, physical // 2)
        why.append(f"hybrid CPU suspected, performance cores estimated at {fast}")
    else:
        fast = physical

    decode_cap = DECODE_CAP_SERVER if host.server_class else DECODE_CAP_CONSUMER
    decode = min(fast, decode_cap)
    if host.environment == "wsl2" and host.hybrid == "no":
        # The physical count here is logical/2 by construction; do not rely on
        # it being the real number of cores.
        why.append("wsl2: physical core count is synthetic")
    # Prompt processing is compute bound: use every fast core, but never ggml's
    # slowest-thread trap on a hybrid package.
    batch_threads = fast if host.hybrid != "no" else physical
    batch_threads = max(decode, min(batch_threads, max_threads))
    decode = max(1, min(decode, max_threads))

    total_gib = total_ram_mib // 1024
    if physical <= 2 or (total_gib and total_gib < 16):
        decode, batch_threads = min(decode, 2 if physical <= 2 else 4), min(batch_threads, 2 if physical <= 2 else 4)
        batch, ubatch = 128, 64
        why.append("small machine: conservative batch")
    elif host.server_class and physical >= SERVER_MIN_PHYSICAL and total_gib >= 64:
        batch, ubatch = 512, 256
        why.append("server class: larger batch")
    else:
        batch, ubatch = fallback_batch, fallback_ubatch

    candidates = {
        max(1, physical // 2), max(1, (physical * 3) // 4), physical, logical,
        decode, batch_threads, fast,
    }
    if physical >= fallback_threads:
        candidates.add(fallback_threads)
    ordered = tuple(sorted({min(max_threads, max(1, c)) for c in candidates if c > 0}))
    return Recommendation(decode, batch_threads, batch, ubatch, ordered, tuple(why))
