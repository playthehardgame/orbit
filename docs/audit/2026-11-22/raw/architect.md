# Architect audit - Orbit (branch merge/ornith-recovery, read-only)

Inspected: AGENTS.md (headers, state, Server Startup Profile, Principles, Anti-Goals), CLAUDE.md, docs/CORE_CI.md, tests/test_architecture.py (read and run),
native_server/app.py (lifecycle, handler, lock, read_json, run_server), request_logging.py, server_profile.py (topology, fingerprint, cache_ram),
server_calibration.py (grep only), native_llama/model_registry.json (+ sha fields), build_identity.py, llama_provenance + vendor/LLAMA_PROVENANCE.json,
telemetry.py header + docs/TELEMETRY.md, qualification/schema.py RunProvenance, runtime/evidence.py (sha), pyproject.toml, .github/workflows/core.yml,
~/.cache/orbit/server-profiles/*.json (19 records), git remotes/divergence.
NOT inspected: client.py internals (5940 lines), tool_loop/chat/analysis internals, qualification/runner, shell_guardrails/sandbox, model_download retry logic, terminal/, scripts/*, vendor C++ build flags beyond build_cli.py, no live server run.

## Layering / ratchets
Clean areas (VERIFIED): no module-level import cycles; layer rule set enforced by AST over module AND lazy imports (test_architecture.py:107-129);
`runtime` never imports native_llama/native_server/terminal (grep of runtime/*.py found no native_llama import); only 3 reverse edges allowed and ratchet fails both ways (debt paid or added).
Telemetry is genuinely safe: opt-in, allowlisted attributes, no prompts (telemetry.py:1-35).

### AR-001 MAJOR - Ratchet currently RED on working tree (VERIFIED)
tests/test_architecture.py:57 freezes AnalysisRuntime.step at 516; uncommitted diff to analysis_runtime.py makes it 537. I ran the module: 1 failure (537 != 516), 4 pass.
Why: ratchet works as designed (good), but it means the dirty tree cannot pass the gate. Change: either revert growth or raise the frozen value with a documented reason; do not commit with a red ratchet. Risk: none. Verify: rerun test_architecture.py.

### AR-002 MAJOR - God-modules and ratchet blind spot (VERIFIED)
analysis_runtime.py 9278 lines (one module, ~9 dataclasses + AnalysisRuntime; `_run_autonomous_locked` 1199 lines, `step` 537), client.py 5940 lines with NativeLlamaClient from line 366 to 5788 (one class ~5400 lines), tool_loop.run_tool_loop 1809 lines, app.py 2562 (handler + run_server 312 + discovery UI + profile resolution + prewarm).
Ratchet freezes only FUNCTION length >300; module/class size is not ratcheted, so growth by adding 299-line functions or methods to the god class is unconstrained. AGENTS.md records R2-R6 extraction TECHNICAL_STOPs, so do NOT propose big refactors; propose only a cheap ratchet: freeze per-module line count and per-class method count (allowlist = current values). Risk: low. Verify: new test fails when file grows.

### AR-003 MINOR - app.py mixes server, CLI UX and policy; runtime layer imported by server for prompts
native_server/app.py:102-105 imports runtime.analysis_runtime and tool_healing for prompt constants/status. Allowed direction (server -> runtime), but pulls the 9k-line analysis module in at server import (startup memory/latency). Proposed: move ANALYSIS_SYSTEM_PROMPT/ANALYSIS_TOOL_SCHEMA to a small runtime/prompts module. HYPOTHESIS on measurable import cost; verify with `python -X importtime`.

## Production operability (native_server/app.py)
### AR-004 BLOCKER for unattended production - no SIGTERM handling (VERIFIED)
app.py:1670 handles only KeyboardInterrupt; grep finds no SIGTERM handler in src (only shell_guardrails killpg and a comment). `kill`, systemd stop, `wsl --shutdown`, docker stop send SIGTERM: Python dies without running the `finally` (stop_route_shadow, client.close, RequestLogger server_stop). Effects: native context not freed cleanly, no server_stop log row, an in-flight route shadow may be inside llama_decode. Local PC: service managers and WSL restarts are the normal stop path.
Change: signal.signal(SIGTERM, lambda *_: raise KeyboardInterrupt) in run_server main thread, and call httpd.shutdown() from a thread. Risk: low. Verify: start server (stub client), send SIGTERM, assert server_stop row and exit 0.

### AR-005 MAJOR - unbounded concurrency, no backpressure, no timeouts (VERIFIED)
ThreadingHTTPServer (app.py:1658) spawns a thread per connection with no cap; one global threading.Lock (app.py:153, _model_lock 175-187) serialises the model; waiters block indefinitely with no queue depth limit, no 503/429, no per-request deadline, no socket timeout (grep: no settimeout/request_queue_size). A stuck or slow client, or N retrying clients, stacks threads each holding a request body in RAM. Single-model serialisation is correct for CPU; the missing piece is admission control.
Change: bounded waiter counter in _model_lock (reject with 503 + Retry-After when waiters > ORBIT_SERVER_MAX_QUEUE, default e.g. 4), handler `timeout` attribute (socket read timeout, e.g. 30 s) for header/body reads, and acquire(timeout=...). Risk: low-medium (must not break /props polling, which already avoids the lock - app.py:472). Verify: unit test with two concurrent POSTs and queue=1.

### AR-006 MAJOR - request body unbounded, malformed Content-Length crashes handler (VERIFIED)
_read_json (app.py:837-842): int(Content-Length) with no cap and no try/except; rfile.read(length) allocates whatever the client claims. A bad header raises ValueError inside the handler (may become a 400 or a dropped connection depending on call-site; HYPOTHESIS for the call-site). Local risk is lower (default bind 127.0.0.1, app.py:108) but --host is free-form with no auth (app.py:1924); /chat can drive tools. Change: cap (e.g. 32 MiB, env-tunable) -> 413; wrap int() -> 400; warn on stderr when host is non-loopback, optionally require ORBIT_SERVER_TOKEN bearer. Risk: low. Verify: tests with huge/negative/non-numeric Content-Length.

### AR-007 MAJOR - /health is liveness only; handler errors are invisible by default (VERIFIED)
/health returns {"status":"ok"} unconditionally (app.py:578-580), never touches client/model state, so a wedged native context still reports ok; there is no readiness vs liveness split (the socket binds only after load+calibrate+prewarm, app.py:1658, so connection-refused doubles as "loading"; acceptable but opaque for 2-3 minute starts). log_message is a no-op (app.py:802-803) and request logging is off unless --log is given; stdlib handler tracebacks are therefore dropped. Change: /health?ready=1 returning model_loaded, queue depth, last_completion_age; default-on minimal stderr error logging (status>=500, exceptions with type name) independent of --log. Risk: low. Verify: unit test of /health payload; induced handler exception appears on stderr.

### AR-008 MAJOR - request log stores full prompts, unrotated, un-fsynced (VERIFIED)
request_logging.py: payload of every request logged (app.py:810-821) up to 1 MiB per string value, one append-only requests.jsonl, no rotation/size cap/permissions (0644 default umask), flush only (no fsync), no hash chain. Contradicts the privacy posture of telemetry.py (no prompts) - two policies. server_start logs argv only (app.py:1657), not env, resolved profile, model identity or build id. Change: see audit manifest below; add rotation by size and mode 0600; option to log payload hashes+lengths instead of bodies (default for production). Risk: low. Verify: tests on rotation and mode.

### AR-009 MINOR - hardcoded hardware numbers in UX (VERIFIED)
app.py:2150-2153 prints "~31.3 GiB peak RSS / ~18.3 GiB" and "recommended for >=24 GB" as constants measured on the Dell, shown on any PC. Change: derive from model file size + calibration (see AR-012) or label as "measured on reference host". Risk: none.

## Hardware portability / calibration
What exists (VERIFIED, good): topology detect from /proc (server_profile.py:278-345), heuristic floor that degrades toward smaller values, bounded calibration (90 s, warm-up, tie-break to fewest threads, KV cleared), cache keyed by fingerprint, explicit values never measured/cached, cache_ram computed conservatively (server_profile.py:380-420), nothing auto-enables MTP/GPU/ctx/quant. This is a well-designed precedence chain; do not redo it.

### AR-010 MAJOR - no hard admission control for model vs RAM at load (VERIFIED for download; HYPOTHESIS for load path)
RAM arithmetic exists in compute_cache_ram_mib and download_advisory (model_store.py:228), but both are advisory; grep of app.py/client.py for MemAvailable/refuse shows no pre-load check that model_bytes + KV(ctx) + OS reserve <= MemAvailable. Local PC with 16-32 GB loading a 20+ GB model hits the OOM killer (WSL2 kills the VM, losing all processes) instead of a clean refusal. Change: pre-load gate `required = model_bytes*1.1 + kv_estimate(ctx) + 2 GiB reserve`; if > MemAvailable (and no --force-unsafe) exit 2 with a message naming --low-memory/--ctx/smaller model; with mmap the file cache can be reclaimed, so compare to MemTotal-reserve as the hard limit and to MemAvailable as the warning. Risk: medium (false refusal on qualified machines; make the Dell profile pass by test). Verify: unit tests with synthetic meminfo.

### AR-011 MAJOR - topology ignores ISA, cgroup limits, WSL cap, thermal/power (VERIFIED)
detect_topology reads model name, physical/logical cores, MemTotal/MemAvailable/SwapTotal only. No AVX/AVX2/AVX512/AMX/FMA flags (cpuinfo `flags` parsed nowhere - grep), no cgroup cpu.max/memory.max (docker, systemd slices), no sched_getaffinity (os.cpu_count ignores affinity/cgroup quota; os.process_cpu_count in 3.13). Native lib is built with the llama.cpp default (GGML_NATIVE on host; build_cli.py shows no ISA flags) so a build moved to an older CPU can SIGILL; fingerprint (server_profile.py:512-562) does not include ISA flags or build id beyond upstream tag/commit (_backend_identity app.py:1051, ignores patchset_sha256 and ORBIT_LLAMA_BUILD_FLAGS). Hybrid (P/E core) hosts: physical count is from (physical id, core id) pairs, not core type.
Change: add `isa` (sorted subset of flags), `cpu_quota`, `mem_limit` to HostTopology; refuse start (clear error) when the built libggml-cpu variant needs a flag absent; include isa + patchset hash in fingerprint (bump PROFILE_FORMAT_VERSION). Risk: low-medium (cache invalidation => one recalibration). Verify: tests with fixture cpuinfo/cgroup text.

### AR-012 MINOR - calibration record is measurement-only; not auditable (VERIFIED)
Cached JSON has fingerprint, format_version, measurements, chosen threads (19 records inspected). It lacks: the fingerprint's inputs in clear (cannot tell what machine/model a hash is), timestamp, orbit commit, ISA, kernel, candidate list rationale, tie-break decision, resolved ctx/batch/ubatch. Model identity is name:size:mtime (app.py:1115-1135), not a hash - a same-size corrupted/replaced file with preserved mtime is undetected (acceptable for cache keying, NOT for audit). See target design.

## Registry / profiles / versioning
### AR-013 MAJOR - weights are not hash-pinned (VERIFIED)
model_registry.json: only 1 sha256 in the whole registry (an mmproj, line 213); targets give repo/file/glob, no size/sha256/revision. "Verified identity" in model_profiles.py (e.g. QWEN38_VERIFIED_MODEL_NAME/FILE_TYPE/QUANTIZATION, lines 12-58) is GGUF-metadata name + quant, which any fine-tune can copy. AGENTS.md itself says download verification is "dimensional, not a checksum guarantee". Floating HF snapshot glob (`snapshots/*/`) means a re-upload under the same file name silently changes the model while all qualification claims stay attached to the id. Change: add `size` + `sha256` + HF `revision` (commit) per registry file for the qualified models (the schema already supports it, model_registry.py:432 checks size for mmproj); verify lazily (sha256 once on first load, store result next to the file as `<file>.orbit-verified.json` keyed by size+mtime). Risk: low; cost one 20 GB hash once per model (~1-2 min). Verify: tests/test_model_matrix* + new test of mismatch refusal.

### AR-014 MINOR - registry `version: 1` has no migration/compat story (VERIFIED)
Registry is a package-data JSON with version=1 and 18 profiles; test_model_matrix/test_model_capability_matrix enforce cross-consistency (good). No min_orbit_version, no per-entry `qualified_on`/`qualification_report_hash` link, so a user cannot tell whether a model is qualified, research, or merely loadable. Change: add `status` (qualified/research/blocked) and `evidence` (report path+sha) fields; refuse non-qualified for production mode. Risk: low.

## Auditability
Already recorded (VERIFIED): qualification RunProvenance (schema.py:253-266: schema version, fixture-set hash, git revision, profile, model/template identity+hash, backend identity/revision, runtime config, hardware, measurement scope); evidence store with raw_sha256 verified on read (evidence.py:185); vendored llama.cpp provenance (commit, source_tree_sha256, patchset_sha256, LLAMA_PROVENANCE.json); build identity captured once per process (build_identity.py); atomic session writes (sessions.py:164-165); OTel sandbox spans with input/code sha256; Actions pinned to commit SHAs with read-only token (core.yml).

### AR-015 MAJOR - no per-run manifest for ordinary (non-qualification) runs (VERIFIED by absence)
Provenance exists for qualification runs only. A production server run records argv (app.py:1657) but not env overrides (ORBIT_* are many), resolved profile and its source (cli/env/cache/heuristic/qualified), model identity/hash, build id + llama provenance, calibration fingerprint, host topology. /props exposes much of it live but is not persisted. Hence a production answer cannot later be tied to config.
### AR-016 MAJOR - dependency/SBOM and reproducibility gaps (VERIFIED)
pyproject dependencies=[] (stdlib only - a strength) but: no lock/constraints file for the optional otel extra (>= only), no SBOM, native libs are built locally (vendor/lib empty) with no recorded compiler/cmake versions or flags hash in a persisted artefact (ORBIT_LLAMA_BUILD_FLAGS exists but is in KNOWN_UNDOCUMENTED_ENV), CI runs only the core subset on Python 3.11/3.12 (docs/CORE_CI.md) and explicitly not the native/corpus gates, so nothing automated proves the build+load path on a fresh machine. Python 3.12 is what the dev host runs (pycache cpython-312).
### AR-017 MINOR - logs not immutable (VERIFIED)
requests.jsonl and sessions are plain mutable files; evidence index.json is a mutable index. Raw evidence is hash-checked but the index/log has no chain. Cheap fix: per-row `prev_hash` (sha256 of the previous row bytes) in RequestLogger and the run manifest; verification = recompute chain.

## Fork vs upstream
(VERIFIED) origin=playthehardgame/orbit, upstream=guelfoweb/orbit; at last fetch `git rev-list --left-right --count main...upstream/main` = 43 ahead / 0 behind; branch merge/ornith-recovery merges upstream fix branches (fix/ornith-recovery, fix/plan-delivery-guard) not yet in upstream main; 671 commits total, 43 by Riccardo. Last merge commits show a merge-not-rebase policy (good, preserves upstream attribution per CLAUDE.md).
### AR-018 MINOR - drift risk concentrated in god-modules (HYPOTHESIS on conflict rate)
Fork touches analysis_runtime.py/client.py/model_profiles.py/registry which upstream also edits heavily; large files = large conflict surface and the ratchet constants (FROZEN_LONG_FUNCTIONS, KNOWN_UNDOCUMENTED_ENV) are single-line conflict magnets. Policy proposal: keep fork-only changes in additive files (registry rows, config, new modules) and keep a `docs/FORK_DELTA.md`-style list generated from `git diff --stat upstream/main...main`; run `check_core` + test_architecture on every upstream merge; merge only tagged-qualified upstream points, as the branch already does. Keep patching in the fork: SIGTERM/admission/limits (AR-004..006, 010) are generic and could be offered upstream to shrink the delta.

## Proposed target design (smallest sufficient)
Calibration record `~/.cache/orbit/server-profiles/<fingerprint>.json` (extend, bump format to v4):
- identity: fingerprint, format_version, created_at (UTC ISO), orbit_commit, llama_upstream_commit + patchset_sha256, build_flags_hash
- inputs in clear: cpu_model, physical/logical cores, cpu_quota, isa_flags (subset), ram_total/avail MiB, swap, mem_limit, kernel, WSL flag, model {path name, size, mtime, sha256 if verified, quant, arch}, ctx, low_memory/mtp/experts flags
- procedure: candidate list with reason, warm-up row, budget used, score constant, tie margin, winner and why
- decision: resolved threads/threads_batch/batch/ubatch, per-field source
- safety verdict: required_mib vs available_mib, refused_combinations
- self hash: sha256 of canonical JSON of the record minus this field. Verify: `orbit server --show-profile --verify` recomputes self hash and checks fingerprint inputs still match the host.
- Command: `orbit calibrate [--model-id X] [--json]` = the existing sweep without starting a server, written atomically (0600).
Per-run audit manifest `<--log dir>/run-<utc>-<pid>/manifest.json` (written at server_start, atomically, then read-only 0444):
orbit_commit+dirty flag, argv, ORBIT_* env snapshot (names+values, secrets filtered), resolved profile + per-field source, calibration fingerprint + record hash, model identity (sha256 verified, size, registry id, profile id), backend/llama provenance hash, python version, host topology/ISA, registry hash, manifest schema version. Request log rows carry `run_id` and a prev_hash chain; server_stop row stores final chain hash and counters. Verification: `orbit audit verify <dir>` recomputes chain and manifest hash, compares model sha256 to registry. HYPOTHESIS: ~150 lines, no new dependency.

## Confirmations requested from Python expert
1. Does an unhandled ValueError from int(Content-Length) in _read_json (app.py:838) produce 400 or a dropped connection at each call site?
2. Whether SIGTERM really skips `finally` in run_server on 3.12 (default action is immediate termination; confirm by experiment with a stub).
3. Confirm no pre-load RAM check exists in client.py load path (I did not read it).
4. Thread-safety of `_waiters` / `__dict__.get` pattern (app.py:162-196) and of route-shadow stop vs request arrival.
5. Whether analysis_runtime.py working-tree diff (3 ins/2 del, step 516->537 lines? ) is intended and who owns resetting the ratchet.
