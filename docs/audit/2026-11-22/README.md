# Local-production review, 2026-11-22

Purpose: make Orbit reliable for running local agents on ordinary PCs, with an
audit trail. Host used: WSL2, 22 logical CPUs, 31 GiB RAM, CPU only.

## Method and provenance

1. Upstream `guelfoweb/orbit` was fetched (read-only). `main` was already merged
   (#393-#395). Two unmerged upstream branches were evaluated:
   - `fix/plan-delivery-guard` (1 commit): merged cleanly.
   - `fix/ornith-recovery` (5 commits): merged with two conflicts, resolved by
     keeping the fork's runtime-owned FINISH ownership (`analysis_finish_request`)
     and the #377 dossier tests; the superseded `analysis_evidence_request`
     mechanism was dropped. Upstream PR #1 (June MTP prototype) was not taken: it
     conflicts with the fork's existing `native_server` and is stale.
2. Two independent read-only reviews were run (Python expert, software
   architect). Their complete reports are in `raw/` (content as delivered; only trailing whitespace was stripped so the repository's whitespace check passes). Every claim there
   is marked VERIFIED or HYPOTHESIS by its author; the coordinator re-verified the
   ones acted on, by reproducing them or by tests added with the fix.
3. Fixes were made one finding at a time, each with its own tests, on branch
   `hardening/local-production`, one commit per change.

## Changes made (commit order)

| Commit | Finding | Change | Evidence |
|---|---|---|---|
| merge `142c636` | upstream | `fix/plan-delivery-guard` | suite 6788 OK |
| merge `b19f522`, fix `161c326` | upstream, AR-001 | `fix/ornith-recovery`; FINISH marker hardening; `step()` kept at its frozen 516 lines by extracting helpers (no behaviour change) | `tests/test_analysis_ornith_recovery.py`, `tests/test_architecture.py` |
| `9d115ff` | PY-001, AR-006, PY-002/AR-004 | bounded shell output (8 MiB/stream); Content-Length validation (`ORBIT_MAX_REQUEST_BYTES`); SIGTERM/SIGHUP clean stop | `tests/test_shell_output_bounds.py`, `tests/test_native_server_hardening.py` |
| `c1cae6d` | AR-015, AR-008 (part) | `server_start` run manifest; log dir 0700, file 0600 | `tests/test_run_manifest.py` |
| `3c8e075` | PY-004/AR-010 (advisory) | memory-fit advisory at startup | `tests/test_memory_fit_advisory.py` |
| `d745ac5` | PY-002 | startup sweep of abandoned analysis workspaces (`ORBIT_ANALYSIS_SWEEP_DAYS`) | `tests/test_analysis_workspace_sweep.py` |
| `6ed64a3` | AR-005 (part) | cap on concurrent POSTs, HTTP 503 (`ORBIT_MAX_INFLIGHT_REQUESTS`) | `tests/test_native_server_hardening.py` |

## Finding status

Fixed: PY-001, PY-002, AR-001, AR-004, AR-006, AR-015.

Partly addressed (what remains is stated):
- PY-004 / AR-010: advisory only. A hard refusal would contradict the qualified
  beyond-RAM mmap profiles (TECHNICAL_STOP decisions); it needs a decision.
- PY-005 / AR-005: body cap and in-flight cap done; socket and lock timeouts not
  added (a timeout on the handler socket could cut long streaming responses and
  needs a live test).
- AR-008 / PY-011: permissions done; prompts are still logged verbatim and the
  log is not rotated, hashed or fsynced.
- AR-013: the manifest can record the model SHA-256 (`ORBIT_AUDIT_HASH_MODEL=1`);
  the registry still pins only one hash.

Open, not changed (no evidence yet that a change is safe or measured):
- PY-003 / AR-011: topology (WSL2 reports 11 physical cores, no machine model, no
  ISA/cgroup awareness). Needs measurement on the target PCs; changing inference
  defaults without it would break the repository's measured-defaults rule.
- PY-006 to PY-010, PY-012 to PY-014, AR-002, AR-003, AR-007, AR-009, AR-012,
  AR-014, AR-016, AR-017, AR-018: see `raw/` for evidence and proposed fixes.

## Not verified

- No live ANALYSIS run on the frozen malware corpus (absent on this host); the
  suite ran with `ORBIT_ALLOW_MISSING_CORPUS=1`.
- Effectiveness of `fix/plan-delivery-guard` with a real model on a real
  unanswerable PLAN request: covered by unit tests and fixtures only.
- Throughput: live smokes (MiniCPM5 2B, Granite 3B) were compared before and
  after the first merge only; one run each, so timings are indicative.

## Phase 2 (same day): follow-up changes

| Area | Change | Evidence |
|---|---|---|
| Calibration | `host_class.py`: platform-class rules (WSL2/VM/container quota, hybrid P/E cores, ISA, server class) feed the heuristic tier and the sweep's candidates; precedence unchanged, unknown hosts keep the legacy rules | `tests/test_host_class.py`; real sweep below |
| Timeouts | `ORBIT_SOCKET_READ_TIMEOUT` (408) and `ORBIT_MODEL_LOCK_TIMEOUT` (503), both off by default | `tests/test_native_server_hardening.py` |
| Logging | stdlib `logging` rotating 0600 file, content modes, secret scrubbing, hash chain + `verify` CLI | `tests/test_request_log_framework.py` |
| Memory | `ORBIT_MEMORY_GUARD=warn\|refuse\|off` (default warn) | `tests/test_memory_guard.py` |
| Supply chain | size/sha256 pins for 18 registry files (Gemma mmproj left unpinned on purpose), verified after download, CycloneDX SBOM generator | `tests/test_model_download_integrity.py`, `tests/test_sbom.py` |
| Structure | complexity/nesting ratchet, ADR 0001, `detect_native_model_profile` 626 to 40 lines behind a 2446-case golden | `tests/test_architecture.py`, `tests/test_model_profile_characterization.py` |

Pins not applied: `qwen38-27b-q4-k-m` (registry file name does not exist upstream,
the repo has `...UD-Q4_K_M.gguf`) and `qwen38-flash-next-ud-iq1-m` (split model,
path mismatch). Both need a registry fix first.

### Measured calibration sweep (this host, MiniCPM5-2B Q4_K_M, WSL2, Core Ultra 7 155H)

| threads | decode tok/s | prefill tok/s |
|---|---|---|
| 5 | 17.6 | 51.8 |
| 6 | 18.8 | 52.6 |
| 8 | 15.9 | 64.4 |
| 11 (old heuristic) | 9.5 | 49.9 |
| 22 | 0.09 | 9.6 |

The new heuristic picks 5 and the sweep confirmed 5 (6 and 8 within noise). The old
heuristic would have chosen 11, 46% slower decode. A single model and run: not a
general benchmark.
