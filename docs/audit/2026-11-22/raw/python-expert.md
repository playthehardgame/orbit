# Python expert audit (read-only) - orbit @ merge/ornith-recovery
Host: WSL2, Core Ultra 7 155H (22 logical), 31 GiB RAM, 8 GiB swap. Claims marked VERIFIED = read and/or run; HYPOTHESIS = reasoning only.

## Findings

### PY-001 MAJOR - exec_shell_full_command buffers unbounded stdout/stderr in RAM
- File: src/orbit/runtime/shell_guardrails.py:416-434 (`_run_shell_command`, `process.communicate(timeout=...)`); caps MAX_SHELL_TIMEOUT=15 (line 29), MAX_SHELL_OUTPUT_BYTES=12_000 (line 31) are applied only AFTER the full output is in memory.
- Evidence (VERIFIED, ran): `_run_shell_command("yes", timeout=2)` -> TimeoutExpired with len(stdout)=1,247,649,792 bytes, process maxrss 4,793 MiB. At the 15 s cap this extrapolates to ~9+ GiB (OOM on a 16 GB PC, WSL2 VM kill).
- Why: a model-issued `yes`, `cat /dev/zero`, `find /` or a `cat bigfile` can take down the whole agent runtime on a consumer machine.
- Fix: replace communicate() with a selector read loop (as analysis_sandbox.py:582-592 already does) with a hard cap (e.g. 2 MiB/stream), killpg on exceed, and report "output truncated". Optionally RLIMIT_AS/RLIMIT_FSIZE in a preexec.
- Risk: low; callers only use the first 12 KB, but `_postprocess_shell_full_output`/search helpers may expect full text for `grep -c` etc. - keep cap well above 12 KB.
- Verify: new unit test asserting `yes` with timeout=2 keeps RSS growth < 64 MiB and returns bounded output; re-run tests/test_shell_guardrails*.py.

### PY-002 MAJOR - No SIGTERM/SIGHUP handling in `orbit server`; workspaces and native context leak
- File: src/orbit/native_server/app.py:1657-1675 (only `except KeyboardInterrupt` around serve_forever; signal.signal used only for SIGINT during prewarm, lines 1609-1636). grep for SIGTERM/atexit across src (excluding vendor): only shell_guardrails (child kill) and telemetry atexit.
- Evidence (VERIFIED): `ls -d /tmp/orbit-analysis-session-*` -> 100 residue dirs (oldest Nov 14); AnalysisWorkspace docstring (analysis_runtime.py:2550-2556) admits "After abnormal process death the residue is an ordinary temporary directory". `wsl --shutdown`, systemd stop, `kill`, terminal close all send SIGTERM/SIGHUP, which skip the `finally` (client.close(), request_logger.close(), stop_route_shadow()).
- Why: production deployments are stopped by SIGTERM, not Ctrl-C. Leaks malware-sample copies (source/ dirs, mode 0700 but persistent) and leaves jsonl log unflushed state/route-shadow thread mid-decode.
- Fix: install SIGTERM (and SIGHUP) handler that raises KeyboardInterrupt/SystemExit in main thread (`signal.signal(SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))` or call httpd.shutdown from a thread); in terminal/analysis_mode.py register atexit for workspace.close(); add startup sweep that deletes `orbit-analysis-session-*` older than N hours owned by the user.
- Risk: low-medium (signal handlers + threads; keep main-thread only). Sweep must not delete a live session of another process: use age + lock/pid file.
- Verify: start server (fake client) in subprocess, send SIGTERM, assert exit code and that client.close called; test that workspace dir removed.

### PY-003 MAJOR - CPU thread heuristic and "qualified" defaults are Dell-specific; WSL2 topology is mis-detected
- Files: src/orbit/native_server/server_profile.py:86-100 (QUALIFIED_NOREPACK_MACHINE="Dell Pro 5 14 P514260" matched on /sys/devices/virtual/dmi/id/product_name), :160-195 (QUALIFIED_STARTUP_PROFILES), :300-345 (detect_topology), :425-465 (heuristic_profile).
- Evidence (VERIFIED, ran `detect_topology()`): `HostTopology(cpu_model='Intel(R) Core(TM) Ultra 7 155H', physical_cores=11, logical_cpus=22, ..., machine_model='')`; thread_candidates -> [5,6,8,11,22]; heuristic_profile -> threads=11. /proc/cpuinfo under WSL2 reports 22 `core id` lines but only 11 unique ids (hybrid P/E cores wrongly paired as SMT), and no DMI product_name.
  Consequences: (a) physical_cores is a guess on hybrid/WSL2 (real: 6P+8E+2LP = 16 cores); (b) `machine_model==''` so the Dell/Ornith no-repack default (resolve_cpu_repack) never applies on WSL2/other PCs -> llama.cpp repack ON by default for a ~21 GB Q4_K_M model duplicates weights into anon memory (the exact overflow the comment at line 90-93 describes) on a 16-32 GB PC. (c) QUALIFIED_STARTUP_PROFILES keyed by machine means Qwen3.8 Flash Next (74.5 GB beyond-RAM) gets no safe lazy/mmap/ctx profile elsewhere.
- Why: first-run behaviour on a typical PC is un-qualified; the measured calibration (server_calibration.py) mitigates threads but not repack/memory.
- Fix: (1) key the no-repack default on memory-fit instead of machine name: repack OFF when model_bytes > ~0.45*MemTotal (or model_bytes + 20% > MemAvailable); keep Dell match as additional. (2) add ORBIT_PHYSICAL_CORES / use `lscpu -p` + /sys/devices/system/cpu/cpu*/topology/thread_siblings_list when present; cap heuristic threads for hybrid CPUs at P-core count if detectable. (3) document WSL2 `.wslconfig` (memory=, swap=, processors=) in README since MemTotal inside WSL2 is 50% host RAM by default.
- Risk: medium - changes inference defaults; must re-qualify with AGENTS.md oracles (IBAN.js) and must not contradict the TECHNICAL_STOPs; do not change the qualified Dell pair.
- Verify: tests/test_server_profile*.py, `orbit server --show-profile`, plus a memory-fit unit test with synthetic HostTopology (total 16 GiB, model 21 GiB).

### PY-004 MAJOR - No pre-load RAM-fit / low-memory guard before loading the model
- Files: src/orbit/native_server/app.py:1210-1260 (model_bytes is only fed to cache_ram arithmetic), grep of `available_ram_mib|MemAvailable` shows use only in server_profile.py, system_info.py, terminal/runtime_status.py (display).
- Evidence (VERIFIED by grep): nothing compares model file size/ctx KV against MemAvailable/MemTotal before llama_model_load; compute_cache_ram_mib returns the floor (MIN_CACHE_RAM_MIB) and proceeds. Default ctx fixed at 8192 (app.py:1152) irrespective of RAM.
- Why: on a 16 GB PC a 20 GB model mmap'd goes to constant page-fault thrash (or WSL2 OOM kill) instead of a clear error; operators see "hangs".
- Fix: at startup compute needed = model_bytes (mmap'd, counts as page cache) + KV(ctx) + reserve; if needed > MemTotal: refuse unless `--allow-oversubscribe`/ORBIT_ALLOW_OVERSUBSCRIBE=1 (the qualified beyond-RAM Qwen3.8 Flash Next profile is explicitly allowed); if > MemAvailable: warn and reduce default ctx (4096). Print the decision in --show-profile.
- Risk: medium (could block qualified beyond-RAM mmap use) -> whitelist qualified lazy-mmap profiles.
- Verify: unit test with fake topology; `--show-profile` output.

### PY-005 MINOR/MAJOR - Native HTTP server has no request timeout or body cap
- File: src/orbit/native_server/app.py:838-841 `_read_json` -> `int(Content-Length)`; `self.rfile.read(length)` with no max; BaseHTTPRequestHandler.timeout is None (VERIFIED: `BaseHTTPRequestHandler.timeout -> None`, ThreadingHTTPServer.daemon_threads True). Server binds 127.0.0.1 by default (DEFAULT_HOST line 108) and `--host` is user-selectable with no auth.
- Why: a stalled client holds a thread forever (while the model lock is requested only later, so inference is not blocked, but threads/FDs accumulate); a huge Content-Length allocates GB. `--host 0.0.0.0` on a LAN-connected PC exposes an unauthenticated generate endpoint (HYPOTHESIS: no auth check seen in the handler paths I read; not exhaustively audited).
- Fix: `timeout = 60` on handler class (socket timeout for header/body reads; streaming writes unaffected), cap Content-Length (e.g. 32 MiB -> 413), reject non-loopback --host without an explicit flag/ORBIT_SERVER_ALLOW_REMOTE=1 and warn.
- Risk: low; long prompts need cap above realistic ctx (8k tokens ~ 100 KB).
- Verify: socket test sending partial body; unit test for 413.

### PY-006 MINOR - analysis sandbox: preexec_fn in a multi-threaded process and unbounded source read
- File: src/orbit/runtime/analysis_sandbox.py:557-563 (`preexec_fn=_preexec_limits`), :218-255 (`_uid_task_count` globbing every /proc/*/status in the forked child), :513,:593 (`source.read_bytes()` twice, no size cap).
- Evidence (VERIFIED read): preexec_fn runs Python code (file reads, glob, allocations) between fork and exec in a process that has an HTTP server thread pool and native threads -> documented deadlock hazard (Python docs: "preexec_fn is not safe in the presence of threads"). /proc scan costs ~ms per task on busy WSL2 and runs per sandbox call. The input file is read fully into memory twice (before/after tamper check), so a multi-GB sample in analysis mode uses 2x its size in RAM (HYPOTHESIS that nothing upstream caps size; I did not find a cap in analysis_sandbox/analysis_bootstrap grep for MAX_SOURCE/MAX_INPUT/st_size).
- Fix: compute `_uid_task_count()` in the parent before Popen and capture the number in a closure so the child only calls setrlimit; use sha256 streaming with chunked read and compare mtime/size+hash instead of holding bytes; consider `process_group=0` (3.11+) / prlimit-wrapper instead of preexec_fn.
- Risk: low-medium (tests/test_analysis_sandbox.py exercises it: 48 tests OK in 30 s VERIFIED with ORBIT_ANALYSIS_EXTRA_PROCESSES=128).
- Verify: same test module; add a test with a 200 MiB sparse input asserting RSS.

### PY-007 MINOR - sandbox Popen failure paths can leak the child
- File: analysis_sandbox.py:565-600: the try/finally (`finally: program_owner.cleanup()`) never kills the process group. Exceptions in the read loop (KeyboardInterrupt, MemoryError) or `process.wait(timeout=5)` raising TimeoutExpired (grandchild holding pipes after killpg is unlikely) leave bwrap running until its own 15 s CPU rlimit / --die-with-parent. Mitigated by `--die-with-parent` and RLIMIT_CPU=14 (VERIFIED read), so impact is small.
- Fix: in `finally`, `if process.poll() is None: os.killpg(process.pid, SIGKILL); process.wait()`; also close stdout/stderr unconditionally.
- Risk: very low. Verify: test that injects KeyboardInterrupt in selector.select and asserts no live pgid.

### PY-008 MINOR - pdftotext/strings subprocess output unbounded
- File: src/orbit/runtime/file_tools.py:373-400 (`subprocess.run(... capture_output=True, timeout=PDF_EXTRACT_TIMEOUT)`); size check on extracted text happens after capture (line ~369).
- Why: a crafted/large PDF can produce hundreds of MB of text into RAM before the MAX_CHUNK_FILE_BYTES test. HYPOTHESIS magnitude (not run; no PDF). 
- Fix: stream with Popen and byte cap, or pre-check file size; `strings` fallback likewise.
- Verify: unit test with fake large output. Risk: low.

### PY-009 MINOR - fetch_url/web search have no private-address (SSRF) filtering
- File: src/orbit/runtime/web.py:196-228 (`urlopen` of arbitrary model-chosen URL, redirects followed by default).
- Why: prompt-injected content can make the agent fetch http://127.0.0.1:<llama port>, cloud metadata or router admin pages and echo them into context. Practical impact limited because the unrestricted exec_shell_full_command can already run curl (so this is defence-in-depth, not a boundary). Network is already denied in analysis mode (network_retrieval_denied, VERIFIED read).
- Fix: resolve host, reject loopback/link-local/private ranges unless ORBIT_FETCH_ALLOW_PRIVATE=1; re-check on redirect via custom HTTPRedirectHandler.
- Risk: low; breaks fetching local dev servers (document env override). Verify: new unit test with 127.0.0.1 URL.

### PY-010 MINOR - shell tool passes the full parent environment (secrets) to model-run commands
- File: shell_guardrails.py:~393 `env = dict(os.environ)` (only HOME/PWD overridden). The model can `env`/`printenv` and put HF_TOKEN, API keys, SSH_AUTH_SOCK in context/logs (request log `--log` writes full messages up to 1 MiB/field, request_logging.py:12-29).
- Fix: allowlist env (PATH, LANG, LC_*, TERM, TMPDIR, HOME=workdir) with ORBIT_SHELL_ENV_PASSTHROUGH for additions; redact obvious token patterns in request logger.
- Risk: low-medium: commands relying on e.g. proxies/venv vars; keep PATH+VIRTUAL_ENV. Verify: test that `env` output lacks a planted FOO_TOKEN.

### PY-011 MINOR - request log is unbounded (no rotation)
- File: src/orbit/native_server/request_logging.py:36-51: appends forever to requests.jsonl, each field up to 1 MiB, full prompts and outputs (privacy + disk). Opt-in via --log (VERIFIED), so only matters when enabled.
- Fix: size-based rotation (e.g. 50 MiB x 3), chmod 0600 on creation, document that it stores prompts. Risk: low. Verify: unit test writing past the cap.

### PY-012 MINOR - Calibration/profile cache: hard-coded reference constants
- Files: server_profile.py:69-71 FALLBACK_THREADS=6, BATCH=256, UBATCH=128 (NUC numbers); server_calibration.py TIE_MARGIN=0.10, reference turn 210/60 tokens (tuned on Dell). Measured calibration itself is well bounded (90 s budget, swap-growth rejection, warm-up, atomic cache write with fsync, tmp cleanup on BaseException - VERIFIED read). Clean area apart from the above defaults.
- Why: on 8-core/16 GB CPUs batch 256/ubatch 128 is OK (heuristic already drops to 128/64 for <16 GiB or <=2 cores - VERIFIED). No env override gap: ORBIT_THREADS, ORBIT_THREADS_BATCH, ORBIT_BATCH, ORBIT_UBATCH, ORBIT_CPU_REPACK exist. Missing: ORBIT_CTX env (only --ctx), ORBIT_CALIBRATION=off / budget env.
- Fix: add ORBIT_CTX and ORBIT_CALIBRATE=0/ORBIT_CALIBRATION_BUDGET env for service-style launches. Risk: low. Verify: tests/test_server_profile*.py.

### PY-013 MINOR - broad `except Exception` (125 in src) mostly diagnostics; no bare `except:`
- Evidence (VERIFIED): `grep -rn "except:" src` -> none. `except Exception` + `pass` occurrences listed in telemetry.py (by contract), qualification/runner.py:400, backend/llama_server.py:532,576, analysis_runtime.py:7764, terminal/history.py, repl_input.py, streaming.py:283 - all annotated as diagnostics/UI. Not audited individually beyond these. Retry loops are bounded (web.py SEARCH_MAX_ATTEMPTS=3, backoff tuple; download resume bounded per AGENTS.md #387). Clean.

### PY-014 INFO - tests: environment sensitivity
- Sandbox test module needs /usr/bin/bwrap (present here) and benefits from ORBIT_ANALYSIS_EXTRA_PROCESSES=128 (WSL2 NPROC undercount, code comment lines 56-66). 100 leaked `/tmp/orbit-analysis-session-*` dirs show tests/real runs do not always clean (some test helper patterns in tests/test_workflow_mode.py:68, test_automatic_analysis_routing.py:73 only match names). Suggest a conftest-style tearDown cleanup. HYPOTHESIS that the 100 dirs come from tests vs real runs. Not run: full suite, native-lib tests.

## Clean areas (with evidence)
- Telemetry (src/orbit/telemetry.py): opt-in via ORBIT_OTEL, allowlisted attributes, regex-restricted values, exception messages never recorded, failures swallowed (VERIFIED read lines 1-140, call sites grep). No finding.
- Sandbox flags: bwrap --unshare-all --disable-userns --die-with-parent --new-session --clearenv --cap-drop ALL, rlimits CPU/AS/FSIZE/NOFILE/NPROC, 15 s wall clock with killpg SIGKILL, hard 2 MiB output cap, temp dirs cleaned in finally (VERIFIED read; 48 sandbox tests pass).
- Profile cache write is atomic and failure-tolerant. Shell process-tree termination (SIGTERM then SIGKILL on pgid) correct.

## Inspected
AGENTS.md (head), CLAUDE.md, server_profile.py (resolve/heuristic/topology/cache), server_calibration.py (header+constants), app.py (run_server, handler, _read_json), analysis_sandbox.py (full), shell_guardrails.py (execution part), web.py (fetch), file_tools.py (pdf), telemetry.py (header), request_logging.py, analysis_runtime.py AnalysisWorkspace, scripts/suggest-server-profile.sh, uncommitted diff in analysis_runtime.py (small guard tweak on FINISH rehydration; not assessed in depth).

## NOT inspected
native_llama/client.py load path internals (only close/cancel), persistent_mtp.py, bindings ctypes lifetime/ABI, model_download.py resume logic details, sessions/evidence-store growth (analysis_runtime.py is 8000+ lines; EvidenceStore memory growth unaudited), llama_server.py backend retries beyond grep, build_cli.py, the full test suite, any live model/server run.
