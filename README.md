# Orbit

> **Nota sul fork.** Questo fork è uno studio preliminare per affiancare Orbit
> a strumenti come [llmwiki](https://github.com/KnowledgeGarden/llmwiki), con
> l'obiettivo di costruire un piccolo *second brain* locale: un ambiente
> personale, eseguibile offline e basato su modelli locali, capace di
> organizzare conversazioni, documenti, evidenze e conoscenza operativa. Non è
> una dichiarazione di produzione o di qualifica semantica generale; le
> capacità effettive restano limitate ai profili, ai test e alle evidenze
> riportati in questo README e nella documentazione di qualifica.

Orbit is a Python local-AI runtime for CPU-only machines. It combines local
chat, model-selected tools, file workflows and static artifact analysis in a
terminal client. Linux x86_64 is the qualified platform.

The runtime manages tools, sessions, evidence, verification and bounded repair.
The integrated llama.cpp backend handles tokenization, inference, streaming and
model state. Orbit builds its own vendored backend; no separate llama.cpp
installation or external inference server is required.

## Install and quick start

You need Python 3.11 or newer, Git, CMake and a C/C++ build toolchain. ANALYSIS
also requires bubblewrap. Memory and storage requirements depend on the model
and workload.

On Debian/Ubuntu with Python 3.11 or newer:

```bash
sudo apt install git build-essential cmake python3-venv bubblewrap
git clone https://github.com/playthehardgame/orbit.git
cd orbit
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 scripts/build_native.py
```

Start the local server:

```bash
orbit server
```

Choose a model from the menu. If it is missing, Orbit offers to download it.
To store models on another disk, [set the model directory](#configuration-and-model-store)
first. Orbit selects the startup profile; normal use needs no tuning flags.
Loading and any startup warm-up depend on the model. Wait for `listening on`
before connecting.

For temporary request diagnostics, pass a directory to `--log`. Orbit creates
the directory when needed and appends structured request/response events to
`requests.jsonl`; omitting the option keeps logging disabled:

```bash
orbit server --log workdir/server-log
```

To select non-greedy native generation at startup, pass a non-negative
temperature; the default `0` keeps greedy decoding. The value is server-wide:
it overrides any per-request temperature, grammar-constrained calls (required
tool calls, Qwen3-Coder artifacts) stay greedy, and it cannot be combined with `--mtp`. Keep the
default `0` for qualification runs, which assume greedy decoding:

```bash
orbit server --temperature 0.7
```

Leave the server terminal running. In a second terminal, open the same checkout:

```bash
. .venv/bin/activate
orbit
```

The client connects to the local server. Exiting the client leaves the server
running; stop the server with Ctrl-C when finished. See `orbit --help` and
`orbit server --help` for options.

`/status` shows the client and server Orbit commits. If both are known and
different, startup and `/status` display a non-blocking build-mismatch warning.
The native server also publishes its identity in `/props` and its startup log.
Identity is captured from the installed source checkout once per process, so
updating a checkout does not relabel an already-running server. Packages without
Git metadata and older/external servers report an unknown commit. This is a
revision diagnostic, not an integrity check of local edits or compatibility
certification; it never restarts or disconnects a server.

## Recent developments

- **LFM2.5 8B-A1B** is available through the exact verified
  `LFM2.5-8B-A1B-UD-Q4_K_M.gguf`, `LFM2.5-8B-A1B-Q8_0.gguf` and
  `LFM2.5-8B-A1B-UD-Q6_K.gguf` profiles from
  `unsloth/LFM2.5-8B-A1B-GGUF`.
- LFM2.5 has a dedicated `lfm2moe` identity and an isolated rolling route-cache
  identity, so its checkpoints are never shared with Qwen, Ornith or other
  model families. The cache path is enabled in the runtime, while real-model
  round-trip qualification is still pending.
- The model store is configurable across discovery, download and server startup;
  split GGUF downloads support per-shard validation, resume and atomic
  publication.
- Qwen3.8 Flash Next has an exact verified split-GGUF profile, rolling route
  reuse and aligned startup route prewarm. These optimizations remain
  model-specific and do not generalize automatically to every local model.

## Supported models and qualification

The [model registry](src/orbit/native_llama/model_registry.json) contains these
verified native profiles. Verification applies to the exact model,
quantization, template and tested workflows, not every related variant.
This list is not a performance ranking.

| Model | Verified quantization |
|---|---|
| Gemma 4 26B-A4B | Q4_0 |
| Ornith 1.5 35B-A3B | Q4_K_M |
| Qwen 3.6 35B-A3B | Q4_K_M |
| Qwen3-Coder 30B-A3B Instruct | Q4_K_M |
| Qwen3.8 27B | Q4_K_M |
| MiniCPM5 2B | Q4_K_M |
| Qwen3.8 Flash Next | UD-IQ1_M, three shards |
| LFM2.5 8B-A1B | UD-Q4_K_M, Q8_0, UD-Q6_K |
| Underdog Saluki 27B 1.0 | IQ2_XS (IQ2-mix GGUF) |
| Qwen3.8 9B Distill | Q4_K_M |
| ScopeGuard-2 4B | Q8_0 (mradermacher GGUF; scope-guard classifier, identity only, not qualified) |
| NousResearch Arete 27B | Q4_K_M (bartowski GGUF; applicability only, not qualified) |
| Qwen3.8 27B TurboFCFusion Heretic NEO-CODER | Q4_K_M (DavidAU GGUF, plain non-MTP file; identity only, not qualified) |
| Gemma 4 E4B Heretic | Q4_K_M (BF16 mmproj; image/audio smoke-tested) |
| Granite 4.2 3B | Q4_K_M, Q8_0 |
| Granite 4.2 8B | Q4_K_M, Q6_K |

**Qualification report diagnostics.** A fixture stopped by an exception carries
`error_detail` (bounded message and traceback tail) in the report JSON. The `route` call
of a failing route fixture, and the first route-phase call of a failing `ask_auto`-based
fixture (full-request, document and artifact fixtures; recorded by a pass-through backend
proxy, so routing, prompts and results are unchanged), carry `output_excerpt`, the raw
model output (or tool calls as JSON when the content is empty) capped at 2000
characters with control characters stripped; it is model output stored locally in the report under the output directory,
so treat reports as local artifacts. The model-matrix driver keeps the last 256 KiB of
the server log as `<model-id>.server.log` for non-PASS rows (`--keep-server-logs` keeps
PASS logs too) and records its path as `server_log_path`.

**Backend/CHAT qualification is separate from ANALYSIS semantic qualification.**
A working model integration and valid tool calls do not establish that its
explanations are correct.

Qwen3.8 9B Distill Q4_K_M uses its own verified profile; CHAT route calls use
the rolling route checkpoint, taken at the turn boundary because its template
drops the generation prompt's tail from history, after a real-model state
round-trip qualification. MTP and the fixed-head route prefix are not
qualified; see the
[checkpoint](docs/checkpoints/qwen38-9b-distill-qualification.md).

Granite 4.2 3B and 8B use the verified native Granite 4.2 profile. Their
qualified route-prefix reuse is enabled for the quantizations listed above;
other Granite variants remain unsupported until separately verified.

Ornith has bounded ANALYSIS qualification on specific retained tests, but
retained reports also contain factual errors. It is not generally qualified
for accurate explanations across the semantic corpus.

Qwen3.8 Flash Next UD-IQ1_M retains backend/CHAT qualification on the tested
Dell configuration. It did not pass the retained ANALYSIS semantic cases.
That result does not extend to every Qwen model or quantization.

LFM2.5 8B-A1B uses a distinct verified profile and rolling route-cache identity;
its cache round-trip still requires real-model qualification before performance
or broader workflow claims are made.

Underdog Saluki 27B 1.0 (a Qwen3.8-27B derivative) uses its own verified
profile, separate from the full Qwen3.8 27B Q4_K_M profile, and CHAT route
calls use the rolling route checkpoint after a real-model state round-trip
qualification. MTP, the fixed-head route prefix and the optional `mmproj` are
not qualified; see the
[Saluki checkpoint](docs/checkpoints/underdog-saluki-qualification.md).

NousResearch Arete 27B (a Qwen3.8-27B finetune, bartowski Q4_K_M GGUF) has its
own verified profile, matched against the identity read from the real GGUF
header. Its embedded chat template is identical to the reviewed Saluki one. It
has not been run through Orbit: rolling route reuse, the fixed-head route
prefix, MTP and the optional `mmproj` are all off and unqualified.

DavidAU Qwen3.8 27B TurboFCFusion Heretic NEO-CODER (a Qwen3.8-27B finetune/merge,
plain non-MTP Q4_K_M GGUF) has its own verified profile, matched against the
identity read from the real GGUF header (`general.name` is "Qwen3.8 27B Brainwaves
NM HERETIC BR LOA1", not the repository name). Its embedded chat template is
different from the Saluki/Arete one and was reviewed by reading it, not by a live
run. It has not been run through Orbit: rolling route reuse, the fixed-head route
prefix, MTP and the optional `mmproj` are all off and unqualified. The repository's
MTP files are not covered by this profile.

See the [qualification scope](docs/ANALYSIS_QUALIFICATION.md) and
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md) for evidence and limits.
No model in this table is presented as generally semantically qualified for
ANALYSIS.

## Chat and tools

Ask a question or request a task, for example:
`Use system_info to describe this computer.` Tools are on by default and can
run local shell commands; the model chooses when to use them. Use `/tools off`
for chat without tools and `/tools on` to restore access.

| Client command | Purpose |
|---|---|
| `/help` | List commands and their syntax. |
| `/status` | Inspect the active model and runtime settings. |
| `/read <source> [prompt]` | Read a local document or URL. |
| `/reset` | Clear the conversation and saved session. |
| `/exit` | Close the client. |

## ANALYSIS and its limits

Use `/analysis path/to/artifact` to investigate one local artifact, and `/chat`
to return to chat. ANALYSIS combines deterministic extraction and decoding with
model-guided investigation. Generated Python actions run in a resource-bounded
bubblewrap sandbox with a read-only input and network access denied.

The canonical Markdown report puts attested indicators and decoded values
before unverified model answers. It retains provenance, evidence coverage,
open questions and the reason the investigation stopped. A complete report
records the work performed; it does not mean the investigation answered every
question or established all of the artifact's behavior.

Source acquisition, delivery to the model and proof of a conclusion are distinct.
Review model interpretations against the evidence, especially claims about
remote content or behavior not observed. The
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md) records required facts,
known errors and unknowns; automatic integrity checks do not replace semantic
review.

## Configuration and model store

Downloads and the server use the same model directory: `models/` in a checkout,
or `~/.cache/orbit/models` outside one. To choose a writable location:

```bash
orbit config models-dir /mnt/data/orbit-models
orbit config models-dir
```

The first command creates the directory if needed and saves the setting; the
second shows the effective directory. Existing models are not moved. When
copying models, retain the `<owner>--<repo>/` subdirectories.

Precedence is an explicit command's `--models-dir`, then `ORBIT_MODELS_DIR`,
then the saved setting, then the default.

You can also download before starting the server, for example:

```bash
orbit download ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
```

For a multi-shard GGUF, request its first shard. Orbit downloads the complete
set, reports per-shard progress, resumes interrupted transfers and reuses
complete shards. Keep the shards together; they appear as one model in the
menu. Rerun the same command to finish an interrupted set. Exact repository and
GGUF names are in the registry; options are in `orbit download --help`.

Single files and shards retain a `.part` beside the destination across
interruptions. Rerun the same command to resume; progress includes the bytes
already downloaded. Transfers use a 30-second socket inactivity timeout and
at most four attempts with bounded backoff. Only a complete transfer matching
the declared size is published by atomic rename. Existing final files are reused.
See [download recovery](docs/DOWNLOAD_RECOVERY.md) for server compatibility and
integrity limits.

### Server limits and shutdown

`orbit server` rejects a request whose `Content-Length` is malformed or larger
than 64 MiB with HTTP 400 before reading the body. Set `ORBIT_MAX_REQUEST_BYTES`
to a positive byte count to change the limit; invalid or non-positive values
fall back to the default. `SIGTERM` and `SIGHUP` (a service stop, `kill`,
`wsl --shutdown`) stop the server through the same path as Ctrl-C: the model
context is freed and `server_stop` is logged. At most 64 POST requests are handled or queued at once; further ones
get HTTP 503 (`server busy`) so a client can back off, and `/health` stays
available. `ORBIT_MAX_INFLIGHT_REQUESTS` changes the cap. Shell tool output is captured up
to 8 MiB per stream; anything beyond that is read and discarded, and a marker
says so.

With `--log DIR`, the log directory is created owner-only (`0700`) and
`requests.jsonl` is `0600`, because prompts are logged verbatim. Its first row,
`server_start`, is the run manifest: argv, alias, model file path/size/mtime,
build identity, Python and host details, memory, and every `ORBIT_*` variable in
effect (values of names containing TOKEN, KEY, SECRET, PASSWORD or CREDENTIAL are
replaced by `[redacted]`). Set `ORBIT_AUDIT_HASH_MODEL=1` to also record the
model file's SHA-256; this reads the whole file and delays startup accordingly.

At startup the server removes analysis workspaces (`orbit-analysis-session-*` in
the temporary directory, owned by the same user) that a crashed or killed
process left behind and that have not been modified for 7 days. Set
`ORBIT_ANALYSIS_SWEEP_DAYS` to change the threshold; `0` disables the sweep.

**Timeouts (all off by default).** `ORBIT_SOCKET_READ_TIMEOUT=<seconds>` bounds
how long a client may take to send its request (headers and body); a stalled
body gets HTTP 408 and the connection is closed. The limit is lifted before
generation starts, so streaming responses are never cut by it.
`ORBIT_MODEL_LOCK_TIMEOUT=<seconds>` bounds how long a request waits for the
model (generation is serialized); past it the request gets HTTP 503
(`model busy`). Non-positive or invalid values mean "no timeout".

**Memory guard.** `ORBIT_MEMORY_GUARD=warn` (default) prints an advisory line
when the model plus KV window will not fit in the RAM free right now. `refuse`
stops the start with an error instead (`--show-profile` still only reports);
`off` silences it. The default stays advisory because beyond-RAM models are
qualified to run mmap'd from disk.

**Log framework.** The request log uses Python's `logging` with a rotating,
owner-only file: `ORBIT_LOG_MAX_BYTES` (default 64 MiB; rotation is disabled
with `ORBIT_LOG_BACKUPS=0`) and `ORBIT_LOG_BACKUPS` (default 7).
`ORBIT_LOG_CONTENT=full|redact|none` controls prompt retention: `full`
(default) keeps text verbatim, `redact` and `none` replace message/prompt text
with its length and SHA-256. Credentials (bearer tokens, `hf_`/`sk-`/`gh*_`
keys, private-key headers) are scrubbed in every mode. Each row carries `seq`
and a SHA-256 hash chained to the previous row, across rotated files and
restarts; `python -m orbit.native_server.request_logging verify DIR` reports
removed, reordered or edited rows.

**Model integrity.** The packaged registry pins the size and SHA-256 of model
files (from Hugging Face LFS metadata). `orbit download` verifies a fresh
download against its pin and deletes the file on a mismatch.

**Platform-aware tuning.** The heuristic thread/batch choice recognises WSL2,
VMs, containers (cgroup quota and CPU affinity), hybrid CPUs (performance vs.
efficiency cores), ISA features and server-class parts. Decode threads track
performance cores up to a bandwidth-bound ceiling; prompt processing uses all
fast cores. It only feeds the heuristic tier and the calibration sweep's
candidate list; explicit settings, qualified profiles, the cache and
measurements still take precedence. See `src/orbit/native_server/host_class.py`.

## Development and tests

For the lightweight, artifact-free [core CI gate](docs/CORE_CI.md), run:

```bash
python3 scripts/check_core.py
```

The project uses `unittest`. From the checkout, run the non-live suite:

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Some qualification tests require external, hash-pinned corpus files. Follow
[AGENTS.md](AGENTS.md) for provisioning and contribution gates. ANALYSIS changes
must preserve the [deterministic cross-sample gate](tests/test_analysis_cross_sample_gate.py);
claims of semantic improvement also require comparison against the
[semantic baseline](docs/ANALYSIS_SEMANTIC_BASELINE.md), with no regressions on
applicable criteria.

The [tool-intent qualification gate](docs/TOOL_INTENT_QUALIFICATION.md) checks
24 frozen bilingual intents with tool execution intercepted. Its retained-response
replay runs offline and emits JSON and Markdown capability results.
