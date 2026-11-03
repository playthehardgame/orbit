# Qwen3.8 9B Distill qualification

## Identity

- profile: `orbit-qwen38-9b-distill-native-v1`
- registry id: `qwen38-9b-distill-q4-k-m`
- source: `empero-ai/Qwen3.8-9B-Distill-GGUF`
- file: `Qwen3.8-9B-Q4_K_M.gguf` (5,780,090,176 B)
- GGUF SHA-256: `df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a`
  (matches the repository `SHA256SUMS`)

| Field | Verified value |
|---|---|
| `general.architecture` | `qwen35` |
| `general.name` | `Qwen3.8 9B` |
| `general.file_type` | `15` (`Q4_K_M`) |
| `qwen35.context_length` | `262144` |
| `qwen35.block_count` | `33` |
| `tokenizer.ggml.model` / `pre` | `gpt2` / `qwen35` |
| EOS / padding | `248046` / `248044` (no BOS id key; `add_bos_token=false`) |
| embedded template SHA-256 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

Only Q4_K_M is registered. Q5_K_M, Q6_K, Q8_0 and BF16 have different
`file_type` values and are not qualified. Drift fails closed with
`qwen38_9b_distill_*` reasons. The profile uses the Qwen XML tool envelope
(`<tool_call>/<function=>/<parameter=>`) and supports thinking.

## Qualification result (2026-10-12)

Native Orbit server, CPU-only WSL2, context 4096, six threads, thinking off,
temperature 0, model loaded with `--model`.

| Gate | Result |
|---|---|
| GGUF load and profile detection | PASS (`/props` reports the profile, `Q4_K_M`) |
| deterministic chat (`DISTILL_CHAT_OK`) | PASS |
| required native tool call, empty argument object | PASS (`get_current_time`, `{}`) |
| MTP | intentionally disabled (file carries `nextn_predict_layers=1`, unqualified) |
| fixed-head route-prefix reuse | intentionally disabled |
| rolling route checkpoint | PASS at the turn boundary (2026-10-13, below) |
| multimodal/mmproj | none shipped |

Observed speed (shared host, not a benchmark): prefill ~21 tok/s, decode
5-7 tok/s. The model card recommends temperature 0.6 / top_p 0.95 / top_k 20;
qualification used greedy decoding.

## Rolling route checkpoint qualification (2026-10-13)

State round-trip on this GGUF (`qwen35` hybrid, ChatML route prefix of 1058
tokens plus a 35-token suffix, `llama_state_seq_get_data` / `set_data`):

| Check | Result |
|---|---|
| checkpoint size / capture / restore | 87,385,692 B / 0.17 s / 0.09 s |
| same-layout repeat is deterministic | equal |
| clear + restore, then suffix: logits hash, argmax, greedy continuation | equal |
| 300 unrelated tokens, no restore: logits | differ |
| restore over the contaminated live state: logits, argmax, continuation | equal |
| `seq_rm(seq0, 1058, -1)` | `false` (refused) |

**Why a whole-prompt checkpoint is not enough here.** The embedded template
renders a historical assistant turn without the empty `<think>\n\n</think>\n\n`
block that the route prompt ends with as its generation prompt. The route-1
prompt is therefore not a strict prefix of the route-2 prompt (first differing
character at the start of the assistant turn) and the exact-prefix rule
correctly rejects a checkpoint taken at the end of the prompt. Measured with
only the eligibility gate enabled, every route call had `cached=0`, and the
post-final shadow skipped every turn (`checkpoint_not_a_prefix_of_head`).

**Fix.** `ROLLING_ROUTE_BOUNDARY_PROFILE_IDS = {Qwen3.8 9B Distill}` (a subset
of `ROLLING_ROUTE_PROFILE_IDS`): the CHAT route checkpoint is taken at the turn
boundary, before the generation prompt, using the renderer-reported
generation prompt and `rolling_capture_boundary` (re-tokenized head accepted
only if it is a strict token prefix of the prompt). The same exact-prefix rule
still authorizes every restore. The older checkpoint is kept when a new
boundary does not extend it, and the miss is recorded. Every other profile
keeps the whole-prompt capture unchanged; the post-final shadow is not used.

Live 4-turn CHAT session (`hi` / `hello again` / capital of France, one word /
and of Italy?), tools on, thinking off, max-tokens 32, greedy, fresh server per
run, same host and settings as above:

| Turn | reuse on: in / eval / cached | anchor off (`ORBIT_KV_PREFIX_ANCHOR=off`) |
|---|---|---|
| 1 | 1091 / 1091 / 0 | 1091 / 1091 / 0 |
| 2 | 1133 / 82 / 1051 | 1133 / 1133 / 0 |
| 3 | 1227 / 155 / 1072 | 1227 / 1227 / 0 |
| 4 | 1261 / 142 / 1119 | 1261 / 1261 / 0 |
| session cache | 3242 (69%) over 8 calls | 0 (0%) |

Both runs produced byte-identical responses. Server diagnostics show the
fixed-head anchor `model_profile_ineligible` on every route call, as intended;
the reuse comes entirely from the rolling checkpoint. Wall times were taken on
a shared host and are not a benchmark; token and cache counts are.

Known limits: only the `thinking=off` route/final switch preserves the
checkpoint; capture and restore transiently hold 2-3 copies of the ~87 MB
blob; a route prompt whose boundary the tokenizer splits mid-token falls back
to a cold prefill. Evidence is kept untracked under
`workdir/diag/qwen38_9b_distill_rolling_route/`.

Offline regression coverage: `tests/test_qwen38_rolling_route_reuse.py`,
`tests/test_analysis_rolling_kv_step_slot.py` (route boundary capture) and
`tests/test_native_model_profiles.py`.
