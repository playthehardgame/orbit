# Underdog Saluki 27B qualification

## Identity

The repository contains a dedicated native profile:

- profile: `orbit-qwen38-saluki-native-v1`
- registry id: `underdog-saluki-27b-iq2-xs`
- source: `ConwayResearch/Underdog-Saluki-27B-1.0`
- file: `Underdog-Saluki-27B-1.0-IQ2-mix.gguf`
- GGUF SHA-256: `4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb`

The verified GGUF identity is intentionally separate from the full-size
Qwen3.8 Q4_K_M profile:

| Field | Verified value |
|---|---|
| `general.architecture` | `qwen35` |
| `general.name` | `Qwen3.8-27B GSQ-RCO` |
| `general.file_type` | `20` (`IQ2_XS`) |
| `general.quantization_version` | `2` (recorded, not checked by detection) |
| `qwen35.context_length` | `262144` |
| `qwen35.block_count` | `64` |
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen35` |
| BOS / EOS / padding | `248044` / `248046` / `248044` |
| embedded template SHA-256 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |

Template drift, model-name drift, tokenizer drift, quantization drift and
geometry drift fail closed. The profile uses the reviewed Qwen XML tool
envelope and supports thinking. CHAT route calls use the rolling route
checkpoint (qualified below); the fixed-head route prefix
(`route_prefix_reuse_supported`) and MTP remain disabled. The optional Saluki
`mmproj` was not qualified and is not registered.

## Qualification result

Performed on 2026-10-09 using the local GGUF above, CPU-only, native Orbit
server, context 4096, six threads, batch 256, ubatch 128, repack off, thinking
off, temperature 0:

| Gate | Result |
|---|---|
| GGUF load and profile detection | PASS |
| `/props` reports the Saluki profile and template identity | PASS |
| deterministic chat (`SALUKI_CHAT_OK`) | PASS |
| native tool call with an empty JSON argument object | PASS |
| MTP | intentionally disabled |
| fixed-head route-prefix reuse | intentionally disabled |
| rolling route checkpoint | PASS (2026-10-12, below) |
| multimodal/mmproj | not qualified |

Observed tool response:

```json
{"name":"get_current_time","arguments":"{}"}
```

## Rolling route checkpoint qualification (2026-10-12)

Same GGUF (SHA-256 above), CPU-only WSL2 host, context 4096, six threads,
batch 256, ubatch 128, repack off, thinking off, greedy decoding.

`qwen35` here is a hybrid memory: full attention plus Gated DeltaNet
recurrent layers. A partial `llama_memory_seq_rm` is refused, so without a
whole-sequence checkpoint every route call after the tools-free final call
falls to a cold prefill. Admission follows the Qwen3.8 Flash Next precedent:
membership in `ROLLING_ROUTE_PROFILE_IDS` only. The rolling identity already
carries the profile id, model path, template hash and GGUF metadata, so Saluki
checkpoints are never shared with the full Qwen3.8 27B profile.

State round-trip (`llama_state_seq_get_data` / `set_data`, ChatML route
prefix of 1058 tokens plus a 35-token suffix):

| Check | Result |
|---|---|
| checkpoint size / capture / restore | 226,257,244 B / 0.37 s / 0.21 s |
| same-layout repeat is deterministic | equal |
| clear + restore, then suffix: logits hash, argmax, greedy continuation | equal |
| 300 unrelated tokens, no restore: logits | differ |
| restore over the contaminated live state: logits, argmax, continuation | equal |
| `seq_rm(seq0, 1058, -1)` | `false` (refused) |

Live 4-turn CHAT session (`hi` / `hello again` / capital of France, one word /
and of Italy?), tools on, max-tokens 32, fresh server per run:

| Turn | reuse on: in / eval / cached, wall | anchor off (`ORBIT_KV_PREFIX_ANCHOR=off`) |
|---|---|---|
| 1 | 1091 / 1091 / 0, 3m11s | 1091 / 1091 / 0, 4m06s |
| 2 | 1141 / 83 / 1058, 35 s | 1141 / 1141 / 0, 4m19s |
| 3 | 1237 / 154 / 1083, 27 s | 1237 / 1237 / 0, 2m41s |
| 4 | 1279 / 148 / 1131, 28 s | 1279 / 1279 / 0, 2m41s |
| session cache | 3272 (69%) over 8 calls | 0 (0%) |

Both runs produced byte-identical responses with identical per-turn output
token counts (14 / 25 / 1 / 2). Server diagnostics show the fixed-head anchor
`model_profile_ineligible` on every route call, as intended; the reuse comes
entirely from the rolling checkpoint. Wall times were taken on a shared host
and are not a comparable benchmark; token and cache counts are.

Known limits are those of the shared rolling mechanism: only the
`thinking=off` route/final switch preserves the checkpoint, and capture and
restore transiently hold 2-3 copies of the ~226 MB blob. Evidence is kept
untracked under `workdir/diag/saluki_rolling_route/`.

Offline regression coverage is in `tests/test_native_model_profiles.py`,
`tests/test_native_model_registry.py` and
`tests/test_qwen38_rolling_route_reuse.py`.
