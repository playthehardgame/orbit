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
| rolling route checkpoint | not qualified (not in `ROLLING_ROUTE_PROFILE_IDS`) |
| multimodal/mmproj | none shipped |

Observed speed (shared host, not a benchmark): prefill ~21 tok/s, decode
5-7 tok/s. The model card recommends temperature 0.6 / top_p 0.95 / top_k 20;
qualification used greedy decoding.
