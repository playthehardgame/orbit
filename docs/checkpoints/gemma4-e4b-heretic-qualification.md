# Gemma 4 E4B Heretic qualification

## Identity

- profile: `orbit-gemma4-e4b-heretic-native-v1`
- registry id: `gemma4-e4b-heretic-q4-k-m`
- source: `llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF`
- file: `gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` (5,335,289,696 B)
- GGUF SHA-256: `f9dbbe3bdf396f65aec4e801516a1b868a510d96225417a8aa06763e3c5b97dc`
- mmproj: `gemma-4-E4B-it-mmproj-BF16.gguf` (991,552,000 B), SHA-256
  `ffa64aebf7144bbadd5a1e143b77794d88e76a81cc0168d91de0770eb49e88e0`
  (arch `clip`, `gemma4v` vision + `gemma4a` audio, projection_dim 2560)

| Field | Verified value |
|---|---|
| `general.architecture` | `gemma4` |
| `general.name` | `Gemma 4 E4B It Ultra Uncensored Heretic` |
| `general.file_type` | `15` (`Q4_K_M`) |
| `gemma4.context_length` / `block_count` | `131072` / `42` |
| `gemma4.embedding_length_per_layer_input` | `256` (per-layer-embedding E-series) |
| `tokenizer.ggml.model` | `gemma4` |
| BOS / EOS / padding | `2` / `1` / `0` |
| embedded template SHA-256 | `781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf` |

Only this Q4_K_M is registered. Q5_K_S, Q5_K_M, Q6_K, Q8_0 and BF16 have other
`file_type` values and are not qualified. Drift fails closed with
`gemma4_e4b_heretic_*` reasons. Any other per-layer-embedding Gemma 4 file
(E2B, other E4B builds) no longer inherits the 26B-A4B profile and is refused
with `gemma4_e_series_unqualified`. The profile uses the Orbit Gemma 4 renderer
(`orbit-gemma4`) and the `gemma4-native` tool envelope; MTP, final-prefix reuse
and route-prefix reuse are off (their qualification is specific to the 26B-A4B and was not repeated here).

## Qualification result (2026-10-31)

Native Orbit server started with `--model-id gemma4-e4b-heretic-q4-k-m`,
CPU-only WSL2, context 4096, six threads, `--think off`, temperature 0.

| Gate | Result |
|---|---|
| registry resolution, GGUF + mmproj load | PASS (`/props`: profile verified, `Q4_K_M`, `supports_vision` and `supports_audio` true) |
| deterministic chat (`GEMMA_E4B_CHAT_OK`) | PASS with a visible reasoning preamble (see below) |
| tool call | PASS in the Gemma raw envelope: `<|tool_call>call:get_current_time{}<tool_call|>` (`tool_choice=auto`, 300 max tokens) |
| image input (solid red PNG, "dominant color") | PASS: "red" |
| audio input (2 s 440 Hz WAV, "describe") | PASS: "a low, sustained humming tone" |
| MTP / prefix reuse / route-prefix reuse | intentionally disabled |

Observed speed (shared host, not a benchmark): image answer 9 s, audio 20 s.

## Findings and limits

- **Runtime metadata whitelist.** The first live start failed with
  `gemma4_e4b_heretic_metadata_identity_mismatch` although unit tests and a raw
  header read passed: the server only sees keys in `PROFILE_METADATA_KEYS`, and
  `gemma4.*` keys were missing. Added `gemma4.context_length`,
  `gemma4.block_count`, `gemma4.embedding_length_per_layer_input`.
- **Visible reasoning with thinking off.** The finetune writes a plain-text
  reasoning preamble before the answer ("The user has requested ...") and
  before tool calls; with the answer appended without separator ("...is
  red.Red"). The renderer's empty thought channel does not suppress it. Not
  fixed: model behaviour, not a profile identity issue.
- **`tool_choice=required`** returns "required tool decoding needs a native
  chat bridge profile": the renderer is `orbit-gemma4`, not the Jinja bridge.
  The server returns the raw Gemma envelope in `content` (no server-side parse for this renderer); the runtime parses it.
- Multimodal checks used one synthetic image and one synthetic tone, not a
  corpus; they show the projector path works, not answer quality.
- Not run: agent-loop qualification (`qualify_fresh.py`), thinking on.
