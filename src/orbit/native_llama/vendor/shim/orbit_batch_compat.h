// Orbit: adapter from the legacy llama_batch (still the shims' native decode
// currency) to upstream's common_batch, which common_speculative_process()
// takes since the llama_batch_ext rework (llama.cpp Nov 2026). Token batches only:
// the speculative path is fed the target decode batch, never a raw embd batch.
#pragma once

#include <vector>

#include "common.h"
#include "llama.h"

// `last_output_only` mirrors the target decode when the target flagged only the last
// row as an output (fill_target_prefill_batch). Upstream's masked nextn rows (gemma4,
// shared memory) are indexed through the target's own output map, so the batch handed
// to common_speculative_process() must carry the same output flags the target decode
// had; otherwise the lookup misses (nullptr) or reads the wrong row.
static inline common_batch orbit_common_batch(llama_context * ctx, const llama_batch & src,
                                              bool last_output_only = false) {
    common_batch out(ctx);
    for (int32_t i = 0; i < src.n_tokens; ++i) {
        const bool output = last_output_only ? (i == src.n_tokens - 1)
                                             : (src.logits ? src.logits[i] != 0 : false);
        const int32_t n_seq = src.n_seq_id ? src.n_seq_id[i] : 1;
        if (n_seq > 1) {
            out.add(src.token[i], src.pos[i],
                    std::vector<llama_seq_id>(src.seq_id[i], src.seq_id[i] + n_seq), output);
        } else {
            out.add(src.token[i], src.pos[i], src.seq_id ? src.seq_id[i][0] : 0, output);
        }
    }
    return out;
}

