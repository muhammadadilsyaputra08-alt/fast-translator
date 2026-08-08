#pragma once
#include "fllm_format.h"
#include "kv_cache.h"
#include <vector>

namespace transformer_mixed {

// Exposed at namespace scope (not file-local) so neon_selftest.cpp can
// cross-check the NEON path against the scalar reference on-device — this
// sandbox has no ARM cross-compiler, so NEON correctness can't be verified
// here and must be proven empirically on real hardware instead. See
// notes/NEON_OPTIMIZATION_FINDINGS.md.
void matmul_int8_scalar(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void matmul_int8_neon(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out);
#endif

// Same shape as transformer_bitnet::RunState.
struct RunState {
    std::vector<float> x, xb, xb2;
    std::vector<float> hb, hb2;
    std::vector<float> q, k, v;
    std::vector<float> att;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    void init(const fllm::TransformerConfig& c);
};

// Milestone 5: mixed-precision forward pass. Attention weights (wq/wk/wv/wo)
// are read as standard symmetric INT8 (real multiply-accumulate, one scale
// per layer) — much higher fidelity than ternary. FFN weights (w1/w2/w3)
// stay ternary (add/sub/skip), same as transformer_bitnet. Embedding/norms/
// RoPE stay fp32, same convention as the other two forward passes.
//
// Reuses the same fllm::QuantizedTransformerWeights storage — the wq/wk/
// wv/wo int8_t arrays just hold real -127..127 magnitudes instead of
// {-1,0,1} sign codes; the ternary FFN arrays are unchanged.
class MixedPrecisionTransformer {
public:
    enum class FfnPrecision { Ternary, Int8 };

    explicit MixedPrecisionTransformer(const fllm::FllmModel& model, FfnPrecision ffn_precision = FfnPrecision::Ternary);
    float* forward(int token, int pos);

    // Processes n_tokens consecutive positions (start_pos..start_pos+n_tokens-1)
    // in one call, batching the Q/K/V/FFN matmuls across all n_tokens so each
    // weight row is loaded once and reused across the batch instead of once
    // per token — this is the building block speculative decoding needs
    // ("verify several draft tokens in one pass"). Returns one logits vector
    // per input position, in order. Causal masking is preserved: position
    // start_pos+t attends to everything in the KV cache up to and including
    // itself, same as calling forward() n_tokens times sequentially would.
    std::vector<std::vector<float>> forward_batch(const std::vector<int>& tokens, int start_pos);

    // Same as forward(), but the KV cache is bounded: once the number of
    // cached positions exceeds `evict_threshold`, RocketKVEvictor decides
    // which positions survive (recent window + highest L2-norm "salient"
    // keys from layer 0, applied uniformly across all layers so every
    // layer keeps the same set of time-steps). Evicted positions are
    // physically removed from the cache arrays -- real memory freed, not
    // just masked out. `true_pos` is the token's real absolute position
    // (used for RoPE), independent of how many earlier positions have
    // already been evicted.
    float* forward_evictable(int token, int true_pos, const kvcache::RocketKVEvictor& evictor);

    // Bytes currently held by the KV cache (sum over all layers of
    // active_positions_.size() * dim * 2 * sizeof(float)) -- lets callers
    // measure real memory savings from eviction, not just infer it.
    size_t kv_cache_bytes() const;

    const fllm::TransformerConfig& config() const { return config_; }

private:
    const fllm::TransformerConfig& config_;
    const fllm::QuantizedTransformerWeights& w_;
    RunState state_;
    FfnPrecision ffn_precision_;

    // Evictable-cache state (only used by forward_evictable()). Growable,
    // one entry per currently-active position, shared indexing across all
    // layers (evicting a position removes it from every layer at once).
    std::vector<int> active_positions_;
    std::vector<std::vector<float>> evict_key_cache_;   // [n_layers][num_active * dim]
    std::vector<std::vector<float>> evict_value_cache_; // [n_layers][num_active * dim]
};

} // namespace transformer_mixed
