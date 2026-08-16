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
// o_begin/o_end select a sub-range of output neurons [o_begin, o_end) to
// compute; default (o_end=-1) means "the full [0, d_out) range" -- kept as
// trailing optional params so existing single-range callers (e.g.
// neon_selftest.cpp, which cross-checks these two functions directly) don't
// need to change. Used internally by matmul_int8() to split work across
// threads (core/include/thread_pool.h): each thread computes a disjoint
// sub-range of output rows, so results are bit-identical to the single-
// threaded full-range call -- no floating-point reduction is ever split
// across threads.
void matmul_int8_scalar(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                         int o_begin = 0, int o_end = -1);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void matmul_int8_neon(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                       int o_begin = 0, int o_end = -1);
#endif

// Same shape as transformer_bitnet::RunState. q sized by `dim` (n_heads *
// head_size); k/v and the KV cache sized by `kv_dim` (n_kv_heads *
// head_size) to support GQA — when n_kv_heads == n_heads, kv_dim == dim,
// same as before GQA support was added.
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
    // NOTE: MHA-only (assumes n_kv_heads == n_heads) — not yet updated for
    // GQA. Not on the production path (only used by the speculative-
    // decoding experiments in notes/SPECULATIVE_DECODING_FINDINGS.md,
    // already concluded not worth pursuing further on this hardware), so
    // left as-is rather than risk errors under time pressure. Update this
    // if speculative decoding is revisited for a GQA model.
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

    // Clears RocketKV eviction state (evict_key_cache_/evict_value_cache_/
    // active_positions_). MUST be called before starting a new, unrelated
    // generate() call -- otherwise forward_evictable() keeps appending onto
    // the previous call's cache instead of starting fresh, causing later
    // generations to attend over stale tokens from earlier conversations.
    void reset_evictable_cache();

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
