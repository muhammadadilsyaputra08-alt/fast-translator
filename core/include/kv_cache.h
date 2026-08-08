#pragma once
#include <vector>
#include <cstddef>

namespace kvcache {

// A single KV entry for one token at one layer: key + value vectors.
struct KVEntry {
    std::vector<float> key;
    std::vector<float> value;
};

// MiniCache: merges KV states across adjacent layers when they are highly
// similar (cosine similarity above a threshold), storing one shared entry
// plus a small per-layer residual instead of two full entries.
// This is a simplified, CPU-testable version of the technique described in
// the MiniCache paper (cross-layer KV merging in the middle-to-deep layers).
class MiniCacheCompressor {
public:
    explicit MiniCacheCompressor(float merge_similarity_threshold = 0.95f)
        : threshold_(merge_similarity_threshold) {}

    // Attempts to merge entry `b` into `a` if similar enough.
    // Returns true if merged (caller should drop `b` and keep `a` + residual).
    bool try_merge(const KVEntry& a, const KVEntry& b, KVEntry* merged_out, std::vector<float>* residual_out) const;

    static float cosine_similarity(const std::vector<float>& x, const std::vector<float>& y);

private:
    float threshold_;
};

// RocketKV-style: token-level eviction for long contexts. Keeps the top-N
// tokens by an importance score (here: L2 norm of the key vector, as a
// simple proxy for attention-worthiness) and evicts the rest once the
// sequence exceeds `evict_threshold` tokens.
class RocketKVEvictor {
public:
    explicit RocketKVEvictor(size_t evict_threshold, size_t keep_top_n)
        : evict_threshold_(evict_threshold), keep_top_n_(keep_top_n) {}

    // Returns indices (into `entries`) that should be KEPT after eviction.
    std::vector<size_t> select_kept_indices(const std::vector<KVEntry>& entries) const;

    // True once the cache has grown past the point where eviction should
    // run. Exposed so callers can decide when to invoke select_kept_indices
    // without duplicating the threshold value.
    bool should_evict_check(size_t current_size) const { return current_size > evict_threshold_; }

private:
    size_t evict_threshold_;
    size_t keep_top_n_;
};

} // namespace kvcache
