#include "kv_cache.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace kvcache {

float MiniCacheCompressor::cosine_similarity(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0f;
    double dot = 0.0, nx = 0.0, ny = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        dot += static_cast<double>(x[i]) * y[i];
        nx += static_cast<double>(x[i]) * x[i];
        ny += static_cast<double>(y[i]) * y[i];
    }
    if (nx == 0.0 || ny == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(nx) * std::sqrt(ny)));
}

bool MiniCacheCompressor::try_merge(const KVEntry& a, const KVEntry& b, KVEntry* merged_out, std::vector<float>* residual_out) const {
    float sim_k = cosine_similarity(a.key, b.key);
    float sim_v = cosine_similarity(a.value, b.value);
    float sim = std::min(sim_k, sim_v);

    if (sim < threshold_) return false;

    // Merge: average key/value, store b's deviation as a residual so the
    // original can be approximately reconstructed if needed.
    KVEntry merged;
    merged.key.resize(a.key.size());
    merged.value.resize(a.value.size());
    for (size_t i = 0; i < a.key.size(); ++i) merged.key[i] = 0.5f * (a.key[i] + b.key[i]);
    for (size_t i = 0; i < a.value.size(); ++i) merged.value[i] = 0.5f * (a.value[i] + b.value[i]);

    std::vector<float> residual;
    residual.reserve(b.key.size() + b.value.size());
    for (size_t i = 0; i < b.key.size(); ++i) residual.push_back(b.key[i] - merged.key[i]);
    for (size_t i = 0; i < b.value.size(); ++i) residual.push_back(b.value[i] - merged.value[i]);

    if (merged_out) *merged_out = merged;
    if (residual_out) *residual_out = residual;
    return true;
}

std::vector<size_t> RocketKVEvictor::select_kept_indices(const std::vector<KVEntry>& entries) const {
    std::vector<size_t> all_idx(entries.size());
    std::iota(all_idx.begin(), all_idx.end(), 0);

    if (entries.size() <= evict_threshold_) return all_idx; // nothing to evict yet

    // Score each entry by L2 norm of its key vector (proxy for salience).
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        double norm = 0.0;
        for (float v : entries[i].key) norm += static_cast<double>(v) * v;
        scored.emplace_back(static_cast<float>(std::sqrt(norm)), i);
    }

    // Always keep the most recent tokens (last keep_top_n_/2) plus the
    // highest-scoring ones from the rest — mirrors RocketKV's hybrid
    // "recent window + top salient" eviction policy.
    size_t recent_window = keep_top_n_ / 2;
    size_t recent_start = entries.size() > recent_window ? entries.size() - recent_window : 0;

    std::vector<bool> keep(entries.size(), false);
    for (size_t i = recent_start; i < entries.size(); ++i) keep[i] = true;

    std::vector<std::pair<float, size_t>> candidates;
    for (size_t i = 0; i < recent_start; ++i) candidates.push_back(scored[i]);
    std::sort(candidates.begin(), candidates.end(), std::greater<>());

    size_t remaining_budget = keep_top_n_ > recent_window ? keep_top_n_ - recent_window : 0;
    for (size_t i = 0; i < candidates.size() && i < remaining_budget; ++i) keep[candidates[i].second] = true;

    std::vector<size_t> kept;
    for (size_t i = 0; i < entries.size(); ++i) if (keep[i]) kept.push_back(i);
    return kept;
}

} // namespace kvcache
