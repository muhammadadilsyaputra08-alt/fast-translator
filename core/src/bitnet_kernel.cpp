#include "bitnet_kernel.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace bitnet {

TernaryWeights quantize_ternary(const std::vector<float>& w, float* out_alpha) {
    double sum_abs = 0.0;
    for (float v : w) sum_abs += std::fabs(v);
    float alpha = w.empty() ? 0.0f : static_cast<float>(sum_abs / w.size());
    float threshold = 0.5f * alpha;

    TernaryWeights out(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] > threshold) out[i] = 1;
        else if (w[i] < -threshold) out[i] = -1;
        else out[i] = 0;
    }
    if (out_alpha) *out_alpha = alpha;
    return out;
}

std::vector<float> matmul_ternary_int8(
    const TernaryWeights& weights,
    int out_dim,
    int in_dim,
    const std::vector<int8_t>& activations,
    float activation_scale,
    float weight_alpha) {

    std::vector<float> result(out_dim, 0.0f);
    for (int o = 0; o < out_dim; ++o) {
        // Ternary weights mean the "multiply" degenerates into add/sub/skip,
        // which is the entire point of BitNet: no real multiplication needed.
        int32_t acc = 0;
        const int8_t* row = &weights[static_cast<size_t>(o) * in_dim];
        for (int i = 0; i < in_dim; ++i) {
            int8_t wv = row[i];
            if (wv == 1) acc += activations[i];
            else if (wv == -1) acc -= activations[i];
            // wv == 0 contributes nothing
        }
        result[o] = static_cast<float>(acc) * activation_scale * weight_alpha;
    }
    return result;
}

std::vector<int8_t> quantize_int8(const std::vector<float>& w, float* out_scale) {
    float max_abs = 0.0f;
    for (float v : w) max_abs = std::max(max_abs, std::fabs(v));
    float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;

    std::vector<int8_t> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        int q = static_cast<int>(std::lround(w[i] / scale));
        q = std::max(-127, std::min(127, q));
        out[i] = static_cast<int8_t>(q);
    }
    if (out_scale) *out_scale = scale;
    return out;
}

} // namespace bitnet
