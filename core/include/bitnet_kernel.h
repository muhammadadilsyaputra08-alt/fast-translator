#pragma once
#include <cstdint>
#include <vector>

namespace bitnet {

// Ternary weight packing: each weight is one of {-1, 0, +1}, stored as int8
// for simplicity here (a production kernel would pack 2 bits/weight, but the
// arithmetic and correctness are identical — the packing is a memory-layout
// optimization, not a numerical one).
using TernaryWeights = std::vector<int8_t>;

// Quantize a float weight matrix to ternary {-1,0,+1} using the BitNet
// absmean thresholding scheme: alpha = mean(|W|), threshold = 0.5 * alpha.
TernaryWeights quantize_ternary(const std::vector<float>& w, float* out_alpha);

// W1.58A8 matmul: ternary weights (1.58-bit) x int8-quantized activations.
// weights: [out_dim x in_dim] ternary, row-major.
// activations: [in_dim] int8 quantized (with a known scale).
// Returns: [out_dim] float32 (dequantized) output.
std::vector<float> matmul_ternary_int8(
    const TernaryWeights& weights,
    int out_dim,
    int in_dim,
    const std::vector<int8_t>& activations,
    float activation_scale,
    float weight_alpha);

// Standard symmetric INT8 weight quantization (255 levels, not ternary):
// scale = max(|W|) / 127, q = round(W / scale) clamped to [-127, 127].
// Used for the "Attention: INT8" tier in the mixed-precision scheme —
// much higher fidelity than ternary, at 8x the storage per weight instead
// of ~1.58x.
std::vector<int8_t> quantize_int8(const std::vector<float>& w, float* out_scale);

} // namespace bitnet
