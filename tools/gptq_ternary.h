#pragma once
#include <vector>
#include <cstdint>

namespace gptq {

// A calibration Hessian: H = X^T X + damping*I, computed from calibration
// activation vectors X [n_samples x d_in] that feed into a given weight
// matrix. Shared across all output rows of that matrix (matches the GPTQ
// paper: the Hessian only depends on the layer's *input* distribution).
struct Hessian {
    int d_in = 0;
    std::vector<double> H;    // [d_in x d_in], row-major
    std::vector<double> Hinv; // [d_in x d_in], row-major inverse

    // Builds H from a set of calibration activation vectors and inverts it
    // (Gauss-Jordan with partial pivoting; d_in is small enough here —
    // at most 768 — that this runs in well under a second per layer).
    static Hessian from_activations(const std::vector<std::vector<float>>& samples, int d_in, double damping);
};

// Quantizes weight matrix W [d_out x d_in] (row-major, modified in place as
// scratch) into ternary codes {-1,0,+1} using the GPTQ sequential-column
// algorithm: quantize column i, measure the error, and propagate it into
// not-yet-quantized columns (i+1..d_in-1) via the Hessian inverse — this is
// what lets a small number of "sacrificial" weights absorb the rounding
// error of the whole layer instead of every weight independently
// contributing its own uncorrected error (which is what made the naive
// absmean-threshold approach collapse to ~56% relative error).
//
// `alpha` is a single scale for the WHOLE matrix (not per-row), matching
// the existing .fllm storage convention (one alpha per layer per tensor).
// It's computed upfront (e.g. via bitnet::quantize_ternary's absmean rule)
// and held fixed through the sequential pass, same as GPTQ fixing its
// quantization grid before running the column sweep.
void quantize_ternary_gptq(
    std::vector<float>& W, // [d_out x d_in], mutated (used as scratch during compensation)
    int d_out, int d_in,
    float alpha,
    const Hessian& hess,
    std::vector<int8_t>& out_codes);   // [d_out x d_in]

} // namespace gptq

