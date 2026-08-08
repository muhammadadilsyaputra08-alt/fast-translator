#include "gptq_ternary.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace gptq {

Hessian Hessian::from_activations(const std::vector<std::vector<float>>& samples, int d_in, double damping) {
    Hessian h;
    h.d_in = d_in;
    h.H.assign(static_cast<size_t>(d_in) * d_in, 0.0);

    // H = X^T X accumulated over all calibration samples.
    for (const auto& x : samples) {
        for (int i = 0; i < d_in; ++i) {
            double xi = x[i];
            if (xi == 0.0) continue;
            double* row = &h.H[static_cast<size_t>(i) * d_in];
            for (int j = 0; j < d_in; ++j) row[j] += xi * x[j];
        }
    }

    // Damping: add a small multiple of the mean diagonal value, standard
    // GPTQ practice to keep the Hessian well-conditioned/invertible even
    // when calibration sample count is modest relative to d_in.
    double mean_diag = 0.0;
    for (int i = 0; i < d_in; ++i) mean_diag += h.H[static_cast<size_t>(i) * d_in + i];
    mean_diag = (d_in > 0) ? mean_diag / d_in : 1.0;
    double eps = damping * std::max(mean_diag, 1e-6);
    for (int i = 0; i < d_in; ++i) h.H[static_cast<size_t>(i) * d_in + i] += eps;

    // Gauss-Jordan inversion with partial pivoting. d_in <= 768 here, so
    // O(d_in^3) is a fraction of a second per layer at this scale.
    std::vector<double> A = h.H; // working copy [d_in x d_in]
    std::vector<double> I(static_cast<size_t>(d_in) * d_in, 0.0);
    for (int i = 0; i < d_in; ++i) I[static_cast<size_t>(i) * d_in + i] = 1.0;

    for (int col = 0; col < d_in; ++col) {
        int pivot_row = col;
        double best = std::fabs(A[static_cast<size_t>(col) * d_in + col]);
        for (int r = col + 1; r < d_in; ++r) {
            double v = std::fabs(A[static_cast<size_t>(r) * d_in + col]);
            if (v > best) { best = v; pivot_row = r; }
        }
        if (pivot_row != col) {
            for (int c = 0; c < d_in; ++c) {
                std::swap(A[static_cast<size_t>(col) * d_in + c], A[static_cast<size_t>(pivot_row) * d_in + c]);
                std::swap(I[static_cast<size_t>(col) * d_in + c], I[static_cast<size_t>(pivot_row) * d_in + c]);
            }
        }
        double pivot = A[static_cast<size_t>(col) * d_in + col];
        if (std::fabs(pivot) < 1e-12) pivot = (pivot >= 0 ? 1e-12 : -1e-12); // guard singular
        for (int c = 0; c < d_in; ++c) {
            A[static_cast<size_t>(col) * d_in + c] /= pivot;
            I[static_cast<size_t>(col) * d_in + c] /= pivot;
        }
        for (int r = 0; r < d_in; ++r) {
            if (r == col) continue;
            double factor = A[static_cast<size_t>(r) * d_in + col];
            if (factor == 0.0) continue;
            for (int c = 0; c < d_in; ++c) {
                A[static_cast<size_t>(r) * d_in + c] -= factor * A[static_cast<size_t>(col) * d_in + c];
                I[static_cast<size_t>(r) * d_in + c] -= factor * I[static_cast<size_t>(col) * d_in + c];
            }
        }
    }
    h.Hinv = std::move(I);
    return h;
}

namespace {
float ternary_round(float v, float alpha) {
    float threshold = 0.5f * alpha;
    if (v > threshold) return alpha;
    if (v < -threshold) return -alpha;
    return 0.0f;
}
}

void quantize_ternary_gptq(
    std::vector<float>& W, int d_out, int d_in,
    float alpha,
    const Hessian& hess,
    std::vector<int8_t>& out_codes) {

    out_codes.assign(static_cast<size_t>(d_out) * d_in, 0);

    // Sequential column sweep with GPTQ-style error compensation.
    for (int col = 0; col < d_in; ++col) {
        double d = hess.Hinv[static_cast<size_t>(col) * d_in + col];
        if (std::fabs(d) < 1e-12) d = (d >= 0 ? 1e-12 : -1e-12);

        for (int r = 0; r < d_out; ++r) {
            float* row = &W[static_cast<size_t>(r) * d_in];
            float w = row[col];
            float q = ternary_round(w, alpha);

            out_codes[static_cast<size_t>(r) * d_in + col] =
                (q > 0.0f) ? 1 : (q < 0.0f ? -1 : 0);

            double err = (static_cast<double>(w) - q) / d;

            // Propagate the error into not-yet-quantized columns of this
            // same row, weighted by the corresponding row of Hinv — this
            // is the step that's missing from naive independent rounding,
            // and it's what lets the layer's *output* stay close to the
            // unquantized output even though individual weights are crude.
            if (err != 0.0) {
                const double* hinv_row = &hess.Hinv[static_cast<size_t>(col) * d_in];
                for (int j = col + 1; j < d_in; ++j) {
                    row[j] -= static_cast<float>(err * hinv_row[j]);
                }
            }
        }
    }
}

} // namespace gptq
