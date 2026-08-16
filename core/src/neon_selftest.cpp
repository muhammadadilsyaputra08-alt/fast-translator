#include "neon_selftest.h"
#include "transformer_mixed.h"
#include <vector>
#include <random>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace neon_selftest {

std::string run() {
    std::ostringstream report;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    report << "NEON: compiled in (matmul_int8 uses vectorized path)\n\n";

    std::mt19937 rng(42); // fixed seed: reproducible across runs
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> wdist(-127, 127);

    // Sizes deliberately include non-multiples of 8 (the NEON loop
    // processes 8 elements at a time) so the scalar remainder-handling
    // path actually gets exercised, not just the vectorized main loop.
    std::vector<int> d_in_sizes = {8, 16, 100, 288, 768, 5, 13};
    int d_out = 4;

    double overall_max_diff = 0.0;
    int total_checks = 0, total_mismatches = 0;

    for (int d_in : d_in_sizes) {
        std::vector<float> x(d_in);
        std::vector<int8_t> w(static_cast<size_t>(d_out) * d_in);
        for (auto& v : x) v = xdist(rng);
        for (auto& v : w) v = static_cast<int8_t>(wdist(rng));
        float scale = 0.037f;

        std::vector<float> out_scalar(d_out), out_neon(d_out);
        transformer_mixed::matmul_int8_scalar(out_scalar.data(), x.data(), w.data(), scale, d_in, d_out);
        transformer_mixed::matmul_int8_neon(out_neon.data(), x.data(), w.data(), scale, d_in, d_out);

        double max_diff_this_size = 0.0;
        for (int o = 0; o < d_out; ++o) {
            double diff = std::fabs(out_scalar[o] - out_neon[o]);
            max_diff_this_size = std::max(max_diff_this_size, diff);
            total_checks++;
            // Tolerance: float32 accumulation order differs between scalar
            // (sequential) and NEON (two parallel lanes summed at the end),
            // so bit-exact equality isn't expected -- a small relative
            // tolerance is the correct bar, not zero difference.
            double tol = 1e-4 * (std::fabs(out_scalar[o]) + 1.0);
            if (diff > tol) total_mismatches++;
        }
        overall_max_diff = std::max(overall_max_diff, max_diff_this_size);
        report << "d_in=" << d_in << " (" << (d_in % 8) << " remainder): max diff = " << max_diff_this_size << "\n";
    }

    report << "\noverall max diff: " << overall_max_diff << "\n";
    report << "mismatches beyond tolerance: " << total_mismatches << " / " << total_checks << "\n";
    report << (total_mismatches == 0 ? "RESULT: PASS -- NEON output matches scalar reference"
                                      : "RESULT: FAIL -- NEON path has a bug, do not trust its output");
#else
    report << "NEON: NOT compiled in on this build (matmul_int8 uses scalar path).\n"
              "This is expected on non-ARM builds; the arm64-v8a Android build should show NEON compiled in.";
#endif

    return report.str();
}

} // namespace neon_selftest
