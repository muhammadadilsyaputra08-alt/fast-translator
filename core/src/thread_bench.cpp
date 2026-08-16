#include "thread_bench.h"
#include "transformer_mixed.h"
#include "thread_pool.h"
#include <chrono>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace transformer_mixed {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void matmul_int8_neon(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                       int o_begin, int o_end);
#endif
void matmul_int8_scalar(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                         int o_begin, int o_end);
}

namespace thread_bench {

namespace {

void run_kernel(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                 int o_begin, int o_end) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    transformer_mixed::matmul_int8_neon(out, x, w, scale, d_in, d_out, o_begin, o_end);
#else
    transformer_mixed::matmul_int8_scalar(out, x, w, scale, d_in, d_out, o_begin, o_end);
#endif
}

// Rata-rata waktu satu matmul (dim x dim, sama seperti wq/wo TinyLlama)
// dengan pool berukuran `n_threads`, dirata-rata dari `iterations` run.
// n_threads=1 dijalankan tanpa pool sama sekali (baseline single-thread
// murni, tanpa overhead sinkronisasi apa pun).
double bench_one(int n_threads, int dim, int iterations,
                  const std::vector<float>& x, const std::vector<int8_t>& w, float scale) {
    std::vector<float> out(dim);

    if (n_threads <= 1) {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iterations; ++it) {
            run_kernel(out.data(), x.data(), w.data(), scale, dim, dim, 0, dim);
        }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    }

    threadpool::ThreadPool pool(static_cast<size_t>(n_threads));
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        pool.parallel_for(dim, [&](int begin, int end) {
            run_kernel(out.data(), x.data(), w.data(), scale, dim, dim, begin, end);
        });
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
}

} // namespace

std::string run() {
    constexpr int dim = 2048;      // ukuran wq/wo TinyLlama (dim x dim)
    constexpr int iterations = 30; // per konfigurasi -- cukup untuk rata-rata stabil

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> distf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> disti(-127, 127);

    std::vector<float> x(dim);
    for (auto& v : x) v = distf(rng);
    std::vector<int8_t> w(static_cast<size_t>(dim) * dim);
    for (auto& v : w) v = static_cast<int8_t>(disti(rng));
    float scale = 0.02f;

    unsigned hw = std::thread::hardware_concurrency();

    std::ostringstream out;
    out << "Thread benchmark -- matmul " << dim << "x" << dim << " INT8, "
        << iterations << " iterasi per konfigurasi\n";
    out << "hardware_concurrency() = " << hw << "\n\n";

    std::vector<int> configs = {1, 2, 3, 4, 6, 8};
    double best_ms = -1;
    int best_n = 1;
    for (int n : configs) {
        if (n > 1 && static_cast<unsigned>(n) > hw + 2) continue; // skip yang jauh melebihi core fisik
        double ms = bench_one(n, dim, iterations, x, w, scale);
        out << "threads=" << n << ": " << ms << " ms/matmul\n";
        if (best_ms < 0 || ms < best_ms) { best_ms = ms; best_n = n; }
    }

    out << "\nPaling cepat: threads=" << best_n << " (" << best_ms << " ms/matmul)\n";
    if (best_n == 1) {
        out << "-> Single-thread menang. Threading dinonaktifkan untuk device ini "
               "(overhead sinkronisasi/core lambat > manfaat paralelisasi).\n";
    } else {
        out << "-> Set jumlah thread pool ke " << best_n << " (bukan hardware_concurrency() penuh).\n";
    }
    return out.str();
}

} // namespace thread_bench
