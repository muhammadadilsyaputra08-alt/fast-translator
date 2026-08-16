#include "transformer_mixed.h"
#include <iostream>
#include <random>
#include <vector>

namespace transformer_mixed {
void matmul_int8_scalar(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                         int o_begin, int o_end);
void matmul_int8(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out);
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> distf(-2.0f, 2.0f);
    std::uniform_int_distribution<int> disti(-127, 127);

    // Ukuran-ukuran yang sengaja tidak habis dibagi jumlah thread, plus
    // kasus di bawah & di atas ambang paralelisasi (kMinParallelWork=64).
    std::vector<std::pair<int,int>> sizes = {
        {16, 16}, {64, 63}, {128, 129}, {256, 257}, {2048, 2048}, {2048, 256}, {512, 32000}
    };

    bool all_ok = true;
    for (auto [d_in, d_out] : sizes) {
        std::vector<float> x(d_in);
        for (auto& v : x) v = distf(rng);
        std::vector<int8_t> w(static_cast<size_t>(d_in) * d_out);
        for (auto& v : w) v = static_cast<int8_t>(disti(rng));
        float scale = 0.013f;

        std::vector<float> out_ref(d_out), out_parallel(d_out);
        transformer_mixed::matmul_int8_scalar(out_ref.data(), x.data(), w.data(), scale, d_in, d_out, 0, d_out);
        transformer_mixed::matmul_int8(out_parallel.data(), x.data(), w.data(), scale, d_in, d_out);

        bool identical = (out_ref == out_parallel);
        std::cout << "d_in=" << d_in << " d_out=" << d_out << ": "
                  << (identical ? "[PASS] bit-identik" : "[FAIL] BEDA dari referensi single-thread") << "\n";
        if (!identical) all_ok = false;
    }

    // Panggil berkali-kali beruntun (simulasi forward pass nyata: ~7 matmul
    // per layer x 22 layer) untuk kasih ThreadSanitizer banyak kesempatan
    // menangkap race condition kalau ada, bukan cuma sekali jalan.
    std::cout << "\nMenjalankan 200 panggilan matmul_int8(dim=2048) beruntun...\n";
    std::vector<float> x2(2048), out2(2048);
    for (auto& v : x2) v = distf(rng);
    std::vector<int8_t> w2(static_cast<size_t>(2048) * 2048);
    for (auto& v : w2) v = static_cast<int8_t>(disti(rng));
    for (int i = 0; i < 200; ++i) {
        transformer_mixed::matmul_int8(out2.data(), x2.data(), w2.data(), 0.02f, 2048, 2048);
    }
    std::cout << "[PASS] 200 panggilan selesai tanpa crash/hang\n";

    if (!all_ok) { std::cout << "\nADA KEGAGALAN\n"; return 1; }
    std::cout << "\nSemua ukuran cocok dengan referensi single-thread.\n";
    return 0;
}
