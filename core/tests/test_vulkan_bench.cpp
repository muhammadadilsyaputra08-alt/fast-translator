#include "vulkan_bench.h"
#include <iostream>
#include <string>

// CI hanya bisa menguji KOREKTNES shader (via Mesa llvmpipe, software Vulkan)
// -- bukan klaim kecepatan, karena itu sepenuhnya bergantung pada GPU nyata
// yang tidak ada di runner CI. Kalau tidak ada Vulkan device sama sekali di
// runner, itu bukan kegagalan test (skip), karena environment CI bervariasi.
int main() {
    std::string report = vulkan_bench::run();
    std::cout << report << "\n";

    if (report.find("[GAGAL]") != std::string::npos) {
        std::cout << "[SKIP] Tidak ada Vulkan device di runner ini -- korektnes tidak bisa diuji, "
                     "tapi ini bukan kegagalan CI.\n";
        return 0;
    }
    if (report.find("[FAIL korektnes") != std::string::npos) {
        std::cout << "[FAIL] Shader Vulkan menghasilkan output BEDA dari referensi CPU.\n";
        return 1;
    }
    if (report.find("[PASS korektnes]") == std::string::npos) {
        std::cout << "[FAIL] Tidak ketemu penanda PASS/FAIL korektnes di laporan -- format berubah?\n";
        return 1;
    }
    std::cout << "[PASS] Shader Vulkan korektnes terverifikasi.\n";
    return 0;
}
