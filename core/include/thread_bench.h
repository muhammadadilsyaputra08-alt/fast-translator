#pragma once
#include <string>

namespace thread_bench {

// Menjalankan matmul_int8 (2048x2048, ukuran sama dengan wq/wo TinyLlama)
// berkali-kali dengan beberapa konfigurasi jumlah thread berbeda (1, 2, 3,
// 4, 6, 8 -- mencakup skenario "cuma pakai core besar" sampai "pakai semua
// core"), lalu melaporkan rata-rata waktu per konfigurasi. Dibuat karena
// hasil di device (Dimensity 6300, big.LITTLE 2xA76+6xA55) menunjukkan
// threading naif (equal-split ke semua 8 core) justru memperlambat
// (1,31 tok/s vs baseline 1,43 tok/s) -- daripada menebak ulang jumlah
// thread yang optimal, ukur langsung di hardware sebenarnya.
std::string run();

} // namespace thread_bench
