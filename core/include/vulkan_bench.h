#pragma once
#include <string>

namespace vulkan_bench {

// Mengukur overhead round-trip NYATA per panggilan compute Vulkan (upload
// x -> dispatch -> wait -> readback out) untuk matmul 2048x2048, dengan GPU
// context & buffer bobot yang sudah disiapkan SEKALI di awal (mensimulasikan
// kondisi produksi: bobot sudah resident di GPU dari saat load model, cuma
// aktivasi kecil yang dikirim tiap panggilan). Dibandingkan terhadap
// baseline CPU single-thread untuk operasi yang sama.
//
// Ini gerbang keputusan SEBELUM integrasi penuh ke transformer_mixed.cpp:
// kalau overhead per-dispatch di sini sudah lebih mahal dari compute-nya
// sendiri, backend Vulkan tidak akan menang di produksi (matmul_int8
// dipanggil ~150x per token) -- sama seperti pelajaran dari CPU threading
// (Bagian 16-19).
//
// Mengembalikan string laporan manusiawi, atau pesan error kalau device
// tidak punya Vulkan compute yang berfungsi (mis. driver GPU tidak
// mendukung, atau headless).
std::string run();

} // namespace vulkan_bench
