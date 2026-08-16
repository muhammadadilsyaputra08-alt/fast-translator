# Benchmark NPU/QNN: Status Jujur

## Ringkasan: Belum ada backend NPU sungguhan — semua eksekusi saat ini CPU

`Backend.QNN` di `Types.kt` dan `engine::Backend::QNN` di C++ **cuma label/enum value**, tidak
pernah benar-benar merutekan komputasi ke NPU. `MixedPrecisionTransformer::forward()` dan
`forward_evictable()` keduanya adalah kode C++ murni (scalar loop), jalan di CPU device tidak
peduli `Backend` apa yang dipilih saat `load()`. Ini sudah ditandai ❌ di `PROJECT_HANDOFF.md` §3
sejak awal — bukan temuan baru, tapi perlu ditegaskan sebelum melangkah ke "benchmark NPU".

## Yang Sudah Bisa Dilakukan Sekarang: Benchmark CPU Nyata di Device

Ditambahkan tombol **"Benchmark (measure real tok/s)"** di `MainActivity.kt` — generate 80 token,
ukur wall-clock time nyata, hitung tok/s. Ini **baseline CPU asli** dari HP Anda (bukan simulasi
sandbox saya lagi ~52 tok/s CPU x86) — nomor Android/ARM yang sebenarnya belum pernah kita ukur.

### Hasil Terukur (device fisik, dikonfirmasi 2025)
```
Benchmark: 81 tokens in 6342 ms = 12.77 tok/s (CPU reference, no NPU)
```

**Perbandingan dengan sandbox saya (CPU x86, single-thread):**
| Environment | tok/s | Catatan |
|---|---|---|
| Sandbox Claude (x86 CPU) | ~51-52 | Server-grade CPU, tanpa optimisasi SIMD |
| Device Android nyata (ARM CPU) | **12.77** | ~4× lebih lambat dari sandbox |

**Kenapa device ARM lebih lambat dari sandbox x86:** wajar dan diharapkan — core CPU mobile
(bahkan flagship) punya clock speed & IPC lebih rendah dari CPU server/desktop x86, ditambah kode
kita masih scalar loop murni (`matmul_int8`/`matmul_ternary` di `transformer_mixed.cpp`) tanpa
optimisasi ARM NEON/SIMD sama sekali. Ini bukan tanda ada yang salah — ini baseline CPU paling
dasar, belum ada optimisasi apapun diterapkan.

**Implikasi untuk prioritas ke depan:** 12.77 tok/s inilah angka yang harus dikalahkan kalau nanti
ada: (a) optimisasi NEON/SIMD di kernel matmul (potensi speedup signifikan, murni software, tidak
butuh SDK NPU), atau (b) backend QNN/NPU sungguhan (lihat Opsi B di bawah). (a) jauh lebih murah
untuk dicoba lebih dulu dibanding (b).

## Opsi Kalau Ingin Angka NPU Sungguhan

### Opsi C (baru, termurah): ARM NEON/SIMD di kernel matmul
Optimisasi `matmul_int8`/`matmul_ternary` di `transformer_mixed.cpp` pakai intrinsic ARM NEON
(vektor 4-8 elemen sekaligus per instruksi, bukan scalar satu-per-satu). **Tidak butuh SDK
tambahan** — NDK Anda sudah menyertakan header NEON secara default untuk target `arm64-v8a`. Ini
murni software, bisa saya kerjakan & uji lewat cross-compile g++ dengan flag ARM di sandbox (meski
tidak bisa dijalankan langsung di sini, korektnes logika bisa divalidasi via compile check),
baru diukur kecepatan sungguhan lewat tombol Benchmark yang sudah terbukti bekerja. Potensi
speedup 2-4× murni dari software, tanpa NPU, tanpa SDK tambahan.

### Opsi A: Qualcomm AI Hub (cloud, tanpa perlu SDK lokal)
Sudah dibahas sebelumnya di percakapan ini — aihub.qualcomm.com bisa profile model di device
Snapdragon asli tanpa Anda punya device fisik. **Tapi catatan penting:** AI Hub menerima model
dalam format standar (ONNX/TorchScript/TFLite), bukan `.fllm` custom kita. Supaya bisa dites di
sana, perlu:
1. Export arsitektur transformer kita (dim=288, 6 layer, dst.) ke ONNX.
2. Upload ke AI Hub, profiling jalan di device farm mereka.
3. **Hasil yang didapat mengukur arsitektur transformer generiknya**, BUKAN pipeline `.fllm` +
   kuantisasi INT8 + RocketKV + GBNF kita yang sudah dibangun — AI Hub akan pakai jalur kuantisasi/
   kompilasi mereka sendiri untuk NPU (biasanya lewat ONNX Runtime + QNN Execution Provider).

Ini berguna untuk validasi **"apakah arsitektur ukuran segini bisa dapat speedup NPU"** secara umum,
tapi tidak langsung memvalidasi kode C++ yang sudah kita tulis.

### Opsi B: QNN SDK asli + tulis kernel NPU sungguhan
Ini yang blueprint aslinya maksud ("Custom kernel QNN/Vulkan untuk operasi ternary/INT8"). Butuh:
1. Qualcomm QNN SDK terpasang (Anda download, butuh akun Qualcomm Developer).
2. Tulis kernel HTP (Hexagon Tensor Processor) untuk matmul INT8 kita — ini pekerjaan besar,
   beda paradigma dari CPU scalar loop yang kita tulis sekarang (perlu belajar QNN API, HTP
   intrinsics, tensor descriptor, dst).
3. Ganti `matmul_int8`/`matmul_ternary` di `transformer_mixed.cpp` dengan pemanggilan kernel QNN,
   dikondisikan oleh `Backend` yang dipilih.

Ini realistis 1-2 minggu kerja minimal untuk versi awal yang benar, dan **saya tidak bisa
mengerjakannya di sandbox ini** (tidak ada SDK QNN, tidak ada device/emulator NPU).

## Rekomendasi
Dengan baseline 12.77 tok/s sekarang terukur nyata di device: **Opsi C (NEON/SIMD) adalah langkah
paling masuk akal berikutnya** — murah (tidak perlu SDK/akses baru), dan bisa langsung diverifikasi
lewat tombol Benchmark yang sudah terbukti bekerja. Opsi A (AI Hub) worth dicoba kalau penasaran
potensi speedup umum di NPU, tapi tidak mengubah kode kita. Opsi B baru masuk akal kalau proyek ini
mau dibawa ke tahap produksi sungguhan dengan target performa spesifik dan siap investasi 1-2 minggu.

## Catatan Sampingan (ditemukan saat audit ini)
`KVCacheConfig` yang di-pass ke `FastModel.load()` dari Kotlin saat ini **tidak dipakai** oleh
native layer — `engine::FastEngine::load()` hanya menerima `model_path` dan `backend`; konfigurasi
KV-cache sesungguhnya berasal dari apa yang di-bake ke dalam file `.fllm` saat `pack_fllm.cpp`
dijalankan (lihat `KV_CACHE_COMPRESSION_FINDINGS.md`). Ini bukan bug yang berbahaya (tidak
menyebabkan crash/salah hasil), tapi API Kotlin-nya menyiratkan kontrol yang sebenarnya tidak
berpengaruh — worth diperbaiki (baik dengan benar-benar menyalurkan parameter itu, atau
menghapusnya dari API supaya tidak menyesatkan) di iterasi berikutnya.
