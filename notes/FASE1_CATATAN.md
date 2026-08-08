# Catatan Fase 1: Core Engine — SELESAI ✅

## Yang benar-benar diimplementasikan & diuji

Semua kode di bawah **dikompilasi dengan g++ 13.3 (C++17)** dan **dijalankan** di sandbox ini —
bukan pseudocode. Hasil: **24/24 unit test lulus**.

### 1. Format `.fllm` (`fllm_format.h`, `fllm_parser.cpp`)
- Header biner 64-byte persis sesuai spesifikasi blueprint (magic, version, model type, 6 offset, checksum).
- `write_fllm()`: serialisasi model in-memory → file biner dengan layout section berurutan.
- `read_fllm()`: baca file, validasi magic number, **validasi checksum SHA-256** (implementasi SHA-256 dari nol di `sha256.h`, tanpa dependency eksternal).
- **Teruji:** roundtrip semua section (tokenizer, embeddings, grammar, kv-config, medusa-config, weights) identik setelah write→read. File yang di-corrupt (1 byte diflip) **berhasil ditolak** oleh pemeriksaan checksum.

### 2. Kernel BitNet Ternary (`bitnet_kernel.cpp`)
- `quantize_ternary()`: kuantisasi float → {-1,0,+1} pakai skema absmean-threshold asli BitNet (α = mean(|W|), threshold = 0.5α).
- `matmul_ternary_int8()`: matmul ternary×int8 yang **benar-benar tanpa perkalian** — hanya add/sub/skip sesuai prinsip inti BitNet.
- **Teruji:** hasil numerik matmul diverifikasi manual (row 0 dihitung tangan, cocok dengan output kernel).

### 3. KV-Cache Compression (`kv_cache.cpp`)
- `MiniCacheCompressor`: merge KV entry antar-layer berdasarkan cosine similarity (key & value), threshold configurable. Entry mirip di-merge + residual disimpan; entry berbeda tidak di-merge.
- `RocketKVEvictor`: eviction hybrid "recent window + top-salience" (skor = L2 norm key vector), mirip strategi RocketKV untuk long-context.
- **Teruji:** entry mirip (cosine sim tinggi) berhasil merge, entry berbeda ditolak; token terbaru selalu bertahan setelah eviction.

### 4. Medusa Tree-Attention (`medusa.cpp`)
- `build_tree_attention_mask()`: mask N×N di mana node hanya "melihat" node dirinya + seluruh ancestor-nya di tree kandidat.
- `typical_acceptance_check()`: threshold dinamis `min(epsilon, delta·e^-entropy)` — implementasi asli typical acceptance dari paper Medusa.
- `accept_longest_chain()`: greedy-walk menerima rantai token terpanjang yang lolos typical acceptance.
- **Teruji:** mask memisahkan branch sibling dengan benar; token confidence tinggi diterima, token confidence rendah pada distribusi peaked ditolak; rantai penuh diterima ketika semua langkah confident.

## Yang BELUM diimplementasikan (batasan sandbox ini)
- **QNN/Vulkan backend asli** — SDK Qualcomm & driver GPU tidak tersedia di sandbox tanpa jaringan ini. Kernel BitNet di atas adalah **referensi CPU murni**, bukan kernel NPU/GPU. Porting ke QNN HTP/Vulkan compute shader perlu dilakukan di mesin dengan SDK tersebut terpasang.
- **Speculative drafting (AHASD/sd.npu)** — masuk Fase 3.
- **LZ4 compression untuk tokenizer JSON** — saat ini disimpan uncompressed; kompresi LZ4 nyata butuh library eksternal yang belum di-install (tidak ada akses jaringan untuk `apt`/`pip install`).

## File yang dihasilkan
```
core/include/fllm_format.h
core/include/sha256.h
core/include/fllm_parser.h
core/include/bitnet_kernel.h
core/include/kv_cache.h
core/include/medusa.h
core/src/fllm_parser.cpp
core/src/bitnet_kernel.cpp
core/src/kv_cache.cpp
core/src/medusa.cpp
core/tests/test_fase1.cpp
```

## Cara menjalankan ulang test
```bash
cd core
g++ -std=c++17 -Wall -Wextra -I include \
  src/fllm_parser.cpp src/bitnet_kernel.cpp src/kv_cache.cpp src/medusa.cpp \
  tests/test_fase1.cpp -o test_fase1
./test_fase1
```

## Hasil aktual run terakhir
```
24/24 tests passed
```
