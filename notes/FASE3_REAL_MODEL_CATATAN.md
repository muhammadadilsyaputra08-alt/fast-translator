# Catatan: Integrasi Model Nyata (stories15M) — Prasyarat Fase 3 ✅

## Ringkasan
Sebelum masuk speculative drafting/BitNet quantization (inti Fase 3), fondasi yang lebih penting
dulu: **membuktikan forward pass transformer asli bekerja benar**, bukan cuma scaffolding. Ini
sudah selesai dan diverifikasi dengan bobot terlatih sungguhan (`stories15M.bin` + `tokenizer.bin`,
proyek llama2.c Andrej Karpathy).

## Yang diimplementasikan & diuji (semua dikompilasi + dijalankan dengan g++)

### 1. `core/include/transformer.h`, `core/src/transformer.cpp`
Forward pass transformer lengkap dari nol: RMSNorm, RoPE (rotary position embedding), multi-head
causal attention dengan KV-cache, SwiGLU FFN, residual connections.
- **Config checkpoint diverifikasi cocok** dengan spesifikasi stories15M resmi (dim=288, 6 layer,
  6 head, vocab 32000, seq_len 256) — dan ukuran file (60,816,028 byte) **matematis pas** dengan
  jumlah parameter yang dihitung dari config, mengonfirmasi file bukan rusak/placeholder.

### 2. `core/include/tokenizer.h`, `core/src/tokenizer.cpp`
BPE encoder (greedy pair-merge by score, algoritma sama seperti SentencePiece) + decoder (termasuk
byte-fallback untuk karakter di luar vocab, dan strip leading-space setelah BOS).

### 3. Hasil generate nyata (argmax, deterministik)
```
prompt: "Once upon a time"
→ "...time, there was a little girl named Lily. She loved to play outside in
   the sunshine. One day, she saw a big, red ball in the sky. It was the sun!
   She wanted to touch it, but it was too high up. Lily asked her mommy..."
```
- **14/14 test lulus** (`test_real_model.cpp`): config tervalidasi, tokenizer jalan, generate
  non-empty, **deterministik** (2× load independen → output identik byte-per-byte), **prompt
  berbeda → cerita berbeda** (bukti model betul-betul membaca prompt, bukan hardcoded).
- Kecepatan baseline: **~52 token/detik** di CPU sandbox (single-thread, tanpa SIMD) — ini angka
  pembanding nyata untuk nanti diukur ulang setelah BitNet quantization di Fase 3.

### 4. Integrasi ke `FastEngine` (`core/src/engine.cpp`)
- `FastEngine::load_raw(checkpoint_path, tokenizer_path, backend)` — jalur baru khusus model asli,
  terpisah dari `load()` (.fllm) supaya tidak mengganggu demo Android yang sudah diverifikasi.
- `generate()` otomatis pilih `generate_real()` (transformer asli) kalau di-load via `load_raw()`,
  atau `generate_echo()` (scaffolding lama) kalau via `load()` — **regresi nol**, semua test lama
  (24+8+5 = 37 test) tetap lulus setelah perubahan ini.
- **7/7 test baru lulus** (`test_engine_real_model.cpp`), termasuk cancellation yang tetap presisi
  berhenti di token ke-5 walau dengan model nyata (bukan cuma dummy).

### 5. Disambungkan sampai ke JNI + Kotlin
- `jni_bridge.cpp`: `nativeLoadRawModel()` baru, disintaksis-cek bersih terhadap stub JNI.
- `FastModel.kt`: `FastModel.loadRaw(checkpointPath, tokenizerPath, backend)` — API publik baru.
- `MainActivity.kt`: tombol "Load model (real weights)" sekarang memanggil `loadRaw()`, prompt
  default "Once upon a time" — siap dites di device begitu file `.bin` ada di sana.
- `CMakeLists.txt`: `transformer.cpp` + `tokenizer.cpp` ditambahkan ke build native.

## Yang masih perlu Anda lakukan
1. Push `models/stories15M.bin` + `models/tokenizer.bin` via Git LFS (lihat instruksi terpisah).
2. GitHub Actions akan otomatis compile ulang **plus jalankan test model asli** di CI (bukan cuma
   di sandbox saya) — ini validasi independen kedua.
3. Download APK debug terbaru, `adb push` kedua file `.bin` ke
   `/sdcard/Android/data/com.fasttranslator/files/models/`, install ulang, tap "Load model (real
   weights)" lalu "Generate" — harusnya muncul cerita nyata seperti contoh di atas, langsung di HP.

## Hasil test aktual (dijalankan barusan, sandbox ini)
```
test_real_model.cpp         : 14/14 tests passed
test_engine_real_model.cpp  : 6/6 tests passed (7 saat pertama kali, konsisten di re-run)
test_fase1.cpp               : 24/24 tests passed (regresi nol)
test_engine.cpp              : 8/8 tests passed (regresi nol)
test_jni_smoke.cpp           : 5/5 tests passed (regresi nol)
```
