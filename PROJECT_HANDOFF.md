# PROJECT HANDOFF: FastTranslator Engine

**Terakhir diupdate:** Sesi ini (setelah bugfix callbackFlow + benchmark CPU nyata terkonfirmasi device)
**Tujuan dokumen:** Supaya sesi Claude manapun (atau developer manapun) bisa lanjut kerja tanpa
harus baca ulang seluruh riwayat chat. Baca dokumen ini dulu sebelum menyentuh kode.

---

## 1. RINGKASAN SATU LAYAR

Proyek membangun **FastTranslator Engine**: mesin inferensi LLM/SLM di Android, mengikuti blueprint
awal (format `.fllm`, Medusa, BitNet, KV-cache compression, dst). Setelah eksperimen ekstensif:

- ✅ **Pipeline Android lengkap jalan di device fisik** (load model → generate teks) — terverifikasi
  dengan screenshot nyata dari HP pengguna.
- ✅ **KV-cache compression (RocketKV) & Grammar-guided sampling (GBNF) terintegrasi produksi**,
  keduanya terbukti bekerja di device fisik (screenshot: JSON grammar dipaksa dari karakter
  pertama, memori KV-cache turun 70%).
- ✅ **Bug nyata ditemukan & diperbaiki dari device**: `callbackFlow` Kotlin tidak pernah `close()`
  otomatis, bikin UI macet meski generation sudah selesai — lihat `BUGFIX_CALLBACKFLOW_NEVER_CLOSES.md`.
- ✅ **Benchmark CPU nyata terukur di device fisik**: **12.77 tok/s** (ARM) — baseline resmi untuk
  segala optimisasi ke depan.
- ✅ **Kuantisasi produksi: INT8** (bukan ternary/BitNet seperti blueprint asli) — keputusan berbasis
  data empiris setelah ternary terbukti gagal total (lihat §4).
- ⚠️ **Speculative decoding: diimplementasikan & benar, tapi tidak mempercepat** di model/hardware
  saat ini (0.89×, lebih lambat) — lihat §5 untuk alasan teknisnya.
- ❌ **Belum ada backend NPU/QNN sungguhan** — semua eksekusi (termasuk di device fisik) masih CPU
  reference murni, `Backend.QNN` cuma label. Opsi termurah berikutnya: NEON/SIMD (lihat
  `NPU_BENCHMARK_STATUS.md`).
- ⚠️ **ARM NEON/SIMD ditulis, TAPI korektnes belum terverifikasi** — sandbox saya x86, tidak ada
  akses jaringan untuk cross-compiler ARM. Ada self-test on-device ("Verify NEON correctness")
  yang harus dijalankan di device Anda dulu sebelum hasil ini bisa dipercaya — lihat
  `NEON_OPTIMIZATION_FINDINGS.md`.
- ❌ **Model saat ini (`stories15M`) cuma bisa generate cerita anak sederhana**, belum benar-benar
  "translator". Nama proyek "FastTranslator" belum tercermin di kapabilitas model.

---

## 2. KRONOLOGI LENGKAP (Blueprint → Sekarang)

### Fase 1: Core Engine (C++, CPU sandbox, tanpa model asli)
- Format biner `.fllm` (header 64-byte v1, lalu diperluas ke 80-byte v2 di Fase 3).
- SHA-256 checksum dari nol (tanpa dependency eksternal).
- Kernel BitNet ternary (add/sub/skip, tanpa perkalian asli).
- KV-cache compression: MiniCache (merge by cosine similarity) + RocketKV (eviction hybrid).
- Medusa tree-attention mask + typical acceptance sampling.
- **24/24 unit test lulus** (`core/tests/test_fase1.cpp`).

### Fase 2: Android Integration
- JNI bridge (`jni_bridge.cpp`) dengan threading benar (`AttachCurrentThread`, global ref, cancellation
  via atomic flag per-generation).
- Kotlin API 3-langkah: `FastModel.load()` → `.generate()` (Flow) → `.close()`.
- GitHub Actions CI (karena user tidak pakai Android Studio) — build APK debug (auto-signed) +
  release (unsigned, verifikasi saja).
- **Terverifikasi di device fisik nyata** (screenshot: "Model loaded OK", generate menampilkan teks).

### Fase 3a: Model Asli (stories15M, llama2.c format)
- User upload `stories15M.bin` (60MB, dim=288, 6 layer, 6 head, vocab 32000) + `tokenizer.bin`.
- Forward pass transformer dari nol: RMSNorm, RoPE, multi-head causal attention, SwiGLU FFN.
- Tokenizer BPE (encode greedy pair-merge, decode dengan byte-fallback).
- **Hasil pertama:** teks koheren nyata ("...there was a little girl named Lily...") — fp32, 52 tok/s CPU.
- Diverifikasi juga di device fisik lewat `load_raw()` (jalur sementara, sudah di-deprecate).

### Fase 3b: Kuantisasi — SAGA PANJANG, BANYAK EKSPERIMEN GAGAL (semua didokumentasikan jujur)
User awalnya minta ternary/BitNet sesuai blueprint. Hasil eksperimen berurutan:

1. **Ternary naif** (absmean threshold): perplexity 35,099 vs fp32 27.11 — output "time time time..." (rusak total).
2. **Audit layer-wise** (`tools/audit_quantizer.cpp`): error 52-56% seragam di semua layer/tensor —
   bukan bug lokal, keterbatasan fundamental grid 3-level.
3. **Evaluator perplexity objektif** (`tools/eval_perplexity.cpp`): dibangun supaya penilaian kualitas
   tidak lagi cuma "baca teks, kelihatan jelek/bagus".
4. **GPTQ-style calibrated quantization** (`tools/gptq_ternary.cpp`, Hessian dari 8 prompt kalibrasi):
   perplexity turun ke 11,822 — membantu tapi masih 436× lebih buruk dari fp32.
5. **Mixed precision** (attention INT8, FFN ternary): perplexity 15,093 — **lebih buruk** dari GPTQ-ternary-penuh.
   Temuan: FFN adalah sumber error dominan, bukan attention.
6. **INT8 di semua tempat** (eksperimen penentu): perplexity **27.01** — nyaris identik fp32 (27.11)!
   → **Ini jadi keputusan produksi final.**

**Kesimpulan:** BitNet ternary butuh model yang dilatih native dalam ternary (quantization-aware
training) — post-training quantization dari checkpoint fp32 gagal total untuk model sekecil 15M
parameter, bahkan dengan kalibrasi GPTQ. INT8 standar terbukti hampir lossless.

### Fase 3c: Konsolidasi Produksi
- `tools/pack_fllm.cpp` → resmi jadi packer INT8 (semua 7 tensor bobot).
- `FastEngine::load()` di `core/src/engine.cpp` otomatis pakai jalur INT8 terkuantisasi kalau
  `.fllm` punya `has_transformer = true`.
- `MainActivity.kt` dikembalikan ke **satu file `.fllm`** (bukan lagi 2 file `.bin` terpisah) —
  sesuai desain 3-langkah asli.
- **Diverifikasi di device fisik**: screenshot menunjukkan generate teks koheren identik kualitas fp32.
- Regresi nol: 38+ unit test lama tetap lulus setelah refactor besar ini.

### Fase 3d-2: KV-Cache Compression — Terintegrasi Produksi (hasil POSITIF)
- Audit dulu: cek similaritas cosine antar-layer K-vector → nyaris nol (-0.06 s/d 0.10) → **MiniCache
  di-skip** (asumsi dasarnya tidak berlaku untuk model 6-layer ini, beda dari model besar 32+ layer
  yang jadi basis paper aslinya).
- **RocketKV diimplementasi penuh**: `forward_evictable()` di `transformer_mixed.cpp`, cache berbasis
  slot aktif (bukan array posisi tetap) — token yang di-evict benar-benar dibuang dari memori.
- **Hasil: 70.3% pengurangan memori KV-cache**, korektnes bit-identik saat eviction tidak aktif,
  output tetap koheren penuh bahkan setelah melewati ambang eviction di tengah generate 150 token.
- **Tersambung ke `FastEngine::load()`** otomatis lewat `kv_config.enable_rocketkv` di `.fllm` —
  tidak ada perubahan API Kotlin/JNI sama sekali.

### Fase 3d: Speculative Decoding (real implementation, honest negative result)
- `forward_batch()` di `transformer_mixed.cpp`: verifikasi N token sekaligus dalam 1 pass (GEMM batch).
  **Teruji bit-identik** dengan pemanggilan sekuensial (`max_abs_diff = 0`).
- Prompt Lookup Decoding (`test_speculative_decoding.cpp`): drafting dari n-gram berulang, tanpa
  perlu extra trained heads (beda dari Medusa yang butuh head khusus).
- **Hasil: 0.89× — LEBIH LAMBAT**, bukan lebih cepat, meski output identik/benar.
- Sebab: model 15M sudah muat di cache CPU (bukan memory-bandwidth-bound), CPU single-thread
  (bukan GPU/NPU paralel) → tidak ada "kapasitas nganggur" untuk dieksploitasi speculative decoding.
- Kode tetap disimpan sebagai kapabilitas siap pakai untuk model lebih besar / NPU asli nanti.

---

## 3. STATUS PER KOMPONEN BLUEPRINT ASLI

| Komponen Blueprint | Status | Catatan |
|---|---|---|
| Format `.fllm` | ✅ Selesai (v2) | Checksum SHA-256, section transformer config+weights |
| Medusa multi-token | ⚠️ Logika ada, tak terpakai produksi | Modul teruji standalone (Fase 1), belum disambung ke generate loop nyata |
| Speculative drafting | ✅ Diimplementasi, negatif hasilnya | Lihat §2 Fase 3d — benar tapi tak mempercepat di kondisi saat ini |
| KV-cache compression (RocketKV) | ✅ Terintegrasi produksi | 70% pengurangan memori, koherensi terjaga, otomatis aktif via `.fllm` config — lihat `notes/KV_CACHE_COMPRESSION_FINDINGS.md` |
| KV-cache compression (MiniCache) | ❌ Di-skip (diaudit dulu) | Similaritas antar-layer nyaris nol untuk model 6-layer ini — tidak berlaku, bukan bug |
| Backend QNN/Vulkan/ExecuTorch | ❌ Belum ada | Semua forward pass = CPU reference C++. Tidak ada SDK QNN/Vulkan di sandbox ini |
| BitNet W1.58A8 | ❌ Diganti INT8 | Ternary terbukti gagal (lihat Fase 3b), INT8 dipakai sebagai gantinya |
| Grammar-guided sampler (GBNF) | ✅ Terintegrasi produksi | Parser + constrained decoding dari nol, terbukti memaksa JSON valid — lihat `notes/GBNF_GRAMMAR_FINDINGS.md`. ~10× lebih lambat dari sampling bebas (scan seluruh vocab tiap langkah) |
| Android 3-step API | ✅ Selesai & teruji device fisik | `FastModel.load()/.generate()/.close()` |
| GitHub Actions CI | ✅ Selesai | Build APK, pack model, verifikasi perplexity otomatis |

---

## 4. STRUKTUR REPO SAAT INI

```
fasttranslator-engine/
├── .github/workflows/build.yml       — CI: compile test, pack model INT8, verifikasi perplexity, build APK
├── .gitattributes                    — Git LFS untuk *.bin
├── core/
│   ├── include/                      — semua header (fllm_format, engine, transformer*, tokenizer, dst)
│   ├── src/                          — implementasi
│   │   ├── fllm_parser.cpp           — writer/reader .fllm v2
│   │   ├── engine.cpp                — FastEngine: load()=.fllm produksi INT8, load_raw()=fp32 dev/ref
│   │   ├── transformer.cpp           — forward pass fp32 murni (referensi/dev)
│   │   ├── transformer_bitnet.cpp    — forward pass ternary murni (riset, tak dipakai produksi)
│   │   ├── transformer_mixed.cpp     — forward pass PRODUKSI (INT8) + forward_batch()
│   │   ├── tokenizer.cpp             — BPE encode/decode, load dari file ATAU dari bytes embedded
│   │   ├── bitnet_kernel.cpp         — quantize_ternary(), quantize_int8(), matmul primitif
│   │   ├── kv_cache.cpp, medusa.cpp  — modul Fase 1, belum tersambung ke generate produksi
│   ├── tests/                        — semua unit test (lihat §5 untuk cara jalanin)
├── tools/
│   ├── pack_fllm.cpp                 — PACKER PRODUKSI (INT8 semua tensor)
│   ├── pack_fllm_calibrated.cpp      — riset: GPTQ-style (non-produksi, disimpan sbg referensi)
│   ├── pack_fllm_mixed.cpp           — riset: mixed precision (non-produksi)
│   ├── audit_quantizer.cpp           — Milestone 1: audit layer-wise
│   ├── eval_perplexity.cpp           — Milestone 2: evaluator objektif (fp32/fllm/mixed/int8all mode)
│   ├── gptq_ternary.h/.cpp           — Milestone 4: algoritma GPTQ-style
│   ├── make_dummy_model.cpp          — generator model dummy (untuk test awal Fase 2, bukan produksi)
├── android/                          — project Gradle lengkap (Kotlin DSL), Jetpack ComponentActivity
├── models/                           — TIDAK ADA DI ZIP (lihat §6, disimpan via Git LFS di repo user)
├── notes/                            — semua catatan/dokumentasi per fase (lihat daftar di bawah)
└── PROJECT_HANDOFF.md                — dokumen ini
```

### Daftar file `notes/` (baca berurutan untuk paham kronologi lengkap):
1. `FASE1_CATATAN.md`
2. `FASE2_CATATAN.md`
3. `FASE3_REAL_MODEL_CATATAN.md` (integrasi model asli pertama kali)
4. `FASE3_QUANTIZATION_FINDINGS.md` (saga kuantisasi lengkap, tabel perbandingan)
5. `SPECULATIVE_DECODING_FINDINGS.md` (speculative decoding: benar tapi tidak mempercepat)
6. `KV_CACHE_COMPRESSION_FINDINGS.md` (RocketKV: hasil positif, terintegrasi produksi)
7. `GBNF_GRAMMAR_FINDINGS.md` (grammar-guided sampling: hasil positif, terintegrasi produksi)
8. `NPU_BENCHMARK_STATUS.md` (status jujur backend NPU + cara ukur CPU real di device)
9. `BUGFIX_CALLBACKFLOW_NEVER_CLOSES.md` (bug nyata dari device: Flow hang selamanya)
10. `NEON_OPTIMIZATION_FINDINGS.md` (ARM NEON/SIMD: kode ditulis, korektnes BELUM terverifikasi
    di sandbox — butuh tombol "Verify NEON correctness" di device — terbaru)

---

## 5. CARA MENJALANKAN ULANG SEMUA TEST (verifikasi environment baru)

Semua ini butuh `g++` (C++17) saja, TIDAK butuh Android SDK/NDK untuk core testing:

```bash
cd core

# Fase 1 (24 test, tidak butuh model asli)
g++ -std=c++17 -Wall -Wextra -pthread -I include \
  src/fllm_parser.cpp src/bitnet_kernel.cpp src/kv_cache.cpp src/medusa.cpp \
  src/engine.cpp src/transformer.cpp src/transformer_mixed.cpp src/tokenizer.cpp \
  tests/test_fase1.cpp -o /tmp/t1 && /tmp/t1

# Engine layer (8 test, tidak butuh model asli)
g++ [sama seperti di atas] tests/test_engine.cpp -o /tmp/t2 && /tmp/t2

# BUTUH model asli (stories15M.bin + tokenizer.bin — lihat §6):
g++ -std=c++17 -O2 -I include src/transformer.cpp src/tokenizer.cpp \
  tests/test_real_model.cpp -o /tmp/t3 && /tmp/t3 <path_stories15M.bin> <path_tokenizer.bin>
```

Referensi lengkap urutan compile ada di `.github/workflows/build.yml` — itu source of truth paling
akurat karena dijalankan otomatis tiap push.

---

## 6. CHECKLIST: APA YANG PERLU DISIAPKAN/DILAMPIRKAN SAAT PINDAH SESI

### A. WAJIB dilampirkan ulang ke sesi baru:
1. **File zip project ini** (`fasttranslator-engine.zip`) — berisi SEMUA source code, tapi
   **TIDAK berisi file model** (terlalu besar untuk chat upload).
2. **Dokumen ini (`PROJECT_HANDOFF.md`)** — supaya sesi baru langsung paham konteks tanpa
   scroll ribuan baris chat history.

### B. TIDAK perlu upload ulang (sudah ada di sistem lain, tinggal disambungkan):
3. **`stories15M.bin` + `tokenizer.bin`** — SUDAH ada di repo GitHub user via Git LFS
   (folder `models/`). Sesi baru tidak perlu ini di-upload ulang KECUALI sandbox baru butuh
   compile+test lokal (dalam hal ini user perlu upload lagi ke chat karena sandbox tidak
   auto-akses ke GitHub user).
4. **Model produksi (`model.fllm`, hasil `pack_fllm`)** — dihasilkan otomatis oleh GitHub Actions
   tiap push, didownload sebagai artifact bernama `model-fllm-production`. Tidak perlu disimpan manual.

### C. Konteks penting yang HARUS disampaikan ke sesi baru (kalau tidak pakai dokumen ini):
5. **Kredensial/akses yang TIDAK dimiliki sandbox Claude:**
   - Tidak ada akses internet/jaringan sama sekali (semua download harus lewat user).
   - Tidak ada Android SDK/NDK terpasang (makanya pakai GitHub Actions, bukan build lokal).
   - Tidak ada QNN SDK, Vulkan driver, atau device NPU fisik.
6. **Alur kerja yang sudah disepakati:** Claude tulis kode → user push ke GitHub → GitHub Actions
   build & test otomatis → user download artifact (APK/model) → user `adb push` & test di HP →
   user laporkan hasil (idealnya dengan screenshot) → Claude lanjut iterasi.
7. **Struktur repo GitHub user** (asumsi, perlu dikonfirmasi ulang di sesi baru kalau berubah):
   - Root repo = isi zip ini.
   - `models/stories15M.bin` dan `models/tokenizer.bin` via Git LFS.
   - Workflow CI di `.github/workflows/build.yml` jalan otomatis tiap push ke branch `main`.

### D. Pertanyaan yang sebaiknya ditanyakan di awal sesi baru (kalau user langsung minta "lanjutkan"):
- "Apakah kode di GitHub sudah sinkron dengan zip terakhir yang saya berikan?" (kalau user sempat
  edit manual atau ada commit lain di antara sesi).
- "Apakah GitHub Actions run terakhir sukses?" (screenshot/link run kalau ada masalah).
- "Mau lanjut ke arah mana?" — merujuk ke §7 di bawah untuk opsi yang masih terbuka.

---

## 7. OPSI LANJUTAN YANG MASIH TERBUKA (belum dikerjakan)

Diurutkan dari yang paling actionable tanpa sumber daya tambahan, ke yang butuh paling banyak:

1. ~~Sambungkan KV-cache compression ke generate produksi~~ ✅ **SELESAI** (RocketKV; MiniCache
   di-skip setelah audit — lihat `notes/KV_CACHE_COMPRESSION_FINDINGS.md`).
2. ~~Grammar-guided sampler (GBNF) sungguhan~~ ✅ **SELESAI** — lihat `notes/GBNF_GRAMMAR_FINDINGS.md`.
3. ~~Benchmark nyata di NPU~~ ✅ **CPU baseline terukur nyata di device**: 12.77 tok/s (ARM,
   ~4× lebih lambat dari sandbox x86 saya ~52 tok/s — wajar, belum ada optimisasi SIMD).
   NPU sungguhan masih belum ada (`Backend.QNN` masih label). Opsi termurah berikutnya: NEON/SIMD
   di kernel matmul (potensi 2-4×, tanpa SDK tambahan) — lihat `notes/NPU_BENCHMARK_STATUS.md`.
4. **Model yang lebih capable** (supaya "FastTranslator" benar-benar bisa translate, bukan cuma
   generate cerita anak) — butuh model instruct kecil (mis. Qwen2-0.5B-Instruct), arsitektur
   kemungkinan beda (GQA, RoPE theta beda, dll) sehingga perlu porting parser/forward-pass baru.
   User perlu download & upload model (bisa >1GB, mungkin perlu Git LFS/chunking).
5. **Medusa production integration** — modul sudah ada (Fase 1), tapi ekspektasi harus realistis:
   speculative decoding (prinsip serupa) sudah terbukti tidak menang di CPU/model kecil ini
   (§2 Fase 3d) — Medusa kemungkinan besar akan mengalami hal serupa. Worth diuji untuk
   kelengkapan, prioritas rendah.
6. **True bit-packing untuk ternary** (kalau nanti kembali ke ternary dengan model native-trained) —
   saat ini prototype masih 1 byte/bobot untuk ternary maupun INT8, belum packing 2-bit asli.

---

## 8. PRINSIP KERJA YANG SUDAH TERBUKTI EFEKTIF (lanjutkan pola ini)

- **Selalu compile & jalankan test nyata** sebelum klaim sesuatu "berhasil" — jangan simulasi.
- **Laporkan hasil negatif dengan jujur** (lihat Fase 3b dan speculative decoding) — ini yang
  bikin proyek ini kredibel, bukan sekadar demo yang terlihat bagus tapi rapuh.
- **Pisahkan kode eksperimen dari kode produksi** (`tools/pack_fllm_calibrated.cpp` dkk disimpan
  sebagai referensi riset, bukan dihapus, tapi juga tidak dipakai default).
- **Setiap klaim performa harus didukung angka terukur** (perplexity, tok/s, ms) — bukan "kelihatan
  lebih cepat" atau asumsi dari teori saja.
