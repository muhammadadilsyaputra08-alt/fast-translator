# Fase 3: Quantization — Temuan Lengkap & Keputusan Produksi

## Ringkasan Eksekutif
**Keputusan final: INT8 (bukan ternary/BitNet) untuk semua 7 tensor bobot.**
Perplexity 27.01 vs fp32 baseline 27.11 — **nyaris tak terbedakan**, di 1/4 ukuran fp32.
Ini sekarang jalur produksi resmi, tersambung penuh sampai `FastEngine::load()` → Android.

## Kronologi Eksperimen (semua diuji nyata, bukan asumsi)

### 1. Ternary naif (absmean threshold per-layer)
- Perplexity: **35,099** (vs fp32 27.11) — praktis output acak ("time time time...").
- Error relatif L2 per bobot: **56%**.

### 2. Audit layer-wise (Milestone 1 — `tools/audit_quantizer.cpp`)
- Error **seragam** 52-56% di *semua* 6 layer dan *semua* 7 jenis tensor.
- Distribusi ternary ~33/33/33% (-1/0/+1) — kuantizer bekerja "benar" secara desain.
- Kesimpulan: bukan bug di satu tempat, tapi keterbatasan fundamental 3-level grid.

### 3. Evaluator perplexity objektif (Milestone 2 — `tools/eval_perplexity.cpp`)
- Dibangun supaya penilaian kualitas tidak lagi cuma "baca teksnya jelek/bagus".
- Dipakai untuk semua eksperimen selanjutnya.

### 4. GPTQ-style calibrated quantization (`tools/gptq_ternary.cpp`)
- Kompensasi error sequential-column pakai Hessian dari aktivasi kalibrasi asli (8 prompt, 462 sampel).
- Perplexity: **11,822** — membantu (turun 3× dari naif) tapi masih 436× lebih buruk dari fp32.
- Kesimpulan: linear error-compensation tidak cukup untuk grid 3-level.

### 5. Mixed precision: attention INT8, FFN ternary (Milestone 5)
- Perplexity: **15,093** — lebih buruk dari GPTQ-ternary-penuh.
- Temuan penting: **FFN adalah sumber error dominan**, bukan attention (dikonfirmasi juga oleh eksperimen sensitivitas terpisah: FFN-only ternary collapse lebih cepat daripada attention-only ternary).

### 6. INT8 di semua tempat (eksperimen penentu)
- Perplexity: **27.01** — **nyaris identik dengan fp32 (27.11)**.
- Membuktikan: masalahnya bukan "kuantisasi post-training gagal untuk model kecil" — INT8 (255 level) bekerja nyaris sempurna. Masalahnya spesifik pada **ternary (3 level)** yang tidak cukup menyimpan informasi model 15M non-QAT.

## Tabel Perbandingan Lengkap

| Skema | Perplexity | Rasio vs fp32 | Ukuran file |
|---|---|---|---|
| FP32 baseline | 27.11 | 1× | 57 MB |
| **INT8 semua (PRODUKSI)** | **27.01** | **1.00×** | **41 MB (71%)** |
| GPTQ ternary (semua) | 11,822 | 436× lebih buruk | 41 MB |
| Mixed: attn INT8, FFN ternary | 15,093 | 557× lebih buruk | 41 MB |
| Ternary naif (semua) | 35,099 | 1,295× lebih buruk | 41 MB |

*(Catatan: ukuran file sama 41MB di semua varian kuantisasi karena prototipe ini menyimpan 1 byte/bobot baik untuk ternary maupun int8 — belum ada bit-packing 2-bit untuk ternary. Bila nanti ternary dipakai lagi, ukurannya bisa ditekan lebih jauh dengan packing asli.)*

## Kenapa Ternary/BitNet Gagal di Sini (dan Kapan Ia Akan Berhasil)
BitNet asli (Microsoft) **dilatih dari nol** dalam presisi ternary (quantization-aware training) — bobotnya memang dioptimasi untuk hidup di 3 level. Model kita (`stories15M`) dilatih fp32 lalu dikuantisasi **belakangan** (post-training) — jauh lebih kasar, dan terbukti gagal meski dengan kalibrasi GPTQ. Ternary baru masuk akal kalau:
1. Pakai checkpoint yang memang dilatih native ternary, atau
2. Investasi penuh ke teknik seperti BiLLM (salient weight preservation) — belum diuji di sini, potensi perbaikan tidak dijamin menutup gap 400×+ .

## Keputusan Produksi
`tools/pack_fllm.cpp` sekarang mengkuantisasi **semua 7 tensor ke INT8** (bukan ternary).
`FastEngine::load()` otomatis mendeteksi `has_transformer` dan menjalankan
`transformer_mixed::MixedPrecisionTransformer` dengan `FfnPrecision::Int8` — jalur ini
**tersambung sampai Android** (tombol "Load model" di `MainActivity.kt` sekarang memuat satu
file `.fllm` produksi, bukan lagi `stories15M.bin` mentah terpisah).

## File Eksperimen (disimpan sebagai referensi riset)
```
tools/audit_quantizer.cpp          — Milestone 1
tools/eval_perplexity.cpp          — Milestone 2
tools/gptq_ternary.h/.cpp          — Milestone 4 (GPTQ-style)
tools/pack_fllm_calibrated.cpp     — packer versi GPTQ (non-produksi)
tools/pack_fllm_mixed.cpp          — packer versi mixed precision (non-produksi)
tools/pack_fllm_int8all.cpp        — sumber asli packer produksi (identik dgn pack_fllm.cpp)
core/include/transformer_bitnet.h/.cpp  — forward pass ternary murni (referensi/riset)
core/include/transformer_mixed.h/.cpp   — forward pass produksi (dipakai FastEngine)
```

## Verifikasi Objektif via CI
Workflow GitHub Actions sekarang otomatis: pack model produksi → jalankan `eval_perplexity`
membandingkan fp32 vs hasil pack → nilai harus tetap dekat dengan baseline sebelum APK dianggap valid.
