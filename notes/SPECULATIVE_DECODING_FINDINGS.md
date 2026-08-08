# Speculative Decoding: Implementasi Nyata & Temuan Jujur

## Status: BERHASIL diimplementasikan, TERBUKTI BENAR, TAPI TIDAK MEMPERCEPAT pada model/hardware ini

## Yang dibangun (semua nyata, teruji, bukan simulasi)

### 1. `forward_batch()` — verifikasi multi-token dalam satu pass
Ditambahkan ke `transformer_mixed.cpp`: memproses N token sekaligus dengan matmul ter-batch
(GEMM, bukan GEMV berulang), menjaga causal masking & KV-cache yang benar untuk setiap posisi.
- **Teruji:** hasil logits **identik bit-per-bit** (`max_abs_diff = 0`) dengan memanggil `forward()`
  satu-per-satu secara sekuensial. Ini pembuktian korektnes paling ketat yang bisa dilakukan.
- Speedup batching murni (tanpa speculative drafting, cuma isolasi efek batching): **1.01×** — nyaris nol.

### 2. Prompt Lookup Decoding — drafting nyata tanpa perlu extra trained heads
`test_speculative_decoding.cpp`: mencari n-gram berulang di teks yang sudah digenerate, mengambil
kelanjutan sebelumnya sebagai draft token, verifikasi lewat `forward_batch()`, terima sebanyak
mungkin prefix yang cocok dengan prediksi model (sama seperti Medusa's typical acceptance, tapi
tanpa perlu head tambahan yang di-training khusus).
- **Teruji:** output speculative **identik** dengan output baseline sekuensial (pembuktian bahwa
  algoritma accept/reject-nya benar, tidak mengubah hasil akhir).
- Acceptance rate: 43.75% (7 dari 16 token draft diterima) — masuk akal untuk prompt yang sengaja
  dibuat repetitif.

## Hasil Benchmark (jujur, bukan yang diharapkan)

| Metode | Tokens/detik | vs Baseline |
|---|---|---|
| Sequential (baseline) | 51.07 | 1× |
| Speculative (lookup + batch verify) | 45.70 | **0.89× (lebih lambat)** |

## Kenapa Ini Terjadi (Bukan Bug — Realitas Engineering)

Speculative decoding secara historis terbukti mempercepat inferensi karena dua faktor:
1. **Model besar = memory-bandwidth-bound.** Setiap token butuh load ulang jutaan/miliaran bobot
   dari memori — ini biaya dominan, bukan komputasinya. Batching verifikasi mengamortisasi biaya
   load tersebut ke beberapa token sekaligus.
2. **Hardware paralel (GPU/NPU) yang under-utilized** saat generate token satu-satu — speculative
   decoding mengisi kapasitas nganggur itu dengan kerja verifikasi paralel yang "gratis".

**Model kita (15M parameter, ~41MB) sudah cukup kecil untuk hidup di cache CPU**, dan pengujian
dilakukan di **CPU single-thread** (sandbox ini, bukan NPU Snapdragon target asli). Tidak ada
bottleneck memory-bandwidth untuk diamortisasi, tidak ada kapasitas paralel nganggur untuk diisi.
Hasilnya: overhead menghitung kandidat draft yang ditolak (56% di antaranya) lebih mahal daripada
penghematan dari yang diterima.

**Klaim blueprint "Medusa 2-3× speedup"** diukur pada model 1.5B–13B di GPU/NPU — kondisi yang
sangat berbeda dari eksperimen ini. Klaim itu kemungkinan valid **untuk model yang lebih besar dari
milik kita, di hardware yang ditargetkan blueprint (Snapdragon NPU via QNN)** — tapi itu tidak bisa
divalidasi di sandbox ini karena tidak ada akses SDK QNN/device fisik (lihat percakapan sebelumnya).

## Rekomendasi
Untuk model & hardware saat ini (15M, CPU), **generate sekuensial biasa adalah pilihan yang benar**
— lebih sederhana DAN lebih cepat daripada speculative decoding. Modul speculative decoding yang
sudah dibangun (`forward_batch`, prompt-lookup drafting) disimpan sebagai kapabilitas siap pakai:
kalau nanti dipasangkan dengan model lebih besar atau dijalankan di NPU asli (bukan CPU sandbox),
worth diuji ulang — arsitekturnya sudah benar, cuma belum menang di kondisi saat ini.

## File
```
core/include/transformer_mixed.h    — deklarasi forward_batch()
core/src/transformer_mixed.cpp      — implementasi batched matmul + forward_batch()
core/tests/test_batch_correctness.cpp    — pembuktian batch == sequential (exact)
core/tests/test_speculative_decoding.cpp — prompt-lookup decoding + benchmark end-to-end
```
