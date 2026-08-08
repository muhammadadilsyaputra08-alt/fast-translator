# Optimisasi ARM NEON/SIMD: Implementasi & Batasan Verifikasi Jujur

## Status: Kode ditulis & dites sebisa mungkin di sandbox — korektnes NYATA butuh device Anda

## Kenapa Ini Beda dari Fitur Sebelumnya

Semua fitur sebelumnya (KV-cache, GBNF, dst.) bisa saya compile DAN jalankan penuh di sandbox
(CPU x86). **NEON tidak bisa** — ini instruksi khusus ARM, sandbox saya x86. Saya cek akses
cross-compiler ARM (`apt-get install g++-13-aarch64-linux-gnu`) — **gagal, 403 Forbidden**,
mengonfirmasi tidak ada akses jaringan sama sekali di sandbox ini. Jadi kode NEON di bawah ini
**ditulis dengan hati-hati mengikuti pola standar yang sudah mapan, tapi belum pernah benar-benar
dijalankan oleh siapapun sampai Anda coba di device.**

## Yang Dilakukan untuk Mitigasi Risiko

### 1. Compile-time guard (`#ifdef __ARM_NEON`)
Kode NEON hanya aktif di target ARM. Di sandbox x86 saya, otomatis fallback ke `matmul_int8_scalar`
(kode yang sudah lama teruji) — **tidak ada resiko merusak apapun yang sudah jalan**, karena target
Android kita `arm64-v8a` (NEON wajib ada di semua CPU AArch64, tidak perlu flag compile tambahan).

### 2. On-device self-test (bukan cuma "percaya kode saya benar")
Tombol baru **"Verify NEON correctness"** di app — jalankan kapan saja (tidak perlu load model
dulu), membandingkan hasil `matmul_int8_neon` vs `matmul_int8_scalar` pada 7 ukuran input berbeda
(termasuk yang tidak habis dibagi 8, supaya jalur "sisa" NEON ikut teruji, bukan cuma jalur SIMD
utamanya), toleransi numerik wajar (urutan akumulasi float NEON vs scalar memang beda, bukan bug).
Melaporkan PASS/FAIL + angka error nyata — bukan cuma "tidak crash".

### 3. Diverifikasi di sandbox sejauh mungkin
- Kode fallback scalar (`FASTAI_HAVE_NEON` tidak terdefinisi) dikompilasi & dijalankan penuh —
  bersih, regresi nol di semua test lama.
- Modul self-test sendiri dikompilasi & dijalankan di x86 — melaporkan "NEON not compiled in"
  dengan benar (bukti bahwa mekanisme deteksi platform bekerja, meski NEON-nya sendiri belum teruji).
- CI (`test_neon_selftest.cpp`) otomatis assert laporan itu benar tiap push.

## Yang TIDAK Bisa Saya Klaim
Saya **tidak bisa** bilang "NEON sudah terbukti benar" sampai Anda menjalankan tombol **"Verify NEON
correctness"** di device dan melaporkan hasilnya. Kode intrinsic (`vld1_s8`, `vmovl_s8`, `vmovl_s16`,
`vcvtq_f32_s32`, `vmlaq_f32`, `vaddvq_f32`) ditulis mengikuti pola widening int8→int16→int32→float32
yang standar dan terdokumentasi baik, tapi detail seperti urutan lane, edge-case ukuran ganjil, dsb.
riskan salah tanpa eksekusi nyata.

## Cara Verifikasi (Langkah Anda)
1. Push project ini, tunggu CI build APK.
2. Install, buka app — **tidak perlu load model dulu**.
3. Tap **"Verify NEON correctness"**.
4. Kalau muncul `RESULT: PASS` → kode benar, lanjut ke langkah 5.
   Kalau muncul `RESULT: FAIL` → kirim screenshot ke saya, itu bug nyata yang perlu saya perbaiki
   (dengan detail angka `max diff` per ukuran, sangat membantu diagnosis).
5. Kalau PASS, tap **"Benchmark"** lagi — bandingkan tok/s baru vs baseline sebelumnya (12.77 tok/s)
   untuk lihat speedup nyata dari optimisasi ini.

## File
```
core/include/transformer_mixed.h    — deklarasi matmul_int8_scalar/matmul_int8_neon
core/src/transformer_mixed.cpp      — implementasi NEON + scalar fallback
core/include/neon_selftest.h
core/src/neon_selftest.cpp          — self-test on-device
core/tests/test_neon_selftest.cpp   — CI: verifikasi graceful fallback di x86
```

## Update Setelah Anda Menguji
*(bagian ini akan diisi setelah hasil dari device Anda)*
- [ ] Verify NEON correctness: PASS / FAIL
- [ ] Benchmark tok/s setelah NEON: _____ tok/s (baseline sebelumnya: 12.77 tok/s)
