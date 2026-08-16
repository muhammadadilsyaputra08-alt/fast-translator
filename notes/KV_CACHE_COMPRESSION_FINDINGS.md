# KV-Cache Compression: Temuan & Integrasi Produksi

## Ringkasan: Hasil POSITIF, tersambung penuh ke produksi ✅

Beda dari saga kuantisasi ternary — ini kisah sukses lurus. Kedua modul dari Fase 1 diaudit dulu
sebelum diimplementasi penuh (prinsip yang sama: jangan investasi besar tanpa validasi awal).

## MiniCache: diaudit, terbukti TIDAK BERLAKU untuk model ini — di-skip

**Sebelum implementasi penuh**, saya cek asumsi dasar MiniCache: apakah representasi KV antar-layer
yang bertetangga cukup mirip untuk di-merge tanpa kehilangan informasi berarti?

```
Cosine similarity K-vector, layer l vs l+1 (stories15M, 6 layer):
  layer 0 vs 1: -0.023
  layer 1 vs 2: -0.056
  layer 2 vs 3: -0.034
  layer 3 vs 4:  0.104
  layer 4 vs 5: -0.054
```

**Nyaris nol di semua pasangan.** Paper MiniCache aslinya diuji di model besar (32+ layer) di mana
representasi jenuh di layer tengah-dalam. Model 6-layer kita: setiap layer melakukan transformasi
substansial dan berbeda — tidak ada redundansi untuk dieksploitasi. Memaksa merge di sini akan
menyuntikkan noise, bukan mengompresi secara berguna. **Keputusan: skip, hemat waktu.**

## RocketKV: diimplementasi penuh, diuji, TERBUKTI BEKERJA

Eviction berbasis posisi token (bukan antar-layer) — tidak bergantung asumsi similaritas yang gagal
di atas. Diimplementasikan sebagai `forward_evictable()` di `transformer_mixed.cpp`: cache KV
sekarang berbasis daftar slot aktif (bukan array indeks tetap), sehingga token yang di-evict
benar-benar dibuang dari memori, bukan cuma di-mask.

### Hasil Uji (`core/tests/test_kv_eviction.cpp`)

**Test 1 — Korektnes:** dengan threshold besar (eviction tak pernah aktif), `forward_evictable()`
harus identik `forward()` biasa.
```
max diff: 0 (bit-identik)
```

**Test 2 — Generate 60 token dengan eviction agresif (threshold=20, keep=12):**
```
KV cache size -- baseline (64 posisi, tanpa kompresi): 884,736 bytes
KV cache size -- dengan eviction: 262,656 bytes
Pengurangan memori: 70.3%
```
Output tetap cerita koheren penuh (bukan degenerate), meski isi ceritanya sedikit berbeda dari
baseline setelah titik eviction (diharapkan — konteks yang dilihat model memang berubah).

**Test end-to-end lewat `FastEngine::load()` produksi**, generate 150 token (melewati
`evict_threshold=128`):
> "...there was a little girl named Lily. She loved to play outside in the sunshine. One day, she
> saw a big, red ball in the sky. It was the sun!... Lily asked her mommy, 'Can you help me touch
> the sunshine?' Her mommy said, 'Sure, let's go outside and try to touch it together.'..."

Tetap koheren penuh sampai akhir, walau sempat melewati ambang eviction di tengah generate.

## Integrasi Produksi

- `core/include/kv_cache.h`: tambah `RocketKVEvictor::should_evict_check()`.
- `core/include/transformer_mixed.h` / `.cpp`: tambah `forward_evictable()` dan `kv_cache_bytes()`.
- `core/include/engine.h` / `core/src/engine.cpp`: `FastEngine` otomatis membaca
  `fllm_model_.kv_config.enable_rocketkv` saat `load()`; kalau aktif, `generate_quantized()`
  otomatis pakai `forward_evictable()` alih-alih `forward()` biasa. **Tidak ada perubahan API
  publik** — Kotlin/JNI tidak perlu disentuh sama sekali, semua transparan di balik `.fllm`.
- `tools/pack_fllm.cpp`: `kv_config` sekarang `enable_minicache=false` (jujur, karena tidak dipakai),
  `enable_rocketkv=true`, `rocketkv_evict_threshold=128` (beri ruang untuk generate pendek ~60 token
  seperti demo Android, tetap membatasi memori untuk generate panjang).

## Regresi
38+ test lama tetap lulus setelah integrasi (`test_fase1.cpp` 24, `test_engine.cpp` 8,
`test_engine_real_model.cpp` 6) + JNI smoke test 5/5.

## File
```
core/tests/test_kv_eviction.cpp   — test korektnes + pengukuran memori nyata
```
