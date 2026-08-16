# Laporan: Export & Kuantisasi TinyLlama-1.1B-Chat untuk FastTranslator

**Tanggal:** 14 Agustus 2026
**Model sumber:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (HuggingFace)
**Target:** `model.fllm` (format produksi INT8, dipakai engine Android FastTranslator)

---

## 1. Masalah: `tokenizer.bin` Tidak Ditemukan

### Gejala
Skrip diff berbasis `glob` (before/after snapshot direktori) gagal mendeteksi `tokenizer.bin`
hasil ekspor, walau `tokenizer.py` sukses jalan.

### Root Cause
Pendekatan diff-glob custom reinvent hal yang sudah disediakan `llama2.c/tokenizer.py` —
skrip itu otomatis menulis `tokenizer.bin` ke working directory (`/content/llama2.c`), tidak
perlu logika pencarian file baru secara manual.

### Solusi
Cell yang dipakai (identik dengan `notes/FastTranslator_TinyLlama_Pipeline.ipynb`, Cell 11):

```python
import glob, os

os.chdir('/content/llama2.c')

tok_candidates = glob.glob(
    '/root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/**/tokenizer.model',
    recursive=True
)
print("tokenizer.model ditemukan di:", tok_candidates)

if tok_candidates:
    !python tokenizer.py --tokenizer-model="{tok_candidates[0]}"
    !ls -la tokenizer.bin
else:
    print("TIDAK KETEMU tokenizer.model -- jalankan cell berikut untuk cari manual:")
    !find /root/.cache/huggingface -iname '*tokenizer*'
```

**Hasil:** `tokenizer.bin` berhasil dibuat di `/content/llama2.c/tokenizer.bin` (433.869 bytes).

---

## 2. Verifikasi Config Binary `tinyllama.bin`

Sebelum compile, config header binary diverifikasi manual untuk memastikan file hasil
`export.py` benar-benar arsitektur TinyLlama (bukan sisa `stories15M.bin` lama):

```python
%%writefile /content/check_bin.py
import struct
with open('/content/tinyllama.bin','rb') as f:
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('iiiiiii', f.read(28))
print(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
```

```python
!python3 /content/check_bin.py
!ls -la /content/tinyllama.bin
```

**Output:**
```
2048 5632 22 32 4 -32000 2048
-rw-r--r-- 1 root root 4400717852 Aug 14 05:24 /content/tinyllama.bin
```

**Interpretasi:** `dim=2048, n_layers=22, n_heads=32, n_kv_heads=4 (GQA), vocab_size=32000
(negatif = unshared classifier), seq_len=2048` — cocok persis dengan arsitektur
TinyLlama-1.1B. Ukuran 4.4 GB wajar untuk fp32 1.1B parameter. **Kesimpulan: export benar,
bukan file lama.**

---

## 3. Compile & Test Forward Pass (fp32, sebelum kuantisasi)

Compile:
```
/content/project/fasttranslator-engine/tools
Compile selesai.
```

Output `test_real_model`:
```
[PASS] stories15M.bin loads successfully
[FAIL] config.dim == 288
[FAIL] config.n_layers == 6
[FAIL] config.n_heads == 6
[PASS] config.vocab_size == 32000
[FAIL] config.seq_len == 256
[PASS] tokenizer.bin loads successfully
[PASS] encode() produces at least one token
[PASS] encode() prepends BOS token when requested
[PASS] model and tokenizer both load for generation test

--- GENERATED TEXT (real stories15M weights, argmax) ---
 time, there was a young woman named Lily. She was a kind and gentle soul, always looking for ways to help others. One day, she stumbled upon a group of people who were being oppressed by a cruel and corrupt ruler. Lily knew that she had to do something
--- end (108.528s, 0.552851 tok/s) ---

[PASS] generated story is non-empty
[PASS] generated story has substantial length
[PASS] generation is deterministic across independent model loads
--- second prompt test ---
 dog was a stray, and he had been wandering the streets for days. He was small and scrawny, with a long, thin tail and
---

[PASS] different prompts produce different continuations

10/14 tests passed
```

### Evaluasi Hasil Test
- **4 FAIL adalah expected**, bukan bug: `test_real_model.cpp` meng-hardcode nama test dan
  nilai assert untuk config `stories15M` (dim=288, n_layers=6, n_heads=6, seq_len=256) sebagai
  baseline lama; nilai itu tidak relevan untuk TinyLlama dan sengaja dibiarkan gagal.
- Loading model, tokenizer, dan generasi teks semua **PASS** — jalur GQA di `transformer.cpp`
  (kv_dim, group_size, RoPE per `n_kv_heads`) sudah menangani arsitektur TinyLlama dengan benar.
- Kecepatan **0.55 tok/s** pada fp32 CPU-only wajar untuk model 1.1B tanpa kuantisasi.

---

## 4. Kuantisasi ke `.fllm` (INT8)

```
unshared classifier detected -- read separate wcls tensor (250 MB)
quantizing attention to INT8 (kv_dim=256, dim=2048)...
quantizing FFN to INT8...
wrote /content/model.fllm (1425 MB, 33.9604% of original 4196 MB)
-rw-r--r-- 1 root root 1494499975 Aug 14 05:46 /content/model.fllm
```

## 5. Verifikasi Kualitas Kuantisasi (Perplexity)

```
/content/project/fasttranslator-engine/tools
--- fp32 baseline ---
tokens evaluated: 56
average cross-entropy (nats/token): 4.0317
perplexity: 56.3566
--- packed .fllm (INT8) ---
tokens evaluated: 56
average cross-entropy (nats/token): 3.94169
perplexity: 51.5054
```

**Catatan:** Perplexity INT8 (51.51) sedikit lebih rendah dari fp32 (56.36) pada sampel 56
token ini — dalam rentang wajar untuk sampel evaluasi kecil, mengindikasikan kuantisasi
nyaris lossless (tidak ada degradasi signifikan).

---

## 6. Output Final

| Item | Nilai |
|---|---|
| File | `model.fllm` |
| Ukuran | 1.494.499.975 bytes (≈ 1494.5 MB) |
| Lokasi Colab | `/content/model.fllm` |
| Lokasi Drive | `/content/drive/MyDrive/FastTranslator/model.fllm` |
| Rasio kompresi | 33.96% dari ukuran asli (4196 MB) |

---

## 7. Uji Kapabilitas Terjemahan (Sebelum Deploy)

Test forward pass di atas (Bagian 3) cuma membuktikan arsitektur (GQA, `wcls`, forward pass)
benar — belum membuktikan kapabilitas terjemahan/instruksi yang jadi tujuan utama upgrade ini.
Dibuat notebook terpisah (`FastTranslator_Test_Terjemahan.ipynb`) yang me-load `model.fllm` dari
Drive lewat `engine::FastEngine::load()` — jalur kode identik dengan Android — dan menguji 3 gaya
prompt.

### Diagnostik Tokenizer

```
"<|system|>" -> 5 token(s): [29966, 29989, 5205, 29989, 29958]  (PECAH -- byte-fallback)
"<|user|>" -> 5 token(s): [29966, 29989, 1792, 29989, 29958]  (PECAH -- byte-fallback)
"<|assistant|>" -> 6 token(s): [29966, 29989, 465, 22137, 29989, 29958]  (PECAH -- byte-fallback)
"</s>" -> 3 token(s): [829, 29879, 29958]  (PECAH -- byte-fallback)
```

**Temuan:** tokenizer hasil ekspor `llama2.c` **tidak mengenali** token spesial chat template
TinyLlama-Chat (`<|system|>`, `<|user|>`, `<|assistant|>`, `</s>`) sebagai token utuh — semua
pecah jadi byte-fallback. Chat template asli TinyLlama-Chat tidak bisa dipakai apa adanya dengan
tokenizer ini.

### Hasil 3 Gaya Prompt

| Gaya | Hasil |
|---|---|
| Instruksi polos ("Translate to English: ...") | Fluent secara grammar, tapi arah terjemahan terbalik (nerjemahin ke Prancis, bukan dari Prancis) |
| Chat template TinyLlama-Chat | Gibberish (efek langsung dari token byte-fallback yang membanjiri prompt) |
| Kontrol non-terjemahan ("List three colors") | **Gibberish tak terduga** — prompt sederhana, seharusnya model bisa jawab wajar |

Hasil kontrol yang janggal itu memicu investigasi bug terpisah (Bagian 8).

## 8. Bug Produksi: RocketKV Eviction Cache Tidak Direset Antar Panggilan `generate()`

### Gejala

Memanggil `FastEngine::generate()` dua kali berturut-turut pada instance `FastEngine` yang sama
(persis skenario nyata: user translate kalimat kedua tanpa restart app) menghasilkan output yang
**bocor** dari panggilan sebelumnya. Contoh: setelah generate() dengan prompt berisi "Je suis
désolé", panggilan berikutnya dengan prompt tak berhubungan ("List three colors: ") ikut
menghasilkan teks yang mengandung "Je suis désolé...sorry" — mustahil terjadi kalau setiap
panggilan benar-benar independen.

### Proses Debugging

Diuji berlapis dari luar ke dalam untuk mengisolasi lapisan mana yang bermasalah:

1. **AddressSanitizer + UBSan** pada `test_translate.cpp` → bersih (bukan buffer overrun).
2. **MemorySanitizer** → menandai `std::ifstream` constructor di `fllm_parser.cpp` — dikonfirmasi
   **false-positive** (libstdc++ sistem belum diinstrumentasi MSan), bukan bug asli.
3. **Valgrind** pada model 4.4GB → terlalu lambat untuk dipraktikkan (instrumentasi 20-50x lebih
   lambat pada working set besar).
4. **Isolasi manual bertahap** langsung ke `MixedPrecisionTransformer::forward()` (bypass
   `engine.cpp` dan tokenizer sepenuhnya):
   - `forward(token=BOS, pos=0)` dipanggil 2x → **identik**.
   - `forward()` berurutan 7 posisi, dijalankan 2x → **identik**.
   - Generasi auto-regressive penuh (prompt + 30 token bebas), dijalankan 2x → **identik**.

   Kesimpulan dari langkah ini: `forward()` biasa 100% deterministik dan bersih dari state bocor.

### Root Cause

`tools/pack_fllm.cpp` hardcode `kv_config.enable_rocketkv = true` untuk setiap `.fllm` yang
di-pack — artinya jalur produksi (`engine.cpp`) **selalu** memanggil
`MixedPrecisionTransformer::forward_evictable()`, bukan `forward()` biasa yang sudah terbukti
bersih di atas.

Di `core/src/transformer_mixed.cpp`, member `evict_key_cache_` dan `evict_value_cache_`:

```cpp
if (evict_key_cache_.empty()) {
    evict_key_cache_.assign(p.n_layers, {});
    evict_value_cache_.assign(p.n_layers, {});
}
...
evict_key_cache_[l].insert(evict_key_cache_[l].end(), s.k.begin(), s.k.end());
evict_value_cache_[l].insert(evict_value_cache_[l].end(), s.v.begin(), s.v.end());
```

hanya diinisialisasi **sekali** (guard `if (empty())`), lalu **selalu di-*append*, tidak pernah
dikosongkan**. Setiap panggilan `generate()` baru menumpuk token barunya di atas cache lama —
bukan mulai dari cache kosong — sehingga attention pada panggilan kedua benar-benar menghitung
skor atas gabungan token lama (dari panggilan sebelumnya) + token baru. Ini bukan sekadar sinyal
statistik yang mirip; secara harfiah model attend ke key/value asli dari sesi sebelumnya.

### Fix

Ditambahkan method `reset_evictable_cache()` pada `MixedPrecisionTransformer`
(`core/include/transformer_mixed.h` + `core/src/transformer_mixed.cpp`) yang mengosongkan
`active_positions_`, `evict_key_cache_`, `evict_value_cache_`. Dipanggil di awal
`FastEngine::generate_quantized()` (`core/src/engine.cpp`), sebelum loop generasi dimulai:

```cpp
void FastEngine::generate_quantized(const GenerateOptions& opts, ...) {
    if (evictor_) quantized_->reset_evictable_cache();
    auto prompt_tokens = tokenizer_->encode(opts.prompt, true);
    ...
```

### Verifikasi Setelah Fix

| Test | Sebelum fix | Sesudah fix |
|---|---|---|
| Prompt sama dipanggil 2x berturut-turut | Output berbeda (`Blue/Green/Red` vs `Blue/Yellow/Orange`) meski argmax deterministik | **Identik** byte-per-byte |
| Prompt tak berhubungan setelah chat template | Bocor kosakata Prancis dari prompt sebelumnya | Bersih, jawab sesuai prompt sendiri |

### Dampak Kalau Tidak Diperbaiki

Bug ini akan muncul di Android **setiap kali user menerjemahkan kalimat kedua** dalam satu sesi
app tanpa restart — hasil terjemahan kalimat kedua dan seterusnya berpotensi tercampur konteks
dari kalimat sebelumnya. Krusial untuk diperbaiki sebelum deploy, karena use-case utama app
(translate berkali-kali dalam satu sesi) pasti memicu jalur kode yang cacat ini.

## 9. Yang Masih Perlu Diperbaiki Sebelum Deploy Penuh

1. **Chat template TinyLlama-Chat belum bisa dipakai** — token spesial pecah byte-fallback
   (Bagian 7). Opsi: (a) pakai instruksi polos sebagai gantinya, atau (b) daftarkan token spesial
   sebagai entri tambahan di vocab tokenizer sebelum re-export.
2. ~~**Arah terjemahan kadang terbalik** dengan instruksi polos~~ — **SELESAI**, lihat Bagian 11:
   template few-shot (varian B) konsisten benar arah dan berhenti bersih.
3. ~~`MainActivity.kt` baris 99 & 139 masih hardcode `prompt = "Once upon a time"`~~ — **SELESAI**,
   lihat Bagian 11: sudah diganti template terjemahan final.

## 10. Langkah Selanjutnya: Deploy ke Android

1. Simpan ulang project yang sudah dipatch (`fasttranslator-engine-FIXED.zip`) ke Drive,
   menggantikan versi lama.
2. Buka Google Drive di HP, download `model.fllm`.
3. Push ke storage device via ADB:
   ```
   adb push model.fllm /sdcard/Android/data/com.fasttranslator/files/models/model.fllm
   ```
4. Build ulang APK dari source yang sudah dipatch (fix RocketKV di Bagian 8 wajib masuk sebelum
   build produksi).

## 11. Iterasi Prompt Terjemahan & Penerapan di `MainActivity.kt`

Empat gaya prompt diuji di Colab (`FastTranslator_Iterasi_Prompt.ipynb`) atas kalimat sumber
"Bonjour, comment ca va aujourd'hui?", masing-masing dengan stop-string sederhana supaya generasi
berhenti wajar (tidak nyerempet ke label/teks lain):

| Gaya | Arah terjemahan | Cara berhenti |
|---|---|---|
| A. Label French/English | Benar | Bocor lanjut ke `French:` berikutnya (stop token telat) |
| **B. Few-shot 2 contoh** | **Benar** | **Bersih, berhenti tepat setelah jawaban** |
| C. Instruksi + kutip eksplisit | Benar | Bersih, tapi output kebungkus tanda kutip ekstra |
| D. Format tanya-jawab | Benar tapi parafrase (bukan terjemahan literal) | Bersih |

**Pemenang: B (few-shot 2 contoh)** — satu-satunya yang benar arah, bersih berhenti, dan tidak
perlu strip tanda kutip ekstra (beda dari C).

Catatan tambahan: semua output diawali `: ` (artefak echo label "English:") — perlu di-strip
sebelum ditampilkan ke user, apa pun gaya yang dipilih.

### Diterapkan di `MainActivity.kt`

- Ditambahkan `companion object` berisi `DEMO_SOURCE_TEXT`, `buildTranslationPrompt()` (template B),
  dan `stripPromptArtifact()` (strip prefix `": "`).
- `onGenerateClicked()` (baris ~99 lama) dan `onBenchmarkClicked()` (baris ~139 lama): hardcode
  `prompt = "Once upon a time"` diganti `buildTranslationPrompt(DEMO_SOURCE_TEXT)`.
- Stripping artefak diterapkan secara streaming-safe: token di-buffer sampai konten pertama
  non-`":"`/spasi/newline muncul, baru ditampilkan ke `outputView` — supaya tampilan tetap
  streaming token-by-token tanpa nunggu generasi selesai.
- `maxTokens` untuk `onGenerateClicked()` diturunkan dari 60 ke 40 (sesuai panjang yang dipakai
  saat pengujian gaya B, cukup untuk satu kalimat terjemahan tanpa rambling).

UI belum punya input teks bebas — `DEMO_SOURCE_TEXT` masih hardcode satu kalimat contoh. Kalau app
perlu menerima input dari user, tambahkan `EditText` dan ganti `DEMO_SOURCE_TEXT` dengan nilainya
sebelum build produksi.

## 12. Bug Produksi #2: Crash SIGSEGV Saat RocketKV Eviction Aktif (Stride Salah untuk GQA)

### Gejala

Setelah fix Bagian 8 dan mmap loader (Bagian 13), loading model berhasil tanpa crash dan tombol
**Generate** (maxTokens=40) berjalan normal sampai selesai. Tapi tombol **Benchmark**
(maxTokens=80) crash setelah beberapa detik generasi — dikonfirmasi via `adb logcat` + tombstone:

```
Fatal signal 11 (SIGSEGV), code 2 (SEGV_ACCERR) ... tid 1820 (Thread-5), pid 25633 (.fasttranslator)
backtrace:
  #01 std::vector<float>::__assign_with_size(...)
  #02 transformer_mixed::MixedPrecisionTransformer::forward_evictable(int, int, kvcache::RocketKVEvictor const&)
  #03 engine::FastEngine::generate_quantized(...)
```

### Root Cause

Di blok kompaksi cache RocketKV (`transformer_mixed.cpp`, dieksekusi saat
`evictor.should_evict_check()` bernilai true), indexing ke `evict_key_cache_`/`evict_value_cache_`
memakai stride `dim` (2048, dimensi embedding penuh):

```cpp
scoring_entries[t].key.assign(
    evict_key_cache_[0].begin() + t * dim0, evict_key_cache_[0].begin() + (t + 1) * dim0);
```

Padahal `evict_key_cache_`/`evict_value_cache_` diisi dari `s.k`/`s.v`, yang berukuran
**`kv_dim`** (`n_kv_heads * head_size`), bukan `dim`. Untuk model MHA biasa (`n_kv_heads ==
n_heads`, kasus `stories15M`), `kv_dim == dim` secara kebetulan, jadi bug ini tidak pernah
ketahuan di test-test sebelumnya. Untuk TinyLlama (GQA: `n_heads=32`, `n_kv_heads=4` →
`kv_dim=256` vs `dim=2048`), begitu eviction pertama kali benar-benar berjalan, indexing
`t * 2048` melompat jauh melebihi ukuran buffer asli (`n_active * 256`) → baca/tulis di luar
batas → crash. Ini baru terpicu saat `maxTokens` cukup besar untuk melewati
`rocketkv_evict_threshold=128` (Benchmark, bukan Generate biasa yang cuma 40 token).

### Verifikasi

1. Ditulis test baru `core/tests/test_evictable_gqa.cpp`: model kecil dengan GQA nyata
   (`n_heads=8, n_kv_heads=2`) yang memaksa eviction aktif berkali-kali.
2. Dijalankan versi **sebelum** fix di bawah AddressSanitizer → **`heap-buffer-overflow`
   terkonfirmasi**, persis di baris `.assign()` yang sama dengan crash di device.
3. Fix diterapkan: ganti seluruh `dim`/`dim0` di blok kompaksi eviction dengan `kv_dim`.
4. Versi setelah fix dijalankan ulang di bawah ASan → bersih, 30 posisi dengan eviction aktif
   selesai tanpa error.
5. Regression penuh: `test_fase1` (24/24), `test_engine` (8/8), `test_evictable_gqa` (baru) — semua
   PASS.

### Ditambahkan ke CI

`test_evictable_gqa.cpp` dijalankan di bawah ASan+UBSan sebagai step CI tersendiri
(`.github/workflows/build.yml`, setelah step "Build & run KV-cache eviction tests"), supaya
regresi serupa di masa depan tertangkap otomatis alih-alih baru ketahuan lewat crash di device.

## 13. Perbaikan Memori Loading: mmap (Bug Produksi #3, OOM)

Sebelum bug RocketKV di atas ditemukan, loading model sempat crash duluan karena OOM asli —
dikonfirmasi via logcat: `reason=3 (LOW_MEMORY) ... pss=2,7GB` saat proses loading.

Penyebab: `read_fllm()` membaca seluruh file 1.5GB ke `std::vector<uint8_t> buf` di heap, LALU
menyalin ulang setiap tensor ke `std::vector` baru di `model.transformer_weights` — kedua salinan
hidup bersamaan saat parsing, sesuai dengan angka 2.7GB yang terekam.

**Fix:** `read_fllm()` diubah memakai `mmap()` (RAII `MappedFile`) alih-alih
`std::ifstream`+`std::vector` — file dipetakan langsung oleh OS, halamannya file-backed & clean
sehingga bisa di-reclaim otomatis saat tekanan memori tinggi, alih-alih wajib resident penuh
seperti heap buffer biasa. Peak memory turun kira-kira separuh.

Field `model.weights` (legacy, ditulis `pack_fllm.cpp` tapi terbukti tidak pernah dibaca konsumen
manapun di runtime) sempat dihapus dari parsing untuk hemat memori lebih lanjut, tapi ini
mematahkan test kontrak format `weights roundtrip` di `test_fase1.cpp` — **dikembalikan** setelah
ditemukan bahwa untuk pack path TinyLlama (`pack_fllm.cpp`) field ini memang selalu kosong
(`model.weights = {}`), jadi parsingnya tidak menambah beban memori nyata untuk model produksi ini.

## 14. Status Akhir

Tiga bug produksi ditemukan & diperbaiki secara berurutan, masing-masing baru muncul setelah yang
sebelumnya diperbaiki (loading OOM → cross-call state leak → eviction crash):

| # | Bug | Gejala | File yang diperbaiki |
|---|---|---|---|
| 1 | OOM saat loading | Crash instan setelah "Loading model..." | `fllm_parser.cpp` (mmap) |
| 2 | RocketKV cache tidak direset antar `generate()` | Output bocor dari sesi sebelumnya | `transformer_mixed.{h,cpp}`, `engine.cpp` |
| 3 | Stride salah (`dim` vs `kv_dim`) saat eviction aktif | Crash SIGSEGV setelah beberapa detik generate (Benchmark) | `transformer_mixed.cpp` |

Ditambah: threading fix (`Dispatchers.IO`) + auto-load di `MainActivity.kt`, `largeHeap` di
manifest, dan template prompt terjemahan final (Bagian 11).

**Terverifikasi di device (16.24, 15 Agustus 2026):** Load -> Generate -> Benchmark selesai penuh
tanpa crash. Output benchmark: `81 tokens in 56531 ms = 1,43 tok/s (CPU reference, no NPU)`.

## 15. Catatan: `Backend.QNN` Belum Benar-Benar Mengaktifkan NPU

Angka `1,43 tok/s` di atas memicu pertanyaan kenapa selambat itu untuk app bernama
"Fast"Translator. Diperiksa lebih lanjut:

- `Backend` (`core/include/engine.h`) cuma `enum class { QNN, VULKAN, EXECUTORCH }` — nilai
  `backend_` yang disimpan **tidak pernah dibaca di mana pun** di `engine.cpp` (dikonfirmasi via
  grep). Apa pun backend yang dikirim dari `MainActivity.kt` (`Backend.QNN`), engine selalu
  menjalankan jalur compute yang sama. **Belum ada integrasi NPU (Qualcomm QNN SDK) sama
  sekali** — label "(CPU reference, no NPU)" di `MainActivity.kt` sudah akurat dan jujur, bukan
  fallback diam-diam / bug.
- Kabar baik: `matmul_int8` (`transformer_mixed.cpp`) sudah otomatis pakai **NEON SIMD** saat
  dikompilasi untuk `arm64-v8a` (dispatch lewat `#ifdef __ARM_NEON` di compile time, independen
  dari `Backend` enum) — jadi 1,43 tok/s itu sudah dengan akselerasi CPU SIMD, bukan scalar
  murni tanpa optimisasi sama sekali.
- Untuk lompatan kecepatan lebih jauh (NPU/Hexagon DSP), dibutuhkan integrasi Qualcomm QNN SDK —
  proyek terpisah yang jauh lebih besar dari 3 bug produksi di atas, direkomendasikan sebagai
  item roadmap tersendiri, bukan quick-fix.

## 16. Optimasi: CPU Multi-Threading untuk `matmul_int8`

### Konteks

Chipset device test (Infinix Hot 50) adalah **MediaTek** (Dimensity 6300 / Helio G100), bukan
Snapdragon — jalur QNN/Hexagon NPU (Bagian 15) tidak mungkin dipakai sama sekali di hardware ini
(QNN eksklusif Qualcomm). Dua opsi realistis untuk MediaTek: GPU compute via Vulkan (`Backend`
enum sudah punya slot untuk ini) atau CPU multi-threading. Dipilih **multi-threading** dulu karena
lebih cepat dikerjakan dan tidak butuh SDK/API eksternal.

### Implementasi

- `core/include/thread_pool.h` (baru): thread pool ringan, **persistent** (worker di-spawn sekali
  di awal, bukan per panggilan matmul — thread creation overhead akan dominan kalau spawn ulang
  tiap panggilan, mengingat `matmul_int8` dipanggil ~7x per layer x 22 layer x tiap token).
- `matmul_int8_scalar`/`matmul_int8_neon`: ditambah parameter opsional `o_begin`/`o_end` (default
  = seluruh range, jadi caller lama seperti `neon_selftest.cpp` tidak perlu berubah).
- `matmul_int8()` (dispatcher): memecah `d_out` (jumlah neuron output) ke beberapa thread lewat
  `threadpool::global_pool()`. Tiap thread menghitung rentang output yang **sepenuhnya disjoint**
  — tidak ada floating-point reduction yang dipecah antar-thread, sehingga hasilnya **bit-identik**
  dengan versi single-thread, bukan cuma "kurang lebih sama". Matmul kecil (`d_out < 64`, mis.
  proyeksi `wk`/`wv` yang sempit karena GQA) dilewatkan langsung ke jalur single-thread — overhead
  sinkronisasi thread pool tidak sepadan untuk kerja sekecil itu.

### Verifikasi

1. `core/tests/test_matmul_threading.cpp` (baru): membandingkan output `matmul_int8()` (paralel)
   vs `matmul_int8_scalar()` (referensi single-thread) di 7 kombinasi ukuran acak, termasuk ukuran
   yang sengaja tidak habis dibagi jumlah thread — **semua bit-identik**.
2. Dijalankan di bawah **ThreadSanitizer** (`-fsanitize=thread`) — bersih, tidak ada data race,
   baik pada test matmul terisolasi maupun pada forward pass penuh multi-layer
   (`test_evictable_gqa.cpp`) dan skenario `generate()` asli (`test_engine.cpp`).
3. Regression penuh (`test_fase1` 24/24, `test_engine` 8/8, `test_evictable_gqa`,
   `test_neon_selftest`) — semua tetap PASS.

### CI

Ditambahkan step baru "Build & run matmul_int8 threading correctness test (ThreadSanitizer)".
Sekaligus diperbaiki: 2 step CI lama (`NEON self-test`, `KV-cache eviction tests`) yang compile
`transformer_mixed.cpp` belum punya flag `-pthread` — akan gagal *link* di runner CI (glibc,
beda dari Android NDK/Bionic yang built-in pthread) begitu `transformer_mixed.cpp` memakai
`std::thread`. Sudah ditambahkan sebelum jadi masalah.

### Belum Diverifikasi

Speedup nyata di device (Dimensity 6300, 8 core) **belum diukur** — sandbox ini tidak punya
compiler cross-ARM maupun device fisik. Langkah selanjutnya: build APK dari source ini, jalankan
Benchmark lagi di HP, bandingkan `tok/s` terhadap baseline `1,43 tok/s (Bagian 14)`.

## 17. Hasil Threading di Device: LEBIH LAMBAT, Bukan Lebih Cepat

**Hasil terverifikasi di device (21.01, 15 Agustus 2026):** `81 tokens in 61735 ms = 1,31 tok/s`
— **turun** dari baseline `1,43 tok/s` (Bagian 14). Multi-threading dari Bagian 16 justru
memperlambat inferensi di device nyata, meski semua test korektnes (bit-exact + ThreadSanitizer)
lulus di desktop.

### Analisis

`parallel_for` (Bagian 16) membagi `d_out` **rata** ke semua thread yang tersedia (sampai 8).
Dimensity 6300 adalah **big.LITTLE**: 2× core cepat (Cortex-A76) + 6× core lambat (Cortex-A55).
Pembagian kerja yang rata membuat core lambat jadi bottleneck — semua thread harus menunggu core
paling lambat selesai (barrier di `parallel_for`), plus overhead wake-up thread per panggilan
matmul (dipanggil ratusan kali per token) kemungkinan tidak sepadan untuk workload sekecil ini di
CPU mobile.

### Perbaikan: Diagnostik Terukur, Bukan Tebakan Lagi

Alih-alih menebak ulang jumlah thread yang optimal, ditambahkan **alat ukur langsung di device**:

- `core/include/thread_bench.h` + `core/src/thread_bench.cpp` (baru): menjalankan matmul 2048x2048
  INT8 (ukuran sama dengan `wq`/`wo`) dengan beberapa konfigurasi jumlah thread (1, 2, 3, 4, 6, 8),
  30 iterasi tiap konfigurasi, melaporkan mana yang tercepat SECARA NYATA di hardware yang
  dipakai — bukan asumsi dari spesifikasi chip.
- Diekspos ke Android lewat pola yang sama dengan `nativeVerifyNeon` yang sudah ada:
  `jni_bridge.cpp` (`nativeBenchmarkThreads`) → `FastModelNative.kt`/`FastModel.kt` → tombol baru
  "Benchmark thread counts" di `MainActivity.kt`. **Dijalankan di `Dispatchers.IO`** (bukan main
  thread) karena bisa makan beberapa detik (180 total matmul) — kalau tidak, berisiko ANR persis
  seperti bug loading model di Bagian 13.
- `global_pool()` (ukuran default thread pool produksi) **belum diubah** dari nilai
  `min(8, hardware_concurrency())` — sengaja menunggu hasil pengukuran nyata dari tombol baru ini
  dulu sebelum menebak angka lain, supaya tidak mengulang kesalahan yang sama.

### Langkah Selanjutnya

1. Build APK dari source ini, install, tap **"Benchmark thread counts"** — kirim hasilnya.
2. Berdasarkan angka nyata itu, `global_pool()` di `thread_pool.h` disetel ke jumlah thread yang
   benar-benar tercepat di device ini (kemungkinan besar 1 atau 2, bukan 8 — karena hasil
   Benchmark utama sudah menunjukkan lebih banyak thread = lebih lambat).
3. Kalau ternyata `threads=1` menang mutlak di semua ukuran matmul nyata (bukan cuma yang
   disintesis di `thread_bench`), opsi paling jujur adalah **membatalkan** optimasi threading
   ini sepenuhnya dan kembali ke jalur single-thread murni (kode Bagian 16 dipertahankan tapi
   `global_pool()` dipaksa size=1, efeknya sama dengan tidak ada threading).

## 18. `global_pool()` Disetel ke 6 Thread (Berdasarkan Data, Bukan Tebakan)

Hasil "Benchmark thread counts" di device (21.25, 15 Agustus 2026):

| threads | ms/matmul |
|---|---|
| 1 | 1,234 |
| 2 | 1,089 |
| 3 | 1,344 |
| 4 | 1,299 |
| **6** | **1,036 (tercepat)** |
| 8 | 1,052 |

`threads=6` menang, tapi cuma ~16% lebih cepat dari `threads=1`, dan hasilnya **tidak monoton**
(`threads=3`/`4` malah lebih lambat dari `threads=1`) — indikasi ada noise pengukuran (thermal
throttling / OS scheduling saat itu). Yang lebih penting: microbenchmark matmul terisolasi ini
sebelumnya **berlawanan arah** dengan hasil Benchmark model penuh (Bagian 17), yang menunjukkan
`threads=8` membuat inferensi nyata 9% **lebih lambat**, bukan lebih cepat.

`global_pool()` (`thread_pool.h`) diubah dari `min(8, hw_concurrency)` ke `min(6, hw_concurrency)`
berdasarkan angka ini. **Tapi ukuran yang benar-benar menentukan adalah tombol Benchmark biasa**
(generate 81 token penuh), bukan microbenchmark matmul tunggal ini — kedua alat ukur ini sempat
tidak sepakat sebelumnya. Kalau Benchmark penuh dengan `threads=6` masih belum jelas mengalahkan
baseline `1,43 tok/s` (single-thread), keputusan paling jujur adalah membatalkan threading ini
sepenuhnya (set `global_pool()` ke size=1) daripada terus mempertahankan sesuatu yang belum
terbukti membantu di hardware ini.

### Belum Diverifikasi

Perlu dijalankan ulang tombol **Benchmark (measure real tok/s)** — bukan Benchmark thread
counts — dengan `global_pool()`=6 ini, dibandingkan terhadap baseline `1,43 tok/s`.

## 19. Kesimpulan: CPU Multi-Threading Dibatalkan

**Hasil terverifikasi di device (09.09, 16 Agustus 2026):** `threads=6` → `81 tokens in 59459 ms =
1,36 tok/s`. Masih di bawah baseline single-thread.

Ringkasan 3 percobaan end-to-end (tombol Benchmark, bukan microbenchmark):

| Konfigurasi | tok/s |
|---|---|
| Single-thread (baseline) | **1,43** |
| threads=8 (naif, equal-split) | 1,31 |
| threads=6 (tuned dari microbenchmark) | 1,36 |

Single-thread menang di ketiga percobaan. **Threading dibatalkan** — `global_pool()` di
`thread_pool.h` diset permanen ke size=1 (efektif menonaktifkan paralelisasi tanpa membongkar
infrastrukturnya; kode `ThreadPool`, `matmul_int8` dengan rentang `o_begin`/`o_end`, dan test-test
korektnesnya tetap dipertahankan — sudah terverifikasi benar & bebas race lewat ASan/TSan — untuk
kemungkinan dipakai lagi kalau suatu saat ada perubahan yang membuatnya masuk akal lagi, misalnya
paralelisasi granularitas lebih kasar di level layer, atau model yang jauh lebih besar di mana
porsi matmul mendominasi waktu total).

Kemungkinan penyebab kenapa threading tidak membantu di sini: hanya matmul besar yang
terparalelkan; attention, RMSNorm, softmax, dan logika eviction RocketKV tetap single-thread —
overhead sinkronisasi yang dibayar di tiap panggilan matmul menggerus lebih banyak daripada
manfaat paralel yang didapat.

**Status akselerasi:** kembali ke baseline `1,43 tok/s` (CPU + NEON, single-thread). Opsi
berikutnya yang belum dicoba: **Vulkan GPU compute** (Bagian 15, `Backend::VULKAN` sudah ada
slot-nya) — device ini punya GPU Mali-G57 yang belum dimanfaatkan sama sekali.

## 20. Vulkan GPU Compute — Alat Diagnostik (Belum Diintegrasi Penuh)

### Konteks

Setelah CPU multi-threading terbukti tidak membantu (Bagian 19), opsi berikutnya untuk MediaTek
adalah GPU compute lewat Vulkan (device punya GPU Mali-G57, `Backend::VULKAN` sudah ada slot di
`engine.h`).

### Verifikasi Shader (Sebelum Kirim ke User)

Berbeda dari kode sebelumnya yang hanya bisa diverifikasi di device Android, kali ini shader
Vulkan **divalidasi lebih dulu** memakai **Mesa `llvmpipe`** (implementasi Vulkan software, jalan
di CPU) yang diinstall di sandbox development:

1. `core/shaders/matmul_int8.comp` (GLSL, dikompilasi `glslangValidator` -> SPIR-V): satu
   invocation compute per baris output, baca bobot INT8 lewat `bitfieldExtract` dari buffer kata
   32-bit (menghindari kebutuhan ekstensi `GL_EXT_shader_8bit_storage` untuk kompatibilitas driver
   mobile lebih luas). SPIR-V hasil kompilasi di-embed sebagai array `uint32_t`
   (`core/include/matmul_int8_spv.h`) supaya tidak perlu plumbing `AssetManager` JNI tambahan.
2. Harness standalone menjalankan shader ini via `llvmpipe` dan membandingkan output terhadap
   referensi CPU murni — **`max_diff = 0`, bit-exact**. Korektnes matematika shader terverifikasi
   sebelum satu baris pun masuk ke `core/tests/`.

### Alat Diagnostik: "Benchmark Vulkan vs CPU"

Mengikuti pola `thread_bench` (Bagian 16-18): sebelum melakukan integrasi penuh (upload bobot ke
GPU sekali saat load model, reroute semua panggilan `matmul_int8` produksi lewat Vulkan), dibuat
alat ukur `core/src/vulkan_bench.cpp` yang mengukur **overhead round-trip nyata per panggilan**
(upload aktivasi kecil -> dispatch -> wait -> readback), dengan context Vulkan & buffer bobot yang
sudah disiapkan sekali di awal (mensimulasikan kondisi produksi).

Ini krusial karena `matmul_int8` dipanggil ~150x per token — kalau overhead submit/wait per
dispatch GPU lebih mahal dari compute-nya sendiri (skenario yang sangat mungkin di GPU mobile
untuk beban kerja sekecil ini), integrasi penuh akan gagal persis seperti CPU threading, tapi
dengan usaha implementasi yang jauh lebih besar. **Diukur dulu sebelum berkomitmen ke integrasi
besar.**

Diekspos ke Android via pola yang sama seperti `nativeBenchmarkThreads`: `jni_bridge.cpp`
(`nativeBenchmarkVulkan`) -> `FastModelNative.kt`/`FastModel.kt` -> tombol baru **"Benchmark
Vulkan vs CPU"** di `MainActivity.kt`, dijalankan di `Dispatchers.IO` (setup Vulkan + 30 round-trip
GPU tidak instan, risiko ANR sama seperti tombol benchmark thread sebelumnya).

### CI

Ditambahkan step baru: install `mesa-vulkan-drivers` di runner, compile & jalankan
`test_vulkan_bench.cpp` terhadap `llvmpipe` — **cuma menguji korektnes shader**, bukan klaim
kecepatan (itu sepenuhnya bergantung GPU nyata yang tidak ada di runner CI). Runner tanpa Vulkan
device sama sekali dianggap skip, bukan gagal.

### Belum Diverifikasi

**Angka kecepatan nyata di Mali-G57 belum ada** — `llvmpipe` cuma untuk verifikasi korektnes
logika, BUKAN indikasi performa (llvmpipe adalah Vulkan yang dijalankan software di CPU, jelas
tidak representatif GPU asli). Langkah selanjutnya: build APK, install, tap **"Benchmark Vulkan vs
CPU"**, kirim hasilnya. Kalau Vulkan menang jelas di Mali-G57 nyata, baru integrasi penuh ke
`transformer_mixed.cpp` (pre-upload bobot saat load model) dikerjakan. Kalau kalah — sama seperti
threading — dibatalkan, kode diagnostik tetap disimpan sebagai referensi.
