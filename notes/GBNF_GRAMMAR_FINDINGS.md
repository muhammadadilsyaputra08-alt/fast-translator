# Grammar-Guided Sampling (GBNF): Implementasi & Integrasi Produksi

## Ringkasan: Hasil POSITIF, tersambung penuh ke produksi ✅

Parser GBNF + constrained decoding dibangun dari nol (bukan library eksternal), diuji dengan
41 test korektnes murni + 4 test end-to-end via engine produksi. Terbukti **memaksa model kecil
(stories15M, tidak pernah dilatih JSON) menghasilkan sintaks JSON valid**, dengan verifikasi
independen bukan cuma "lihat outputnya kelihatan bagus".

## Yang Dibangun

### 1. Parser GBNF (`core/src/gbnf_parser.cpp`)
Recursive-descent parser mendukung: literal string (`"..."`), character class (`[abc]`, `[a-z]`,
`[^"]`), alternasi (`|`), rule reference, grouping (`(...)`), dan quantifier (`?`, `*`, `+`) —
quantifier & grup di-desugar jadi rule sintetis saat parsing, supaya runtime engine cuma perlu
menangani 3 elemen dasar (char range, rule ref, end).

### 2. State Machine Runtime (`core/src/gbnf_runtime.cpp`)
Pushdown automaton dengan multiple parallel stack (menangani ambiguitas grammar) — pendekatan
klasik yang sama dipakai llama.cpp untuk GBNF sampling. `epsilon_close()` **di-memoisasi** (cache
per-signature) — tanpa ini, kecepatan generate anjlok drastis (lihat catatan performa di bawah).

### 3. Integrasi ke `engine.cpp`
`generate_quantized()` sekarang: kalau `opts.grammar_gbnf` tidak kosong, setiap langkah sampling
melakukan **mask seluruh vocab** (32000 token) — tiap token kandidat di-decode, karakternya
disimulasikan lewat grammar state; token yang tidak valid di-set logit `-inf` sebelum argmax/
temperature sampling. Setelah token dipilih, grammar state maju sesuai karakter yang benar-benar
di-emit.

## Hasil Uji

### Korektnes murni (41/41 test, `test_gbnf.cpp`)
Literal, alternasi, char class (termasuk negasi `[^"]`), quantifier `?`/`*`/`+`, rule reference,
grouping — semua diuji dengan kasus positif & negatif. **Termasuk grammar JSON asli** yang dipakai
`Grammar.JSON` di Kotlin API (`Types.kt`) — parse sukses, match kasus valid, reject kasus invalid.

### End-to-end lewat `FastEngine` produksi (4/4 test, `test_grammar_sampling.cpp`)
- **Kontrol (tanpa grammar):** model TIDAK spontan menghasilkan sintaks JSON — membuktikan hasil di
  bawah ini bukan kebetulan.
- **Dengan grammar JSON dipaksa:** output diawali `{"..."` — grammar memaksa struktur JSON sejak
  karakter pertama, meski `stories15M` tidak pernah dilatih data JSON.
- **Verifikasi independen** (grammar instance baru, terpisah dari yang dipakai engine saat generate):
  me-replay seluruh output karakter-per-karakter — **setiap karakter yang pernah di-emit terbukti
  valid sebagai prefix JSON**, di generate 15 token maupun 55 token.
- Objek JSON belum tentu "tertutup" dalam token budget pendek (grammar `[^"]*` untuk isi string
  memang mengizinkan apa saja kecuali kutip — jadi string panjang belum tentu langsung ditutup) —
  ini bukan kegagalan grammar, cuma belum ada `"` yang ter-sample untuk menutup.

## Catatan Performa (jujur)

Constrained sampling **~10× lebih lambat** dari sampling bebas (butuh scan seluruh 32000 vocab
tiap langkah, decode + simulasi grammar per kandidat token):

| Optimisasi | ms/token | vs baseline |
|---|---|---|
| Implementasi awal (tanpa cache) | 596 ms | ~30× lebih lambat |
| + dedup hasil (ternyata tidak cukup, overhead string compare) | 1754 ms | lebih lambat lagi! |
| + memoisasi `epsilon_close` (fix yang benar) | 189 ms | ~10× lebih lambat dari tanpa grammar |

Pelajaran: dedup di akhir tidak mencegah komputasi berulang terjadi — harus dicegah di sumbernya
(memoisasi input, bukan filter output). `MainActivity.kt` diset default `maxTokens=25` untuk tombol
grammar (vs 60 untuk tombol bebas) supaya tetap responsif di device.

## Integrasi Produksi
- `core/include/gbnf.h`, `core/src/gbnf_parser.cpp`, `core/src/gbnf_runtime.cpp` — modul baru.
- `core/src/engine.cpp`: `generate_quantized()` otomatis pakai grammar constraint kalau
  `opts.grammar_gbnf` diisi. Tidak ada perubahan API publik Kotlin — `Grammar.JSON`/`Grammar.Custom`
  yang sudah ada di `Types.kt` sekarang **benar-benar berfungsi**, bukan cuma API kosong.
- `android/app/src/main/cpp/CMakeLists.txt`: tambah 2 file gbnf.
- `MainActivity.kt`: tombol baru "Generate (JSON grammar)" untuk demo langsung di device.
- CI: `test_gbnf.cpp` (murni, tanpa model) + `test_grammar_sampling.cpp` (end-to-end lewat model
  produksi) otomatis jalan tiap push.

## Regresi
38+ test lama tetap lulus (`test_fase1.cpp` 24, `test_engine.cpp` 8) + JNI smoke test 5/5.

## File
```
core/include/gbnf.h
core/src/gbnf_parser.cpp
core/src/gbnf_runtime.cpp
core/tests/test_gbnf.cpp                — 41 test korektnes murni
core/tests/test_grammar_sampling.cpp    — 4 test end-to-end + verifikasi independen
```
