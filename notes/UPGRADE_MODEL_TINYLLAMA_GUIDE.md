# Upgrade Model: Panduan Ekspor TinyLlama-1.1B-Chat

## Status Kesiapan Engine
✅ **GQA (Grouped Query Attention) sudah didukung** di `transformer.cpp` (referensi fp32) dan
`transformer_mixed.cpp` (produksi INT8) — diperlukan karena TinyLlama pakai GQA, beda dari
`stories15M` yang MHA biasa. **Regresi nol**: `stories15M` diuji ulang penuh (42 test), hasil
identik byte-per-byte dengan sebelum perubahan ini.

✅ `pack_fllm.cpp` (packer produksi) sudah diperbarui untuk membaca ukuran `wk`/`wv` yang benar
di bawah GQA (`kv_dim = n_kv_heads × head_size`, lebih sempit dari `dim` biasa).

## Yang Perlu Anda Lakukan (di komputer Anda, bukan HP)

### 1. Install dependency Python (sekali saja)
```bash
pip install torch transformers sentencepiece --break-system-packages
```

### 2. Download & ekspor TinyLlama ke format llama2.c
```bash
git clone https://github.com/karpathy/llama2.c.git
cd llama2.c
python export.py tinyllama1.1b.bin --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
Proses ini akan otomatis download model dari HuggingFace (~2.2GB) dan mengekspornya ke format
biner yang **sama persis** dengan `stories15M.bin` yang sudah kita pakai — tidak perlu ubah kode
parser sama sekali untuk bagian format file, cuma untuk arsitektur GQA (sudah selesai di atas).

### 3. Dapatkan tokenizer yang sesuai
TinyLlama pakai tokenizer SentencePiece bawaan dari checkpoint HuggingFace-nya, **bukan**
`tokenizer.bin` dari `stories15M` (vocab beda total). Repo llama2.c punya script terpisah:
```bash
python tokenizer.py --tokenizer-model=<path ke tokenizer.model dari HF cache>
```
Atau, kalau tersedia, ekspor otomatis dari `export.py` biasanya juga menghasilkan `tokenizer.bin`
yang sesuai — cek folder output setelah langkah 2.

### 4. Upload ke chat ini
Upload kedua file (`tinyllama1.1b.bin` + `tokenizer.bin` yang sesuai) ke percakapan ini seperti
sebelumnya. **Catatan ukuran:** file `.bin` ini akan jauh lebih besar dari `stories15M.bin`
(~2.2GB fp32 vs 57MB) — cek dulu apakah upload chat mendukung ukuran segini. Kalau terlalu besar,
kabari saya, kita cari cara lain (mis. Anda quantize dulu ke fp16 sebelum upload, potong jadi 2×
lebih kecil).

## Yang Saya Kerjakan Setelah Anda Upload
1. Verifikasi config checkpoint (dim, n_layers, n_heads, **n_kv_heads**, vocab_size) cocok dengan
   ekspektasi TinyLlama (dim=2048, n_layers=22, n_heads=32, n_kv_heads=4, vocab_size=32000).
2. Test forward pass fp32 dulu (`transformer.cpp`) — pastikan model menghasilkan teks koheren
   sebelum masuk kuantisasi.
3. Ukur perplexity baseline fp32 (`eval_perplexity.cpp`) sebagai acuan.
4. Pack ke `.fllm` produksi (INT8) — cek ulang apakah kualitas tetap terjaga seperti `stories15M`
   (yang dulu hasilnya nyaris lossless) atau perlu penyesuaian, karena model 1.1B jauh lebih besar
   dan mungkin berbeda karakteristiknya dari yang 15M.
5. Test prompt instruksi/terjemahan nyata (mis. "Translate to English: Selamat pagi") — ini
   pembuktian paling penting bahwa upgrade ini benar-benar berguna, bukan cuma model lebih besar.

## Ekspektasi Realistis
- **Ukuran file `.fllm` hasil packing**: jauh lebih besar dari `stories15M` (41MB) — kemungkinan
  ratusan MB hingga >1GB tergantung berapa besar tabel embedding TinyLlama (vocab 32000 × dim 2048).
- **Kecepatan generate**: akan jauh lebih lambat dari 86.63 tok/s yang kita capai untuk model 15M —
  model 1.1B adalah ~73× lebih banyak parameter, realistis mungkin turun ke single-digit tok/s di
  CPU tanpa NPU. Ini trade-off yang perlu Anda terima demi kapabilitas instruksi/terjemahan nyata.
- **Waktu load model**: kemungkinan beberapa detik di device (bukan lagi < 1 detik seperti
  `stories15M`), karena ukuran file jauh lebih besar.
