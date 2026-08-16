# Upgrade Model via Google Colab: Panduan Lengkap (Trinity: Colab + Drive + HuggingFace)

## Kenapa Ini Bekerja
Repo GitHub Anda sudah berisi **seluruh kode C++** yang saya tulis (`core/`, `tools/`) — jadi Colab
tidak perlu file baru dari saya. Alurnya:

```
HuggingFace (download model)  →  Colab (export + compile C++ + pack INT8)  →  Google Drive (simpan
hasil kecil)  →  Download ke HP  →  adb push  →  Selesai, tanpa upload apapun ke chat ini.
```

File besar (checkpoint fp32 TinyLlama, ~4GB) **tidak pernah** menyentuh chat ini maupun disk lokal
Anda — semuanya di server Colab, dibuang begitu sesi berakhir. Yang keluar dari Colab cuma
`model.fllm` hasil kuantisasi INT8 (jauh lebih kecil).

## Langkah 1: Buka Colab, Buat Notebook Baru
1. Buka **colab.research.google.com**
2. New Notebook
3. Runtime → Change runtime type → pastikan pilih CPU biasa sudah cukup (tidak perlu GPU untuk
   export/quantize; GPU cuma mempercepat kalau load model pakai `torch.cuda`, tapi export ke
   format kita murni operasi CPU/disk)

## Langkah 2: Paste & Jalankan Cell Ini Satu-Per-Satu

### Cell 1 — Setup & clone repo Anda sendiri
```python
!pip install torch transformers sentencepiece --quiet
!git clone https://github.com/karpathy/llama2.c.git
!git clone https://github.com/muhammadadilsyaputra08-alt/fast-translator.git
```
*(Ganti URL repo kedua dengan URL repo GitHub Anda yang sebenarnya kalau beda.)*

### Cell 2 — Download & ekspor TinyLlama dari HuggingFace
```python
%cd /content/llama2.c
!python export.py /content/tinyllama.bin --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
Ini akan download langsung dari HuggingFace ke storage Colab (bukan ke komputer Anda), lalu
export ke format biner yang sama persis dengan `stories15M.bin`.

### Cell 3 — Cek tokenizer yang dihasilkan
```python
!ls -la /content/llama2.c/tokenizer.bin 2>/dev/null || echo "tokenizer.bin tidak otomatis dibuat, lihat langkah 3b"
```
Kalau tidak ada otomatis, jalankan:
```python
!python tokenizer.py --tokenizer-model=/root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*/tokenizer.model
```
*(Sesuaikan path wildcard sesuai output `!find /root/.cache/huggingface -name "tokenizer.model"` kalau perlu.)*

### Cell 4 — Cek ukuran & config hasil ekspor (verifikasi sebelum lanjut)
```python
import struct, os
print("ukuran file:", os.path.getsize("/content/tinyllama.bin") / 1e9, "GB")
with open("/content/tinyllama.bin", "rb") as f:
    dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq = struct.unpack("<7i", f.read(28))
    print(f"dim={dim} hidden={hidden} n_layers={n_layers} n_heads={n_heads} n_kv_heads={n_kv_heads} vocab={vocab} seq_len={seq}")
```
**Kirim screenshot output cell ini ke saya** sebelum lanjut — saya perlu konfirmasi angka-angka
ini cocok dengan yang kode C++ saya harapkan (terutama `n_kv_heads` harus < `n_heads` untuk
membuktikan GQA benar-benar terdeteksi).

### Cell 5 — Compile kode C++ dari repo Anda (persis kode yang sudah saya tulis & uji)
```python
%cd /content/fast-translator/tools
!g++ -std=c++17 -O2 -I ../core/include \
  ../core/src/fllm_parser.cpp ../core/src/bitnet_kernel.cpp \
  pack_fllm.cpp -o pack_fllm
!g++ -std=c++17 -O2 -I ../core/include \
  ../core/src/transformer.cpp ../core/src/tokenizer.cpp \
  ../core/tests/test_real_model.cpp -o test_real_model
```

### Cell 6 — Test forward pass fp32 dulu (SEBELUM kuantisasi)
```python
!./test_real_model /content/tinyllama.bin /content/llama2.c/tokenizer.bin
```
**Ini langkah paling penting** — kalau test ini gagal atau outputnya tidak masuk akal (bukan
kalimat bahasa Inggris koheren), JANGAN lanjut ke kuantisasi. Kirim screenshot/output ke saya dulu.

### Cell 7 — Pack ke `.fllm` produksi (INT8)
```python
!./pack_fllm /content/tinyllama.bin /content/llama2.c/tokenizer.bin /content/model.fllm
```

### Cell 8 — Verifikasi kualitas setelah kuantisasi (bandingkan dengan fp32)
```python
!g++ -std=c++17 -O2 -I ../core/include \
  ../core/src/transformer.cpp ../core/src/transformer_bitnet.cpp ../core/src/transformer_mixed.cpp \
  ../core/src/tokenizer.cpp ../core/src/fllm_parser.cpp ../core/src/kv_cache.cpp \
  eval_perplexity.cpp -o eval_perplexity
!./eval_perplexity fp32 /content/tinyllama.bin /content/llama2.c/tokenizer.bin
!./eval_perplexity int8all /content/model.fllm
```
Bandingkan angka perplexity kedua baris ini — kalau selisihnya kecil (seperti `stories15M`: 27.11
vs 27.01), kuantisasi berhasil nyaris lossless. Kalau selisihnya besar, kabari saya.

### Cell 9 — Simpan hasil ke Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/model.fllm /content/drive/MyDrive/model.fllm
print("Selesai! Cek Google Drive Anda, ada file model.fllm")
```

## Langkah 3: Dari Google Drive ke HP
1. Buka app **Google Drive** di HP Anda, cari `model.fllm`, tap **Download** (atau titik tiga →
   Download) — tersimpan ke folder Download HP.
2. Sambungkan HP ke PC via USB, jalankan:
   ```cmd
   adb shell mkdir -p /sdcard/Android/data/com.fasttranslator/files/models
   adb push model.fllm /sdcard/Android/data/com.fasttranslator/files/models/model.fllm
   ```
   *(File sumbernya ambil dari folder Download HP yang sudah ke-mount, atau langsung `adb pull`
   duluan dari Drive lewat browser PC lalu `adb push` dari PC — mana yang lebih mudah buat Anda.)*

## Kalau Ada yang Gagal di Tengah Jalan
Kirim **screenshot cell yang error** (bukan cuma pesan errornya, tapi juga nomor cell & kode di
atasnya) — saya bisa diagnosis dan kasih perbaikan persis seperti pola kerja kita selama ini
dengan log CI GitHub Actions.

## Estimasi Waktu & Biaya
- Semua langkah di atas: gratis (Colab free tier cukup, tidak perlu GPU/Colab Pro).
- Download model dari HuggingFace: beberapa menit tergantung koneksi server Colab (biasanya cepat).
- Total waktu keseluruhan: perkiraan 15-30 menit kalau tidak ada hambatan.
