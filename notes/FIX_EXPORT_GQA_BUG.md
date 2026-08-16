# Fix: `export.py` Gagal untuk Model GQA (TinyLlama)

## Akar Masalah
`permute_reverse()` di `llama2.c/export.py` dipakai untuk memutar ulang layout RoPE dari format
HuggingFace ke format llama2.c — dipanggil untuk `wq` (query, 32 head penuh) **dan** `wk` (key).
Tapi TinyLlama pakai GQA: `wk` cuma py 4 head (bukan 32), jadi bentuknya `[256, 2048]`
(4 head × 64 dim head × 2048), bukan `[2048, 2048]` seperti `wq`. Kalau fungsi `permute_reverse`
dipanggil untuk `wk` dengan asumsi 32 head (default yang sama dipakai untuk `wq`), terjadi
mismatch — persis error yang Anda dapat.

## Langkah 1: Cell Diagnostik (jalankan di Colab, kirim hasilnya ke Claude)
```python
!sed -n '440,490p' /content/llama2.c/export.py
```
Kirim output cell ini — supaya patch yang saya kasih presisi sesuai baris kode aslinya, bukan
tebakan.

## Langkah 2: Coba Patch Ini Dulu (kemungkinan besar sudah benar)
Jalankan cell ini **setelah** `git clone` tapi **sebelum** `python export.py` — cell ini
menambal fungsi `permute_reverse` supaya sadar GQA (pakai `n_kv_heads` yang benar untuk `wk`,
bukan `n_heads` penuh):

```python
import re

path = '/content/llama2.c/export.py'
with open(path) as f:
    src = f.read()

# Cari baris pemanggilan wk yang salah (pakai permute_reverse tanpa parameter GQA)
old_line = "layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']))"

if old_line in src:
    # Tambal: panggil dengan n_kv_heads dan kv_dim eksplisit, bukan default n_heads/dim penuh.
    new_line = (
        "n_kv_heads_local = getattr(config, 'num_key_value_heads', n_heads)\n"
        "    kv_dim_local = n_kv_heads_local * (dim // n_heads)\n"
        "    layer.attention.wk.weight = nn.Parameter(permute_reverse(\n"
        "        hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'], n_kv_heads_local, kv_dim_local, dim))"
    )
    src = src.replace(old_line, new_line)
    with open(path, 'w') as f:
        f.write(src)
    print("Patch berhasil diterapkan.")
else:
    print("Baris yang dicari tidak ditemukan persis -- kemungkinan versi export.py Anda beda.")
    print("Kirim hasil cell diagnostik (Langkah 1) di atas ke Claude untuk patch yang presisi.")
```

## Langkah 3: Kalau Patch di Atas Tidak Match
Kemungkinan struktur kode di versi `export.py` yang ter-clone berbeda dari yang saya ingat
(repo publik bisa berubah dari waktu ke waktu). **Ini wajar** — kirim output Langkah 1 (cell
diagnostik), saya kasih patch yang cocok persis dengan baris kode Anda, sama seperti pola kerja
kita dengan log CI GitHub Actions selama ini.

## Setelah Patch Berhasil
Jalankan ulang Cell 3 (Download & Ekspor TinyLlama) dari notebook — tidak perlu ulang dari Cell 1,
Drive sudah ter-mount dan project sudah terekstrak, cukup lanjut dari situ.
