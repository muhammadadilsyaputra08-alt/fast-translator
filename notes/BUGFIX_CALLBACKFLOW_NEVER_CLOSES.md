# Bug Fix: `callbackFlow` Tidak Pernah Selesai Secara Alami

## Ditemukan dari: screenshot device nyata (bukan sandbox/CI)

User menjalankan tombol "Benchmark" — teks output selesai penuh (~80 token, cerita utuh), tapi
status text tetap "Benchmarking..." selamanya. Ini mengindikasikan `collect{}` di Kotlin tidak
pernah return, padahal semua token sudah diterima.

## Akar Masalah

`FastModel.generate()` pakai `callbackFlow { ... }`. Sebelumnya:
```kotlin
val callback = TokenCallback { token -> trySend(token) }
val generationHandle = FastModelNative.nativeGenerate(...)
awaitClose { FastModelNative.nativeCancelGeneration(...) }
```

`awaitClose` cuma jalan saat **collector** membatalkan (mis. scope dibatalkan) — bukan saat
generation **native selesai dengan sendirinya** (mencapai `max_tokens` atau EOS). JNI bridge
sebelumnya memanggil `onToken()` per token lalu thread native selesai begitu saja, **tidak pernah
memberi sinyal ke Kotlin bahwa sudah selesai**. Akibatnya `Flow`-nya tetap "terbuka" menunggu token
berikutnya yang tidak akan pernah datang — `collect{}` hang selamanya, meski semua data sebenarnya
sudah lengkap diterima.

## Kenapa Tidak Ketahuan dari Testing Sebelumnya

- **Stub JNI test saya** (`core/tests/stub_jni/jni.h`) punya `CallVoidMethod` sebagai no-op — tidak
  pernah benar-benar mensimulasikan semantik `callbackFlow`/coroutine JVM asli. Test itu cuma
  memverifikasi bridge C++ tidak crash, bukan bahwa Kotlin Flow-nya benar secara perilaku.
- Tombol "Generate" biasa **juga kena bug ini** (status "Generation complete." mestinya muncul
  tapi tidak pernah kelihatan) — cuma tidak disadari sebelumnya karena teks output tetap muncul
  penuh, dan tidak ada yang secara khusus mengecek apakah status akhir benar-benar berubah.

## Perbaikan

1. **`FastModel.kt`**: `TokenCallback` diubah dari `fun interface` (1 method) jadi `interface`
   biasa dengan 2 method: `onToken(token)` dan `onComplete()`. Saat `onComplete()` dipanggil,
   Kotlin memanggil `close()` pada `callbackFlow` — inilah yang bikin `collect{}` akhirnya return.
2. **`jni_bridge.cpp`**: setelah `eng->generate(...)` benar-benar selesai (baris ini return artinya
   native generation loop sudah berhenti, entah karena max_tokens, EOS, atau dibatalkan), thread
   worker sekarang memanggil `callback.onComplete()` sebelum exit.

## Dampak
Bug ini mempengaruhi **semua** pemanggilan `generate()` — Generate biasa, Generate JSON grammar,
dan Benchmark — bukan cuma tombol Benchmark. Setelah fix, status text "Generation complete."/hasil
tok/s benchmark akan benar-benar muncul, dan resource coroutine tidak lagi menggantung tanpa batas
waktu menunggu Flow yang tidak pernah ditutup.

## Verifikasi
- JNI smoke test (`test_jni_smoke.cpp`) tetap 5/5 lulus setelah perubahan signature callback.
- **Dikonfirmasi lewat device fisik nyata**: tombol Benchmark sekarang menampilkan hasil final
  dengan benar — "Benchmark: 81 tokens in 6342 ms = 12,77 tok/s (CPU reference, no NPU)" — bukan
  macet di "Benchmarking..." lagi. Status "Generation complete." pada tombol Generate biasa juga
  ikut terkonfirmasi bekerja normal.

## Pelajaran untuk Sesi Berikutnya
Stub JNI (`core/tests/stub_jni/jni.h`) berguna untuk cek sintaks C++ bridge, tapi **tidak bisa
menangkap bug yang sifatnya perilaku Kotlin Flow/coroutine**. Kelas bug seperti ini cuma kelihatan
dari testing di device fisik/JVM asli — pertimbangkan ini kalau ada laporan "macet"/"hang" lain
dari device meski semua test sandbox lulus.
