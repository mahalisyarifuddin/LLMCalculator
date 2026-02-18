[English](README.md) | **Bahasa Indonesia**

# LLMCalculator
*VRAM usage, simplified.*

## Pendahuluan
LLMCalculator adalah alat berbasis peramban (browser) berkas tunggal untuk memperkirakan ukuran Model Bahasa Besar (LLM) maksimum yang dapat dimuat dalam memori GPU Anda. Dirancang untuk penggemar AI, peneliti, dan perencana perangkat keras, alat ini membantu memvisualisasikan bagaimana kuantisasi, panjang konteks, dan overhead sistem memengaruhi kapasitas model.

Antarmuka mendukung **Bahasa Inggris** dan **Bahasa Indonesia**.

## Cara Kerja
Kalkulator memperkirakan penggunaan memori berdasarkan:

1.  **Ukuran VRAM**: Total memori GPU yang tersedia (misalnya, 24GB, 80GB).
2.  **Kuantisasi**: Presisi bobot model (FP32, FP16/BF16, FP8/INT8, INT4/FP4, FP2). Presisi yang lebih rendah mengurangi penggunaan memori tetapi dapat memengaruhi kualitas.
3.  **Jendela Konteks**: Jumlah maksimum token yang diproses model. Konteks yang lebih besar membutuhkan lebih banyak memori KV cache.
4.  **KV Cache**: Memori yang diperlukan untuk menyimpan status Key-Value untuk jendela konteks. Ini juga mendukung kuantisasi terpisah untuk KV cache.
5.  **Overhead Sistem**: Buffer cadangan tetap (1.5 GB) untuk OS, tampilan, dan overhead framework.

Alat ini secara berulang menghitung jumlah parameter maksimum (dalam Miliar) yang muat dalam sisa VRAM setelah memperhitungkan overhead dan KV cache.

## Mulai Cepat
1.  Unduh `LLMCalculator.html`.
2.  Buka di peramban modern apa pun (Chrome, Edge, Firefox, Safari).
3.  Atur **Memori GPU** (Ukuran VRAM) Anda menggunakan penggeser.
4.  Pilih **Tipe GPU** (Diskrit atau Apple Silicon).
5.  Pilih **Presisi Model** (Kuantisasi) dan presisi **KV Cache**.
6.  Sesuaikan **Jendela Konteks** (misalnya, 8K, 32K token).
7.  Lihat estimasi **Parameter Maksimum** dan rincian memori.
8.  Gunakan **Preset Cepat** untuk mensimulasikan konfigurasi perangkat keras populer seperti RTX 4090, A100, H200, atau M4 Max.

## Fitur Utama
-   **Dukungan Multi-bahasa**: Beralih antara Bahasa Inggris dan Indonesia.
-   **Kalkulasi Real-time**: Pembaruan instan saat Anda menyesuaikan penggeser dan menu drop-down.
-   **Rincian Memori Mendetail**: Memvisualisasikan penggunaan untuk Overhead Sistem, KV Cache, dan Bobot Model.
-   **Preset Perangkat Keras**: Konfigurasi satu klik untuk GPU umum.
-   **Opsi Lanjutan**: Dukungan untuk berbagai format kuantisasi (hingga FP2) dan presisi KV cache terpisah.
-   **Estimasi Model**: Secara dinamis mengestimasi arsitektur model (Lapisan dan Ukuran Tersembunyi) berdasarkan konfigurasi LLM modern (misalnya Llama 3, Qwen 2.5) untuk jumlah parameter yang dihitung.
-   **Berkas HTML tunggal**: Tidak ada instalasi, tidak ada dependensi, bekerja sepenuhnya offline.
-   **Desain responsif**: Bekerja di desktop, tablet, dan perangkat seluler.

## Kasus Penggunaan
-   **Perencanaan Perangkat Keras**: menentukan GPU mana yang akan dibeli untuk menjalankan model tertentu.
-   **Pemilihan Model**: Memilih ukuran model dan kuantisasi yang tepat untuk perangkat keras Anda yang ada.
-   **Edukasi**: Memahami hubungan antara parameter model, panjang konteks, dan kebutuhan memori.

## Privasi & Data
Semua perhitungan terjadi secara lokal di peramban Anda. Tidak ada data yang dikirim ke server mana pun. Alat ini sepenuhnya offline setelah dimuat.

## Lisensi
Lisensi MIT. Lihat LICENSE untuk detailnya.

## Kontribusi
Kontribusi, masalah, dan saran dipersilakan. Silakan buka issue untuk mendiskusikan ide atau mengirimkan PR.
