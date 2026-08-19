[English](README.md) | **Bahasa Indonesia**

# LLMCalculator
*VRAM usage, simplified.*

## Pendahuluan
LLMCalculator adalah alat berbasis peramban (browser) berkas tunggal untuk memperkirakan ukuran Model Bahasa Besar (LLM) maksimum yang dapat dimuat dalam memori GPU Anda. Dirancang untuk penggemar AI, peneliti, dan perencana perangkat keras, alat ini membantu memvisualisasikan bagaimana kuantisasi, panjang konteks, dan overhead sistem memengaruhi kapasitas model.

Antarmuka mendukung **Bahasa Inggris** dan **Bahasa Indonesia**.

## Cara Kerja
Kalkulator memperkirakan penggunaan memori berdasarkan:

1.  **Ukuran VRAM**: Total memori GPU yang tersedia (misalnya, 24GB, 80GB).
2.  **Kuantisasi**: Presisi bobot model (FP32, FP16/BF16, FP8/INT8, MXFP4, NVFP4, INT4/FP4, FP2). MXFP4 (~0,53 byte/param) adalah format native GPT-OSS dan Kimi K3 serta tersedia untuk expert MiMo-V2.5-Pro; NVFP4 (~0,56 byte/param) adalah format FP4 Blackwell-native yang dikirim Poolside Laguna dan Step-3.7-Flash. Presisi yang lebih rendah mengurangi penggunaan memori tetapi dapat memengaruhi kualitas.
3.  **Mode Konteks**: Pilih **Normal** untuk chat lokal normal, prompt dokumen, atau fitur lokal tertanam, atau **Agentik** untuk satu agen lokal yang berjalan lama.
4.  **Jendela Konteks**: Dalam mode Normal, ini adalah kapasitas konteks/KV lokal terkonfigurasi. Dalam mode Agentik, penggeser yang sama berlabel **Konteks Kerja Aktif** dan mewakili konteks kerja agen yang terbatas. Kedua mode memakai nilai penggeser secara langsung dalam formula KV cache berbasis arsitektur yang sama.
5.  **KV Cache**: Memori yang diperlukan untuk menyimpan status Key-Value bagi satu konteks aktif. Presisinya dapat diatur terpisah dari bobot model.
6.  **Overhead Sistem**:
    -   **GPU Diskrit (NVIDIA / AMD / Intel ARC)**: Deduksi yang dapat dikonfigurasi (default 1.5 GB) untuk OS dan tampilan. Dapat disesuaikan melalui penggeser (0–16 GB). Server headless Linux: set ~0.5 GB. *Catatan Intel ARC:* VRAM GDDR6 terdedikasi (model sama seperti NVIDIA/AMD), butuh **Resizable BAR** aktif di BIOS/UEFI; gunakan IPEX-LLM (SYCL/Level Zero) atau backend Vulkan/SYCL llama.cpp untuk inferensi LLM. Dukungan software masih berkembang, lebih rumit daripada CUDA (Ollama via fork IPEX-LLM) — contoh Arc Pro B60 24GB default 0.8 GB workstation.
    -   **Apple Silicon (M/A-series)**: Driver Metal macOS mencadangkan ~25% Memori Terpadu secara default (dapat diubah secara sistem melalui `sysctl iogpu.wired_limit_mb`).
    -   **Snapdragon (Windows on Arm)**: Overhead OS + driver yang dapat dikonfigurasi (default 3.0 GB). Windows membatasi memori GPU bersama hingga 50% dari total RAM — penggeser overhead memungkinkan Anda menyesuaikan dengan penggunaan OS + latar belakang sistem Anda. Ini adalah batas dinamis, bukan cadangan tetap.

Alat ini secara berulang menghitung jumlah parameter maksimum (dalam Miliar) yang muat dalam sisa VRAM setelah memperhitungkan overhead dan KV cache.

## Mulai Cepat
1.  Unduh `LLMCalculator.html`.
2.  Buka di peramban modern apa pun (Chrome, Edge, Firefox, Safari).
3.  Atur **Memori GPU** (Ukuran VRAM) Anda menggunakan penggeser.
4.  Pilih **Tipe GPU** (Diskrit — NVIDIA / AMD / Intel ARC, Apple Silicon, atau Snapdragon).
5.  Pilih **Presisi Model** (Kuantisasi) dan presisi **KV Cache**.
6.  Pilih **Normal** atau **Agentik**, lalu sesuaikan **Jendela Konteks** / **Konteks Kerja Aktif** (misalnya, 8K, 32K token).
7.  Lihat estimasi **Parameter Maksimum** dan rincian memori.
8.  Gunakan **Preset Cepat** untuk mensimulasikan konfigurasi perangkat keras populer di seluruh spektrum ekonomi — dari kartu 4 GB untuk pelajar hingga GPU 192 GB untuk bisnis:
    - **Budget / Entry (4–12 GB)**: GTX 1650 4GB, Arc A380 6GB, RTX 4060 8GB, RTX 5050 Laptop 8GB, Arc B570 10GB, RTX 3060 12GB (Steam #1, nilai terbaik <$250 bekas), Arc B580 12GB ($249 budget baru terbaik)
    - **Mainstream / Enthusiast (16–32 GB)**: Arc A770 16GB (termurah 16GB), RTX 5070 Ti 16GB, RTX 4090 24GB, Arc Pro B60 24GB ($599 workstation value), RTX 5090 32GB
    - **Workstation / Server (48–192 GB)**: L40S 48GB, A100 80GB, RTX PRO 6000 96GB, H200 141GB, B200 192GB
    - **Apple / Mobile**: MacBook Neo 8GB, M4 Max 128GB (MacBook Pro), M3 Ultra 192GB (Mac Studio), Snapdragon X Elite 32GB

## Fitur Utama
-   **Dukungan Multi-bahasa**: Beralih antara Bahasa Inggris dan Indonesia.
-   **Kalkulasi Real-time**: Pembaruan instan saat Anda menyesuaikan penggeser dan menu drop-down.
-   **Logika Arsitektur Cerdas**: Mekanisme attention spesifik (MHA, GQA, MLA) dan aturan reservasi memori spesifik-GPU (Diskrit termasuk Intel ARC, Unified vs Shared).
-   **Overhead yang Dapat Disesuaikan**: Atur deduksi memori sistem secara presisi untuk server Linux headless vs desktop Windows.
-   **Rincian Memori Mendetail**: Memvisualisasikan penggunaan untuk Overhead Sistem, KV Cache, dan Bobot Model.
-   **Mode Konteks Normal / Agentik yang Sederhana**: Satu penggeser konteks logaritmik mengestimasi satu konteks inferensi lokal aktif. Mode Agentik hanya mengubah label dan panduan, bukan kalkulasi KV cache.
-   **Preset Perangkat Keras (21 preset, 4 GB–192 GB)**: Konfigurasi satu klik mencakup seluruh stack — Budget (GTX 1650, Arc A380/B570/B580, RTX 3060/4060), Mainstream (Arc A770, RTX 5070 Ti/4090/5090, Arc Pro B60), Workstation/Server (L40S, A100, RTX PRO 6000 96GB, H200, B200), dan Apple/Mobile (M4 Max, M3 Ultra, Snapdragon). Diverifikasi terhadap Steam HW Survey 2025–2026, Tom's Hardware, Intel ARK.
-   **Opsi Lanjutan**: Dukungan untuk berbagai format kuantisasi (hingga FP2, termasuk MXFP4 dan NVFP4) dan presisi KV cache terpisah.
-   **Estimasi Model & Atensi Otomatis Multi-Generasi**: Secara dinamis mengestimasi arsitektur model (Lapisan dan Ukuran Tersembunyi) serta mekanisme atensi (GQA/MQA termasuk kepala ringkas 64-d, atensi sliding-window/global hibrida, MFA, MLA, CSA/HCA, dan CLA-2) melalui sintesis perhitungan dari semua generasi keluarga LLM modern: Llama (Gen 1–4), Gemma (Gen 1–4), Qwen (Gen 1–3), MiniCPM (Gen 1–5), G9 (v1–v3), Ling (1.0–2.0), Inkling (Small/276B, Base/975B), DeepSeek (V1–V4, R1), Tencent Hunyuan / Hy (dense 0,5B–7B, A13B, Large/A52B, Hy3), GLM (1–4), Kimi (V1–K3), **Xiaomi MiMo (7B, V2-Flash, V2.5, V2.5-Pro), StepFun (Step3-VL-10B, Step-3, Step-3.5/3.7 Flash), Muse Glimmer 30B, IBM Granite (Code, 3.0–3.3, 4.0, 4.1, SWASH)**, GPT-OSS, Cohere Command/Aya/North, Poolside Laguna, bobot terbuka Mistral, Arcee Trinity, dan MiniMax. Formula KV khusus keluarga memperhitungkan dimensi K/V asimetris dan SWA-128 MiMo, SWA-512 dan MFA Step, SWA-2048 Muse, serta lapisan GQA/MQA 64-d dan SWASH Granite.
-   **Kategori Ukuran Model Standar (Artificial Analysis)**: Mengelompokkan model ke dalam 4 tingkatan standar: **Sangat Kecil (<4B)**, **Kecil (4B–40B)**, **Sedang (40B–150B)**, dan **Besar (150B+)** berdasarkan taksonomi Artificial Analysis.
-   **Berkas HTML tunggal**: Tidak ada instalasi, tidak ada dependensi, bekerja sepenuhnya offline.
-   **Desain responsif**: Bekerja di desktop, tablet, dan perangkat seluler.

## Kasus Penggunaan
-   **Perencanaan Perangkat Keras**: menentukan GPU mana yang akan dibeli untuk menjalankan model tertentu.
-   **Pemilihan Model**: Memilih ukuran model dan kuantisasi yang tepat untuk perangkat keras Anda yang ada.
-   **Edukasi**: Memahami hubungan antara parameter model, panjang konteks, dan kebutuhan memori.

### Perencanaan konteks lokal dan KV cache
Penggeser konteks adalah kapasitas token terkonfigurasi yang digunakan langsung oleh formula KV berbasis arsitektur kalkulator. Ini serupa dengan ukuran konteks prompt yang dikonfigurasi melalui [`llama.cpp --ctx-size`](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md). **Normal** menggambarkan satu chat lokal, prompt dokumen, atau fitur lokal tertanam. **Agentik** menggambarkan konteks kerja terbatas untuk satu agen lokal yang berjalan lama dan memakai kalkulasi yang sama persis.

Riwayat proyek jangka panjang sebaiknya disimpan di luar konteks aktif dalam artefak, catatan terstruktur, dan sistem retrieval, dengan kompaksi untuk menyegarkan set kerja. Pendekatan ini mengikuti [panduan rekayasa konteks agen](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Jendela nominal yang lebih besar bukan jaminan kualitas: [riset context rot](https://research.trychroma.com/context-rot) menemukan bahwa performa model dapat menjadi kurang andal saat panjang input bertambah.

> Kalkulator ini mengestimasi satu model lokal dan satu konteks inferensi aktif. Layanan multi-pengguna, agen paralel, offload KV ke CPU, dan perilaku cache spesifik runtime memerlukan perencanaan kapasitas terpisah.

Kalkulator tetap menggunakan model arsitektur terinterpolasi: hasilnya adalah estimasi model bobot terbuka lokal yang masuk akal dan muat dalam anggaran memori serta konteks terpilih. Ini bukan validator kecocokan checkpoint yang presisi atau perencana kapasitas server inferensi.

## Privasi & Data
Semua perhitungan terjadi secara lokal di peramban Anda. Tidak ada data yang dikirim ke server mana pun. Alat ini sepenuhnya offline setelah dimuat.

## Lisensi
Lisensi MIT. Lihat LICENSE untuk detailnya.

## Kontribusi
Kontribusi, masalah, dan saran dipersilakan. Silakan buka issue untuk mendiskusikan ide atau mengirimkan PR.
