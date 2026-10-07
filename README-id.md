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
3.  **Jendela Konteks / Konteks Kerja Aktif**: Satu penggeser berlabel ganda menetapkan kapasitas KV lokal terkonfigurasi untuk chat, prompt dokumen, fitur tertanam, atau satu langkah inferensi agen aktif. Riwayat proyek jangka panjang tidak termasuk dalam estimasi KV GPU ini.
4.  **KV Cache**: Memori yang diperlukan untuk menyimpan status Key-Value bagi satu konteks aktif. Presisinya dapat diatur terpisah dari bobot model.
5.  **Overhead Sistem**:
    -   **GPU Diskrit (NVIDIA / AMD / Intel ARC)**: Deduksi yang dapat dikonfigurasi (default 1.5 GB) untuk OS dan tampilan. Dapat disesuaikan melalui penggeser (0–16 GB). Server headless Linux: set ~0.5 GB. *Catatan Intel ARC:* VRAM GDDR6 terdedikasi (model sama seperti NVIDIA/AMD), butuh **Resizable BAR** aktif di BIOS/UEFI; gunakan IPEX-LLM (SYCL/Level Zero) atau backend Vulkan/SYCL llama.cpp untuk inferensi LLM. Dukungan software masih berkembang, lebih rumit daripada CUDA (Ollama via fork IPEX-LLM) — contoh Arc Pro B60 24GB default 0.8 GB workstation.
    -   **Apple Silicon (M/A-series)**: Driver Metal macOS mencadangkan ~25% Memori Terpadu secara default — `recommendedMaxWorkingSetSize` Metal ≈75% RAM (≈⅔ pada Mac ≤32 GB dengan macOS lama); dapat diubah secara sistem melalui `sudo sysctl iogpu.wired_limit_mb`, dengan mengorbankan cadangan OS.
    -   **NVIDIA RTX Spark (Windows on Arm)**: Memakai aturan anggaran GPU resmi NVIDIA untuk superchip N1X: **anggaran GPU = carveout khusus + memori shared**, dengan shared = (memori sisa setelah carveout − 16 GB), dibatasi 50–80% dari sisa tersebut. Sisanya adalah memori khusus CPU yang tidak pernah bisa dipakai alokasi GPU. Pilih **Carveout GPU Khusus** dari OEM bila Anda mengetahuinya (Task Manager → Dedicated GPU memory); default 0 GB adalah batas bawah konservatif. Penggeser mengatur **ruang desktop / driver** di dalam anggaran (default 1.5 GB) karena NVIDIA memperingatkan agar tidak mengalokasikan seluruh anggaran. Contoh (tanpa carveout): 24 GB → anggaran GPU 12 GB, 64 GB → 48 GB, 128 GB → 102.4 GB (hingga 112 GB dengan carveout ≥48 GB).
    -   **SoC Arm Lain (Snapdragon X/X2 · DGX Spark · Linux)**: Cadangan OS/firmware tetap yang dapat dikonfigurasi (default 3.0 GB). Snapdragon X/X2 di Windows: ~3 GB untuk jalur CPU/NPU. Windows membatasi offload GPU Adreno (llama.cpp OpenCL) hingga 50% RAM. DGX Spark / GB10 di DGX OS: ~12 GB tanpa desktop, atau ~16 GB bila desktop berjalan (OS melihat ≈119 GiB dari 128 GB; ≈116 GiB tersedia tanpa desktop dibanding ≈112 GiB dengan desktop). DGX Spark memakai keluarga chip yang sama dengan RTX Spark, tetapi Linux membuka hampir seluruh memorinya untuk CUDA, sehingga aturan anggaran Windows tidak berlaku. Lihat [pembaruan RTX Spark & SoC Arm 2026](ARM_SOC_RTX_SPARK_UPDATE_2026.md).

Alat ini menelusuri kurva arsitektur terinterpolasi untuk menemukan jumlah parameter tertinggi yang muat setelah memperhitungkan overhead dan KV cache. Byte bobot serta byte KV berbasis arsitektur dikonversi secara konsisten ke GiB biner secara internal; label memori UI tetap memakai konvensi “GB” yang lazim pada pemasaran GPU. Lihat [audit matematika konteks dan memori 2026](CONTEXT_MATH_AUDIT_2026.md).

## Mulai Cepat
1.  Unduh `LLMCalculator.html`.
2.  Buka di peramban modern apa pun (Chrome, Edge, Firefox, Safari).
3.  Atur **Memori GPU** (Ukuran VRAM) Anda menggunakan penggeser.
4.  Pilih **Tipe GPU** (Diskrit — NVIDIA / AMD / Intel ARC, Apple Silicon, NVIDIA RTX Spark, atau SoC Arm Lain). Untuk RTX Spark, pilih juga **Carveout GPU Khusus** dari OEM bila diketahui.
5.  Pilih **Presisi Model** (Kuantisasi) dan presisi **KV Cache**.
6.  Sesuaikan **Jendela Konteks / Konteks Kerja Aktif** (misalnya, 8K atau 32K token).
7.  Lihat estimasi **Parameter Maksimum** dan rincian memori.
8.  Gunakan **Preset Cepat** untuk mensimulasikan konfigurasi perangkat keras populer di seluruh spektrum ekonomi — dari kartu 4 GB untuk pelajar hingga GPU pusat data 192 GB dan workstation memori terpadu 256 GB untuk bisnis:
    - **Budget / Entry (4–12 GB)**: GTX 1650 4GB, Arc A380 6GB, RTX 4060 8GB, RTX 5050 Laptop 8GB, Arc B570 10GB, RTX 3060 12GB (Steam #1, nilai terbaik <$250 bekas), Arc B580 12GB ($249 budget baru terbaik)
    - **Mainstream / Enthusiast (16–32 GB)**: Arc A770 16GB (termurah 16GB), RTX 5070 Ti 16GB, RTX 4090 24GB, Arc Pro B60 24GB ($599 workstation value), RTX 5090 32GB
    - **Workstation / Server (48–192 GB)**: L40S 48GB, A100 80GB, RTX PRO 6000 96GB, H200 141GB, B200 192GB
    - **Apple Silicon (8–256 GB)**: MacBook Neo 8GB, M5 Max 128GB (MacBook Pro / Mac Studio), M5 Ultra 256GB (Mac Studio)
    - **SoC Arm (24–128 GB)**: RTX Spark 24GB (Surface Laptop Ultra dasar), RTX Spark 64GB, RTX Spark 128GB (N1X teratas), DGX Spark 128GB (NVFP4, DGX OS), Snapdragon X2 Elite Extreme 48GB

## Fitur Utama
-   **Dukungan Multi-bahasa**: Beralih antara Bahasa Inggris dan Indonesia.
-   **Kalkulasi Real-time**: Pembaruan instan saat Anda menyesuaikan penggeser dan menu drop-down.
-   **Logika Arsitektur Cerdas**: Mekanisme attention spesifik (MHA, GQA, MLA) dan aturan reservasi memori spesifik-GPU (Diskrit termasuk Intel ARC, memori terpadu Apple, anggaran carveout + shared NVIDIA RTX Spark, SoC Arm dengan cadangan tetap).
-   **Overhead yang Dapat Disesuaikan**: Atur deduksi memori sistem secara presisi untuk server Linux headless vs desktop Windows.
-   **Rincian Memori Mendetail**: Memvisualisasikan penggunaan untuk Overhead Sistem, KV Cache, dan Bobot Model.
-   **Satu Penggeser Konteks Serbaguna**: Penggeser logaritmik **Jendela Konteks / Konteks Kerja Aktif** mengestimasi satu konteks inferensi lokal aktif tanpa toggle mode yang redundan.
-   **Preset Perangkat Keras (25 preset, 4 GB–256 GB)**: Konfigurasi satu klik mencakup seluruh stack — Budget (GTX 1650, Arc A380/B570/B580, RTX 3060/4060), Mainstream (Arc A770, RTX 5070 Ti/4090/5090, Arc Pro B60), Workstation/Server (L40S, A100, RTX PRO 6000 96GB, H200, B200), Apple Silicon (MacBook Neo, M5 Max, M5 Ultra), dan SoC Arm (RTX Spark 24/64/128GB, DGX Spark, Snapdragon X2 Elite). Diverifikasi terhadap Steam HW Survey 2025–2026, Tom's Hardware, Intel ARK.
-   **Opsi Lanjutan**: Dukungan untuk berbagai format kuantisasi (hingga FP2, termasuk MXFP4 dan NVFP4) dan presisi KV cache terpisah.
-   **Estimasi Model & Atensi Otomatis Multi-Generasi**: Secara dinamis mengestimasi arsitektur model (Lapisan dan Ukuran Tersembunyi) serta mekanisme atensi (GQA/MQA termasuk kepala ringkas 64-d, atensi sliding-window/global hibrida, MFA, MLA, MLA+DSA, atensi linear/softmax hibrida, CSA/HCA, dan CLA-2) melalui sintesis perhitungan dari semua generasi keluarga LLM modern: Llama (Gen 1–4), Gemma (Gen 1–4, **Gemma 3 12B/27B, Gemma 4 26B-A4B / 31B**), Qwen (Gen 1–3, **Gen 3-Next, Gen 3.5 (0,8B–397B), Gen 3.6, Gen 3.8 (27B, Flash-Next 180B, Max 2,4T)**), MiniCPM (Gen 1–5), G9 (v1–v3), Ling (1.0–2.0 mini/flash/1T, Ring-linear 2.0, **2.5/2.6 1T, 3.0-flash**), Inkling (Small/276B, Base/975B), DeepSeek (V1–V4, V4.1-Flash, R1), Tencent Hunyuan / Hy (dense 0,5B–7B, A13B, Large/A52B, Hy3, **Hy4-preview 770B**), GLM (1–4, **4.5/4.5-Air, 4.6, 4.7/4.7-Flash, 5→5.3**), Kimi (V1–K3), **Xiaomi MiMo (7B, V2-Flash, V2.5, V2.5-Pro), StepFun (Step3-VL-10B, Step-3, Step-3.5/3.7 Flash), Muse Glimmer 30B, IBM Granite (Code, 3.0–3.3, 4.0, 4.1, SWASH)**, GPT-OSS, Cohere Command/Aya/North, Poolside Laguna, bobot terbuka Mistral, Arcee Trinity, dan MiniMax. Formula KV khusus keluarga memperhitungkan dimensi K/V asimetris dan SWA-128 MiMo, SWA-512 dan MFA Step, SWA-2048 Muse, serta lapisan GQA/MQA 64-d dan SWASH Granite. Generasi terbaru menambah tujuh bentuk KV: **Qwen3-Next / Qwen3.5** hanya menyimpan token pada lapisan Gated Attention (1 dari 4) karena lapisan Gated DeltaNet berstate konstan, dan **GLM-5.x** menggabungkan cache laten MLA dengan indexer DeepSeek Sparse Attention 128-d yang dibagi tiap empat lapisan. **Ling 2.5/2.6 dan Ling 3.0** melangkah lebih jauh: satu lapisan gated-MLA per grup 8 (atau 6) lapisan menanggung seluruh cache, sedangkan lapisan lightning-attention / Kimi Delta Attention berstate konstan. **Qwen3.8-Flash-Next** menambahkan indeks Qwen Sparse Attention (1 KV head × 128-d, kompresi 4:1) di atas lapisan atensi penuh 1-dari-4-nya, dan **DeepSeek-V4.1-Flash** adalah encoder-decoder kausal yang lapisan CSA2 Full/Reindex/Reuse-nya berbagi satu cache global ~890 byte per token pada FP4. Generasi yang sudah diperiksa namun sengaja tidak di-anchor (tanpa bobot terbuka atau geometri tidak diungkap) — Qwen 3.7, StepFun Step 5 Preview, Kimi K2.8 Preview, GLM-5.3-Flash/FlashX, Meta Muse Spark 1.2/1.3, MiniMax H3 (model video), DiffusionGemma — tercantum di panel data referensi aplikasi. **Gemma 3 dan Gemma 4** mendapat bentuk berselang-selingnya sendiri: lima lapisan sliding-window-1024 per lapisan global, dengan lapisan global Gemma 4 menyimpan satu vektor K=V terpadu per head pada 512-d. Lihat [pembaruan anchor generasi terbaru 2026](NEXTGEN_ANCHORS_UPDATE_2026.md).
-   **Kategori Ukuran Model Standar (Artificial Analysis)**: Mengelompokkan model ke dalam 4 tingkatan standar: **Sangat Kecil (<4B)**, **Kecil (4B–40B)**, **Sedang (40B–150B)**, dan **Besar (150B+)** berdasarkan taksonomi Artificial Analysis.
-   **Berkas HTML tunggal**: Tidak ada instalasi, tidak ada dependensi, bekerja sepenuhnya offline.
-   **Desain responsif**: Bekerja di desktop, tablet, dan perangkat seluler.

## Kasus Penggunaan
-   **Perencanaan Perangkat Keras**: menentukan GPU mana yang akan dibeli untuk menjalankan model tertentu.
-   **Pemilihan Model**: Memilih ukuran model dan kuantisasi yang tepat untuk perangkat keras Anda yang ada.
-   **Edukasi**: Memahami hubungan antara parameter model, panjang konteks, dan kebutuhan memori.

### Perencanaan konteks lokal dan KV cache
Penggeser konteks berlabel ganda adalah kapasitas token terkonfigurasi yang digunakan langsung oleh formula KV berbasis arsitektur kalkulator. Ini serupa dengan ukuran konteks prompt yang dikonfigurasi melalui [`llama.cpp --ctx-size`](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md). Penggeser ini mencakup satu chat lokal, prompt dokumen, fitur tertanam, atau langkah inferensi aktif dalam alur kerja agen lokal jangka panjang.

Riwayat proyek jangka panjang sebaiknya disimpan di luar konteks aktif dalam artefak, catatan terstruktur, dan sistem pengambilan informasi, dengan kompaksi untuk menyegarkan set kerja. Pendekatan ini mengikuti [panduan rekayasa konteks agen](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Jendela nominal yang lebih besar bukan jaminan kualitas: [riset context rot](https://research.trychroma.com/context-rot) menemukan bahwa performa model dapat menjadi kurang andal saat panjang input bertambah.

Berbagai harness sumber terbuka menegaskan batas antara konteks aktif dan keadaan persisten ini dengan cara berbeda: [OpenHands](https://docs.openhands.dev/sdk/guides/context-condenser) mengompaksi riwayat; [SWE-agent](https://swe-agent.com/latest/reference/history_processor_config/) menyaring observasi; [Aider](https://aider.chat/docs/config/options.html) meringkas chat dan membatasi peta repositori; [Cline](https://docs.cline.bot/features/auto-compact), [Roo Code](https://docs.roocode.com/features/intelligent-context-condensing), dan [Goose](https://block.github.io/goose/docs/guides/smart-context-management/) mengompaksi sesi panjang; [LangGraph](https://docs.langchain.com/oss/python/langgraph/add-memory) memisahkan pemangkasan/ringkasan dari keadaan yang disimpan sebagai checkpoint; sedangkan [Letta](https://docs.letta.com/concepts/memory-management/) memisahkan memori dalam konteks dari memori arsip dan recall. Tinjauan lebih luas atas OpenClaw, OpenCode, DeepSeek Harness, Hermes Agent, Prime Agent, Pi, Qwen Code, Gemini CLI, Agent Zero, Crush, dan smolagents menghasilkan kesimpulan kalkulator yang sama; lihat [audit harness 2026](AGENT_CONTEXT_HARNESS_AUDIT_2026.md). Kebijakan harness tersebut tidak menciptakan kapasitas KV GPU tambahan.

> Kalkulator ini mengestimasi satu model lokal dan satu konteks inferensi aktif. Layanan multi-pengguna, agen paralel, offload KV ke CPU, dan perilaku cache spesifik runtime memerlukan perencanaan kapasitas terpisah.

Kalkulator tetap menggunakan model arsitektur terinterpolasi: hasilnya adalah estimasi model bobot terbuka lokal yang masuk akal dan muat dalam anggaran memori serta konteks terpilih. Ini bukan validator kecocokan checkpoint yang presisi atau perencana kapasitas server inferensi.

## Privasi & Data
Semua perhitungan terjadi secara lokal di peramban Anda. Tidak ada data yang dikirim ke server mana pun. Alat ini sepenuhnya offline setelah dimuat.

## Lisensi
Lisensi MIT. Lihat LICENSE untuk detailnya.

## Kontribusi
Kontribusi, masalah, dan saran dipersilakan. Silakan buka issue untuk mendiskusikan ide atau mengirimkan PR.
