# Ringkasan Audit LLMCalculator

Dokumen ini menggabungkan dan merangkum 15 laporan audit dan pembaruan untuk `LLMCalculator.html` yang ditulis antara Agustus dan Oktober 2026. Isinya mencatat asumsi kalkulator, apa yang sudah diverifikasi atau dikoreksi, beserta alasannya. English: [AUDIT.md](AUDIT.md).

Laporan asli lengkap, dengan semua tabel dan sumbernya (sekitar 220 tautan), tetap tersimpan di riwayat git. `git log --oneline -- <FILE>` menampilkan commit yang menyentuh sebuah laporan, dan `git show <commit>:<FILE>` menampilkan isinya. Commit terakhir yang memuat ke-15 laporan adalah `228ba94`. Daftarnya ada di [lampiran](#lampiran-laporan-asli).

## Daftar isi

1. [Cakupan dan invarian](#1-cakupan-dan-invarian)
2. [Matematika memori](#2-matematika-memori)
3. [Overhead sistem per tipe GPU](#3-overhead-sistem-per-tipe-gpu)
4. [Anchor arsitektur estimasi otomatis](#4-anchor-arsitektur-estimasi-otomatis)
5. [Jendela konteks dan harness agen](#5-jendela-konteks-dan-harness-agen)
6. [Preset hardware](#6-preset-hardware)
7. [Praktik verifikasi](#7-praktik-verifikasi)
8. [Sumber utama](#8-sumber-utama)
- [Lampiran: laporan asli](#lampiran-laporan-asli)

---

## 1. Cakupan dan invarian

Kalkulator ini mengestimasi **model open-weight terbesar yang masuk akal** dan muat di memori satu perangkat, bersama **satu konteks KV cache aktif**. Bentuk model (jumlah layer, lebar hidden, tipe atensi) diinterpolasi dari checkpoint nyata (§4). Hasilnya estimasi, bukan validator checkpoint yang presisi ataupun perencana kapasitas server inferensi.

**Tidak dimodelkan:**
- Offload ke CPU, sekuens paralel, layanan multi-pengguna, dan subagen yang berjalan bersamaan.
- Ruang kerja runtime di luar jatah overhead yang dapat diatur.
- Batas konteks pelatihan dan kualitas konteks panjang.

**Invarian yang dijaga setiap audit:**
- **100% offline, satu file:** tanpa state di URL atau hash, tanpa storage, tanpa panggilan jaringan, tanpa permalink (arahan di header file). State hanya ada di memori.
- **Tipe GPU hanya ditambah bila matematika memorinya berbeda.** Intel ARC digabung ke "Discrete GPU" setelah masukan pengguna, dan AMD Ryzen AI Max masuk ke tipe cadangan-tetap yang sudah ada.
- **UI dwibahasa** (Inggris dan Indonesia) dengan kunci teks yang identik.
- **Satuan:** UI menulis "GB", tetapi perhitungannya selalu memakai GiB biner.

---

## 2. Matematika memori

### 2.1 Bobot model dan kuantisasi

```text
GiB bobot = parameter (miliar) × 10^9 × byte per parameter × 1,05 ÷ 2^30
```

- **Faktor 1,05** mencakup metadata, skala kuantisasi, dan alignment. Panduan publik menyebut 1,1–1,2×, jadi 1,05 sedikit optimistis tetapi masih dapat dipertanggungjawabkan. Cek cepat: 7B pada Q4_K_M ≈ 4,4 GB, dibanding file llama.cpp nyata 4,4–5,1 GB.
- **Perbaikan satuan (Agustus 2026):** bobot sebelumnya memakai GB desimal sementara KV memakai GiB biner. Kini keduanya biner, sehingga satu miliar byte = 0,931 GiB.

| Format | Byte/parameter | Catatan |
| --- | --- | --- |
| FP32 | 4,0 | |
| FP16 / BF16 | 2,0 | Default preset pusat data |
| FP8 / INT8 | 1,0 | |
| GGUF Q8_0 | 1,06 | |
| GGUF Q6_K | 0,83 | |
| GGUF Q5_K_M | 0,69 | |
| GGUF Q4_K_M / GPTQ 4-bit | 0,60 | Ujung atas rentang 0,56–0,60, jujur terhadap ukuran file nyata; default preset konsumen |
| NVFP4 | 0,56 | FP4 + skala FP8 per 16 nilai + skala tensor ≈ 4,5 bit; native Blackwell (Poolside Laguna, Step-3.7-Flash) |
| MXFP4 | 0,53 | FP4 + skala 8-bit per 32 nilai ≈ 4,25 bit; native gpt-oss (117B → 62 GB vs resmi 60,8 GB), Kimi K3, dan expert MiMo-V2.5-Pro |
| INT4 / FP4 | 0,5 | Ideal; file GPTQ/AWQ nyata 0,55–0,60 |
| GGUF Q3_K_M | 0,46 | |
| GGUF Q2_K | 0,41 | Ujung atas (realistisnya ≈ 0,31) |
| FP2 | 0,25 | Ideal |

KV cache punya pengaturan presisi sendiri: sama dengan model, FP16, FP8, INT8, atau INT4.

### 2.2 KV cache

Atensi standar (MHA, GQA, MQA), untuk satu sekuens:

```text
byte KV = 2 (K dan V) × layer × token konteks × head KV × dimensi head × byte per elemen
```

Cek cepat pada 8.192 token FP16: kelas Llama 8B = 1,000 GiB, kelas Llama 70B = 2,500 GiB, model MHA 64 layer selebar 12.288 (kelas Command R+) = 24,000 GiB.

Arsitektur baru menyimpan jauh lebih sedikit, jadi anchor tiap keluarga membawa bentuk cache yang persis:

| Bentuk cache | Keluarga | Yang disimpan per token |
| --- | --- | --- |
| MLA | DeepSeek V2/V3/R1, Kimi K2/K3, Mistral Large 3, GLM-4.7-Flash | 512 laten + 64 RoPE = 576 elemen per layer (Mistral Small 4: 256 + 64 = 320) |
| MLA + DSA | GLM-5.x, Hy4-preview | Cache MLA di setiap layer + indexer sparse 128-d di setiap layer ke-4 |
| CSA / HCA | DeepSeek V4-Flash / V4-Pro | K=V bersama 512-d; CSA menyimpan SWA-128 + konteks/4 + indexer 128-d, HCA menyimpan SWA-128 + konteks/128 |
| CED + CSA2 | DeepSeek V4.1-Flash | 44,5 elemen per layer, dikalibrasi ke 890 byte/token pada FP4 menurut kartu modelnya |
| Hybrid sliding window | MiMo V2/V2.5/Pro (192 K + 128 V), Step-3.5/3.7 (12 penuh + 33 SWA-512), Muse Glimmer (13 + 39 SWA-2048), Granite SWASH | Layer penuh menyimpan seluruh konteks; layer jendela menyimpan min(konteks, jendela) |
| Gemma 3 / Gemma 4 | Gemma 3 12B/27B, Gemma 4 26B-A4B/31B | 5 layer SWA-1024 per layer global; layer global Gemma 4 memakai satu vektor K=V terpadu 512-d |
| Hybrid Gated DeltaNet | Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Max | Hanya 1 dari 4 layer (Gated Attention) yang menyimpan token; layer DeltaNet menyimpan state berukuran tetap |
| Qwen Sparse Attention | Qwen3.8-Flash-Next | Hybrid DeltaNet + indeks 128-d atas konteks/4 |
| Hybrid linear Ling | Ling 2.5/2.6 (1 layer MLA per 8), Ling 3.0 (1 per 6) | Satu layer gated-MLA per grup × 576 |
| MFA | Step-3 | Satu K 256-d dan satu V 256-d bersama per layer |
| CLA-2 | Hunyuan-Large | Lebar GQA-8 pada ceil(layer / 2) layer bersama |
| GQA / MQA ringkas | gpt-oss (8 KV × 64-d), Granite (head 64-d; model Code MQA-1), Trinity Nano (GQA-2) | Head KV × dimensi head |

*Mengapa penting:* pada konteks 32K, model Qwen3.5 397B menyimpan KV lebih sedikit daripada model GQA-8 35B, sedangkan GLM-4.6 357B menyimpan sekitar 12× lebih banyak daripada Qwen3.5 berukuran serupa.

**Pilihan konservatif yang sengaja dipertahankan:**
- Layer sliding-window milik gpt-oss, Command A/A+, North, Laguna XS/S, dan Inkling dihitung pada konteks penuh.
- Lightning attention MiniMax Text-01/M1 dihitung sebagai GQA-8.
- Sparse attention (MiniMax MSA, GLM DSA) memangkas komputasi, bukan cache yang disimpan, sehingga cache penuh tetap dihitung.

### 2.3 Solver muat-maksimum dan satuan

Solver lama bergantian antara jumlah parameter dan arsitektur hanya lima kali, sehingga bisa mengembalikan pasangan yang tidak cocok. Penggantinya:
1. menghitung batas atas berbasis bobot saja;
2. memindai turun di sepanjang kurva interpolasi, karena atensi bisa melompat di anchor sehingga memori tidak monoton;
3. memperhalus dengan pencarian biner;
4. menghitung ulang bobot dan KV dari pasangan akhir yang sama.

Override lanjutan (layer dan lebar hidden tetap) diselesaikan langsung. Metrik per token diberi label **MiB/token**, karena biner.

---

## 3. Overhead sistem per tipe GPU

| Tipe GPU | Model | Default dan bukti |
| --- | --- | --- |
| **Discrete GPU (NVIDIA / AMD / Intel ARC)** | Potongan tetap, slider 0–16 GB | 1,5 GB di desktop (konteks CUDA 0,4–0,8 + runtime 0,5–1 + tampilan 0,4–0,8 GB); 0,5 GB di server headless. Intel ARC memakai GDDR6 khusus, jadi matematikanya sama. Kartu ini butuh Resizable BAR dan berjalan lewat IPEX-LLM atau llama.cpp Vulkan/SYCL; Ollama vanilla tidak berjalan di Arc. |
| **Apple Silicon (M/A-series)** | Cadangan 25%, slider terkunci | `recommendedMaxWorkingSetSize` Metal ≈75% RAM (≈⅔ pada Mac ≤32 GB dengan macOS lama). `sudo sysctl iogpu.wired_limit_mb` dapat menaikkannya dengan mengorbankan cadangan OS. |
| **NVIDIA RTX Spark (Windows on Arm)** | Aturan anggaran GPU resmi NVIDIA | Anggaran = carveout C + clamp((M − C) − 16 GB, 50%, 80% dari (M − C)). Sisanya memori khusus CPU yang tidak pernah bisa dipakai GPU. Slider mengatur ruang desktop/driver (default 1,5 GB). Pengaman kalkulator: C ≤ M − 16. Tanpa carveout: 24 GB → 12, 32 → 16, 64 → 48, 96 → 76,8, 128 → 102,4 GB (hingga 112 GB dengan C ≥ 48). |
| **Other Unified Memory (DGX Spark · Ryzen AI Max · Snapdragon)** | Cadangan tetap, default 3 GB | **DGX Spark** di DGX OS: 12 GB tanpa desktop (≈116 dari 119 GiB tetap tersedia), 16 GB dengan desktop. **Ryzen AI Max+ 128 GB:** 16 GB di Linux, karena GTT membuat iGPU menjangkau ~110–117 GB; di Windows iGPU dibatasi carve-out VGM 96 GB, jadi modelkan sebagai Discrete GPU 96 GB. **Snapdragon X/X2:** ~3 GB untuk jalur CPU/NPU; Windows membatasi offload GPU Adreno hingga 50% RAM. |

**Riwayat:**
- **Snapdragon (Agustus 2026):** "reservasi" 50% yang lama keliru. Angka 50% di Windows adalah batas dinamis memori GPU, bukan reservasi, sehingga laptop 32 GB yang menjalankan inferensi CPU diremehkan ~12 GB. Diganti cadangan 3 GB yang dapat diatur.
- **Intel ARC (Agustus 2026):** sempat ditambahkan sebagai tipe `intel_arc`, lalu digabung ke Discrete GPU setelah masukan pengguna karena matematikanya identik.
- **Oktober 2026:** `snapdragon` → `arm_uma` → `uma`, dengan matematika cadangan-tetap yang sama. RTX Spark menjadi satu-satunya tipe dengan matematika baru.
- **Cadangan DGX Spark** dikoreksi dari 14 ke 12 GB berdasarkan pengukuran tanpa desktop.

---

## 4. Anchor arsitektur estimasi otomatis

**Cara kerja:**
- `referenceConfigs.auto` berisi **98 anchor unik yang terurut ketat, dari 0,35B hingga 2,8T**. Tiap anchor berisi ukuran, layer, lebar hidden, dan tipe atensi opsional.
- Layer dan lebar hidden diinterpolasi linear antar-anchor.
- Tipe atensi diambil dari anchor terdekat: 58 anchor membawa override eksplisit dalam 28 bentuk cache. Selain itu berlaku heuristik:
  - hidden ≤ 1.536, atau 2.048 dengan ≥ 49 layer → GQA-4;
  - hidden 2.880 → GQA-8 × 64-d;
  - hidden 7.168 dengan ≥ 61 layer → MLA;
  - selebihnya → GQA-8.

**Metode:**
- Utamakan `config.json` resmi dan laporan teknis ketimbang situs agregator, yang sering tidak sesuai dengan config.
- Sebuah baris bersifat **persis** (cocok dengan checkpoint yang dirilis) atau **rata-rata sintesis**. Baris rata-rata tidak boleh mengklaim config yang tidak cocok.
- Field yang menentukan KV (layer dan tipe atensi) didahulukan daripada lebar hidden.

**Pertumbuhan tabel:**

| Tahap | Anchor | Keluarga yang ditambahkan |
| --- | --- | --- |
| Llama Gen 1–4 + Kimi K3 | 21 | Llama 7B → 405B; Kimi K3 2,8T (93 layer) |
| gpt-oss, Cohere | 28 | gpt-oss 20b/120b; Command R/R+/A/A+, Aya, North |
| Poolside | 31 | Laguna XS.2 / XS 2.1 / S 2.1 / M.1 |
| Arcee, Mistral | 41 | Trinity Nano/Mini/Large; Mixtral → Large 3. Override atensi per anchor mulai dipakai. |
| MiniMax | 44 | Text-01/M1, M2→M2.7, M3 |
| Audit anchor + DeepSeek V4 | 50 | Koreksi di bawah; V4-Flash, V4-Pro, DeepSeek-V2 |
| Tencent Hunyuan / Hy | 54 | Hunyuan 1.8B, A13B, Large (CLA-2), Hy3 |
| MiMo, StepFun, Muse, Granite | 72 | MiMo V2/V2.5/Pro, Step-3/3.5/3.7, Muse Glimmer, keluarga Granite |
| Generasi terbaru (Okt 2026) | 98 | GLM-4.5 → 5.3, Qwen3-Next / 3.5 / 3.6 / 3.8, Ling 2.0 → 3.0, DeepSeek V4.1-Flash, Hy4-preview, Gemma 3/4 |

**Koreksi dari audit anchor Agustus 2026:**
- **Pengklasifikasi MLA:** aturan "≥ 61 layer dan hidden ≥ 6.144" menandai model dense 32–130B sebagai MLA sehingga KV-nya terhitung sekitar 3× terlalu kecil. Kini aturannya mensyaratkan hidden 7.168.
- **Qwen3-0.6B:** hidden 1.152 → 1.024, GQA-8 (KV sebelumnya terhitung 2× terlalu kecil).
- **Baris 4B:** hidden 3.072 → 2.560.
- **Baris 32B:** 62 × 6.144 → 64 × 5.120.
- **Command R+:** kini MHA yang persis (64 × 12.288).
- **GLM-130B:** 70 × 12.288.
- **Komposit 250B rekaan dihapus**, diganti baris persis Qwen3-235B (GQA-4) dan Inkling-Small 276B.
- **Baris 1T dipecah:** Kimi K2 (61 × 7.168, MLA) vs Inkling 975B (66 × 6.144, GQA). Inkling sebelumnya terhitung 3,5× terlalu kecil karena dianggap MLA.
- **DeepSeek-V4-Pro** menyimpan 9,625 GiB pada 1M token dalam BF16 (angka terbit: 9,62). Jika dihitung sebagai MLA, hasilnya 68,6 GiB.

**Keluarga yang dicakup:** Llama, Gemma, Qwen, MiniCPM, G9, Ling/Ring, Inkling, DeepSeek, Tencent Hunyuan/Hy, GLM, Kimi, Xiaomi MiMo, StepFun, Muse Glimmer, IBM Granite, gpt-oss, Cohere (Command, Aya, North), Poolside Laguna, Mistral, Arcee Trinity, dan MiniMax.

**Aturan inklusi:**
- Hanya open weights dengan config yang dipublikasikan.
- Model multimodal hanya dihitung untuk backbone bahasa autoregresifnya.
- Dikecualikan: model khusus API, generator gambar/video/audio, embedding, encoder-decoder (Aya 101), Mamba murni (tanpa KV cache), dan difusi teks.
- Model berukuran sama digabung ke satu anchor. Beberapa ukuran sedikit digeser agar tetap unik (27,5, 35,5, 81,0).

**Diperiksa, tidak di-anchor (Oktober 2026):**
- Tanpa open weights: Qwen 3.7 (khusus API), Meta Muse Spark 1.2/1.3 (tertutup).
- Bobot belum dirilis: StepFun Step 5 Preview.
- Geometri tidak diungkap atau belum dipublikasikan: Kimi K2.8-Preview / K3.x, GLM-5.3-Flash/FlashX, Gemma 4 12B.
- Di luar cakupan: MiniMax H3 (video), DiffusionGemma (difusi teks).
- "Llama 5" tidak ada.

---

## 5. Jendela konteks dan harness agen

- **Satu slider berlabel ganda:** "Jendela Konteks / Konteks Kerja Aktif" (skala log, 512 → 1M token). Slider ini mengatur kapasitas KV untuk satu obrolan, prompt dokumen, fitur tertanam, atau **satu langkah inferensi agen yang aktif**.
- **Tanpa toggle mode:** toggle Normal/Agentik yang lama hanya mengubah kata-kata, jadi dihapus.
- **17 harness sumber terbuka ditinjau:** OpenClaw, OpenCode, DeepSeek Harness, Hermes Agent, Prime Agent, Pi, Qwen Code, Gemini CLI, Agent Zero, Crush, OpenHands, SWE-agent, Aider, Cline/Roo Code/Goose, LangGraph/Deep Agents, Letta, dan smolagents. Semuanya memisahkan konteks aktif dari state jangka panjang:
  - sesi bertahan di luar konteks model;
  - kompaksi mengganti riwayat dengan ringkasan yang lossy dan tidak menambah kapasitas;
  - memori bertingkat (file, store, retrieval);
  - tiap subagen punya konteksnya sendiri;
  - ruang prompt yang benar-benar bisa dipakai bergantung pada harness.
- **Tanpa pengali harness:** karena alasan-alasan itu, kalkulator tidak menambahkan cadangan output, konkurensi, horizon tugas, ataupun pengali khusus harness.

---

## 6. Preset hardware

**Riwayat:**
- 8 preset (kartu termurah 8 GB, tier Apple usang).
- **21** pada Agustus 2026: spektrum "pelajar hingga bisnis" dari 4 sampai 192 GB, plus 5 kartu Intel ARC.
- **25** pada 7 Oktober 2026: Apple M5, RTX Spark, DGX Spark, dan Snapdragon X2.
- **14** pada 8 Oktober 2026.

**Aturan:**
- Hasil hanya bergantung pada memori, tipe GPU, presisi, konteks, dan overhead, sehingga kartu bermemori sama memberi hasil identik. Set lama punya 3 pasangan yang persis duplikat.
- Karena itu setiap tombol mewakili **satu tingkat memori dengan satu model memori**. Tombol menyebut kartu paling umum, dan tooltip saat kursor diarahkan menampilkan padanannya.

| Grup | Tombol | Tipe | Memori | Presisi · konteks | Overhead | Parameter maks |
| --- | --- | --- | --- | --- | --- | --- |
| Konsumen | GTX 1650 4GB | Discrete | 4 GB | Q4 · 4K | 1,0 GB | 5,0B |
| | RTX 5060 / 4060 8GB | Discrete | 8 GB | Q4 · 8K | 1,5 GB | 10,4B |
| | RTX 5070 / 3060 12GB | Discrete | 12 GB | Q4 · 8K | 1,5 GB | 17,1B |
| | RTX 5060 Ti / RX 9070 XT 16GB | Discrete | 16 GB | Q4 · 16K | 1,5 GB | 23,5B |
| | RTX 3090 / 4090 24GB | Discrete | 24 GB | Q4 · 32K | 1,5 GB | 37,2B |
| | RTX 5090 32GB | Discrete | 32 GB | Q4 · 32K | 1,5 GB | 49,6B |
| Terpadu | MacBook Air / Mac mini 16GB | Apple | 16 GB | Q4 · 8K | 25% | 20,3B |
| | Ryzen AI Max+ 128GB | Other Unified | 128 GB | Q4 · 32K | 16 GB | 190,0B |
| | DGX Spark 128GB | Other Unified | 128 GB | NVFP4 · 64K | 12 GB | 207,9B |
| | RTX Spark 128GB | RTX Spark | 128 GB | Q4 · 32K | 25,6 khusus CPU + 1,5 GB | 171,5B |
| | M5 Ultra 256GB | Apple | 256 GB | Q4 · 64K | 25% | 325,1B |
| Pusat data | A100 / H100 80GB | Discrete | 80 GB | FP16 · 32K | 0,5 GB | 37,4B |
| | RTX PRO 6000 96GB | Discrete | 96 GB | FP16 · 64K | 0,5 GB | 43,0B |
| | B200 180GB | Discrete | 180 GB | FP16 · 128K | 0,5 GB | 90,4B |

**Bukti di balik pilihan ini:**
- **Steam Hardware Survey, Agustus–September 2026:** tingkat VRAM 16 GB 27,21%, 8 GB 26,71%, 12 GB 12,99%, dan 4 GB 5,69%, sekitar 71% bila digabung. RTX 5070 di posisi #1 dengan 5,86%.
- **Panduan pembelian LLM lokal 2026:** tingkat 24 dan 32 GB diambil dari panduan ini, yang menyarankan RTX 5060 Ti 16GB sebagai kartu awal, RTX 3090 24GB bekas untuk harga per GB terbaik, dan RTX 5090 32GB sebagai flagship. RX 9070 XT 16GB sekitar $600.
- **Ryzen AI Max+:** ~110 GB di Linux; Windows dibatasi 96 GB (VGM).
- **Apple:** Mac mini M6 16 GB seharga $899.
- **B200:** 180 GB yang terlihat software; 192 GB adalah stack HBM3e fisik.
- **Sewa cloud:** A100 80GB mulai $1,38 dan H100 mulai $2,59–2,79 per GPU-jam.
- **RTX PRO 6000:** naik dari $8.565 saat peluncuran menjadi $13–20 ribu.

**128 GB yang sama, aturan berbeda** (Q4, 32K): DGX Spark 196,9B · Ryzen AI Max+ di Linux 190,0B · RTX Spark 171,5B · Apple M5 Max 163,1B · Ryzen AI Max+ di Windows (Discrete 96 GB) 160,6B. Selisih ~36B inilah alasan preset memori terpadu tidak digabung.

**Tidak dimasukkan:** B300 (288 GB) dan M5 Ultra 512 GB melebihi slider linear 256 GB. Menaikkan batas maksimum akan memampatkan rentang 4–32 GB, tempat sebagian besar pengguna berada.

---

## 7. Praktik verifikasi

Setiap perubahan diperiksa dengan:
- `node --check` pada skrip yang diekstrak.
- **Sapuan anchor** dari 0,1B hingga 2,8T: dimensi positif, KV hingga dan tidak negatif, ukuran unik yang terurut ketat, uji regresi anchor persis, dan pemeriksaan batas atensi di titik tengah antar-anchor.
- **Cek cepat terhadap angka terbit:**
  - KV Llama;
  - DeepSeek V4: 9,62 / 6,72 GiB pada 1M token;
  - Ling 2.6: 11,25 KiB/token;
  - DeepSeek V4.1-Flash: 890 B/token;
  - gpt-oss MXFP4: 60,8 GB;
  - Laguna S 2.1 NVFP4: ~71 GB.
- **Smoke test jsdom:**
  - setiap preset;
  - tabel anggaran RTX Spark;
  - perpindahan tipe GPU;
  - panel metodologi yang dapat dilipat;
  - kesetaraan kunci Inggris/Indonesia;
  - pemindaian kebijakan offline untuk API storage, history, fetch, hash, dan query.
- **Screenshot Chromium headless** dalam bahasa Inggris dan Indonesia, desktop dan seluler, terang dan gelap.

---

## 8. Sumber utama

Pilihan ringkas. Daftar lengkapnya ada di laporan asli (lihat lampiran).

**Matematika memori dan KV cache**
- Dokumentasi completion / konteks llama.cpp — https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md
- Laporan teknis DeepSeek-V2 (MLA) — https://arxiv.org/abs/2405.04434
- Konfigurasi DeepSeek V3 di Hugging Face — https://huggingface.co/docs/transformers/en/model_doc/deepseek_v3
- Sebastian Raschka, perhitungan KV cache — https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
- Will It Run AI, kebutuhan VRAM (byte kuantisasi) — https://willitrunai.com/blog/vram-requirements-for-ai-models
- Kubesimplify, format kuantisasi (BF16, FP8, NVFP4, MXFP4, INT4, GGUF) — https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters

**Overhead dan platform**
- NVIDIA RTX Spark Windows on Arm Porting Guide 0.1.0, Unified Memory Architecture — https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/index.html
- vramcalculator, RTX Spark vs Ryzen AI Max — https://vramcalculator.com/rtx-spark-local-llm/
- NVIDIA Developer Forums, memori DGX Spark yang terpakai tanpa desktop — https://forums.developer.nvidia.com/t/12gb-of-ram-in-use-on-a-freshly-booted-spark-disabling-desktop-mode/347814
- Simon Willison, ulasan DGX Spark — https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/
- Batas memori GPU bersama 50% di Windows — https://stackoverflow.com/questions/79596623/why-does-windows-only-allow-your-gpu-to-use-half-of-your-ram
- Backend OpenCL llama.cpp (Adreno) — https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
- Modem Guides, cek realitas Ryzen AI Max+ 395 — https://www.modemguides.com/blogs/ai-infrastructure/ryzen-ai-max-395-local-llm-reality-check
- AMD, LLM satu triliun parameter pada klaster Ryzen AI Max+ (pengaturan GTT) — https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html
- ModelPiper, `iogpu.wired_limit_mb` di Mac — https://modelpiper.com/blog/iogpu-wired-limit-mb-mac
- Log working set Metal llama.cpp di Mac Studio — https://obrienlabs.medium.com/running-the-70b-llama-2-llm-locally-on-metal-via-llama-cpp-on-mac-studio-m2-ultra-32b3179e9cbe
- Apple Newsroom, M5 Pro dan M5 Max — https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- LocalAIMaster, Intel Arc untuk AI lokal (Resizable BAR, IPEX-LLM) — https://localaimaster.com/blog/intel-arc-a770-local-ai

**Anchor arsitektur (config dan laporan resmi)**
- Llama 3 Herd of Models — https://ar5iv.labs.arxiv.org/html/2407.21783 · config Llama 4 — https://huggingface.co/docs/transformers/en/model_doc/llama4
- Qwen3 — https://huggingface.co/Qwen/Qwen3-235B-A22B · Qwen3-Next — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Laporan DeepSeek V4 — https://arxiv.org/html/2606.19348v1 · vLLM DeepSeek V4 — https://vllm.ai/blog/2026-04-24-deepseek-v4
- Kimi K2.5 — https://huggingface.co/moonshotai/Kimi-K2.5 · Kimi K3 — https://huggingface.co/moonshotai/Kimi-K3
- Inkling — https://huggingface.co/thinkingmachines/Inkling
- gpt-oss-120b — https://huggingface.co/openai/gpt-oss-120b
- Cohere Command R+ — https://huggingface.co/CohereForAI/c4ai-command-r-plus · North Mini Code — https://huggingface.co/CohereLabs/North-Mini-Code-1.0
- Poolside Laguna S 2.1 — https://huggingface.co/poolside/Laguna-S-2.1
- Arcee Trinity Large — https://huggingface.co/arcee-ai/Trinity-Large-Base
- Mistral Small 4 — https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 · Mistral Large 3 — https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512
- MiniMax M3 — https://huggingface.co/MiniMaxAI/MiniMax-M3
- Config Hunyuan-Large — https://huggingface.co/tencent/Tencent-Hunyuan-Large/blob/main/Hunyuan-A52B-Pretrain/config.json · Hy3 — https://huggingface.co/tencent/Hy3
- MiMo-V2.5-Pro — https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- Step-3 — https://huggingface.co/stepfun-ai/step3 · Step-3.5-Flash — https://huggingface.co/stepfun-ai/Step-3.5-Flash
- Muse Glimmer 30B — https://huggingface.co/meta-models/Muse-Glimmer-30B
- IBM Granite — https://huggingface.co/ibm-granite · laporan Granite Code — https://arxiv.org/abs/2405.04324
- Config Gemma 3 27B — https://huggingface.co/google/gemma-3-27b-it · Gemma 4 Technical Report (arXiv:2607.02770)
- Config GLM Z.ai — https://huggingface.co/zai-org · config Ling InclusionAI — https://huggingface.co/inclusionAI

**Konteks dan harness agen**
- OpenClaw: Context — https://docs.openclaw.ai/concepts/context
- OpenCode: Compaction — https://opencode.ai/v2/docs/compaction
- Hermes Agent: Context Compression — https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching
- Letta: Memory Management — https://docs.letta.com/concepts/memory-management/
- LangGraph: Memory — https://docs.langchain.com/oss/python/langgraph/add-memory

**Preset hardware**
- VideoCardz, survei Steam September 2026 — https://videocardz.com/newz/geforce-rtx-5070-becomes-the-most-popular-gpu-on-steam-32gb-ram-reaches-42
- PC Guide, tingkat VRAM Steam Agustus 2026 — https://www.pcguide.com/news/16gb-vram-continues-to-grow-in-latest-steam-survey-widening-the-gap-to-12gb-and-8gb-graphics-card-owners/
- Compute Market, GPU konsumen terbaik untuk LLM lokal 2026 — https://www.compute-market.com/blog/best-consumer-gpu-local-llm-2026
- TechRepublic, Mac mini M6 — https://www.techrepublic.com/article/news-mac-mini-m6-cheat-sheet/
- GPUsmith, NVIDIA B200 (180 GB terlihat software) — https://gpusmith.com/hardware/gpus/nvidia-b200
- GPUPerHour, harga GPU cloud — https://gpuperhour.com/

---

## Lampiran: laporan asli

Ke-15 file dihapus dari working tree pada 2026-10-08 dan digantikan ringkasan ini. Baca salah satunya dengan `git show 228ba94:<FILE>`, atau temukan versi lain dengan `git log --oneline -- <FILE>`.

| File | Tanggal | Topik |
| --- | --- | --- |
| `OVERHEAD_AUDIT.md` | 2026-08-19 | Overhead per tipe GPU; Snapdragon 50% → tetap 3 GB |
| `PRESET_AND_INTEL_AUDIT_2026.md` | 2026-08-19 | Audit perhitungan, perbaikan MLA, preset 8 → 21, Intel ARC |
| `CONTEXT_MATH_AUDIT_2026.md` | 2026-08-19 | Rumus KV, satuan bobot, solver, MiB/token |
| `AGENT_CONTEXT_HARNESS_AUDIT_2026.md` | 2026-08-19 | 17 harness agen; satu slider konteks berlabel ganda |
| `AUTOESTIMATE_AUDIT_2026.md` | 2026-08-19 | Koreksi anchor; DeepSeek V4 (44 → 50 anchor) |
| `KIMI3_LLAMA_MACBOOKNEO_UPDATE_2026.md` | 2026-08-19 | Kimi K3, Llama Gen 1–4, MacBook Neo |
| `GPTOSS_COHERE_NORTH_UPDATE_2026.md` | 2026-08-19 | gpt-oss, MXFP4, Cohere Command/Aya/North (21 → 28) |
| `POOLSIDE_LAGUNA_UPDATE_2026.md` | 2026-08-19 | Poolside Laguna, NVFP4 (28 → 31) |
| `ARCEE_TRINITY_MISTRAL_UPDATE_2026.md` | 2026-08-19 | Arcee Trinity, Mistral, override per anchor (31 → 41) |
| `MINIMAX_UPDATE_2026.md` | 2026-08-19 | MiniMax (41 → 44) |
| `HUNYUAN_HY_UPDATE_2026.md` | 2026-08-19 | Tencent Hunyuan / Hy, CLA-2 (50 → 54) |
| `MIMO_STEPFUN_MUSE_GRANITE_UPDATE_2026.md` | 2026-08-19 | MiMo, StepFun, Muse Glimmer, Granite (54 → 72) |
| `NEXTGEN_ANCHORS_UPDATE_2026.md` | 2026-10-04 | GLM, Qwen3-Next → 3.8, Ling, DeepSeek V4.1, Hy4, Gemma 3/4 (72 → 98) |
| `ARM_SOC_RTX_SPARK_UPDATE_2026.md` | 2026-10-07 | Aturan anggaran RTX Spark, DGX Spark, SoC Arm, preset 21 → 25 |
| `PRESET_CONSOLIDATION_AUDIT_2026.md` | 2026-10-08 | Preset 25 → 14, Ryzen AI Max+, B200 180 GB, penggantian nama `uma` |

Repo ini juga menyimpan dua screenshot verifikasi satu halaman penuh (`verify_en.png`, `verify_id.png`) dari UI lama; keduanya dihapus bersamaan dan juga tersimpan di riwayat git.
