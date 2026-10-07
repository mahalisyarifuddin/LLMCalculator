# Arm SoC Research + NVIDIA RTX Spark Support — 2026-10-07

**Scope:** In-depth research on **NVIDIA RTX Spark** and the other Arm SoCs used in computers in 2026: NVIDIA DGX Spark (GB10), Qualcomm Snapdragon X2, Apple M5, NVIDIA Jetson AGX Thor, Cix P1, MediaTek Kompanio Ultra, Huawei Kirin X90, Raspberry Pi 5 and Ampere workstations. The findings are applied to `LLMCalculator.html` and both READMEs.

**Method:** Web research on 2026-10-07. Sources include NVIDIA's *RTX Spark Windows on Arm Porting Guide*, NVIDIA and Microsoft announcements, Apple Newsroom, measured DGX Spark reports, Qualcomm spec coverage, llama.cpp documentation and launch-day press. Every factual claim below cites a source. Lines marked *Interpretation* are this audit's own reasoning.

**Constraint kept:** The app stays 100% offline and single-file. It adds no URL/hash state, no storage, no network calls and no permalinks, as required by the file-header directive. Following the earlier Intel ARC feedback, a new GPU type is added **only where the math differs**.

---

## Summary

| Area | Before | After |
| --- | --- | --- |
| GPU types | Discrete · Apple Silicon · Snapdragon | Discrete · Apple Silicon · **NVIDIA RTX Spark** (new math) · **Other Arm SoC** (Snapdragon renamed and widened to cover DGX Spark and Linux Arm) |
| RTX Spark memory model | — | NVIDIA's published rule: GPU budget = carveout + clamp(rest − 16 GB, 50–80% of rest), minus a desktop/driver headroom you can set |
| Carveout control | — | OEM carveout selector (0/8/16/32/48/64/96 GB), shown only for RTX Spark |
| Platform hints | — | Help text under the overhead slider for Apple (Metal cap, `iogpu.wired_limit_mb`) and Other Arm (Snapdragon 50% GPU cap, DGX Spark reserve) |
| Presets | 21 presets, 4–192 GB, "Apple / Mobile" tier | **25 presets, 4–256 GB**: the **Apple Silicon** tier moves to M5 Max / M5 Ultra, plus a new **Arm SoC** tier (RTX Spark 24/64/128, DGX Spark, Snapdragon X2 Elite) |

---

## 1. NVIDIA RTX Spark (N1X), Windows on Arm

### 1.1 Hardware and platform

- **Chip:** Blackwell RTX GPU with 6,144 CUDA cores and 5th-gen Tensor Cores (FP4). It connects over NVLink-C2C to a 20-core Grace (Arm) CPU, with up to 1 PFLOP of AI compute and up to 128 GB of unified memory [5][6].
- **Memory subsystem:** 20-core Armv9.2 CPU in two clusters of 5× Cortex-X925 + 5× Cortex-A725. Up to 128 GB of unified LPDDR5X on a 256-bit bus at 300 GB/s, hardware-coherent between CPU and GPU [2]. DGX Spark's GB10 has the same 10× X925 + 10× A725 CPU layout and 256-bit bus [13][14].
- **SKUs:**

  | Variant | CPU | CUDA cores | Max memory | Power |
  | --- | --- | --- | --- | --- |
  | Laptop N1X | 20 cores | 6,144 | 128 GB | 45–80 W |
  | Laptop N1X | 18 cores | 5,120 | 64 GB | — |
  | Desktop N1X | 20 cores | 6,144 | 128 GB | 140 W |

  Launch OEMs: ASUS ProArt P16, Dell XPS 16, HP OmniBook Ultra 16, Lenovo Yoga Pro 9n, Surface Laptop Ultra and MSI Prestige N16 Flip AI+ [4].
- **Pricing and availability:** The Surface Laptop Ultra ships **October 16, 2026**. It costs **$2,599** for the 18-core / 24 GB / 512 GB model and up to **$5,899.99** for the 20-core / 128 GB / 1 TB model [7]. The 24 GB base model uses the 18-core chip with the 5,120-core GPU [8].
- **Software:** CUDA 13.4 (September 9, 2026) adds native Windows on Arm (ARM64/ARM64EC). Linux is not officially supported [9].

### 1.2 The GPU memory budget (key finding)

NVIDIA's porting guide splits physical memory into three logical regions: a **dedicated carveout** (reported by Windows as dedicated GPU memory), **shared system memory**, and **CPU-only system memory**. It then states [1]:

> "After the dedicated carveout is reserved, the remaining memory is divided between shared system memory and CPU-only system memory. The nominal shared system memory size is the post-carveout capacity minus 16 GB. That value is clamped to a minimum of 50% and a maximum of 80% of the post-carveout capacity. CPU-only system memory receives the balance."

So, with total memory M and carveout C:

```
post   = M − C
shared = clamp(post − 16 GB, 0.5 × post, 0.8 × post)
GPU    = C + shared          CPU-only = post − shared  (never usable by the GPU)
```

| Total memory M | GPU budget (no carveout) | CPU-only | Which rule applies |
| --- | --- | --- | --- |
| 16 GB | 8 GB | 8 GB | 50% floor |
| 24 GB (Surface base) | **12 GB** | 12 GB | 50% floor |
| 32 GB | 16 GB | 16 GB | 50% floor = M − 16 |
| 48 GB | 32 GB | 16 GB | M − 16 |
| 64 GB | 48 GB | 16 GB | M − 16 |
| 96 GB | 76.8 GB | 19.2 GB | 80% cap |
| 128 GB | **102.4 GB** | 25.6 GB | 80% cap |

With a carveout on a 128 GB machine, the budget is 105.6 GB (C = 16), 108.8 GB (C = 32) or **112 GB** (C = 48–96). On a 64 GB machine it stays at 48 GB for any carveout up to 32 GB and rises to 56 GB at C = 48. These figures match an independent calculator that implements the same rule [10].

*Interpretation:* RTX Spark is **not** "Apple with a different percentage". Its GPU share grows with capacity: 50% up to 32 GB, M − 16 from 32 to 80 GB, then 80%. Treating the 24 GB base model as 24 GB of VRAM would overstate the usable GPU memory by about 2×.

### 1.3 How the runtimes report it

- `cudaMalloc` fills the carveout first and then spills into shared memory. `cudaMemGetInfo` reports **carveout + shared**, while NVML/`nvidia-smi` reports only the carveout. DXGI reports the combined budget as the LOCAL segment, with NON_LOCAL = 0 [3].
- NVIDIA advises against allocating the full reported budget, because CPU memory can end up smaller than GPU memory and the system can become unresponsive. It recommends using budget-change notifications instead [3].
- Microsoft describes the platform change as "a new higher, smarter limit on total system memory accessible by the GPU" for unified-memory RTX Spark systems [6].

### 1.4 Still unknown or not modeled

- **OEM default carveout:** No source publishes it. Windows Task Manager shows it as "Dedicated GPU memory" [1]. The calculator therefore defaults to **0 GB**, which is the conservative floor of the rule.
- **Windows "IntelligentCarveout":** "Reserved memory for accelerators" strings appeared in Insider build 29648.1000, in the Experimental Future Platforms channel. The feature is not exposed in Settings, and coverage says it targets RTX Spark, AMD Ryzen Halo and other APU-like chips [11][12]. It is unshipped, so it is **not modeled**. The carveout selector already covers its effect.

### 1.5 Applied model

New `rtxspark` GPU type:

- Reported overhead = **CPU-only memory** (from the rule) + **desktop/driver headroom** (slider, default 1.5 GB).
- Usable memory = GPU budget − headroom.
- The headroom keeps NVIDIA's "don't allocate the full budget" advice [3]. It plays the same role as the 1.5 GB desktop default for discrete GPUs, covering the display/DWM, CUDA context and compute buffers.
- The carveout selector is limited so that at least 16 GB remains after the carveout. *Interpretation:* this is a calculator guard against nonsensical inputs, not an NVIDIA rule.

---

## 2. NVIDIA DGX Spark (GB10): same chip family, different OS policy

- **Specs:** 128 GB of unified LPDDR5X on a 256-bit bus at 273 GB/s [14]. NVIDIA says it runs inference on models up to 200B parameters and now offers **64 GB or 128 GB** configurations [13]. NVIDIA's marketplace listed it at **$4,699** as of August 27, 2026 [20].
- **Measured capacity:**
  - CUDA reports **119.68 GB** of GPU memory, and `free -h` shows **119 Gi total / 112 Gi available** with the stock desktop running (7.5 Gi used, 12 Gi buff/cache) [15][16].
  - A headless unit shows `free -g` **119 total / 116 available** after boot [17].
  - Disabling the desktop saves 2–3 GB [18].
  - A newer headless DGX OS setup reports about **121.7 GiB visible** [19].
- **Failure mode:** Unified-memory OOM can hard-lock the host. One user configured KV cache with only 2.9 GiB of host memory left and got `NV_ERR_NO_MEMORY` followed by a frozen node [19]. vLLM's default `gpu_memory_utilization=0.9` also over-allocates KV cache on this machine [18].
- **Applied model:** DGX OS (Linux) exposes almost all memory to CUDA, so the Windows budget rule doesn't apply. DGX Spark uses the fixed-reserve **Other Arm SoC** type.
  - Preset reserve: **12 GB**, from 128 − 116 GiB headless [17].
  - With the desktop running: about **16 GB**, from 128 − 112 [15].
  - *Interpretation:* 12 GB is conservative given the newer ~121.7 GiB reading. At NVFP4 with 64K context the preset gives ≈208B parameters, consistent with NVIDIA's "up to 200B" claim [13].

## 3. Qualcomm Snapdragon X2 Elite (Windows on Arm)

- **Specs:**
  - X2 Elite Extreme (X2E-96-100): 18 cores, 192-bit LPDDR5X-9523, **228 GB/s**, up to 128 GB.
  - X2 Elite X2E-88-100 / X2E-80-100: 128-bit, **152 GB/s** [21].
  - Adreno X2-90 at up to 1.85 GHz, 80 TOPS NPU [22].
- **Windows GPU cap:** WDDM sizes graphics-accessible system memory as `MAX(TotalSystemMemory / 2, 64 MB)` [23], so the Adreno GPU can address at most 50% of RAM. The earlier overhead audit confirmed this on a 32 GB X Elite (≈15.7 GB shared). No source says Microsoft's new RTX Spark limit [6] extends to Snapdragon.
- **LLM runtimes:** llama.cpp's OpenCL backend supports Adreno, including the X2-90 [24]. Community notes say only one GPU model at a time fits in shared memory, NPU paths are limited to about 4K context, and some drivers break OpenCL [25]. Most local inference still runs on the CPU/NPU path.
- **Applied model:** Unchanged math, a configurable fixed reserve (default 3 GB), now under **Other Arm SoC**. The hint warns that GPU offload is capped at 50% of RAM. The preset moves from Snapdragon X 32 GB to **Snapdragon X2 Elite 48 GB**, a mid-tier configuration of a chip that supports up to 128 GB [21].

## 4. Apple Silicon M5 family

- **Specs:**
  - **M5 Pro:** up to 64 GB at 307 GB/s.
  - **M5 Max:** up to 128 GB at 614 GB/s (March 3, 2026) [26][27].
  - **M5 Ultra Mac Studio:** up to 512 GB at **1.2 TB/s**, from **$5,499**. It shipped September 22, 2026, with the 512 GB configuration arriving in late October [28][29].
- **Metal working-set cap:**
  - llama.cpp logs show a 48 GiB (75%) `recommendedMaxWorkingSetSize` on 64 GB Macs and 21,845 MiB (⅔ of 32 GiB) on a 32 GB machine [31].
  - On macOS 26.5.2, a 32 GiB M2 Max measured **78%** [30].
  - Guides cite ⅔ up to 36 GB and ¾ above [32]. `sudo sysctl iogpu.wired_limit_mb` raises the cap at the cost of the OS reserve [30].
- **Applied model:** The flat 25% reserve stays (≈75% usable). The ⅔ behavior of older macOS on small Macs and the sysctl override are documented in the new hint.
- **Presets:** M4 Max → **M5 Max 128 GB**, and M3 Ultra 192 GB → **M5 Ultra 256 GB**. The 512 GB M5 Ultra is above the slider maximum of 256 GB. *Interpretation:* raising the slider maximum would squeeze the 4–24 GB range where most users are, so it is documented instead.

## 5. Other Arm SoCs used in computers (documentation only)

| SoC / system | Memory facts | Calculator mapping |
| --- | --- | --- |
| NVIDIA Jetson AGX Thor (T5000) | 14-core Neoverse-V3AE, 2,560-core Blackwell GPU, 128 GB 256-bit LPDDR5X at 273 GB/s, 40–130 W, $3,499 dev kit at launch [33]; listed at $5,499 [34] | Other Arm SoC (Linux UMA, fixed reserve) |
| Cix P1 (Radxa Orion O6, Orange Pi 6 Plus) | 12-core Armv9, Immortalis-G720 GPU, 30 TOPS NPU (45 TOPS combined) [35]; up to 64 GB LPDDR5 [36] | Other Arm SoC |
| MediaTek Kompanio Ultra 910 (Lenovo Chromebook Plus 14) | 3 nm, 8-core CPU, 11-core Immortalis-G925, 50 TOPS NPU [38]; 16 GB LPDDR5X, $1,269.99 [37] | Other Arm SoC |
| Huawei Kirin X90 (MateBook Fold, HarmonyOS 5) | 32 GB RAM, SMIC 7 nm [39] | Other Arm SoC |
| Raspberry Pi 5 16 GB (BCM2712) | 4× Cortex-A76, 16 GB LPDDR4X-4267, $305 [40] | Other Arm SoC (CPU inference) |
| Ampere Altra workstation (System76 Thelio Astra) | Up to 512 GB DDR4 plus a discrete NVIDIA RTX 6000 Ada [41] | **Discrete GPU**: the model runs in the card's VRAM |

---

## 6. Change applied (`LLMCalculator.html`)

1. **GPU types:** Added `rtxspark` = "NVIDIA RTX Spark (Windows on Arm)". Renamed `snapdragon` → `arm_uma` = "Other Arm SoC (Snapdragon X/X2 · DGX Spark · Linux)", with the same fixed-reserve math. Switching to `arm_uma` still raises a reserve below 2 GB to 3 GB.
2. **Pure helper** `rtxSparkGpuBudget(totalGB, carveoutGB)`: implements §1.2 and returns `{carveout, shared, budget, cpuOnly, rule}`. The comment cites the porting guide.
3. **UI:**
   - `#rtxCarveoutGroup` selector with help text, visible only for RTX Spark.
   - For RTX Spark the slider label changes to "Desktop / Driver Headroom", and the slider shows only the headroom.
   - The results row reads, for example, "CPU-only 25.6 GB (NVIDIA 80% cap) + 1.5 GB headroom · GPU budget 102.4 GB".
   - `#overheadHint` shows the Apple and Other Arm hints.
   - The memory label reads "Unified Memory" for every SoC type.
   - The copied summary includes the carveout.
4. **i18n:** Every new string exists in both `text.en` and `text.id`. The unused `sharedMemoryLabel` and `snapdragonOverhead` were removed. The Formulas, Hardware Presets and Hardware sources sections were updated in both languages.
5. **Presets (25, 4–256 GB):**

| Preset | Type | Memory | Precision / context | Overhead | Usable | Max params |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Neo 8GB | Apple | 8 GB | Q4 / 8K | 2.0 (25%) | 6.0 | 9.6B |
| M5 Max 128GB | Apple | 128 GB | Q4 / 32K | 32.0 (25%) | 96.0 | 163.1B |
| M5 Ultra 256GB | Apple | 256 GB | Q4 / 64K | 64.0 (25%) | 192.0 | 325.1B |
| RTX Spark 24GB | RTX Spark, C = 0 | 24 GB | Q4 / 8K | 12 CPU-only + 1.5 | 10.5 | 17.1B |
| RTX Spark 64GB | RTX Spark, C = 0 | 64 GB | Q4 / 32K | 16 CPU-only + 1.5 | 46.5 | 75.9B |
| RTX Spark 128GB | RTX Spark, C = 0 | 128 GB | Q4 / 32K | 25.6 CPU-only + 1.5 | 100.9 | 171.5B |
| DGX Spark 128GB | Other Arm | 128 GB | NVFP4 / 64K | 12.0 | 116.0 | 207.9B |
| Snapdragon X2 Elite 48GB | Other Arm | 48 GB | Q4 / 16K | 3.0 | 45.0 | 74.8B |

Overhead and usable memory are in GB. Max params come from the default auto-estimate model family.

## Verification

- `node --check` on the extracted `<script>`: OK.
- **jsdom smoke suite, 172 checks, all passing:**
  - Budget table from §1.2, including vramcalculator's 64 GB and 128 GB carveout cases [10], the carveout guard, and monotonic growth from 4 to 256 GB.
  - All 25 presets: one active button, select and slider match the state, no NaN.
  - RTX Spark labels, overhead numbers, carveout change, headroom slider, and the copied summary.
  - Indonesian strings.
  - Switching between all four GPU types (hint and control visibility, Apple slider disabled).
  - `text.en` and `text.id` have the same keys.
  - No storage, URL or network APIs in executable code.
- **Headless Chromium screenshots** of the GPU card (EN RTX Spark, ID DGX Spark), the results panel and the preset grid: no page errors, no truncated labels in the carveout selector.

## References

1. NVIDIA, *RTX Spark Windows on Arm Porting Guide 0.1.0 — Unified Memory Architecture*. https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/index.html
2. NVIDIA, *Porting Guide — System Overview*. https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/overview.html
3. NVIDIA, *Porting Guide — Usage with CUDA APIs*. https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/umacuda.html
4. NVIDIA, *RTX Spark* product page. https://www.nvidia.com/en-us/products/rtx-spark/
5. NVIDIA GeForce News, "NVIDIA at COMPUTEX 2026: NVIDIA RTX Spark, DLSS 4.5, RTX Updates" (May 31, 2026). https://www.nvidia.com/en-us/geforce/news/computex-2026-nvidia-geforce-rtx-announcements/
6. Microsoft Windows Experience Blog, "Introducing a powerful new chapter for Windows PCs, accelerated by NVIDIA RTX Spark" (May 31, 2026). https://blogs.windows.com/windowsexperience/2026/05/31/introducing-a-powerful-new-chapter-for-windows-pcs-accelerated-by-nvidia-rtx-spark/
7. The Verge, "The Surface Laptop Ultra finally has a release date — and a starting price of $2,599" (Oct 7, 2026). https://www.theverge.com/news/1006378/microsoft-surface-laptop-ultra-pricing-release-date
8. VideoCardz, "NVIDIA RTX Spark laptops launch October 16, prices start at $2,599" (Oct 7, 2026). https://videocardz.com/newz/nvidia-rtx-spark-laptops-launch-october-16-prices-start-at-2599
9. Computing for Geeks, "NVIDIA RTX Spark on Linux" (Oct 2026). https://computingforgeeks.com/nvidia-rtx-spark-linux/
10. vramcalculator.com, "RTX Spark vs Ryzen AI Max: How Much Memory the GPU Gets" (Sep 28, 2026). https://vramcalculator.com/rtx-spark-local-llm/
11. TweakTown, "Windows 11 may soon let you control how unified memory is split for gaming and AI" (Aug 25, 2026). https://www.tweaktown.com/news/113245/windows-11-may-soon-let-you-control-how-unified-memory-is-split-for-gaming-and-ai/index.html
12. TechPowerUp, "Windows 11 to Get Unified Memory Control Options Ahead of RTX Spark Launch" (Aug 25, 2026). https://www.techpowerup.com/351881/windows-11-to-get-unified-memory-control-options-ahead-of-rtx-spark-launch
13. NVIDIA, *DGX Spark* product page. https://www.nvidia.com/en-us/products/workstations/dgx-spark/
14. PNY, *NVIDIA DGX Spark* specifications. https://www.pny.com/dgx-spark
15. Simon Willison, "NVIDIA DGX Spark: great hardware, early days for the ecosystem" (Oct 14, 2025). https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/
16. Hacker News discussion of [15] (`free -h` output). https://news.ycombinator.com/item?id=45586776
17. NVIDIA Developer Forums, "12gb of RAM in use on a freshly-booted Spark?! Disabling desktop mode?" (Oct 2025). https://forums.developer.nvidia.com/t/12gb-of-ram-in-use-on-a-freshly-booted-spark-disabling-desktop-mode/347814
18. NVIDIA Developer Forums, "Memory Creep on DGX Spark: Where Your 128 GB Actually Goes (And How to Stop It)" (Mar 2026). https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
19. r/LocalLLaMA, "Serving Deepseek v4 Flash 0731 on 2x DGX Spark — 5-7 GB OS headroom" (Aug 2026). https://www.reddit.com/r/LocalLLaMA/comments/1vig3tw/serving_deepseek_v4_flash_0731_on_2x_dgx_spark_57/
20. Layer3 Labs, *Local AI Hardware Calculator* (DGX Spark price note). https://www.layer3labs.io/tools/local-ai-hardware-calculator
21. Notebookcheck, "Snapdragon X2 Elite Extreme X2E-96-100, X2 Elite X2E-88-100 and X2E-80-100 officially announced" (Sep 24, 2025). https://www.notebookcheck.net/Snapdragon-X2-Elite-Extreme-X2E-96-100-Snapdragon-X2-Elite-X2E-88-100-and-Snapdragon-X2-Elite-X2E-80-100-officially-announced-as-Qualcomm-s-new-laptop-chips.1123248.0.html
22. CPU-Monkey, *Snapdragon X2 Elite Extreme (X2E-96-100)* specs. https://www.cpu-monkey.com/en/cpu-qualcomm_snapdragon_x2_elite_extreme_x2e_96_100
23. Stack Overflow, "Why does Windows only allow your GPU to use half of your RAM?" (quotes Microsoft's `TotalSystemMemoryAvailableForGraphics` formula). https://stackoverflow.com/questions/79596623/why-does-windows-only-allow-your-gpu-to-use-half-of-your-ram
24. llama.cpp, *OpenCL backend* documentation (Adreno). https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
25. zabojad75, *Snapdragon-NPU-GPU-LLM_faster* (community notes). https://github.com/zabojad75/Snapdragon-NPU-GPU-LLM_faster
26. Apple Newsroom, "Apple debuts M5 Pro and M5 Max to supercharge the most demanding pro workflows" (Mar 3, 2026). https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
27. Apple Newsroom, "Apple introduces MacBook Pro with all-new M5 Pro and M5 Max" (Mar 3, 2026). https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/
28. Fello AI, "M5 Ultra Mac Studio: Price, Specs and 512GB Memory" (Sep 2026). https://felloai.com/m5-ultra-mac-studio/
29. Tech Times, "Mac Studio M5 Ultra Ships With 512GB to Run Frontier AI Models Locally" (Aug 26, 2026). https://www.techtimes.com/articles/325575/20260826/mac-studio-m5-ultra-ships-512gb-run-frontier-ai-models-locally.htm
30. ModelPiper, "iogpu.wired_limit_mb on Mac: Raising the Metal Memory Ceiling" (Aug 2026). https://modelpiper.com/blog/iogpu-wired-limit-mb-mac
31. O'Brien Labs, "70B LLaMA 2 LLM local inference on metal via llama.cpp on Mac Studio M2 Ultra" (llama.cpp Metal logs). https://obrienlabs.medium.com/running-the-70b-llama-2-llm-locally-on-metal-via-llama-cpp-on-mac-studio-m2-ultra-32b3179e9cbe
32. LLM Configurator, "VRAM Requirements for Local LLMs" (Apple Silicon section). https://llmconfigurator.com/en/guides/vram-requirements-guide
33. Hackster.io, "NVIDIA Tells Resellers to Open Jetson AGX Thor Developer Kit Orders at $3,499". https://www.hackster.io/news/nvidia-tells-resellers-to-open-jetson-agx-thor-developer-kit-orders-at-3-499-06e8cbf52441
34. Newegg, *NVIDIA Jetson AGX Thor Developer Kit* listing. https://www.newegg.com/nvidia-jetson-agx-thor-developer-kit-128-gb-256-bit-lpddr5x-273-gb-s/p/N82E16813190035
35. CNX Software, "Radxa Orion O6 mini-ITX motherboard is powered by Cix P1 12-core Armv9 SoC with a 30 TOPS AI accelerator" (Dec 18, 2024). https://www.cnx-software.com/2024/12/18/radxa-orion-o6-mini-itx-motherboard-is-powered-by-cix-p1-12-core-armv9-soc-with-a-30-tops-ai-accelerator/
36. CNX Software, "Orange Pi 6 Plus – CIX P1 SBC offers up to 64GB LPDDR5 memory, 45 TOPS of AI performance" (Oct 15, 2025). https://www.cnx-software.com/2025/10/15/orange-pi-6-plus-cix-p1-sbc-64gb-lpddr5-45-tops-ai-performance/
37. PCMag, "Lenovo Chromebook Plus 14 Review" (2026). https://www.pcmag.com/reviews/lenovo-chromebook-plus-14
38. PCMag, "I Tried the First Chromebook With an NPU" (2025). https://www.pcmag.com/news/hands-on-lenovo-chromebook-plus-14-first-npu-mediatek-chromebook
39. Wikipedia, "MateBook Fold". https://en.wikipedia.org/wiki/MateBook_Fold
40. SparkFun, *Raspberry Pi 5 – 16GB*. https://www.sparkfun.com/raspberry-pi-5-16gb.html
41. How-To Geek, "System76 Thelio Astra Combines Linux With a 128-Core ARM CPU" (Oct 22, 2024). https://www.howtogeek.com/system76-thelio-astra-reveal/
