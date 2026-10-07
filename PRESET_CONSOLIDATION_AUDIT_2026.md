# Preset Consolidation Audit — 2026-10-08

**Scope:** Cut the Quick Presets in `LLMCalculator.html` from 25 to a small set that still spans the whole economic range, from 4 GB student laptops to datacenter GPUs and 256 GB unified-memory workstations. Every button kept must be justified by 2026 data on ownership, local-LLM buying habits, availability and price.

**Method:** Web research on 2026-10-07 and 2026-10-08. Sources cover the Steam Hardware Survey (July–September 2026), 2026 local-LLM GPU buying guides, AMD/Framework/Linux documentation for Ryzen AI Max+ (Strix Halo), Apple pricing coverage, NVIDIA OEM memory specs and live cloud-rental price trackers. Every factual claim cites a numbered reference. Lines marked *Interpretation* are this audit's own reasoning.

**Constraint kept:** The app stays a 100% offline single file, as the file-header directive requires: no URL or hash state, no storage, no network calls, no permalinks. Following the earlier Intel ARC feedback, no GPU type was added. The only type change is a **rename with identical math**, described in §5.

---

## Summary

| Area | Before | After |
| --- | --- | --- |
| Presets | 25 in 5 groups (Budget, Mainstream, Workstation/Server, Apple Silicon, Arm SoC) | **14 in 3 groups**: Consumer GPUs (6), Unified Memory (5), Workstation / Datacenter (3) |
| Range | 4–256 GB | 4–256 GB (unchanged) |
| Duplicates | 3 pairs had **identical inputs**, so they gave identical results (RTX 4060 = RTX 5050 Laptop, RTX 3060 = Arc B580, RTX 5070 Ti = Arc A770) | None. One button per memory tier per memory model. Each button names the most common cards and a hover tooltip lists the equivalents. |
| New | — | **Ryzen AI Max+ 128GB** (AMD Strix Halo), a major 2026 local-LLM platform that had no preset |
| Corrected | B200 192GB | **B200 180GB**, the software-visible capacity (192 GB is the physical HBM3e stack) [19] |
| Apple entry | MacBook Neo 8GB | **MacBook Air / Mac mini 16GB**, the 16 GB base of Apple's mainstream Macs [16][18] |
| GPU type | "Other Arm SoC (Snapdragon X/X2 · DGX Spark · Linux)" (`arm_uma`) | "Other Unified Memory (DGX Spark · Ryzen AI Max · Snapdragon)" (`uma`). Same fixed-reserve math, now also covers x86 Strix Halo. |

---

## 1. Why one button per memory tier

The calculator's result depends only on five inputs: memory size, GPU type, precision, context and overhead. Two presets with the same inputs therefore show the same number. In the old set, three pairs were exact duplicates and several more differed only by brand (for example RTX 4090 24GB vs Arc Pro B60 24GB, which differed only in overhead, 1.5 vs 0.8 GB).

*Interpretation:* A preset's job is to answer "what fits on hardware like mine?" in one click. The right unit is therefore **a memory tier under a given memory model**, not a product. Naming two or three popular cards on each button, plus a tooltip of equivalents, keeps coverage broad with fewer buttons. Rarer sizes are one slider move away.

### Selection rules

1. **Consumer GPUs:** keep the four VRAM tiers with the largest Steam shares, plus the two tiers that local-LLM buyers target (24 GB and 32 GB).
2. **Unified memory:** keep one preset per distinct memory rule or platform that matters for local LLMs in 2026. Results differ widely at the same capacity (see §4), so these must not be merged.
3. **Workstation / datacenter:** keep the cheapest widely rented tier, the single-card workstation tier and the current flagship that fits the 4–256 GB slider.
4. **Range:** keep both ends of the economic spectrum, from 4 GB student laptops to 180–256 GB.

---

## 2. Evidence

### 2.1 Consumer GPUs (Steam Hardware Survey + local-LLM guides)

**Most common cards:**
- In September 2026 the RTX 5070 became the most popular GPU on Steam at 5.86% (+2.30 points), followed by the RTX 5060 (4.20%) and RTX 5060 Ti (3.87%). The RTX 50 family reached 18.21%, and no Radeon or Intel card is in the top 10 [1].
- The RTX 3060 held the top spot again in April 2026 (~3.99%) [5].
- An early-2026 survey listed: RTX 4060 4.16%, RTX 3060 4.11%, RTX 4060 Laptop 3.79%, RTX 3050 2.88%, GTX 1650 2.63% [4].
- The RTX 3060 12GB is described as a long-time staple. The RTX 5070 Ti is the top 16 GB-only card at 1.73% (July 2026) [6].

**VRAM tiers:**

| Tier | August 2026 [2] | September 2026 [3] |
| --- | --- | --- |
| 16 GB | 26.92% | 27.21% |
| 8 GB | 25.75% | 26.71% |
| 12 GB | 12.99% | — |
| 4 GB | 5.69% (next-largest tier) | — |

*Interpretation:* These four tiers hold about 71% of surveyed GPUs (August figures). Every other tier, including 6 GB and 10 GB, is smaller than the 4 GB tier. That is why the 6 GB (Arc A380) and 10 GB (Arc B570) presets were dropped, while 4 GB stays for student and budget laptops.

**Local-LLM buying guides:**
- April 2026: the RTX 5060 Ti 16GB ($429–479) is the best entry card, a used RTX 3090 ($699–999) has the best price per GB of VRAM, the RTX 5090 costs $1,999–2,199, and the Arc B580 is the only sub-$300 route to 12 GB [7].
- October 2026: the RTX 3090, 5060 Ti 16GB, 4090 and 5090 are still the picks, but used 3090 prices have risen to about $1,700–2,000 [8].
- The AMD path in September 2026: RX 9070 XT 16GB for about $600 and RX 7900 XTX 24GB for $899 [9].

*Interpretation:* Steam shows what people own and the LLM guides show what they buy for local models. Combining the two gives six consumer buttons: 4, 8, 12 and 16 GB from Steam, and 24 and 32 GB from the LLM guides.

### 2.2 Unified memory

**AMD Ryzen AI Max+ 395 (Strix Halo), 128 GB LPDDR5X at 256 GB/s.**
- A BIOS carve-out (Variable Graphics Memory) is capped at 96 GB, and Windows tops out at that allocation. On Linux, the kernel's GTT limit lets the GPU address about 110 GB, "which is why most heavy users run Ubuntu or Fedora" [10].
- ModelFit (October 2026) also gives about 110 GB on Linux and says Windows is limited to the fixed BIOS carve-out [12].
- AMD's own cluster guide sets `amdgpu.gttsize=120000` and reports "120000M of GTT memory ready" on a Framework Desktop node [11].
- One community setup uses a 124 GiB GTT window with pinned pages capped at 108 GiB [15].
- Prices: the Framework Desktop 128 GB lists at $3,449, up from a $1,999 launch, and was out of stock. A 192 GB Max+ PRO 495 model is listed as coming soon [13]. Other 128 GB mini PCs cost $3,324–4,349 [14].

**Apple.**
- The Mac mini M6 shipped on 22 September 2026: 16 GB at 153 GB/s for $899, or 24/32 GB at 170 GB/s from $1,399 [16][17].
- The M5 MacBook Air starts at 16 GB ($1,199 with the education discount) [18].
- The Mac Studio M5 Ultra offers 96/256/512 GB [16][17] (also in the earlier Arm SoC update, from Apple Newsroom).
- The MacBook Neo (8 GB) is the budget model [18] (price and spec in the earlier [Arm SoC update](ARM_SOC_RTX_SPARK_UPDATE_2026.md)).

**NVIDIA DGX Spark and RTX Spark.** Both rules and their sources are documented in [ARM_SOC_RTX_SPARK_UPDATE_2026.md](ARM_SOC_RTX_SPARK_UPDATE_2026.md):
- DGX Spark: about 116 GiB available headless on DGX OS, so a 12 GB reserve.
- RTX Spark: NVIDIA's carveout + clamped-shared Windows on Arm budget rule.

### 2.3 Workstation and datacenter

**B200 capacity.** The B200's software-visible capacity is 180 GB of HBM3e. This is the figure in NVIDIA's own OEM documentation (Lenovo, Dell), and an 8-GPU HGX B200 board has 1.4 TB in aggregate. "Some earlier marketing referenced 192 GB, which reflects the physical HBM3e stack size before reserved capacity" [19].

**Cloud rental prices, October 2026** [21]:

| GPU | Cheapest secure on-demand price |
| --- | --- |
| A100 SXM4 80GB | $1.38/GPU-hr |
| H100 PCIe | $2.59/GPU-hr |
| H100 SXM | $2.79/GPU-hr (6 providers with stock; 30-day median $2.18) |
| H200 NVL | $3.43/GPU-hr |
| B300 | $7.89/GPU-hr |
| L40S | $0.97/GPU-hr |

- A rental index puts the median H100 price at $3.38/hr in August 2026, down from more than $7/hr in early 2024 [22].
- The B300 has 288 GB per GPU [20], which is above the calculator's 256 GB slider.

**RTX PRO 6000 Blackwell 96GB.** It launched at $8,565 and was listed at $13,250 in June 2026 [23]. NVIDIA raised the official price to $16,000 in August 2026, and street prices in September ran $13,998–19,999 [24]. It runs gpt-oss 120B entirely in its 96 GB of VRAM [24].

*Interpretation:*
- A100/H100 80GB is the cheapest, most widely available datacenter tier, and one button covers both cards because they have the same memory.
- RTX PRO 6000 is the on-premises single-card option.
- B200 180GB is the current flagship that fits the slider.
- H200 141GB and L40S 48GB were dropped because their sizes are one slider move away.

---

## 3. Final preset list (14)

Results are the calculator's own output at the preset settings. "Overhead" is the deduction shown in the memory breakdown.

| # | Button | Tooltip (same memory) | Type | Memory | Precision · context | Overhead | Max params | Why kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GTX 1650 4GB | GTX 1650 · GTX 1050 Ti · RTX 3050 Laptop 4GB | Discrete | 4 GB | Q4 · 4K | 1.0 GB | 5.0B | 4 GB tier 5.69% [2]; GTX 1650 2.63% [4]; student and budget laptops |
| 2 | RTX 5060 / 4060 8GB | RTX 5060 · RTX 4060 · RTX 4060 Laptop · RX 7600 · Arc A750 | Discrete | 8 GB | Q4 · 8K | 1.5 GB | 10.4B | 8 GB tier 26.71% [3]; RTX 5060 4.20% [1], RTX 4060 4.16% [4] |
| 3 | RTX 5070 / 3060 12GB | RTX 5070 · RTX 3060 · RTX 4070 · RX 7700 XT · Arc B580 | Discrete | 12 GB | Q4 · 8K | 1.5 GB | 17.1B | Steam #1 RTX 5070 5.86% [1]; RTX 3060 #1 in April [5]; B580 cheapest 12 GB [7] |
| 4 | RTX 5060 Ti / RX 9070 XT 16GB | RTX 5060 Ti 16GB · RTX 5070 Ti · RTX 5080 · RX 9070 XT · RX 9060 XT 16GB · Arc A770 16GB | Discrete | 16 GB | Q4 · 16K | 1.5 GB | 23.5B | Largest tier 27.21% [3]; best entry LLM card [7]; best AMD value [9] |
| 5 | RTX 3090 / 4090 24GB | RTX 3090 · RTX 4090 · RX 7900 XTX · Arc Pro B60 | Discrete | 24 GB | Q4 · 32K | 1.5 GB | 37.2B | The classic local-LLM tier [7][8][9] |
| 6 | RTX 5090 32GB | RTX 5090 · Radeon AI PRO R9700 | Discrete | 32 GB | Q4 · 32K | 1.5 GB | 49.6B | Consumer flagship [7][8] |
| 7 | MacBook Air / Mac mini 16GB | MacBook Air M5 · Mac mini M6 · MacBook Pro M5 | Apple Silicon | 16 GB | Q4 · 8K | 25% (4.0 GB) | 20.3B | 16 GB base of mainstream Macs; $899 Mac mini M6 [16][18] |
| 8 | Ryzen AI Max+ 128GB | Ryzen AI Max+ 395 · Framework Desktop · GMKtec EVO-X2 · HP Z2 Mini G1a | Other Unified Memory | 128 GB | Q4 · 32K | 16 GB | 190.0B | Leading x86 128 GB local-LLM platform [9][10][13][14] |
| 9 | DGX Spark 128GB | DGX Spark · ASUS Ascent GX10 · Dell Pro Max GB10 | Other Unified Memory | 128 GB | NVFP4 · 64K | 12 GB | 207.9B | Linux GB10 appliance (earlier audit) |
| 10 | RTX Spark 128GB | RTX Spark N1X · Surface Laptop Ultra | RTX Spark | 128 GB | Q4 · 32K | 25.6 CPU-only + 1.5 GB | 171.5B | Windows on Arm budget rule (earlier audit) |
| 11 | M5 Ultra 256GB | Mac Studio M5 Ultra | Apple Silicon | 256 GB | Q4 · 64K | 25% (64 GB) | 325.1B | Top of the slider range [16] |
| 12 | A100 / H100 80GB | H100 80GB · A100 80GB | Discrete | 80 GB | FP16 · 32K | 0.5 GB | 37.4B | Cheapest ≥80 GB rental tier; H100 has the most providers in stock [21][22] |
| 13 | RTX PRO 6000 96GB | RTX PRO 6000 Blackwell | Discrete | 96 GB | FP16 · 64K | 0.5 GB | 43.0B | Single-card 96 GB workstation [23][24] |
| 14 | B200 180GB | B200 · HGX B200 · DGX B200 | Discrete | 180 GB | FP16 · 128K | 0.5 GB | 90.4B | Current flagship within 256 GB; 180 GB usable [19] |

Tooltip equivalents use each card's published memory size. Variants with other sizes, such as the 8 GB RTX 5060 Ti or the 6 GB RTX 3050 Laptop, need the slider.

**Coverage check (Interpretation):**
- The consumer buttons cover the four largest Steam VRAM tiers (≈71% of surveyed GPUs [2]) plus the two LLM-buyer tiers.
- The unified group covers Apple at both ends, AMD Strix Halo, and both NVIDIA GB10-class rules.
- The datacenter group covers rental, on-premises and flagship.
- AMD and Intel cards are named on buttons or in tooltips wherever they share a tier (RX 9070 XT, RX 7900 XTX, Arc B580, Arc A770, Arc Pro B60).

---

## 4. Same memory, different rules: the 128 GB comparison

The calculator's output at a common setting (Q4 GGUF, 32K context) shows why the unified presets cannot be merged even though all of them are "128 GB":

| Platform (GPU type) | Rule | Deducted | Left for weights | Max params |
| --- | --- | --- | --- | --- |
| DGX Spark, headless DGX OS (Other Unified Memory) | fixed 12 GB reserve | 12.0 GB | 115.5 GB | 196.9B |
| Ryzen AI Max+ 395 on Linux (Other Unified Memory) | fixed 16 GB reserve (GTT) | 16.0 GB | 111.5 GB | 190.0B |
| RTX Spark, no carveout (RTX Spark) | NVIDIA budget rule + 1.5 GB headroom | 27.1 GB | 100.7 GB | 171.5B |
| Apple M5 Max (Apple Silicon) | Metal ≈75% cap | 32.0 GB | 95.8 GB | 163.1B |
| Ryzen AI Max+ 395 on Windows (Discrete, 96 GB) | 96 GB VGM carve-out − 1.5 GB | 1.5 GB of 96 | 94.3 GB | 160.6B |

*Interpretation:* The spread from 160.6B to 196.9B (~36B) comes only from OS and driver rules. The DGX Spark preset itself uses NVFP4 at 64K and shows 207.9B.

**Why 16 GB for Strix Halo:** The reserve gives 112 GB, inside the reported Linux range of ~108–117 GiB [10][11][12][15]. 16 GB is also the overhead slider's maximum, so the preset stays representable. On Windows the hint advises Discrete GPU at 96 GB instead [10][12].

---

## 5. Changes applied to `LLMCalculator.html`

1. **Preset markup:** three groups with new tier ids and text keys (EN and ID):
   - `presetConsumer`: "Consumer GPUs (4–32 GB)" / "GPU Konsumen (4–32 GB)"
   - `presetUnified`: "Unified Memory (16–256 GB)" / "Memori Terpadu (16–256 GB)"
   - `presetDatacenter`: "Workstation / Datacenter (80–180 GB)" / "Workstation / Pusat Data (80–180 GB)"

   The old keys `presetBudget`, `presetMainstream`, `presetWorkstation`, `presetApple` and `presetArm` are gone. Each button has a `title` tooltip listing same-memory equivalents (product names only, so it needs no translation).
2. **`applyPreset` dictionary:**
   - The 14 keys are `gtx1650`, `rtx5060`, `rtx5070`, `rtx5060ti`, `rtx3090`, `rtx5090`, `mac16`, `ryzenaimax`, `dgxspark`, `rtxspark128`, `m5ultra`, `h100`, `rtx_pro_6000` and `b200`.
   - Kept presets keep their earlier settings.
   - `b200` now uses 180 GB.
   - The new `ryzenaimax` preset is `uma`, 128 GB, Q4, 32K, with a 16 GB reserve.
3. **GPU type rename (same math):**
   - `arm_uma` becomes `uma`, labelled "Other Unified Memory (DGX Spark · Ryzen AI Max · Snapdragon)". The first words stay readable in the narrow select.
   - Text keys `armUmaOverhead`/`armUmaHint` become `umaOverhead`/`umaHint`. The hint adds Strix Halo guidance: ~16 GB reserve on Linux; on Windows, use Discrete GPU at 96 GB.
   - Switching to the type still raises a reserve below 2 GB to 3 GB.
4. **Reference panel (EN and ID):**
   - The Formulas overhead bullet now names Strix Halo.
   - "Hardware Presets" is rewritten for the 14-preset structure.
   - The "Hardware:" sources line now cites this audit's data.
5. **No change to the core math:** RTX Spark rule, Apple 25%, fixed reserve, KV and weights.

Both READMEs were updated: §5 overhead bullet, Quick Start steps 4 and 8, and Key Features (architecture line and "14 presets"). The Arc Pro B60 overhead example was also removed from the Intel ARC note, because that preset is gone.

### Removed presets and how to reproduce them

| Old preset | Status | Reproduce with |
| --- | --- | --- |
| RTX 4060 8GB, RTX 5050 Laptop 8GB | Merged into **RTX 5060 / 4060 8GB** (identical inputs) | — |
| RTX 3060 12GB, Arc B580 12GB | Merged into **RTX 5070 / 3060 12GB** (identical inputs) | — |
| RTX 5070 Ti 16GB, Arc A770 16GB | Merged into **RTX 5060 Ti / RX 9070 XT 16GB** (identical inputs) | — |
| RTX 4090 24GB | Merged into **RTX 3090 / 4090 24GB** | — |
| Arc Pro B60 24GB | In the 24 GB tooltip | 24 GB button, overhead 0.8 GB |
| A100 80GB | Merged into **A100 / H100 80GB** | — |
| Arc A380 6GB | Removed (6 GB tier < 5.69%) | Discrete, 6 GB, overhead 1.0 |
| Arc B570 10GB | Removed (10 GB tier < 5.69%) | Discrete, 10 GB |
| L40S 48GB | Removed | Discrete, 48 GB, FP16, overhead 0.5 |
| H200 141GB | Removed | Discrete, 141 GB, FP16, 128K, overhead 0.5 |
| B200 192GB | **Corrected to 180 GB** | — |
| MacBook Neo 8GB | Replaced by the 16 GB Mac button | Apple Silicon, 8 GB (6 GB usable, 9.6B at Q4 8K) |
| M5 Max 128GB | Removed (128 GB already has three rules) | Apple Silicon, 128 GB |
| RTX Spark 24GB / 64GB | Removed (README lists the budgets) | RTX Spark, 24 or 64 GB |
| Snapdragon X2 Elite 48GB | Removed | Other Unified Memory, 48 GB, reserve 3 GB |

### Not included, and why

- **288 GB class (B300/GB300) and M5 Ultra 512 GB:** these exceed the 256 GB slider [20]. Raising the maximum of a linear slider compresses the 4–32 GB region where most users are, so the earlier decision to keep 256 GB stands.
- **Separate Intel or AMD buttons:** cards that share a tier are named in the label or tooltip instead. The Intel ARC notes (Resizable BAR, IPEX-LLM) stay in the README.

---

## 6. Verification

- `node --check` on the extracted script: OK.
- jsdom smoke tests: **184/184 passed**. They check:
  - exactly 14 buttons in the expected order and 3 groups; all 18 removed preset keys absent;
  - every button's "…GB" label equals the memory it applies, and every button has a tooltip;
  - per-preset results: Ryzen 16 GB reserve, DGX 12 GB + NVFP4, Mac 25%, B200 180 GB, RTX Spark 27.1 GB;
  - the RTX Spark budget table and carveout behaviour, the `uma` type switch (reserve raised to 3.0 GB, Strix Halo hint);
  - EN and ID tier titles, `text.en`/`text.id` key parity, and the offline policy (no storage, history, fetch, hash or query APIs).
- Chromium (headless) screenshots in EN and ID, at 1280 px and 390 px wide: no page errors. Labels wrap inside their buttons without truncation. Desktop shows one row per group (6 / 5 / 3), mobile shows two columns.

---

## References

1. VideoCardz — "GeForce RTX 5070 becomes the most popular GPU on Steam, 32GB RAM reaches 42%" (September 2026 survey). https://videocardz.com/newz/geforce-rtx-5070-becomes-the-most-popular-gpu-on-steam-32gb-ram-reaches-42
2. PC Guide — "16GB VRAM continues to grow in latest Steam survey, widening the gap to 12GB and 8GB graphics card owners" (August 2026 survey). https://www.pcguide.com/news/16gb-vram-continues-to-grow-in-latest-steam-survey-widening-the-gap-to-12gb-and-8gb-graphics-card-owners/
3. Shattered.io — Steam Hardware Survey: 32GB RAM and RTX 5070 (September 2026). https://shattered.io/steam-hardware-survey-32gb-ram-rtx-5070-2026/
4. TweakTown — "GeForce RTX 5070 joins the RTX 3060 as one of the most popular GPUs among PC gamers" (early-2026 survey table). https://www.tweaktown.com/news/109994/geforce-rtx-5070-joins-the-rtx-3060-as-one-of-the-most-popular-gpus-among-pc-gamers/index.html
5. WhySoGeek — "Steam Hardware Survey 2026: Top GPUs, RAM, Linux". https://whysogeek.com/steam-hardware-survey-2026-trends-explained/
6. SpecClear — Steam Hardware Survey July 2026. https://specclear.com/steam-hardware-survey-july-2026/
7. Compute Market — Best consumer GPU for local LLM 2026. https://www.compute-market.com/blog/best-consumer-gpu-local-llm-2026
8. ComputingForGeeks — Best GPU for LLM (October 2026). https://computingforgeeks.com/best-gpu-for-llm/
9. Compute Market — Best AMD GPU for local LLM inference 2026 (September 2026). https://www.compute-market.com/blog/best-amd-gpu-local-llm-inference-2026
10. Modem Guides — "The Ryzen AI Max+ 395 for Local LLMs: An Honest Reality Check" (June 2026). https://www.modemguides.com/blogs/ai-infrastructure/ryzen-ai-max-395-local-llm-reality-check
11. AMD Developer — "How to Run a One Trillion-Parameter LLM Locally: An AMD Ryzen AI Max+ Cluster Guide" (2026). https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html
12. ModelFit — AMD Ryzen AI Max+ 395 local LLM (October 2026). https://modelfit.io/gpu/ryzen-ai-max-395/
13. LocalAIMaster — Strix Halo (Ryzen AI Max+ 395) guide. https://localaimaster.com/blog/strix-halo-ai-max-395-guide
14. ComputingForGeeks — Ryzen AI Max+ 395 mini PC comparison. https://computingforgeeks.com/ryzen-ai-max-395-mini-pc-comparison/
15. sypherin/strix-halo-setup — README (GTT and kernel parameters, September 2026). https://github.com/sypherin/strix-halo-setup/blob/master/README.md
16. TechRepublic — Mac mini M6 cheat sheet. https://www.techrepublic.com/article/news-mac-mini-m6-cheat-sheet/
17. ModelFit — M6 Mac mini for local LLMs. https://modelfit.io/blog/m6-mac-mini-local-llm/
18. Mashable — "The best MacBook to buy in 2026: Comparing the Air, Neo, and Pro" (August 2026). https://mashable.com/tech/best-macbooks-2026-expert-reviewed
19. GPUsmith — NVIDIA B200 specs & procurement (July 2026). https://gpusmith.com/hardware/gpus/nvidia-b200
20. Thunder Compute — NVIDIA B300 pricing. https://www.thundercompute.com/blog/nvidia-b300-pricing
21. GPUPerHour — Cloud GPU pricing, 31 providers (October 2026). https://gpuperhour.com/
22. Shattered.io — H100/H200/B200 cloud GPU pricing 2026 (September 2026). https://shattered.io/h100-h200-b200-cloud-gpu-pricing-2026/
23. Popular AI — "The RTX PRO 6000 Blackwell for local AI: is 96GB worth $13,000?" (July 2026). https://www.popularai.org/p/rtx-pro-6000-blackwell-local-ai-96gb-vram
24. RunAIHome — RTX PRO 6000 Blackwell for local AI in 2026. https://www.runaihome.com/blog/rtx-pro-6000-blackwell-local-ai-2026/

Earlier audits referenced: [ARM_SOC_RTX_SPARK_UPDATE_2026.md](ARM_SOC_RTX_SPARK_UPDATE_2026.md) (RTX Spark, DGX Spark, Snapdragon, Apple M5 sources) and [PRESET_AND_INTEL_AUDIT_2026.md](PRESET_AND_INTEL_AUDIT_2026.md) (overhead defaults, Intel ARC).
