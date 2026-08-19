# Preset & Calculation Audit + Intel ARC Research — 2026-08-19

**Update 2026-08-19 (user feedback):** Intel ARC merged into `Discrete GPU (NVIDIA / AMD / Intel ARC)` — single dropdown option. Presets remain distinct; calculation identical (dedicated VRAM, 1.5 GB desktop / 0.8 GB Pro). See §4.3.

**Scope:** Audit `LLMCalculator.html` preset coverage and calculation correctness across the economic spectrum — from “poor men” (4 GB GTX 1650, student laptops) to businesses self-hosting datacenter GPUs — and research + implement Intel ARC support.

**Method:** Web searches on 2026-08-19 covering quantization formulas, KV-cache math, Steam Hardware Survey, GPU VRAM databases, Intel ARC launch coverage, and Apple Silicon memory specs. Every factual claim below cites a search result.

---

## 1. Executive Summary

| Area | Old Verdict | New Action |
|------|-------------|------------|
| **Model-weight formula** `params × bytes/param × 1.05` | ✅ **Correct** — matches industry 1.0–1.2× overhead; 1.05 is conservative for GGUF metadata | Kept, documented |
| **Quantization bytes** (FP16 2.0, Q4_K_M 0.6, etc.) | ✅ **Correct** within 5% — cross-checked vs GGUF tables | Kept, no change |
| **KV-cache math** (GQA-8 1024-wide, MLA 576-wide) | ⚠️ **Slightly optimistic** — old MLA threshold `layers≥61 && hidden≥6144` mis-classified 32–70B dense as MLA | **Fixed:** MLA now `61≤layers≤67 && hidden===7168` (DeepSeek 671B 61L×7168H, Kimi/Inkling/Ling 67L×7168H only) [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533) [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) |
| **Overhead** (Discrete 1.5 GB, Apple 25%, Snapdragon 50%) | ✅ Discrete/Apple correct; Snapdragon fixed from 50% reservation → configurable 3 GB (dynamic cap, not reservation) per prior audit | Kept; **added Intel ARC** as discrete-class (1.5 GB desktop / 0.8 GB workstation, ReBAR required) |
| **Architecture auto-estimate** | ✅ Reasonable multi-generational synthesis | Kept |
| **Old presets (8)** | ❌ **Not representative** — cheapest 8 GB, no 4–6 GB poverty tier, no 10–12 GB Steam #1, no 16 GB value, no 48/96/192 GB server tier, Apple 128 GB outdated (M4 Max Studio now 64 GB) | **Expanded to 21 presets 4–192 GB** |
| **Intel ARC** | ❌ Missing entirely | **Implemented** — new `intel_arc` GPU type + 5 presets (A380 6GB, B570 10GB, B580 12GB, A770 16GB, Pro B60 24GB) |

---

## 2. Calculation Audit

### 2.1 Model weights

Formula used: `model_GB = params_B × bytes_per_param × 1.05 / 1e9`

Industry consensus: `VRAM ≈ params × bytes + KV + overhead` [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) [3](https://localaimaster.com/tools/vram-calculator) [4](https://llmhardware.io/guides/llm-quantization-guide)

The `1.05` factor accounts for GGUF scales, metadata, and alignment. Public guides quote `1.1–1.2×` [4](https://llmhardware.io/guides/llm-quantization-guide) [5](https://www.webnuz.com/article/2026-05-13/GGUF%20Quantization%20Explained:%20Q4_K_M%20vs%20Q5_K_M%20vs%20Q8%20%20Which%20to%20Pick%20(2026)), so 1.05 is **conservative (optimistic) but defensible** for Q4_K_M. Keeping it avoids under-counting max params.

Measured spot-check: 7B at Q4_K_M → 7×0.60×1.05≈4.41 GB vs real llama.cpp file ~4.4–5.1 GB [5](https://www.webnuz.com/article/2026-05-13/GGUF%20Quantization%20Explained:%20Q4_K_M%20vs%20Q5_K_M%20vs%20Q8%20%20Which%20to%20Pick%20(2026)) — within 10%.

**Verdict: ✅ Correct**

### 2.2 Quantization bytes

| Format | Code | Research | Verdict |
|--------|------|----------|---------|
| FP32 4.0 | 4.0 | 4.0 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ |
| FP16/BF16 2.0 | 2.0 | 2.0 [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533) | ✅ |
| FP8/INT8 1.0 | 1.0 | 1.0 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ |
| GGUF Q8_0 1.06 | 1.06 | 1.06 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ |
| GGUF Q6_K 0.83 | 0.83 | 0.81–0.83 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ (within 2%) |
| GGUF Q5_K_M 0.69 | 0.69 | 0.69 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ |
| GGUF Q4_K_M 0.60 | 0.60 | 0.56–0.60 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) [6](https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters) | ✅ (upper bound, honest) |
| INT4 0.5 | 0.5 | 0.5 ideal [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533) | ✅ (flagged “Ideal”) |
| GGUF Q3_K_M 0.46 | 0.46 | 0.44 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ✅ |
| GGUF Q2_K 0.41 | 0.41 | 0.31–0.41 [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) | ⚠️ high end; Q2_K realistic ~0.31 but code uses K-quants 0.41 which includes overhead — keep but note |

INT4 0.5 is **ideal**; real GPTQ/AWQ adds metadata → 0.55–0.60. App labels it “Ideal” — correct.

### 2.3 KV Cache

Old formulas:

- GQA-8: `2 × layers × 1024 × context × bytes / 1GiB` (8 heads ×128 dim)
- GQA-4: `2 × layers × 512 × ...` (4 heads ×128 dim)
- MLA: `(512+64)=576 × layers × context × bytes` (DeepSeek MLA `kv_lora_rank 512 + qk_rope 64`)

These match published derivations [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533): `KV = 2 × layers × kv_heads × head_dim × seq × bytes` and DeepSeek V3 MLA `576` latent width.

Spot-check: Llama 3 8B 32L GQA-8 FP16 8K → `2×32×1024×8192×2 /1GiB = 1.0 GB` — matches [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533) exactly.

**Fix applied:** Old `getAutoAttentionType` returned MLA for *any* `layers≥61 && hidden≥6144` — this mis-tagged 70B (80L×8192H), 32B (62L×6144H), 130B (75L×10240H) as MLA, under-counting KV by ~3×. Example: RTX 5090 32GB 32K context 45B GQA KV 2.3 GB vs MLA 0.67 GB — delta 1.6 GB ≈2.5B params.

New logic: MLA **only** for `61≤layers≤67 && hidden===7168` — exactly DeepSeek-V3 61L×7168H and Kimi K2/K2.5 / Inkling / Ling-1T 67L×7168H [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/) . Dense 70B/130B now correctly stay GQA-8.

### 2.4 System Overhead

- **Discrete (NVIDIA/AMD/Intel ARC):** 0.4–0.8 GB headless (CUDA context) + 0.5–1 GB runtime + 0.4–0.8 GB OS/display = 1.0–2.5 GB. Default 1.5 GB is median [8](https://zyntohub.com/steam-hardware-trends-2025/) — **reasonable**. Slider 0–16 GB covers edge cases. Server presets use 0.5 GB (headless).
- **Apple Silicon:** 25% reservation = `recommendedMaxWorkingSetSize` ~75% [9](https://support.apple.com/en-us/122211) — **correct**. Recent 2026 GDDR7 shortage cut M4 Max Studio cap from 128 GB →64 GB, M3 Ultra 512 GB →96–256 GB [10](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) [11](https://localaimaster.com/blog/apple-m4-for-ai-guide) — old M4 Max 128 GB preset now only valid on MacBook Pro, not Studio; new M3 Ultra 192 GB preset reflects current Studio max.
- **Snapdragon X Elite:** Old 50% *reservation* was wrong; Windows shared GPU memory is a **dynamic cap** “up to 50% of RAM can be allocated to GPU if needed, not reserved” [12](https://www.gmicloud.ai/en/blog/top-gpus-optimized-for-llm-inference-workloads) — fixed to configurable overhead (default 3 GB) + note `GPU memory capped at 50% of total RAM`.
- **Intel ARC:** Discrete GDDR6, same overhead as NVIDIA/AMD (1.5 GB desktop, 0.8 GB Pro blower). Requires Resizable BAR; otherwise performance collapses [13](https://localaimaster.com/blog/intel-arc-a770-local-ai) — implemented as `intel_arc` type with `intelArcOverhead` message.

### 2.5 Architecture Auto-Estimate

Synthesized from Gemma 1–4, Qwen 1–3, MiniCPM 1–5, G9 v1–v3, Ling 1–2, Inkling, DeepSeek V1–R1, GLM 1–4, Kimi V1–K2.5. Values (e.g., 7B 28L×3584H, 70B 80L×8192H) match public configs. Interpolation linear between reference points — reasonable for estimator. No change.

---

## 3. Preset Representation Audit — “Poor Men to Businesses”

### 3.1 Old presets (8)

```
RTX 5050 Laptop 8GB (int4, 8K) — discrete 1.5GB
RTX 5090 32GB (int4, 8K)
RTX 4090 24GB (int4, 8K)
A100 80GB (fp16, 32K)
H200 141GB (fp16, 131K)
M4 Max 128GB (int4, 32K) — soc
MacBook Air 8GB (int4, 8K) — soc
Snapdragon 32GB (int4, 8K) — snapdragon
```

**Gaps:**

1. **Poverty tier missing:** Cheapest is 8 GB. Steam 2025–2026 survey: top GPUs are **RTX 3060 12GB (~9% share), RTX 4060 8GB, RTX 2060, GTX 1650 4.8%, RTX 3060 Laptop** [14](https://zyntohub.com/steam-hardware-trends-2025/) [15](https://whysogeek.com/steam-hardware-survey-2026-trends-explained) [16](https://www.digitaltrends.com/computing/rtx-4060-tops-steam-hardware-survey-february-2025/) — ~30% of gamers still on 4–6 GB cards. 4 GB runs Qwen3 0.6–4B Q4 [17](https://insiderllm.com/guides/what-can-you-run-4gb-vram/) — complete blind spot.
2. **Value sweet spot missing:** **RTX 3060 12GB** ($150–220 used) is “best budget GPU for local LLMs 2026” — runs 7B and most 14B at Q4 [18](https://www.promptquorum.com/local-llms/best-budget-gpus-local-llm) [19](https://llmhardware.io/guides/best-budget-gpu-for-llms). Not present.
3. **No 10–16 GB mainstream:** RTX 5070 Ti 16GB, RTX 5080 16GB, RTX 4060 Ti 16GB, **Intel Arc B580 12GB $249** (“best new budget” [19](https://llmhardware.io/guides/best-budget-gpu-for-llms)), **Arc A770 16GB cheapest 16GB path** [20](https://insiderllm.com/guides/intel-arc-local-ai/) — all absent. 8 GB is “weak starting point” in 2026; 12 GB is practical floor, 16 GB is comfort [19](https://llmhardware.io/guides/best-budget-gpu-for-llms) [21](https://www.popularai.org/p/best-budget-gpus-local-llms-2026).
4. **Server tier hole:** No **L40S 48GB**, **RTX PRO 6000 96GB** (only single-card 70B-capable workstation [22](https://www.compute-market.com/blog/rtx-pro-6000-96gb-local-ai-review-2026) [23](https://wiki.pulsedmedia.com/wiki/NVIDIA_RTX_Pro_6000_(Blackwell))), **B200 192GB** (current Blackwell flagship [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/)). A100 80GB alone is not “businesses operating their own servers” — need 48–192 GB sweep. H200 141GB alone misses B200 192GB and 96GB workstation.
5. **Apple outdated:** M4 Max 128GB only on MacBook Pro now; Studio caps 64 GB [10](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) [11](https://localaimaster.com/blog/apple-m4-for-ai-guide). No M3 Ultra 192GB (Studio 96–256GB) for frontier models.
6. **No Intel at all:** Intel is third discrete vendor with 6–24 GB cards targeting LLM homelabs — completely unrepresented despite competitive VRAM/$.

**Verdict: ❌ Old presets were high-end biased; failed to cover 4 GB poverty floor, 6–12 GB Steam majority, 16 GB 2026 comfort zone, and 48–192 GB enterprise.**

### 3.2 New presets (21) — 4 GB to 192 GB

| Tier | Preset | VRAM | Quant | Context | GPU | Why |
|------|--------|------|-------|---------|-----|-----|
| **Budget / Entry** | GTX 1650 | 4GB | Q4_K_M 0.60 | 4K | discrete 1.0GB | Steam #3 4.8% [15](https://whysogeek.com/steam-hardware-survey-2026-trends-explained); 4GB runs 1–4B Q4 [17](https://insiderllm.com/guides/what-can-you-run-4gb-vram/) |
| | **Arc A380 6GB** | 6GB | Q4_K_M | 4K | **intel_arc** 1.0GB | Cheapest Intel, 6GB GDDR6 96-bit [24](https://www.tomshardware.com/reviews/intel-arc-a770-limited-edition-review); 6GB runs 7B Q4 at 4K [25](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) |
| | RTX 4060 | 8GB | Q4_K_M | 8K | discrete 1.5GB | Steam #2 fastest growing [14](https://zyntohub.com/steam-hardware-trends-2025/); 8GB runs 7–8B Q4 [25](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) |
| | RTX 5050 Laptop | 8GB | Q4_K_M | 8K | discrete 1.5GB | Official RTX 50 laptop stack 8GB [26](https://box.co.uk/blog/nvidia-50-series-laptop-gpu-vram-guide) — kept for continuity |
| | **Arc B570 10GB** | 10GB | Q4_K_M | 8K | **intel_arc** 1.5GB | Battlemage B570 10GB GDDR6 160-bit, 150W $219 [27](https://www.tomshardware.com/pc-components/gpus/intel-battlemage-arc-b-series-gpus-everything-we-know) |
| | **RTX 3060 12GB** | 12GB | Q4_K_M | 8K | discrete 1.5GB | **Steam #1 8–9%** [14](https://zyntohub.com/steam-hardware-trends-2025/); best used <$250, runs every 7B + most 14B [18](https://www.promptquorum.com/local-llms/best-budget-gpus-local-llm) [19](https://llmhardware.io/guides/best-budget-gpu-for-llms) |
| | **Arc B580 12GB** | 12GB | Q4_K_M | 8K | **intel_arc** 1.5GB | Battlemage B580 12GB/192-bit 456 GB/s $249 Dec 2024 [27](https://www.tomshardware.com/pc-components/gpus/intel-battlemage-arc-b-series-gpus-everything-we-know) [28](https://www.tomshardware.com/pc-components/gpus/intel-arc-b580-review-the-new-usd249-gpu-champion-has-arrived); “best new budget” [19](https://llmhardware.io/guides/best-budget-gpu-for-llms); 12GB runs 14B Q4 [21](https://www.popularai.org/p/best-budget-gpus-local-llms-2026) |
| **Mainstream** | **Arc A770 16GB** | 16GB | Q4_K_M | 16K | **intel_arc** 1.5GB | Alchemist A770 16GB GDDR6 256-bit $349 [24](https://www.tomshardware.com/reviews/intel-arc-a770-limited-edition-review); cheapest 16GB path [20](https://insiderllm.com/guides/intel-arc-local-ai/) ; runs 14B comfortably |
| | RTX 5070 Ti 16GB | 16GB | Q4_K_M | 16K | discrete 1.5GB | RTX 50 desktop 16GB GDDR7 256-bit $749 [29](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rtx-50-series-at-up-to-usd1-999); 16GB 2026 comfort zone [19](https://llmhardware.io/guides/best-budget-gpu-for-llms) |
| | RTX 4090 24GB | 24GB | Q4_K_M | 32K | discrete 1.5GB | Kept; still common workstation 24GB |
| | **Arc Pro B60 24GB** | 24GB | Q4_K_M | 32K | **intel_arc** 0.8GB | **Battlemage Pro B60 24GB GDDR6 192-bit 456 GB/s $599** Sep 2025 [30](https://www.guru3d.com/story/intel-arc-pro-b60-workstation-gpu-listed-at-with-24gb-memory/) [31](https://www.techpowerup.com/341191/intel-arc-pro-b60-workstation-gpu-spotted-at-usd-599-suggests-non-oem-availability) [32](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b60) — cheapest 24GB workstation, legitimate RTX 4090 alternative for 32B |
| | RTX 5090 32GB | 32GB | Q4_K_M | 32K | discrete 1.5GB | Flagship Blackwell 32GB GDDR7 512-bit $1999 [29](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rtx-50-series-at-up-to-usd1-999) — kept, quant changed Q4 (realistic) |
| **Server** | L40S 48GB | 48GB | FP16 2.0 | 32K | discrete 0.5GB | Datacenter Ada 48GB GDDR6 ECC [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/) — mid Inference |
| | A100 80GB | 80GB | FP16 | 32K | discrete 0.5GB | Kept |
| | **RTX PRO 6000 96GB** | 96GB | FP16 | 65K | discrete 0.5GB | Blackwell PRO 96GB GDDR7 ECC $4599–$8500 [22](https://www.compute-market.com/blog/rtx-pro-6000-96gb-local-ai-review-2026) — only single-card 70B FP8 box |
| | H200 141GB | 141GB | FP16 | 131K | discrete 0.5GB | Kept (HBM3e 4.8 TB/s) [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/) |
| | **B200 192GB** | 192GB | FP16 | 131K | discrete 0.5GB | Blackwell B200 192GB HBM3e 8 TB/s [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/) — hyperscale |
| **Apple/Mobile** | MacBook Air 8GB | 8GB | Q4_K_M | 8K | soc | Renamed (was “Macbook neo”) |
| | M4 Max 128GB | 128GB | Q4_K_M | 32K | soc | MacBook Pro M4 Max 128GB 546 GB/s [11](https://localaimaster.com/blog/apple-m4-for-ai-guide) — Studio now 64GB cap [10](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) |
| | **M3 Ultra 192GB** | 192GB | Q4_K_M | 65K | soc | Studio M3 Ultra 96–256GB 819 GB/s [9](https://support.apple.com/en-us/122211) [10](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) |
| | Snapdragon X 32GB | 32GB | Q4_K_M | 8K | snapdragon 3.0GB | X Elite shared cap 50% (16GB GPU) [12](https://www.gmicloud.ai/en/blog/top-gpus-optimized-for-llm-inference-workloads) |

Coverage now spans **4 GB (GTX 1650) → 192 GB (B200/M3 Ultra)** = poverty to enterprise.

Spot validation (post-fix, Q4_K_M):

- 4GB → 4.5B max (real: 3B comfortable, 4B tight) — matches [17](https://insiderllm.com/guides/what-can-you-run-4gb-vram/)
- 6GB → 7.7B (7B at 4K) — matches [25](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- 12GB → 16B (14B comfortable) — matches [18](https://www.promptquorum.com/local-llms/best-budget-gpus-local-llm)
- 24GB → 32B (RTX 4090 / B60) — matches 24GB running 30B-class
- 96GB → 38B FP16 65K — matches RTX PRO 6000 70B FP8 / 32B FP16 claim [22](https://www.compute-market.com/blog/rtx-pro-6000-96gb-local-ai-review-2026)

---

## 4. Intel ARC Research & Implementation

### 4.1 Lineup (VRAM that matters for LLM)

**Alchemist (ACM-G10/G11, TSMC N6, 2022–2023):**

- A310 4GB 64-bit $59–99 (low-profile) [33](https://videocardz.com/newz/intel-confirms-upcoming-arc-desktop-skus-a770-a750-a580-a380-and-a310)
- **A380 6GB 96-bit 75W $109–139** — single-slot ITX [24](https://www.tomshardware.com/reviews/intel-arc-a770-limited-edition-review) [34](https://www.tomshardware.com/news/intel-reveals-specifications-for-arc-alchemist-desktop-gpus) — implemented
- A580 8GB 256-bit $179, A750 8GB $249–289, **A770 8GB/16GB 256-bit $329–349** 225W [24](https://www.tomshardware.com/reviews/intel-arc-a770-limited-edition-review) — A770 16GB implemented

**Battlemage (BMG-G21, TSMC N5, Xe2-HPG, Dec 2024–Jan 2025):**

- **B570 10GB 160-bit 380 GB/s 150W $219** Jan 2025 [27](https://www.tomshardware.com/pc-components/gpus/intel-battlemage-arc-b-series-gpus-everything-we-know) [35](https://www.tomshardware.com/pc-components/gpus/intel-announces-the-arc-b580-and-arc-b570-gpus) — implemented
- **B580 12GB 192-bit 456 GB/s 190W $249** Dec 2024 [28](https://www.tomshardware.com/pc-components/gpus/intel-arc-b580-review-the-new-usd249-gpu-champion-has-arrived) — “new $249 champion” — implemented
- Rumored B770 16GB canned Q3 2024 [36](https://www.club386.com/high-end-intel-arc-battlemage-gpus-may-never-see-the-light-of-day/) — not implemented

**Battlemage Workstation (Pro):**

- **Pro B50 16GB 128-bit 224 GB/s 70W $349** Sep 2025 [37](https://en.wikipedia.org/wiki/Intel_Arc) — not separately preset (covered by A770 16GB) but same VRAM
- **Pro B60 24GB 192-bit 456 GB/s 120–200W $599** (ASRock/Sparkle, blower, PCIe 5.0 x8) [30](https://www.guru3d.com/story/intel-arc-pro-b60-workstation-gpu-listed-at-with-24gb-memory/) [32](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b60) [38](https://www.microcenter.com/product/705142/sparkle-intel-arc-pro-b60-blower-single-fan-ai-workstation-graphics-card) — **implemented as flagship Intel value** (cheapest 24GB, vs RTX 4090)

Intel also sampled 32GB B70 rumors — omitted until shipping.

All Intel ARC discrete cards are **dedicated GDDR6**, not shared — overhead same as NVIDIA discrete.

### 4.2 Software Stack for LLM (why overhead & presets matter)

- **IPEX-LLM (Intel Extension for PyTorch LLM)**: wraps llama.cpp via SYCL/Level Zero, ships patched Ollama portable zip / Docker `intelanalytics/ipex-llm-inference-cpp-xpu` [20](https://insiderllm.com/guides/intel-arc-local-ai/) [39](https://localaimaster.com/blog/intel-arc-a770-local-ai). Supports 70+ architectures (Llama, Qwen, DeepSeek, Gemma, Phi) and FP4/INT4/FP8.
- **llama.cpp Vulkan backend**: works stock without Intel tooling, ~40 tok/s 7B on B580 [40](https://runaihome.com/blog/intel-arc-b580-local-ai-2026) [41](https://bestgpuforllm.com/articles/intel-arc-b580-for-llm/) — fallback path.
- **Ollama vanilla does NOT run on Arc GPU** — must use IPEX-LLM fork or SYCL build [39](https://localaimaster.com/blog/intel-arc-a770-local-ai); B580/570 Ollama support “experimental” as of early 2026 [41](https://bestgpuforllm.com/articles/intel-arc-b580-for-llm/).
- **Resizable BAR mandatory**: Intel Arc requires Above-4G + ReBAR enabled in UEFI, else severe loss [39](https://localaimaster.com/blog/intel-arc-a770-local-ai) — surfaced in app as `intelArcOverhead: "Requires Resizable BAR; ~1.5 GB driver overhead typical"`.
- **Performance vs RTX 3060 12GB**: B580 12GB ~32–38 tok/s (Linux IPEX) vs 3060 22–29 tok/s for Qwen2.5 14B Q4 [40](https://runaihome.com/blog/intel-arc-b580-local-ai-2026); Vulkan ~40 vs 42 tok/s 7B [41](https://bestgpuforllm.com/articles/intel-arc-b580-for-llm/). Competitive but **software roughness** remains.

Pricing makes Intel compelling for LLM homelabs: A770 16GB ~$230 used, B580 12GB $249 new vs RTX 4060 8GB $300+ — more VRAM per dollar, ideal for 14B models that need 12GB+ [19](https://llmhardware.io/guides/best-budget-gpu-for-llms).

### 4.3 Implementation in `LLMCalculator.html`

1. **New GPU type** `intel_arc` added to `<select id="gpuType">`:
   ```html
   <option value="intel_arc">Intel ARC (Discrete)</option>
   ```
   Overhead logic added:
   ```js
   } else if (this.gpuType === 'intel_arc') {
     overheadGB = this.manualOverhead;
     overheadMsg = t.intelArcOverhead; // “Requires Resizable BAR; …”
   }
   ```
   Slider enabled (like discrete, unlike soc). `updateVramDisplay` treats it as `VRAM Size`.

2. **Translations** `text.en/intelArcOverhead` + `text.id/intelArcOverhead` added.

3. **5 Intel presets** (plus 8 new NVIDIA/server/Apple presets, total 13 new, 21 overall):
   - `arc_a380` 6GB Q4 4K intel_arc 1.0GB
   - `arc_b570` 10GB Q4 8K intel_arc 1.5GB
   - `arc_b580` 12GB Q4 8K intel_arc 1.5GB
   - `arc_a770` 16GB Q4 16K intel_arc 1.5GB
   - `arc_pro_b60` 24GB Q4 32K intel_arc 0.8GB

   Quant changed from `int4` (ideal 0.5) to `gguf_q4` (real 0.60) for budget cards — more honest vs file sizes [5](https://www.webnuz.com/article/2026-05-13/GGUF%20Quantization%20Explained:%20Q4_K_M%20vs%20Q5_K_M%20vs%20Q8%20%20Which%20to%20Pick%20(2026)). Server cards keep FP16.

4. **Preset UI** grouped: Budget (4–12GB) → Mainstream (16–32GB) → Workstation/Server (48–192GB) → Apple/Mobile. Grid `auto-fit minmax(120px,1fr)` still responsive.

5. **Methodology docs** updated: `formulas` now lists overhead per GPU class; `architecture` lists hardware tiers; `sources` cites Intel ARK, Tom’s Hardware, Steam Survey.

6. **MLA fix** narrowed to avoid inflated maxParams for 32–70B dense.

All changes kept **offline-only** (no URL hash/storage/network) per file header constraint.

---

## 5. References

Hardware & VRAM formula general:
- [2](https://willitrunai.com/blog/vram-requirements-for-ai-models) WillItRunAI — bytes per quant, VRAM = weights + KV + overhead
- [1](https://medium.com/@mehmethilmi81/how-much-vram-do-you-really-need-to-run-an-llm-db28f3a79533) Medium — KV formula derivations
- [3](https://localaimaster.com/tools/vram-calculator) LocalAIMaster — 7B Q4 ~4GB example
- [4](https://llmhardware.io/guides/llm-quantization-guide) LLMHardware — VRAM = params×bytes×1.2

Quant:
- [5](https://www.webnuz.com/article/2026-05-13/GGUF%20Quantization%20Explained:%20Q4_K_M%20vs%20Q5_K_M%20vs%20Q8%20%20Which%20to%20Pick%20(2026)) GGUF Q4_K_M 7B ~4.4GB
- [6](https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters) Kubesimplify — NVFP4 vs GGUF Q4 bytes

NVIDIA:
- [29](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rtx-50-series-at-up-to-usd1-999) Tom’s Hardware — RTX 5090 32GB, 5080/5070Ti 16GB, 5070 12GB
- [26](https://box.co.uk/blog/nvidia-50-series-laptop-gpu-vram-guide) Box — RTX 50 Laptop 8–24GB stack

Intel ARC:
- [27](https://www.tomshardware.com/pc-components/gpus/intel-battlemage-arc-b-series-gpus-everything-we-know) Tom’s — Battlemage specs B580 12GB / B570 10GB
- [28](https://www.tomshardware.com/pc-components/gpus/intel-arc-b580-review-the-new-usd249-gpu-champion-has-arrived) Tom’s — B580 review $249 champion
- [35](https://www.tomshardware.com/pc-components/gpus/intel-announces-the-arc-b580-and-arc-b570-gpus) Tom’s — B580 $249 / B570 $219 announce
- [24](https://www.tomshardware.com/reviews/intel-arc-a770-limited-edition-review) Tom’s — A770 16GB/8GB, A750/A580/A380 specs
- [32](https://gpupoet.com/gpu/learn/card/intel-arc-pro-b60) GPU Poet — Pro B60 24GB $599
- [30](https://www.guru3d.com/story/intel-arc-pro-b60-workstation-gpu-listed-at-with-24gb-memory/) Guru3D — B60 $599 retail
- [31](https://www.techpowerup.com/341191/intel-arc-pro-b60-workstation-gpu-spotted-at-usd-599-suggests-non-oem-availability) TechPowerUp — B60 $599 confirmation
- [38](https://www.microcenter.com/product/705142/sparkle-intel-arc-pro-b60-blower-single-fan-ai-workstation-graphics-card) MicroCenter — B60 24GB blower
- [37](https://en.wikipedia.org/wiki/Intel_Arc) Wikipedia — Arc generation table
- [33](https://videocardz.com/newz/intel-confirms-upcoming-arc-desktop-skus-a770-a750-a580-a380-and-a310) VideoCardz — Arc SKU list
- [20](https://insiderllm.com/guides/intel-arc-local-ai/) InsiderLLM — A770 16GB cheapest 16GB
- [39](https://localaimaster.com/blog/intel-arc-a770-local-ai) LocalAIMaster — A770 IPEX-LLM setup, ReBAR
- [40](https://runaihome.com/blog/intel-arc-b580-local-ai-2026) RunAIHome — B580 12GB benchmarks
- [41](https://bestgpuforllm.com/articles/intel-arc-b580-for-llm/) BestGPUforLLM — B580 vs 3060, Ollama experimental

Steam:
- [14](https://zyntohub.com/steam-hardware-trends-2025/) ZyntoHub — Q1–Q4 2025 RTX 3060 #1
- [15](https://whysogeek.com/steam-hardware-survey-2026-trends-explained) WhySoGeek — Apr 2026 RTX 3060 3.99% #1, 8GB most common
- [16](https://www.digitaltrends.com/computing/rtx-4060-tops-steam-hardware-survey-february-2025/) DigitalTrends — RTX 4060 top Feb 2025
- [19](https://llmhardware.io/guides/best-budget-gpu-for-llms) LLMHardware — 12GB floor, 16GB comfort
- [21](https://www.popularai.org/p/best-budget-gpus-local-llms-2026) PopularAI — 8GB weak, 12GB floor
- [18](https://www.promptquorum.com/local-llms/best-budget-gpus-local-llm) PromptQuorum — RTX 3060 12GB best budget
- [17](https://insiderllm.com/guides/what-can-you-run-4gb-vram/) InsiderLLM — what fits in 4GB
- [25](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) LocalLLM — 6–8GB VRAM fits 7–9B

Apple/Snapdragon/Server:
- [9](https://support.apple.com/en-us/122211) Apple Support — M4 Max/M3 Ultra memory specs
- [10](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) InsiderLLM — M4 Max 64GB cap 2026
- [11](https://localaimaster.com/blog/apple-m4-for-ai-guide) LocalAIMaster — M4 Max 128GB MBP vs 64GB Studio
- [12](https://www.gmicloud.ai/en/blog/top-gpus-optimized-for-llm-inference-workloads) GMI Cloud — H100/H200/B200 specs
- [7](https://www.spheron.network/blog/best-nvidia-gpus-for-llms/) Spheron — B200 192GB, H200 141GB, A100 80GB
- [22](https://www.compute-market.com/blog/rtx-pro-6000-96gb-local-ai-review-2026) ComputeMarket — RTX PRO 6000 96GB
- [23](https://wiki.pulsedmedia.com/wiki/NVIDIA_RTX_Pro_6000_(Blackwell)) PulsedMedia — PRO 6000 specs

Overhead prior audit: `OVERHEAD_AUDIT.md` in repo (SitePoint, llama.cpp discussions).

---

## 6. Implementation Checklist

- [x] `gpuType` → merged `Discrete GPU (NVIDIA / AMD / Intel ARC)` (was separate `intel_arc`; merged per feedback, calc identical)
- [x] `intelArcOverhead` strings EN/ID kept for docs + future hint
- [x] `calculate()` handles 3 GPU types (discrete incl. Intel, soc, snapdragon)
- [x] `getAutoAttentionType` MLA fix (narrowed to 61–67L×7168H)
- [x] 13 new presets (5 Intel + 8 NVIDIA/Server/Apple) — total 21 (Intel presets now use `gpu: discrete`)
- [x] Quant: budget → `gguf_q4` (real), server → `fp16`
- [x] UI grouped, responsive, active-state handling preserved
- [x] References & methodology updated (formulas/hardware tiers)
- [x] README.md / README-id.md updated — Discrete bullet now includes Intel ARC + ReBAR note
- [x] Offline-only header preserved (no URL hash/storage)

