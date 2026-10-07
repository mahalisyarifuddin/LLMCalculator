# LLMCalculator Audit Summary

This document consolidates and summarizes the 15 audit and update reports written for `LLMCalculator.html` between August and October 2026. It records what the calculator assumes, what was verified or corrected, and why. Bahasa Indonesia: [AUDIT-id.md](AUDIT-id.md).

The full original reports, with every table and source (about 220 links), remain in git history. `git log --oneline -- <FILE>` lists the commits that touched a report, and `git show <commit>:<FILE>` prints it. The last commit that contains all 15 is `228ba94`. The list is in the [appendix](#appendix-original-reports).

## Contents

1. [Scope and invariants](#1-scope-and-invariants)
2. [Memory math](#2-memory-math)
3. [System overhead by GPU type](#3-system-overhead-by-gpu-type)
4. [Auto-estimate architecture anchors](#4-auto-estimate-architecture-anchors)
5. [Context window and agent harnesses](#5-context-window-and-agent-harnesses)
6. [Hardware presets](#6-hardware-presets)
7. [Verification practice](#7-verification-practice)
8. [Key sources](#8-key-sources)
- [Appendix: original reports](#appendix-original-reports)

---

## 1. Scope and invariants

The calculator estimates the **largest plausible open-weight model** that fits in one device's memory, together with **one active KV-cache context**. The model's shape (layers, hidden width, attention type) is interpolated from real checkpoints (§4). It is an estimate, not an exact checkpoint validator or an inference-server planner.

**Not modeled:**
- CPU offload, parallel sequences, multi-user serving and concurrent subagents.
- Runtime workspaces beyond the configurable overhead allowance.
- Training context limits and long-context quality.

**Invariants kept by every audit:**
- **100% offline single file:** no URL or hash state, no storage, no network calls, no permalinks (file-header directive). State lives in memory only.
- **A GPU type is added only when its memory math differs.** Intel ARC was merged into "Discrete GPU" after user feedback, and AMD Ryzen AI Max sits under the existing fixed-reserve type.
- **Bilingual UI** (English and Bahasa Indonesia) with matching text keys.
- **Units:** the UI says "GB", but the math uses binary GiB throughout.

---

## 2. Memory math

### 2.1 Model weights and quantization

```text
weight GiB = params (billions) × 10^9 × bytes per param × 1.05 ÷ 2^30
```

- **The 1.05 factor** covers metadata, quantization scales and alignment. Public guides quote 1.1–1.2×, so 1.05 is mildly optimistic but defensible. Spot check: 7B at Q4_K_M ≈ 4.4 GB, against real llama.cpp files of 4.4–5.1 GB.
- **Unit fix (August 2026):** weights used decimal GB while KV used binary GiB. Both are binary now, so one billion bytes = 0.931 GiB.

| Format | Bytes/param | Note |
| --- | --- | --- |
| FP32 | 4.0 | |
| FP16 / BF16 | 2.0 | Default for datacenter presets |
| FP8 / INT8 | 1.0 | |
| GGUF Q8_0 | 1.06 | |
| GGUF Q6_K | 0.83 | |
| GGUF Q5_K_M | 0.69 | |
| GGUF Q4_K_M / GPTQ 4-bit | 0.60 | Upper end of 0.56–0.60, honest against real file sizes; default for consumer presets |
| NVFP4 | 0.56 | FP4 + FP8 scale per 16 values + tensor scale ≈ 4.5 bits; Blackwell-native (Poolside Laguna, Step-3.7-Flash) |
| MXFP4 | 0.53 | FP4 + 8-bit scale per 32 values ≈ 4.25 bits; native to gpt-oss (117B → 62 GB vs the official 60.8 GB), Kimi K3 and MiMo-V2.5-Pro experts |
| INT4 / FP4 | 0.5 | Ideal; real GPTQ/AWQ files run 0.55–0.60 |
| GGUF Q3_K_M | 0.46 | |
| GGUF Q2_K | 0.41 | High end (realistic ≈ 0.31) |
| FP2 | 0.25 | Ideal |

The KV cache has its own precision setting: same as model, FP16, FP8, INT8 or INT4.

### 2.2 KV cache

Standard attention (MHA, GQA, MQA), for one sequence:

```text
KV bytes = 2 (K and V) × layers × context tokens × KV heads × head dim × bytes per element
```

Spot checks at 8,192 tokens in FP16: Llama-class 8B = 1.000 GiB, Llama-class 70B = 2.500 GiB, a 64-layer 12,288-wide MHA model (Command R+ class) = 24.000 GiB.

Newer architectures store far less, so each family's anchor carries an exact cache shape:

| Cache shape | Families | What is stored per token |
| --- | --- | --- |
| MLA | DeepSeek V2/V3/R1, Kimi K2/K3, Mistral Large 3, GLM-4.7-Flash | 512 latent + 64 RoPE = 576 elements per layer (Mistral Small 4: 256 + 64 = 320) |
| MLA + DSA | GLM-5.x, Hy4-preview | MLA cache on every layer + a 128-d sparse indexer on every 4th layer |
| CSA / HCA | DeepSeek V4-Flash / V4-Pro | Shared K=V 512-d; CSA keeps SWA-128 + context/4 + a 128-d indexer, HCA keeps SWA-128 + context/128 |
| CED + CSA2 | DeepSeek V4.1-Flash | 44.5 elements per layer, calibrated to the card's 890 bytes/token at FP4 |
| Hybrid sliding window | MiMo V2/V2.5/Pro (192 K + 128 V), Step-3.5/3.7 (12 full + 33 SWA-512), Muse Glimmer (13 + 39 SWA-2048), Granite SWASH | Full layers keep the whole context; window layers keep min(context, window) |
| Gemma 3 / Gemma 4 | Gemma 3 12B/27B, Gemma 4 26B-A4B/31B | 5 SWA-1024 layers per global layer; Gemma 4 global layers use one unified K=V 512-d vector |
| Gated DeltaNet hybrid | Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Max | Only 1 layer in 4 (Gated Attention) caches tokens; DeltaNet layers keep a fixed-size state |
| Qwen Sparse Attention | Qwen3.8-Flash-Next | DeltaNet hybrid + a 128-d index over context/4 |
| Ling hybrid linear | Ling 2.5/2.6 (1 MLA layer per 8), Ling 3.0 (1 per 6) | One gated-MLA layer per group × 576 |
| MFA | Step-3 | One shared 256-d K and one 256-d V per layer |
| CLA-2 | Hunyuan-Large | GQA-8 width on ceil(layers / 2) shared layers |
| Compact GQA / MQA | gpt-oss (8 KV × 64-d), Granite (64-d heads; Code models MQA-1), Trinity Nano (GQA-2) | KV heads × head dim |

*Why it matters:* at 32K context a 397B Qwen3.5 model stores less KV than a 35B GQA-8 model, while a 357B GLM-4.6 stores about 12× more than a similarly sized Qwen3.5.

**Conservative choices kept on purpose:**
- The sliding-window layers of gpt-oss, Command A/A+, North, Laguna XS/S and Inkling are costed at full context.
- MiniMax Text-01/M1 lightning attention is costed as GQA-8.
- Sparse attention (MiniMax MSA, GLM DSA) cuts compute, not the stored cache, so the full cache is still counted.

### 2.3 Max-fit solver and units

The old solver alternated between parameter count and architecture only five times, so it could return a mismatched pair. The replacement:
1. computes the weight-only upper bound;
2. scans downward across the interpolated curve, because attention can jump at anchors and memory is not monotonic;
3. refines with binary search;
4. recomputes weights and KV from the same final pair.

Advanced overrides (fixed layers and hidden width) are solved directly. The per-token metric is labelled **MiB/token**, because it is binary.

---

## 3. System overhead by GPU type

| GPU type | Model | Defaults and evidence |
| --- | --- | --- |
| **Discrete GPU (NVIDIA / AMD / Intel ARC)** | Fixed deduction, slider 0–16 GB | 1.5 GB on a desktop (CUDA context 0.4–0.8 + runtime 0.5–1 + display 0.4–0.8 GB); 0.5 GB on headless servers. Intel ARC uses dedicated GDDR6, so the math is the same. It needs Resizable BAR and runs through IPEX-LLM or llama.cpp Vulkan/SYCL; vanilla Ollama does not run on Arc. |
| **Apple Silicon (M/A-series)** | 25% reserve, slider locked | Metal's `recommendedMaxWorkingSetSize` is ≈75% of RAM (≈⅔ on ≤32 GB Macs with older macOS). `sudo sysctl iogpu.wired_limit_mb` raises it at the cost of the OS reserve. |
| **NVIDIA RTX Spark (Windows on Arm)** | NVIDIA's published GPU-budget rule | Budget = carveout C + clamp((M − C) − 16 GB, 50%, 80% of (M − C)). The rest is CPU-only memory the GPU can never use. The slider sets desktop/driver headroom (default 1.5 GB). Calculator guard: C ≤ M − 16. With no carveout: 24 GB → 12, 32 → 16, 64 → 48, 96 → 76.8, 128 → 102.4 GB (up to 112 GB with C ≥ 48). |
| **Other Unified Memory (DGX Spark · Ryzen AI Max · Snapdragon)** | Fixed reserve, default 3 GB | **DGX Spark** on DGX OS: 12 GB headless (≈116 of 119 GiB stay available), 16 GB with the desktop. **Ryzen AI Max+ 128 GB:** 16 GB on Linux, where GTT lets the iGPU reach ~110–117 GB; on Windows the iGPU is capped at the 96 GB VGM carve-out, so model it as a Discrete GPU with 96 GB. **Snapdragon X/X2:** ~3 GB for the CPU/NPU path; Windows caps Adreno GPU offload at 50% of RAM. |

**History:**
- **Snapdragon (August 2026):** the old 50% "reservation" was wrong. Windows' 50% is a dynamic cap on GPU memory, not a reservation, so a 32 GB laptop running CPU inference was underestimated by ~12 GB. Replaced by a configurable 3 GB reserve.
- **Intel ARC (August 2026):** added as an `intel_arc` type, then merged into Discrete GPU after user feedback because the math is identical.
- **October 2026:** `snapdragon` → `arm_uma` → `uma`, with the same fixed-reserve math. RTX Spark became the only type with new math.
- **DGX Spark** reserve corrected from 14 to 12 GB based on headless measurements.

---

## 4. Auto-estimate architecture anchors

**How it works:**
- `referenceConfigs.auto` holds **98 strictly sorted, unique anchors from 0.35B to 2.8T**. Each anchor is size, layers, hidden width and an optional attention type.
- Layers and hidden width are interpolated linearly between anchors.
- The attention type comes from the nearest anchor: 58 anchors carry an explicit override across 28 cache shapes. Otherwise heuristics apply:
  - hidden ≤ 1,536, or 2,048 with ≥ 49 layers → GQA-4;
  - hidden 2,880 → GQA-8 × 64-d;
  - hidden 7,168 with ≥ 61 layers → MLA;
  - everything else → GQA-8.

**Method:**
- Prefer official `config.json` files and technical reports over aggregator pages, which often disagree with the configs.
- A row is either **exact** (matches a shipped checkpoint) or a **synthesis average**. Average rows must not claim a config they don't match.
- KV-critical fields (layers and attention type) take priority over hidden width.

**Growth of the table:**

| Step | Anchors | Families added |
| --- | --- | --- |
| Llama Gen 1–4 + Kimi K3 | 21 | Llama 7B → 405B; Kimi K3 2.8T (93 layers) |
| gpt-oss, Cohere | 28 | gpt-oss 20b/120b; Command R/R+/A/A+, Aya, North |
| Poolside | 31 | Laguna XS.2 / XS 2.1 / S 2.1 / M.1 |
| Arcee, Mistral | 41 | Trinity Nano/Mini/Large; Mixtral → Large 3. Per-anchor attention overrides introduced. |
| MiniMax | 44 | Text-01/M1, M2→M2.7, M3 |
| Anchor audit + DeepSeek V4 | 50 | Corrections below; V4-Flash, V4-Pro, DeepSeek-V2 |
| Tencent Hunyuan / Hy | 54 | Hunyuan 1.8B, A13B, Large (CLA-2), Hy3 |
| MiMo, StepFun, Muse, Granite | 72 | MiMo V2/V2.5/Pro, Step-3/3.5/3.7, Muse Glimmer, Granite family |
| Next generation (Oct 2026) | 98 | GLM-4.5 → 5.3, Qwen3-Next / 3.5 / 3.6 / 3.8, Ling 2.0 → 3.0, DeepSeek V4.1-Flash, Hy4-preview, Gemma 3/4 |

**Corrections from the August 2026 anchor audit:**
- **MLA classifier:** "≥ 61 layers and hidden ≥ 6,144" tagged dense 32–130B models as MLA and under-counted their KV about 3×. It now requires hidden 7,168.
- **Qwen3-0.6B:** 1,152 → 1,024 hidden, GQA-8 (KV had been under-counted 2×).
- **4B row:** 3,072 → 2,560 hidden.
- **32B row:** 62 × 6,144 → 64 × 5,120.
- **Command R+:** now MHA-exact (64 × 12,288).
- **GLM-130B:** 70 × 12,288.
- **Removed an invented 250B composite**, replaced by exact Qwen3-235B (GQA-4) and Inkling-Small 276B rows.
- **Split the 1T row:** Kimi K2 (61 × 7,168, MLA) vs Inkling 975B (66 × 6,144, GQA). Inkling had been under-counted 3.5× as MLA.
- **DeepSeek-V4-Pro** costs 9.625 GiB at 1M tokens in BF16 (published: 9.62). Costing it as MLA would have given 68.6 GiB.

**Families covered:** Llama, Gemma, Qwen, MiniCPM, G9, Ling/Ring, Inkling, DeepSeek, Tencent Hunyuan/Hy, GLM, Kimi, Xiaomi MiMo, StepFun, Muse Glimmer, IBM Granite, gpt-oss, Cohere (Command, Aya, North), Poolside Laguna, Mistral, Arcee Trinity and MiniMax.

**Inclusion rules:**
- Open weights with a published config only.
- Multimodal models count only for their autoregressive language backbone.
- Excluded: API-only models, image/video/audio generators, embeddings, encoder-decoder (Aya 101), Mamba-only (no KV cache) and text diffusion.
- Same-size models fold into one anchor. A few sizes are nudged to keep entries unique (27.5, 35.5, 81.0).

**Checked, not anchored (October 2026):**
- No open weights: Qwen 3.7 (API-only), Meta Muse Spark 1.2/1.3 (closed).
- Weights pending: StepFun Step 5 Preview.
- Geometry undisclosed or unpublished: Kimi K2.8-Preview / K3.x, GLM-5.3-Flash/FlashX, Gemma 4 12B.
- Out of scope: MiniMax H3 (video), DiffusionGemma (text diffusion).
- "Llama 5" does not exist.

---

## 5. Context window and agent harnesses

- **One dual-label slider:** "Context Window / Active Working Context" (log scale, 512 → 1M tokens). It sets the KV capacity of one chat, document prompt, embedded feature, or **one active agent inference step**.
- **No mode toggle:** the old Normal/Agentic toggle changed only wording, so it was removed.
- **17 open-source harnesses reviewed:** OpenClaw, OpenCode, DeepSeek Harness, Hermes Agent, Prime Agent, Pi, Qwen Code, Gemini CLI, Agent Zero, Crush, OpenHands, SWE-agent, Aider, Cline/Roo Code/Goose, LangGraph/Deep Agents, Letta and smolagents. All of them separate the active context from durable state:
  - sessions persist outside the model context;
  - compaction swaps history for a lossy summary and adds no capacity;
  - memory is tiered (files, stores, retrieval);
  - each subagent has its own context;
  - usable prompt space is harness-specific.
- **No harness multipliers:** for these reasons the calculator adds no output reserve, concurrency, task-horizon or harness-specific multiplier.

---

## 6. Hardware presets

**History:**
- 8 presets (cheapest card 8 GB, outdated Apple tier).
- **21** in August 2026: a "poor to business" spectrum from 4 to 192 GB, plus 5 Intel ARC cards.
- **25** on 7 October 2026: Apple M5, RTX Spark, DGX Spark and Snapdragon X2.
- **14** on 8 October 2026.

**Rule:**
- The result depends only on memory, GPU type, precision, context and overhead, so same-memory cards gave identical results. The old set had 3 exact duplicate pairs.
- Each button therefore stands for **one memory tier under one memory model**. It names the most common cards, and a hover tooltip lists the equivalents.

| Group | Button | Type | Memory | Precision · context | Overhead | Max params |
| --- | --- | --- | --- | --- | --- | --- |
| Consumer | GTX 1650 4GB | Discrete | 4 GB | Q4 · 4K | 1.0 GB | 5.0B |
| | RTX 5060 / 4060 8GB | Discrete | 8 GB | Q4 · 8K | 1.5 GB | 10.4B |
| | RTX 5070 / 3060 12GB | Discrete | 12 GB | Q4 · 8K | 1.5 GB | 17.1B |
| | RTX 5060 Ti / RX 9070 XT 16GB | Discrete | 16 GB | Q4 · 16K | 1.5 GB | 23.5B |
| | RTX 3090 / 4090 24GB | Discrete | 24 GB | Q4 · 32K | 1.5 GB | 37.2B |
| | RTX 5090 32GB | Discrete | 32 GB | Q4 · 32K | 1.5 GB | 49.6B |
| Unified | MacBook Air / Mac mini 16GB | Apple | 16 GB | Q4 · 8K | 25% | 20.3B |
| | Ryzen AI Max+ 128GB | Other Unified | 128 GB | Q4 · 32K | 16 GB | 190.0B |
| | DGX Spark 128GB | Other Unified | 128 GB | NVFP4 · 64K | 12 GB | 207.9B |
| | RTX Spark 128GB | RTX Spark | 128 GB | Q4 · 32K | 25.6 CPU-only + 1.5 GB | 171.5B |
| | M5 Ultra 256GB | Apple | 256 GB | Q4 · 64K | 25% | 325.1B |
| Datacenter | A100 / H100 80GB | Discrete | 80 GB | FP16 · 32K | 0.5 GB | 37.4B |
| | RTX PRO 6000 96GB | Discrete | 96 GB | FP16 · 64K | 0.5 GB | 43.0B |
| | B200 180GB | Discrete | 180 GB | FP16 · 128K | 0.5 GB | 90.4B |

**Evidence behind the choice:**
- **Steam Hardware Survey, August–September 2026:** VRAM tiers 16 GB 27.21%, 8 GB 26.71%, 12 GB 12.99% and 4 GB 5.69%, about 71% together. The RTX 5070 is #1 at 5.86%.
- **2026 local-LLM buying guides:** the 24 and 32 GB tiers come from these guides, which recommend the RTX 5060 Ti 16GB as the entry card, a used RTX 3090 24GB for the best price per GB, and the RTX 5090 32GB as the flagship. The RX 9070 XT 16GB costs ~$600.
- **Ryzen AI Max+:** ~110 GB on Linux; Windows is capped at 96 GB (VGM).
- **Apple:** Mac mini M6 16 GB costs $899.
- **B200:** 180 GB is software-visible; 192 GB is the physical HBM3e stack.
- **Cloud rental:** A100 80GB from $1.38 and H100 from $2.59–2.79 per GPU-hour.
- **RTX PRO 6000:** rose from $8,565 at launch to $13–20K.

**Same 128 GB, different rules** (Q4, 32K): DGX Spark 196.9B · Ryzen AI Max+ on Linux 190.0B · RTX Spark 171.5B · Apple M5 Max 163.1B · Ryzen AI Max+ on Windows (Discrete 96 GB) 160.6B. That ~36B spread is why the unified presets are not merged.

**Not included:** B300 (288 GB) and the M5 Ultra 512 GB exceed the linear 256 GB slider. Raising the maximum would compress the 4–32 GB range, where most users are.

---

## 7. Verification practice

Every change was checked with:
- `node --check` on the extracted script.
- **Anchor sweeps** from 0.1B to 2.8T: positive dimensions, finite non-negative KV, strictly sorted unique sizes, exact anchors regression-tested, and attention boundaries checked at nearest-anchor midpoints.
- **Spot checks against published figures:**
  - Llama KV;
  - DeepSeek V4: 9.62 / 6.72 GiB at 1M tokens;
  - Ling 2.6: 11.25 KiB/token;
  - DeepSeek V4.1-Flash: 890 B/token;
  - gpt-oss MXFP4: 60.8 GB;
  - Laguna S 2.1 NVFP4: ~71 GB.
- **jsdom smoke tests:**
  - every preset;
  - the RTX Spark budget table;
  - GPU-type switching;
  - methodology disclosures;
  - English/Indonesian key parity;
  - an offline-policy scan for storage, history, fetch, hash and query APIs.
- **Headless Chromium screenshots** in English and Indonesian, desktop and mobile, light and dark.

---

## 8. Key sources

A condensed selection. The complete lists live in the original reports (see the appendix).

**Memory math and KV cache**
- llama.cpp completion / context documentation — https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md
- DeepSeek-V2 technical report (MLA) — https://arxiv.org/abs/2405.04434
- Hugging Face DeepSeek V3 configuration — https://huggingface.co/docs/transformers/en/model_doc/deepseek_v3
- Sebastian Raschka, KV cache calculations — https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
- Will It Run AI, VRAM requirements (quantization bytes) — https://willitrunai.com/blog/vram-requirements-for-ai-models
- Kubesimplify, quantization formats (BF16, FP8, NVFP4, MXFP4, INT4, GGUF) — https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters

**Overhead and platforms**
- NVIDIA RTX Spark Windows on Arm Porting Guide 0.1.0, Unified Memory Architecture — https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/0.1.0/uma/index.html
- vramcalculator, RTX Spark vs Ryzen AI Max — https://vramcalculator.com/rtx-spark-local-llm/
- NVIDIA Developer Forums, DGX Spark memory in use when headless — https://forums.developer.nvidia.com/t/12gb-of-ram-in-use-on-a-freshly-booted-spark-disabling-desktop-mode/347814
- Simon Willison, DGX Spark review — https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/
- Windows' 50% GPU shared-memory cap — https://stackoverflow.com/questions/79596623/why-does-windows-only-allow-your-gpu-to-use-half-of-your-ram
- llama.cpp OpenCL backend (Adreno) — https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
- Modem Guides, Ryzen AI Max+ 395 reality check — https://www.modemguides.com/blogs/ai-infrastructure/ryzen-ai-max-395-local-llm-reality-check
- AMD, trillion-parameter LLM on a Ryzen AI Max+ cluster (GTT setup) — https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html
- ModelPiper, `iogpu.wired_limit_mb` on Mac — https://modelpiper.com/blog/iogpu-wired-limit-mb-mac
- llama.cpp Metal working-set logs on Mac Studio — https://obrienlabs.medium.com/running-the-70b-llama-2-llm-locally-on-metal-via-llama-cpp-on-mac-studio-m2-ultra-32b3179e9cbe
- Apple Newsroom, M5 Pro and M5 Max — https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- LocalAIMaster, Intel Arc for local AI (Resizable BAR, IPEX-LLM) — https://localaimaster.com/blog/intel-arc-a770-local-ai

**Architecture anchors (official configs and reports)**
- Llama 3 Herd of Models — https://ar5iv.labs.arxiv.org/html/2407.21783 · Llama 4 config — https://huggingface.co/docs/transformers/en/model_doc/llama4
- Qwen3 — https://huggingface.co/Qwen/Qwen3-235B-A22B · Qwen3-Next — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- DeepSeek V4 report — https://arxiv.org/html/2606.19348v1 · vLLM DeepSeek V4 — https://vllm.ai/blog/2026-04-24-deepseek-v4
- Kimi K2.5 — https://huggingface.co/moonshotai/Kimi-K2.5 · Kimi K3 — https://huggingface.co/moonshotai/Kimi-K3
- Inkling — https://huggingface.co/thinkingmachines/Inkling
- gpt-oss-120b — https://huggingface.co/openai/gpt-oss-120b
- Cohere Command R+ — https://huggingface.co/CohereForAI/c4ai-command-r-plus · North Mini Code — https://huggingface.co/CohereLabs/North-Mini-Code-1.0
- Poolside Laguna S 2.1 — https://huggingface.co/poolside/Laguna-S-2.1
- Arcee Trinity Large — https://huggingface.co/arcee-ai/Trinity-Large-Base
- Mistral Small 4 — https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 · Mistral Large 3 — https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512
- MiniMax M3 — https://huggingface.co/MiniMaxAI/MiniMax-M3
- Hunyuan-Large config — https://huggingface.co/tencent/Tencent-Hunyuan-Large/blob/main/Hunyuan-A52B-Pretrain/config.json · Hy3 — https://huggingface.co/tencent/Hy3
- MiMo-V2.5-Pro — https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- Step-3 — https://huggingface.co/stepfun-ai/step3 · Step-3.5-Flash — https://huggingface.co/stepfun-ai/Step-3.5-Flash
- Muse Glimmer 30B — https://huggingface.co/meta-models/Muse-Glimmer-30B
- IBM Granite — https://huggingface.co/ibm-granite · Granite Code report — https://arxiv.org/abs/2405.04324
- Gemma 3 27B config — https://huggingface.co/google/gemma-3-27b-it · Gemma 4 Technical Report (arXiv:2607.02770)
- Z.ai GLM configs — https://huggingface.co/zai-org · InclusionAI Ling configs — https://huggingface.co/inclusionAI

**Context and agent harnesses**
- OpenClaw: Context — https://docs.openclaw.ai/concepts/context
- OpenCode: Compaction — https://opencode.ai/v2/docs/compaction
- Hermes Agent: Context Compression — https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching
- Letta: Memory Management — https://docs.letta.com/concepts/memory-management/
- LangGraph: Memory — https://docs.langchain.com/oss/python/langgraph/add-memory

**Hardware presets**
- VideoCardz, Steam survey September 2026 — https://videocardz.com/newz/geforce-rtx-5070-becomes-the-most-popular-gpu-on-steam-32gb-ram-reaches-42
- PC Guide, Steam VRAM tiers August 2026 — https://www.pcguide.com/news/16gb-vram-continues-to-grow-in-latest-steam-survey-widening-the-gap-to-12gb-and-8gb-graphics-card-owners/
- Compute Market, best consumer GPU for local LLM 2026 — https://www.compute-market.com/blog/best-consumer-gpu-local-llm-2026
- TechRepublic, Mac mini M6 — https://www.techrepublic.com/article/news-mac-mini-m6-cheat-sheet/
- GPUsmith, NVIDIA B200 (180 GB software-visible) — https://gpusmith.com/hardware/gpus/nvidia-b200
- GPUPerHour, cloud GPU prices — https://gpuperhour.com/

---

## Appendix: original reports

All 15 files were removed from the working tree on 2026-10-08 and replaced by this summary. Read any of them with `git show 228ba94:<FILE>`, or find other versions with `git log --oneline -- <FILE>`.

| File | Date | Topic |
| --- | --- | --- |
| `OVERHEAD_AUDIT.md` | 2026-08-19 | Overhead per GPU type; Snapdragon 50% → fixed 3 GB |
| `PRESET_AND_INTEL_AUDIT_2026.md` | 2026-08-19 | Calculation audit, MLA fix, presets 8 → 21, Intel ARC |
| `CONTEXT_MATH_AUDIT_2026.md` | 2026-08-19 | KV formulas, weight units, solver, MiB/token |
| `AGENT_CONTEXT_HARNESS_AUDIT_2026.md` | 2026-08-19 | 17 agent harnesses; one dual-label context slider |
| `AUTOESTIMATE_AUDIT_2026.md` | 2026-08-19 | Anchor corrections; DeepSeek V4 (44 → 50 anchors) |
| `KIMI3_LLAMA_MACBOOKNEO_UPDATE_2026.md` | 2026-08-19 | Kimi K3, Llama Gen 1–4, MacBook Neo |
| `GPTOSS_COHERE_NORTH_UPDATE_2026.md` | 2026-08-19 | gpt-oss, MXFP4, Cohere Command/Aya/North (21 → 28) |
| `POOLSIDE_LAGUNA_UPDATE_2026.md` | 2026-08-19 | Poolside Laguna, NVFP4 (28 → 31) |
| `ARCEE_TRINITY_MISTRAL_UPDATE_2026.md` | 2026-08-19 | Arcee Trinity, Mistral, per-anchor overrides (31 → 41) |
| `MINIMAX_UPDATE_2026.md` | 2026-08-19 | MiniMax (41 → 44) |
| `HUNYUAN_HY_UPDATE_2026.md` | 2026-08-19 | Tencent Hunyuan / Hy, CLA-2 (50 → 54) |
| `MIMO_STEPFUN_MUSE_GRANITE_UPDATE_2026.md` | 2026-08-19 | MiMo, StepFun, Muse Glimmer, Granite (54 → 72) |
| `NEXTGEN_ANCHORS_UPDATE_2026.md` | 2026-10-04 | GLM, Qwen3-Next → 3.8, Ling, DeepSeek V4.1, Hy4, Gemma 3/4 (72 → 98) |
| `ARM_SOC_RTX_SPARK_UPDATE_2026.md` | 2026-10-07 | RTX Spark budget rule, DGX Spark, Arm SoCs, presets 21 → 25 |
| `PRESET_CONSOLIDATION_AUDIT_2026.md` | 2026-10-08 | Presets 25 → 14, Ryzen AI Max+, B200 180 GB, `uma` rename |

The repository also held two full-page verification screenshots (`verify_en.png`, `verify_id.png`) of an older UI; they were removed at the same time and are likewise in git history.
