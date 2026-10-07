**English** | [Bahasa Indonesia](README-id.md)

# LLMCalculator
*VRAM usage, simplified.*

## Introduction
LLMCalculator is a single-file, browser-based tool for estimating the maximum Large Language Model (LLM) size that can fit in your GPU memory. Designed for AI enthusiasts, researchers, and hardware planners, this tool helps visualize how quantization, context length, and system overhead impact model capacity.

The interface supports both **English** and **Bahasa Indonesia**.

## How It Works
The calculator estimates memory usage based on:

1.  **VRAM Size**: The total GPU memory available (e.g., 24GB, 80GB).
2.  **Quantization**: The precision of model weights (FP32, FP16/BF16, FP8/INT8, MXFP4, NVFP4, INT4/FP4, FP2). MXFP4 (~0.53 bytes/param) is native to GPT-OSS and Kimi K3 and is available for MiMo-V2.5-Pro’s experts; NVFP4 (~0.56 bytes/param) is the Blackwell-native FP4 format shipped by Poolside Laguna and Step-3.7-Flash. Lower precision reduces memory usage but may affect quality.
3.  **Context Window / Active Working Context**: One dual-label slider sets the configured local KV capacity for a chat, document prompt, embedded feature, or one active agent inference step. Long-running project history is not part of this GPU-KV estimate.
4.  **KV Cache**: Memory required to store Key-Value states for one active context. It supports separate quantization from model weights.
5.  **System Overhead**:
    -   **Discrete GPUs (NVIDIA / AMD / Intel ARC)**: A configurable deduction (default 1.5 GB) for OS and display. Adjustable via slider (0–16 GB). Headless Linux servers: set to ~0.5 GB. *Intel ARC notes:* dedicated GDDR6 VRAM (same model as NVIDIA/AMD), requires **Resizable BAR** enabled in BIOS/UEFI; use IPEX-LLM (SYCL/Level Zero) or llama.cpp Vulkan/SYCL backend for LLM inference. Software support is improving but still rougher than CUDA (Ollama via IPEX-LLM fork, not vanilla).
    -   **Apple Silicon (M/A-series)**: macOS Metal driver reserves ~25% of Unified Memory by default — Metal's `recommendedMaxWorkingSetSize` is ≈75% of RAM (≈⅔ on ≤32 GB Macs with older macOS releases); adjustable system-wide via `sudo sysctl iogpu.wired_limit_mb`, at the cost of the OS reserve.
    -   **NVIDIA RTX Spark (Windows on Arm)**: Uses NVIDIA's published GPU-budget rule for the N1X superchip: **GPU budget = dedicated carveout + shared memory**, where shared = (memory left after the carveout − 16 GB), clamped to 50–80% of that remainder. The balance is CPU-only memory that GPU allocations can never use. Pick the OEM **Dedicated GPU Carveout** if you know it (Task Manager → Dedicated GPU memory); the default of 0 GB is the conservative floor. The slider sets **desktop / driver headroom** inside the budget (default 1.5 GB), since NVIDIA warns against allocating the full budget. Examples (no carveout): 24 GB → 12 GB GPU budget, 64 GB → 48 GB, 128 GB → 102.4 GB (up to 112 GB with a ≥48 GB carveout).
    -   **Other Unified Memory (DGX Spark · Ryzen AI Max · Snapdragon)**: A fixed, configurable OS/firmware reserve (default 3.0 GB). DGX Spark / GB10 on DGX OS: ~12 GB headless, or ~16 GB with the desktop running (the OS sees ≈119 GiB of the 128 GB; ≈116 GiB stays available headless versus ≈112 GiB with the desktop). DGX Spark uses the same chip family as RTX Spark, but Linux exposes almost all of its memory to CUDA, so the Windows budget rule doesn't apply. AMD Ryzen AI Max+ 395 (Strix Halo) 128 GB: ~16 GB on Linux, where the kernel's GTT pool lets the iGPU reach roughly 110–117 GB depending on tuning. Windows limits the iGPU to the BIOS Variable Graphics Memory carve-out (96 GB maximum), so model a Windows machine as a Discrete GPU with 96 GB. Snapdragon X/X2 on Windows: ~3 GB for the CPU/NPU path. Windows caps Adreno GPU offload (llama.cpp OpenCL) at 50% of RAM. See the [audit summary](AUDIT.md#3-system-overhead-by-gpu-type).

It searches the interpolated architecture curve for the highest parameter count that fits after overhead and KV cache. Weight bytes and architecture-aware KV bytes are converted consistently to binary GiB internally; UI memory labels retain the familiar GPU-market convention of “GB.” See the [audit summary](AUDIT.md#2-memory-math).

## Quick Start
1.  Download [`LLMCalculator.html`](LLMCalculator.html).
2.  Open it in any modern browser (Chrome, Edge, Firefox, Safari).
3.  Set your **GPU Memory** (VRAM Size) using the slider.
4.  Select the **GPU Type** (Discrete — NVIDIA / AMD / Intel ARC, Apple Silicon, NVIDIA RTX Spark, or Other Unified Memory). For RTX Spark, optionally choose the OEM **Dedicated GPU Carveout**.
5.  Choose the **Model Precision** (Quantization) and **KV Cache** precision.
6.  Adjust **Context Window / Active Working Context** (e.g., 8K or 32K tokens).
7.  View the estimated **Maximum Parameters** and detailed memory breakdown.
8.  Use **Quick Presets** for one-click setups across the full economic spectrum — from 4 GB student laptops to 180 GB datacenter GPUs and 256 GB unified-memory workstations. There are 14 presets, one per common memory tier. Cards with the same memory and GPU type give identical results, so each button names the most common cards (hover a button to see equivalents), and rarer sizes are one slider move away:
    - **Consumer GPUs (4–32 GB)**: GTX 1650 4GB, RTX 5060 / 4060 8GB, RTX 5070 / 3060 12GB, RTX 5060 Ti / RX 9070 XT 16GB, RTX 3090 / 4090 24GB, RTX 5090 32GB
    - **Unified Memory (16–256 GB)**: MacBook Air / Mac mini 16GB, Ryzen AI Max+ 128GB (Strix Halo on Linux), DGX Spark 128GB (NVFP4, DGX OS), RTX Spark 128GB (Windows on Arm), M5 Ultra 256GB (Mac Studio)
    - **Workstation / Datacenter (80–180 GB)**: A100 / H100 80GB, RTX PRO 6000 96GB, B200 180GB (software-visible; 192 GB physical)

## Key Features
-   **Multi-language Support**: Toggle between English and Indonesian, with Auto, Light and Dark themes.
-   **Real-time Calculation**: Instant updates as you adjust sliders and dropdowns.
-   **Architecture-Aware Logic**: Specific attention mechanisms (MHA, GQA, MLA) and GPU-specific memory reservation rules (Discrete incl. Intel ARC, Apple unified memory, NVIDIA RTX Spark carveout + shared budget, fixed-reserve unified memory such as DGX Spark, Ryzen AI Max and Snapdragon).
-   **Adjustable Overhead**: Fine-tune system memory deduction for headless Linux servers vs Windows desktops.
-   **Detailed Memory Breakdown**: Visualizes usage for System Overhead, KV Cache, and Model Weights.
-   **Hardware Presets (14 presets, 4 GB–256 GB)**: One button per common memory tier, chosen from the Steam Hardware Survey (Aug–Sep 2026), 2026 local-LLM buying guides and cloud rental data: Consumer GPUs (GTX 1650 4GB → RTX 5090 32GB), Unified Memory (base 16 GB Macs, Ryzen AI Max+, DGX Spark, RTX Spark, M5 Ultra) and Workstation / Datacenter (A100 / H100, RTX PRO 6000, B200). See the [audit summary](AUDIT.md#6-hardware-presets).
-   **One Dual-Purpose Context Slider**: The logarithmic **Context Window / Active Working Context** slider estimates one active local inference context without a redundant mode toggle; see [Local Context and KV-Cache Planning](#local-context-and-kv-cache-planning).
-   **Advanced Options**: Quantization from FP32 down to FP2 — GGUF (Q2_K–Q8_0), GPTQ 4-bit, FP8/INT8, MXFP4, NVFP4 and INT4/FP4 — plus a separate KV-cache precision, manual attention selection, and custom layer / hidden-size overrides.
-   **Multi-Generational Auto-Estimate Model & Attention Architecture**: Dynamically estimates both model architecture (Layers and Hidden Size) and attention mechanism (GQA/MQA including compact 64-d heads, hybrid sliding-window/global attention, MFA, MLA, MLA+DSA, hybrid linear/softmax attention, CSA/HCA, and CLA-2) by synthesizing calculations across all generation versions of modern LLM families: Llama (Gen 1–4), Gemma (Gen 1–4, **Gemma 3 12B/27B, Gemma 4 26B-A4B / 31B**), Qwen (Gen 1–3, **Gen 3-Next, Gen 3.5 (0.8B–397B), Gen 3.6, Gen 3.8 (27B, Flash-Next 180B, Max 2.4T)**), MiniCPM (Gen 1–5), G9 (v1–v3), Ling (1.0–2.0 mini/flash/1T, Ring-linear 2.0, **2.5/2.6 1T, 3.0-flash**), Inkling (Small/276B, Base/975B), DeepSeek (V1–V4, V4.1-Flash, R1), Tencent Hunyuan / Hy (dense 0.5B–7B, A13B, Large/A52B, Hy3, **Hy4-preview 770B**), GLM (1–4, **4.5/4.5-Air, 4.6, 4.7/4.7-Flash, 5→5.3**), Kimi (V1–K3), **Xiaomi MiMo (7B, V2-Flash, V2.5, V2.5-Pro), StepFun (Step3-VL-10B, Step-3, Step-3.5/3.7 Flash), Muse Glimmer 30B, IBM Granite (Code, 3.0–3.3, 4.0, 4.1, SWASH)**, GPT-OSS, Cohere Command/Aya/North, Poolside Laguna, Mistral open weights, Arcee Trinity, and MiniMax. Family-specific KV formulas account for MiMo’s asymmetric K/V dimensions and SWA-128, Step’s SWA-512 and MFA, Muse’s SWA-2048, and Granite’s 64-d GQA/MQA and SWASH layers. Newer generations add seven more KV shapes: **Qwen3-Next / Qwen3.5** cache tokens only in the 1-in-4 Gated Attention layers (Gated DeltaNet layers hold a constant-size recurrent state), and **GLM-5.x** combines an MLA latent cache with a 128-d DeepSeek Sparse Attention indexer shared across every four layers. **Ling 2.5/2.6 and Ling 3.0** go further: one gated-MLA layer per group of 8 (or 6) layers carries the whole cache while the lightning-attention / Kimi-Delta-Attention layers keep a constant-size state. **Qwen3.8-Flash-Next** adds a Qwen Sparse Attention index (1 KV head × 128-d, 4:1 compression) on top of its 1-in-4 full-attention layers, and **DeepSeek-V4.1-Flash** is a causal encoder-decoder whose CSA2 Full/Reindex/Reuse layers share one ~890 bytes-per-token global cache at FP4. Generations that were checked but deliberately left unanchored (no open weights or undisclosed geometry) — Qwen 3.7, StepFun Step 5 Preview, Kimi K2.8 Preview, GLM-5.3-Flash/FlashX, Meta Muse Spark 1.2/1.3, MiniMax H3 (a video model), DiffusionGemma — are listed in the app's reference panel. **Gemma 3 and Gemma 4** get their own interleaved shapes: five sliding-window-1024 layers per global layer, with Gemma 4's global layers storing a single unified K=V vector per head at 512-d. See the [audit summary](AUDIT.md#4-auto-estimate-architecture-anchors).
-   **Standardized Model Size Buckets (Artificial Analysis)**: Categorizes models into 4 standardized tiers: **Tiny (<4B)**, **Small (4B–40B)**, **Medium (40B–150B)**, and **Large (150B+)** based on Artificial Analysis taxonomy.
-   **Single HTML file**: No installation, no dependencies, works completely offline.
-   **Responsive design**: Works on desktop, tablet, and mobile devices.

## Use Cases
-   **Hardware Planning**: determining which GPU to buy for running specific models.
-   **Model Selection**: Choosing the right model size and quantization for your existing hardware.
-   **Educational**: Understanding the relationship between model parameters, context length, and memory requirements.

## Local Context and KV-Cache Planning
The dual-label context slider is the configured token capacity used directly by the calculator's architecture-aware KV formulas. This is analogous to the prompt-context size configured by [`llama.cpp --ctx-size`](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md). It covers one local chat, document prompt, embedded feature, or active inference step in a long-running local-agent workflow.

Long-lived project history should stay outside the active context in artifacts, structured notes, and retrieval, with compaction used to refresh the working set. This follows current [agent context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). A nominally larger window is not a quality guarantee: [context-rot research](https://research.trychroma.com/context-rot) finds that model performance can become less reliable as input length grows.

Open-source harnesses reinforce this active-context versus durable-state boundary in different ways: [OpenHands](https://docs.openhands.dev/sdk/guides/context-condenser) condenses history; [SWE-agent](https://swe-agent.com/latest/reference/history_processor_config/) filters observations; [Aider](https://aider.chat/docs/config/options.html) summarizes chat and budgets repository maps; [Cline](https://docs.cline.bot/features/auto-compact), [Roo Code](https://docs.roocode.com/features/intelligent-context-condensing), and [Goose](https://block.github.io/goose/docs/guides/smart-context-management/) compact long sessions; [LangGraph](https://docs.langchain.com/oss/python/langgraph/add-memory) separates trimming/summaries from checkpointed state; and [Letta](https://docs.letta.com/concepts/memory-management/) separates in-context from archival and recall memory. A broader review of OpenClaw, OpenCode, DeepSeek Harness, Hermes Agent, Prime Agent, Pi, Qwen Code, Gemini CLI, Agent Zero, Crush, and smolagents reaches the same calculator-level conclusion; see [the audit summary](AUDIT.md#5-context-window-and-agent-harnesses). Those harness policies do not create extra GPU KV capacity.

> This calculator estimates one local model and one active inference context. Multi-user serving, parallel agents, CPU KV offload, and runtime-specific cache behavior require separate capacity planning.

The calculator preserves its interpolated architecture model: it estimates a plausible local open-weight model that fits the selected memory and context budget. It is not an exact checkpoint-fit validator or an inference-server capacity planner.

## Privacy & Data
All calculations happen locally in your browser. No data is sent to any server. The tool is completely offline once loaded.

## Audit & Methodology
Every assumption in the calculator has been checked against official model configs and 2026 hardware data: the memory formulas, overhead rules, architecture anchors and hardware presets. The findings are summarized in one document, [AUDIT.md](AUDIT.md) ([Bahasa Indonesia](AUDIT-id.md)). In the app, the same material sits in collapsible panels at the bottom of the page: Formulas, Reference Data, Hardware Presets and Sources.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contributions
Contributions, issues, and suggestions are welcome. Please open an issue to discuss ideas or submit a PR.
