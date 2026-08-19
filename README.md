**English** | [Bahasa Indonesia](README-id.md)

# LLMCalculator
VRAM usage, simplified.

## Introduction
LLMCalculator is a single-file, browser-based tool for estimating the maximum Large Language Model (LLM) size that can fit in your GPU memory. Designed for AI enthusiasts, researchers, and hardware planners, this tool helps visualize how quantization, context length, and system overhead impact model capacity.

The interface supports both **English** and **Bahasa Indonesia**.

## How It Works
The calculator estimates memory usage based on:

1.  **VRAM Size**: The total GPU memory available (e.g., 24GB, 80GB).
2.  **Quantization**: The precision of model weights (FP32, FP16/BF16, FP8/INT8, MXFP4, NVFP4, INT4/FP4, FP2). MXFP4 (~0.53 bytes/param) is native to GPT-OSS and Kimi K3 and is available for MiMo-V2.5-Pro’s experts; NVFP4 (~0.56 bytes/param) is the Blackwell-native FP4 format shipped by Poolside Laguna and Step-3.7-Flash. Lower precision reduces memory usage but may affect quality.
3.  **Context Mode**: Choose **Normal** for a normal local chat, document prompt, or embedded local feature, or **Agentic** for one long-running local agent.
4.  **Context Window**: In Normal mode, this is the configured local context/KV capacity. In Agentic mode, the same slider is labeled **Active Working Context** and represents the KV capacity for one active inference step in a long-running agent workflow—not its lifetime history. Both modes use the slider value directly in the same architecture-aware KV-cache formula.
5.  **KV Cache**: Memory required to store Key-Value states for one active context. It supports separate quantization from model weights.
6.  **System Overhead:**
    -   **Discrete GPUs (NVIDIA / AMD / Intel ARC)**: A configurable deduction (default 1.5 GB) for OS and display. Adjustable via slider (0–16 GB). Headless Linux servers: set to ~0.5 GB. *Intel ARC notes:* dedicated GDDR6 VRAM (same model as NVIDIA/AMD), requires **Resizable BAR** enabled in BIOS/UEFI; use IPEX-LLM (SYCL/Level Zero) or llama.cpp Vulkan/SYCL backend for LLM inference. Software support is improving but still rougher than CUDA (Ollama via IPEX-LLM fork, not vanilla) — e.g., Arc Pro B60 24GB defaults to 0.8 GB workstation overhead.
    -   **Apple Silicon (M/A-series)**: macOS Metal driver reserves ~25% of Unified Memory by default (adjustable system-wide via `sysctl iogpu.wired_limit_mb`).
    -   **Snapdragon (Windows on Arm)**: Configurable OS + driver overhead (default 3.0 GB). Windows caps shared GPU memory at 50% of total RAM — the overhead slider lets you tune for your system's actual OS + background usage. This is a dynamic cap, not a fixed reservation.

It iteratively calculates the maximum parameter count (in Billions) that fits within the remaining VRAM after accounting for overhead and KV cache.

## Quick Start
1.  Download `LLMCalculator.html`.
2.  Open it in any modern browser (Chrome, Edge, Firefox, Safari).
3.  Set your **GPU Memory** (VRAM Size) using the slider.
4.  Select the **GPU Type** (Discrete — NVIDIA / AMD / Intel ARC, Apple Silicon, or Snapdragon).
5.  Choose the **Model Precision** (Quantization) and **KV Cache** precision.
6.  Choose **Normal** or **Agentic**, then adjust the **Context Window** / **Active Working Context** (e.g., 8K, 32K tokens).
7.  View the estimated **Max Parameters** and detailed memory breakdown.
8.  Use **Quick Presets** to simulate popular hardware configurations across the full economic spectrum — from 4 GB budget cards for students to 192 GB datacenter GPUs for businesses:
    - **Budget / Entry (4–12 GB)**: GTX 1650 4GB, Arc A380 6GB, RTX 4060 8GB, RTX 5050 Laptop 8GB, Arc B570 10GB, RTX 3060 12GB (Steam #1, best value under $250 used), Arc B580 12GB ($249 best new budget)
    - **Mainstream / Enthusiast (16–32 GB)**: Arc A770 16GB (cheapest 16GB), RTX 5070 Ti 16GB, RTX 4090 24GB, Arc Pro B60 24GB ($599 workstation value), RTX 5090 32GB
    - **Workstation / Server (48–192 GB)**: L40S 48GB, A100 80GB, RTX PRO 6000 96GB, H200 141GB, B200 192GB
    - **Apple / Mobile**: MacBook Neo 8GB, M4 Max 128GB (MacBook Pro), M3 Ultra 192GB (Mac Studio), Snapdragon X Elite 32GB

## Key Features
-   **Multi-language Support**: Toggle between English and Indonesian.
-   **Real-time Calculation**: Instant updates as you adjust sliders and dropdowns.
-   **Architecture-Aware Logic**: Specific attention mechanisms (MHA, GQA, MLA) and GPU-specific memory reservation rules (Discrete incl. Intel ARC, Unified vs Shared).
-   **Adjustable Overhead**: Fine-tune system memory deduction for headless Linux servers vs Windows desktops.
-   **Detailed Memory Breakdown**: Visualizes usage for System Overhead, KV Cache, and Model Weights.
-   **Hardware Presets (21 presets, 4 GB–192 GB)**: One-click configuration covering the full stack — Budget (GTX 1650, Arc A380/B570/B580, RTX 3060/4060), Mainstream (Arc A770, RTX 5070 Ti/4090/5090, Arc Pro B60), Workstation/Server (L40S, A100, RTX PRO 6000 96GB, H200, B200), and Apple/Mobile (M4 Max, M3 Ultra, Snapdragon). Verified against Steam HW Survey 2025–2026, Tom's Hardware, Intel ARK.
-   **Simple Normal / Agentic Context Modes**: One logarithmic context slider estimates one active local inference context. Agentic mode changes the label and guidance, not the KV-cache calculation.
-   **Advanced Options**: Support for various quantization formats (GGUF, GPTQ, FP8, MXFP4, NVFP4) and manual architecture overrides.
-   **Multi-Generational Auto-Estimate Model & Attention Architecture**: Dynamically estimates both model architecture (Layers and Hidden Size) and attention mechanism (GQA/MQA including compact 64-d heads, hybrid sliding-window/global attention, MFA, MLA, CSA/HCA, and CLA-2) by synthesizing calculations across all generation versions of modern LLM families: Llama (Gen 1–4), Gemma (Gen 1–4), Qwen (Gen 1–3), MiniCPM (Gen 1–5), G9 (v1–v3), Ling (1.0–2.0), Inkling (Small/276B, Base/975B), DeepSeek (V1–V4, R1), Tencent Hunyuan / Hy (dense 0.5B–7B, A13B, Large/A52B, Hy3), GLM (1–4), Kimi (V1–K3), **Xiaomi MiMo (7B, V2-Flash, V2.5, V2.5-Pro), StepFun (Step3-VL-10B, Step-3, Step-3.5/3.7 Flash), Muse Glimmer 30B, IBM Granite (Code, 3.0–3.3, 4.0, 4.1, SWASH)**, GPT-OSS, Cohere Command/Aya/North, Poolside Laguna, Mistral open weights, Arcee Trinity, and MiniMax. Family-specific KV formulas account for MiMo’s asymmetric K/V dimensions and SWA-128, Step’s SWA-512 and MFA, Muse’s SWA-2048, and Granite’s 64-d GQA/MQA and SWASH layers.
-   **Standardized Model Size Buckets (Artificial Analysis)**: Categorizes models into 4 standardized tiers: **Tiny (<4B)**, **Small (4B–40B)**, **Medium (40B–150B)**, and **Large (150B+)** based on Artificial Analysis taxonomy.
-   **Single HTML file**: No installation, no dependencies, works completely offline.
-   **Responsive design**: Works on desktop, tablet, and mobile devices.

## Use Cases
-   **Hardware Planning**: determining which GPU to buy for running specific models.
-   **Model Selection**: Choosing the right model size and quantization for your existing hardware.
-   **Educational**: Understanding the relationship between model parameters, context length, and memory requirements.

## Privacy & Data
All calculations happen locally in your browser. No data is sent to any server. The tool is completely offline once loaded.

## License
MIT License. See LICENSE for details.

## Contributions
Contributions, issues, and suggestions are welcome. Please open an issue to discuss ideas or submit a PR.

### Local context and KV-cache planning
The context slider is the configured token capacity used directly by the calculator's architecture-aware KV formulas. This is analogous to the prompt-context size configured by [`llama.cpp --ctx-size`](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md). **Normal** describes one local chat, document prompt, or embedded local feature. **Agentic** describes the bounded KV capacity for one active inference step in a long-running local-agent workflow and uses exactly the same calculation.

Long-lived project history should stay outside the active context in artifacts, structured notes, and retrieval, with compaction used to refresh the working set. This follows current [agent context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). A nominally larger window is not a quality guarantee: [context-rot research](https://research.trychroma.com/context-rot) finds that model performance can become less reliable as input length grows.

Open-source harnesses reinforce this active-context versus durable-state boundary in different ways: [OpenHands](https://docs.openhands.dev/sdk/guides/context-condenser) condenses history; [SWE-agent](https://swe-agent.com/latest/reference/history_processor_config/) filters observations; [Aider](https://aider.chat/docs/config/options.html) summarizes chat and budgets repository maps; [Cline](https://docs.cline.bot/features/auto-compact), [Roo Code](https://docs.roocode.com/features/intelligent-context-condensing), and [Goose](https://block.github.io/goose/docs/guides/smart-context-management/) compact long sessions; [LangGraph](https://docs.langchain.com/oss/python/langgraph/add-memory) separates trimming/summaries from checkpointed state; and [Letta](https://docs.letta.com/concepts/memory-management/) separates in-context from archival and recall memory. A broader review of OpenClaw, OpenCode, DeepSeek Harness, Hermes Agent, Prime Agent, Pi, Qwen Code, Gemini CLI, Agent Zero, Crush, and smolagents reaches the same calculator-level conclusion; see [the 2026 harness audit](AGENT_CONTEXT_HARNESS_AUDIT_2026.md). Those harness policies do not create extra GPU KV capacity.

> This calculator estimates one local model and one active inference context. Multi-user serving, parallel agents, CPU KV offload, and runtime-specific cache behavior require separate capacity planning.

The calculator preserves its interpolated architecture model: it estimates a plausible local open-weight model that fits the selected memory and context budget. It is not an exact checkpoint-fit validator or an inference-server capacity planner.
