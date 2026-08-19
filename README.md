**English** | [Bahasa Indonesia](README-id.md)

# LLMCalculator
VRAM usage, simplified.

## Introduction
LLMCalculator is a single-file, browser-based tool for estimating the maximum Large Language Model (LLM) size that can fit in your GPU memory. Designed for AI enthusiasts, researchers, and hardware planners, this tool helps visualize how quantization, context length, and system overhead impact model capacity.

The interface supports both **English** and **Bahasa Indonesia**.

## How It Works
The calculator estimates memory usage based on:

1.  **VRAM Size**: The total GPU memory available (e.g., 24GB, 80GB).
2.  **Quantization**: The precision of model weights (FP32, FP16/BF16, FP8/INT8, INT4/FP4, FP2). Lower precision reduces memory usage but may affect quality.
3.  **Context Window**: The maximum number of tokens the model processes. Larger context requires more KV cache memory.
4.  **KV Cache**: Memory required to store Key-Value states for the context window. It also supports separate quantization for KV cache.
5.  **System Overhead**:
    -   **Discrete GPUs (NVIDIA / AMD / Intel ARC)**: A configurable deduction (default 1.5 GB) for OS and display. Adjustable via slider (0–16 GB). Headless Linux servers: set to ~0.5 GB. *Intel ARC notes:* dedicated GDDR6 VRAM (same model as NVIDIA/AMD), requires **Resizable BAR** enabled in BIOS/UEFI; use IPEX-LLM (SYCL/Level Zero) or llama.cpp Vulkan/SYCL backend for LLM inference. Software support is improving but still rougher than CUDA (Ollama via IPEX-LLM fork, not vanilla) — e.g., Arc Pro B60 24GB defaults to 0.8 GB workstation overhead.
    -   **Apple Silicon (M-series)**: macOS Metal driver reserves ~25% of Unified Memory by default (adjustable system-wide via `sysctl iogpu.wired_limit_mb`).
    -   **Snapdragon (Windows on Arm)**: Configurable OS + driver overhead (default 3.0 GB). Windows caps shared GPU memory at 50% of total RAM — the overhead slider lets you tune for your system's actual OS + background usage. This is a dynamic cap, not a fixed reservation.

It iteratively calculates the maximum parameter count (in Billions) that fits within the remaining VRAM after accounting for overhead and KV cache.

## Quick Start
1.  Download `LLMCalculator.html`.
2.  Open it in any modern browser (Chrome, Edge, Firefox, Safari).
3.  Set your **GPU Memory** (VRAM Size) using the slider.
4.  Select the **GPU Type** (Discrete — NVIDIA / AMD / Intel ARC, Apple Silicon, or Snapdragon).
5.  Choose the **Model Precision** (Quantization) and **KV Cache** precision.
6.  Adjust the **Context Window** (e.g., 8K, 32K tokens).
7.  View the estimated **Max Parameters** and detailed memory breakdown.
8.  Use **Quick Presets** to simulate popular hardware configurations across the full economic spectrum — from 4 GB budget cards for students to 192 GB datacenter GPUs for businesses:
    - **Budget / Entry (4–12 GB)**: GTX 1650 4GB, Arc A380 6GB, RTX 4060 8GB, RTX 5050 Laptop 8GB, Arc B570 10GB, RTX 3060 12GB (Steam #1, best value under $250 used), Arc B580 12GB ($249 best new budget)
    - **Mainstream / Enthusiast (16–32 GB)**: Arc A770 16GB (cheapest 16GB), RTX 5070 Ti 16GB, RTX 4090 24GB, Arc Pro B60 24GB ($599 workstation value), RTX 5090 32GB
    - **Workstation / Server (48–192 GB)**: L40S 48GB, A100 80GB, RTX PRO 6000 96GB, H200 141GB, B200 192GB
    - **Apple / Mobile**: MacBook Air 8GB, M4 Max 128GB (MacBook Pro), M3 Ultra 192GB (Mac Studio), Snapdragon X Elite 32GB

## Key Features
-   **Multi-language Support**: Toggle between English and Indonesian.
-   **Real-time Calculation**: Instant updates as you adjust sliders and dropdowns.
-   **Architecture-Aware Logic**: Specific attention mechanisms (MHA, GQA, MLA) and GPU-specific memory reservation rules (Discrete incl. Intel ARC, Unified vs Shared).
-   **Adjustable Overhead**: Fine-tune system memory deduction for headless Linux servers vs Windows desktops.
-   **Detailed Memory Breakdown**: Visualizes usage for System Overhead, KV Cache, and Model Weights.
-   **Hardware Presets (21 presets, 4 GB–192 GB)**: One-click configuration covering the full stack — Budget (GTX 1650, Arc A380/B570/B580, RTX 3060/4060), Mainstream (Arc A770, RTX 5070 Ti/4090/5090, Arc Pro B60), Workstation/Server (L40S, A100, RTX PRO 6000 96GB, H200, B200), and Apple/Mobile (M4 Max, M3 Ultra, Snapdragon). Verified against Steam HW Survey 2025–2026, Tom's Hardware, Intel ARK.
-   **Advanced Options**: Support for various quantization formats (GGUF, GPTQ, FP8) and manual architecture overrides.
-   **Multi-Generational Auto-Estimate Model & Attention Architecture**: Dynamically estimates both model architecture (Layers and Hidden Size) and attention mechanism (GQA-4 for <2B, GQA-8 for 2B–276B, MLA for 300B+ MoEs) by synthesizing calculations across all generation versions of modern LLM families: Gemma (Gen 1–4), Qwen (Gen 1–3), MiniCPM (Gen 1–5), G9 (v1–v3), Ling (1.0–2.0), Inkling (Gen 1), DeepSeek (V1–V3, R1), GLM (1–4), and Kimi (V1–K2.5).
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
