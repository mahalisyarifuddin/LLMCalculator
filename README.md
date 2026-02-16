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
5.  **System Overhead**: A reserved buffer (20%) for the OS, display, and framework overhead.

It iteratively calculates the maximum parameter count (in Billions) that fits within the remaining VRAM after accounting for overhead and KV cache.

## Quick Start
1.  Download `LLMCalculator.html`.
2.  Open it in any modern browser (Chrome, Edge, Firefox, Safari).
3.  Set your **GPU Memory** (VRAM Size) using the slider.
4.  Select the **GPU Type** (Discrete or Apple Silicon).
5.  Choose the **Model Precision** (Quantization) and **KV Cache** precision.
6.  Adjust the **Context Window** (e.g., 8K, 32K tokens).
7.  View the estimated **Max Parameters** and detailed memory breakdown.
8.  Use **Quick Presets** to simulate popular hardware configurations like RTX 4090, A100, H200, or M4 Max.

## Key Features
-   **Multi-language Support**: Toggle between English and Indonesian.
-   **Real-time Calculation**: Instant updates as you adjust sliders and dropdowns.
-   **Detailed Memory Breakdown**: Visualizes usage for System Overhead, KV Cache, and Model Weights.
-   **Hardware Presets**: One-click configuration for common GPUs.
-   **Advanced Options**: Support for various quantization formats (up to FP2) and separate KV cache precision.
-   **Model Estimation**: Suggests the closest known model architecture (e.g., Llama 3, Mistral, Qwen, DeepSeek) that fits the calculated parameters.
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
