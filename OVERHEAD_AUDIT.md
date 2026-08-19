# Overhead Assumptions Audit Report

**Date**: August 19, 2026
**Audited by**: Web research into industry sources, technical documentation, and community knowledge bases
**Target**: LLMCalculator (`LLMCalculator.html`) — system overhead assumptions for three GPU platform types

---

## Summary of Findings

| Platform | Assumption | Verdict | Confidence |
|---|---|---|---|
| Discrete GPU (NVIDIA/AMD) | Default 1.5 GB overhead (configurable 0–16 GB) | ✅ **Reasonable** | High |
| Apple Silicon (M-series) | 25% of unified memory reserved for system | ✅ **Correct** | High |
| Snapdragon (Windows on Arm) | 50% of shared memory reserved for system | ⚠️ **Misleading / oversimplified** | Medium |

---

## 1. Discrete GPU (NVIDIA/AMD) — Default: 1.5 GB overhead

### Assumption in the tool
```
System Overhead: A configurable deduction (default 1.5 GB) for OS and display.
```
The slider ranges from 0 GB to 16 GB, defaulting to 1.5 GB. The preset for server GPUs (A100, H200) uses 0.5 GB.

### Evidence from web research

**OS/Display overhead:**
- SitePoint (April 2026): *"On a standard Linux or Windows desktop with a display attached, OS overhead consumes 400 to 800 MB (varies by display configuration)"* [1]
- SitePoint (April 2026): *"RTX 4060 Ti 16GB lands around 14GB usable"* (≈2 GB overhead); *"RTX 4070 12GB yields approximately 10.8GB"* (≈1.2 GB overhead) [1]

**CUDA/Vulkan/ROCm backend overhead:**
- LM Studio / Ollama guides (2025–2026): *"System overhead: The backend (llama.cpp, CUDA, Vulkan, etc.) adds roughly 0.5–1GB to VRAM use"* [2][3]

**Headless server (data center GPUs):**
- On headless Linux servers without a display compositor, overhead is minimal (≈0.3–0.8 GB for CUDA context + driver). The 0.5 GB preset for A100/H200 aligns with this.

**Combined total:**
- Desktop with display: **~1.0–2.5 GB** depending on resolution, multi-monitor, compositor, and driver
- Headless server: **~0.3–0.8 GB**

### Verdict: ✅ Reasonable

The 1.5 GB default sits in the middle of the typical range. The user can adjust it (0–16 GB slider), which covers edge cases from headless Linux servers (set to 0.5 GB or less) to memory-heavy multi-monitor Windows desktops (set higher). The presets for A100 (0.5 GB) and H200 (0.5 GB) appropriately reflect headless data-center usage.

**Minor note:** The LM Studio/Ollama overhead of 0.5–1 GB is specific to inference runtimes; the OS/display overhead of 400–800 MB is separate. These could theoretically be additive, meaning a combined total of ~1–2.5 GB on a desktop. The 1.5 GB default is a good middle-ground.

---

## 2. Apple Silicon (M-series) — 25% reserved for system (75% usable as VRAM)

### Assumption in the tool
```
Apple Silicon: Automatically reserves ~25% of Unified Memory for system use.
```
The code: `overheadGB = vramSize * 0.25`, `availableVram = vramSize * 0.75`

### Evidence from web research

**Primary source — Apple's Metal API behavior:**
- Stencel.io (April 2025, verified by multiple technical sources): *"macOS does not allow the GPU to use all 128 GB for graphics/compute tasks by default. In practice, about **75% of the physical memory** is the recommended maximum for GPU usage"* [4]
- The Metal API method `recommendedMaxWorkingSetSize` reflects exactly this ~75% cap [4][6]
- This is controlled by the kernel parameter `iogpu.wired_limit_mb`, which defaults to ~75% of physical RAM and can be overridden via `sudo sysctl iogpu.wired_limit_mb=N` [4][5][6]

**Corroborating community knowledge:**
- llama.cpp maintainers (ggml-org discussions): *"By default, you can use ~75% of the total RAM with the GPU"* [7]
- Solidaitech (April 2026): *"~75% of your unified memory is usable as GPU VRAM — macOS reserves the rest by default"* [5]
- Reddit r/LocalLLM: *"The GPU gets to use up to about 75% of the total RAM for configurations over 36 GiB total RAM"* [8]
- GitHub devnote (2025–2026): Documents `iogpu.wired_limit_mb` default = ~75% [6]

**Minor nuance — lower threshold:**
- Some sources suggest the cap is ~67% (2/3) for systems with ≤36 GB of RAM, and ~75% for >36 GB. The tool's flat 25% overhead is a simplification, but a reasonable one since most Apple Silicon machines used for LLM work have ≥32 GB.

### Verdict: ✅ Correct

The 25% reservation (75% usable) is well-documented and matches the default macOS Metal driver behavior exactly. This is the best-supported assumption in the tool.

---

## 3. Snapdragon (Windows on Arm) — 50% reserved for system

### Assumption in the tool
```
Snapdragon (Windows on Arm): Automatically reserves ~50% of Shared Memory for system partitioning.
```
The code: `overheadGB = vramSize * 0.50`, `availableVram = vramSize * 0.50`

### Evidence from web research

**What Windows/iGPU shared memory actually does:**
- On Windows, integrated GPUs (including Snapdragon's Adreno) have no dedicated VRAM. They use system RAM via "Shared GPU Memory" [9]
- Windows caps shared GPU memory at **up to 50% of total system RAM** as a **maximum allocation limit** [9][10]
- However, this is **NOT a reservation** — it's a cap on how much the GPU *can* allocate. The GPU only uses what it needs, and the CPU retains access to the rest [10]

**Snapdragon X Elite specifics:**
- Reddit r/snapdragon (July 2024): A user with 32 GB Snapdragon X Elite reports *"npu/gpu both report 15.7 GB shared memory"* — confirming the 50% cap [11]
- Official Snapdragon account response: *"Other integrated GPUs and Unified Memory Access (UMA) setups also show non-CPU processors like GPU and NPU as sharing ~8GB. This does **not** translate to the Memory graph in Task Manager automatically showing 50% memory usage"* [11]
- The AMD iGPU pattern is the same: *"The iGPU can use up to 50% of your system RAM as shared VRAM regardless of the UMA buffer setting — **if the CPU doesn't need that capacity**"* [10]

**Current state of Snapdragon LLM support:**
- Multiple sources (2025–2026) indicate that GPU/NPU acceleration for LLMs on Snapdragon X Elite is limited and problematic:
  - llama.cpp GPU (Vulkan) support on Snapdragon runs into memory issues [12]
  - AI-focused reviewer (macroco.de, April 2026): *"DO NOT buy a Copilot+ PC with Snapdragon X Elite"* for local LLM inference; most models run on CPU only [13]
  - LLM inference on Snapdragon is primarily CPU-based (via llama.cpp CPU backend or Windows ML CPU path) [12][13]
- CPU-based inference means the model uses regular system memory, not "shared GPU memory"

**What 50% actually means in practice:**
- On a 32 GB Snapdragon system:
  - Windows OS uses **~2–4 GB** at idle
  - Up to **16 GB** (50%) can be allocated to GPU if needed (it's a cap, not a reservation)
  - The remaining ~12 GB + the GPU's 16 GB pool (when not in use by GPU) are available for CPU tasks
  - For CPU-based LLM inference: **effectively ~28 GB available** (32 GB - ~4 GB OS overhead)
  - For GPU-based LLM inference: **up to 16 GB** maximum (limited by the 50% cap)

### Verdict: ⚠️ Misleading / Oversimplified

**The problem:** The tool treats Snapdragon the same way as Apple Silicon — "X% reserved for system, Y% available for model" — but the actual mechanism is fundamentally different:

| Aspect | Apple Silicon | Snapdragon X Elite |
|---|---|---|
| Mechanism | **Hard reservation** by macOS Metal driver | **Dynamic cap** by Windows shared GPU memory |
| GPU can use | 75% of total (hard limit by default) | Up to 50% of total (not always allocated) |
| CPU can use | 25% always reserved for system | 100% - OS usage - GPU usage at that moment |
| OS baseline usage | ~3–4 GB | ~2–4 GB |
| True usable for CPU LLM | ~75% of total (~3-4 GB OS overhead) | ~Total - ~4 GB OS overhead (much more than 50%) |
| True usable for GPU LLM | ~75% of total (confirmed) | Up to 50% of total (cap, but rarely used) |

**Impact of the current assumption:** For a 32 GB Snapdragon system:
- Tool reports: 16 GB overhead + 16 GB available
- Reality for CPU inference: ~4 GB overhead + ~28 GB available (tool underestimates by 12 GB!)
- Reality for GPU inference: ~4 GB OS + up to 16 GB GPU cap (tool overestimates GPU availability)

**Recommendation:** The 50% model is a reasonable approximation **if and only if** the user is running LLM inference exclusively on the GPU/NPU with the DirectML/Qualcomm AI Hub backend. However, since current Snapdragon LLM support is primarily CPU-based (with GPU support being immature), the 50% assumption significantly underestimates available memory.

A better approach might be:
1. Use a fixed OS overhead (~3 GB default, similar to discrete GPU) instead of a percentage-based reservation
2. Or add a note clarifying that the 50% figure represents the Windows shared GPU memory cap, not a system reservation
3. Or differentiate between CPU backend (use total RAM - OS overhead) and GPU backend (use up to 50% cap)

---

## References

[1] SitePoint, "10GB VRAM Local LLM: The Complete Setup Guide", April 2026
[2] localllm.in, "LM Studio VRAM Requirements for Local LLMs", October 2025
[3] localllm.in, "Ollama VRAM Requirements: Complete 2026 Guide", February 2026
[4] Stencel.io, "Apple silicon limitations with usage on local LLM", April 2025
[5] Solidaitech.com, "Your Mac's RAM is its GPU: How Much Unified Memory for Local AI?", April 2026
[6] GitHub - ivanopcode/devnote-override-macos-metal-vram-cap, 2025–2026
[7] ggml-org/llama.cpp Discussion #4167, "Performance of llama.cpp on Apple Silicon M-series"
[8] Reddit r/LocalLLM, "Can someone explain technically why Apple shared memory is so great", August 2025
[9] spheron.network, "What Is Shared GPU Memory? Dedicated vs Shared", January 2026
[10] Reddit r/AMDLaptops, "iGPU config uma buffer size", November 2023
[11] Reddit r/snapdragon, "Snapdragon X elite laptop only has 8GB of usable memory", July 2024
[12] ggml-org/llama.cpp Discussion #8273, "Performance of llama.cpp on Snapdragon X Elite/Plus", July 2024
[13] macroco.de, "The AI laptop that could not: My Snapdragon X Elite NPU Debacle", April 2026
---

## Implementation Status (Aug 19, 2026)

The following changes have been implemented in `LLMCalculator.html` to address the audit findings:

| Change | Detail |
|---|---|
| **Snapdragon overhead model** | Changed from fixed `50%` percentage-based reservation to configurable fixed overhead using `this.manualOverhead` (same as discrete GPU mode) |
| **Overhead slider enabled for Snapdragon** | Slider is now enabled when Snapdragon is selected (only disabled for Apple Silicon, where 25% is a hard macOS Metal limit) |
| **Default overhead for Snapdragon** | Snapdragon hardware preset now uses `overhead: 3.0`. When switching GPU type to Snapdragon, overhead auto-sets to 3.0 GB if current value is < 2.0 GB |
| **Clearer labeling** | Overhead row in memory breakdown shows subtitle: <q>GPU memory capped at 50% of total RAM</q> when Snapdragon is selected |
| **README.md updated** | Both English and Indonesian READMEs updated to reflect the corrected overhead model |
