# Model-Synthesis Update: Poolside Laguna — 2026-08-19

**Scope:** The entire Poolside **Laguna** series added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs:

1. **Laguna XS.2** (2026-04, Apache 2.0) and **Laguna XS 2.1** (2026-06, OpenMDW-1.1) — 33B-A3B.
2. **Laguna M.1** (2026-06, Apache 2.0, + `-base` checkpoint) — 225B-A23B.
3. **Laguna S 2.1** (2026-07, OpenMDW-1.1, + `-base` checkpoint) — 118B-A8B, 1M context.
4. New **NVFP4 (~0.56 bytes/param)** quantization option — Poolside ships NVFP4/FP8/INT4 variants for every Laguna size, and NVFP4 is the format its local deployments actually use.

All changes keep the app offline-only (no URL hash/storage/network) per the file-header constraint.

---

## 1. Verified architecture data (from poolside config.json + safetensors param counts)

| Model | Total params (safetensors) | Active | Layers | Hidden | Attention | Expert layout | Context |
|---|---|---|---|---|---|---|---|
| Laguna XS.2 | 33.44B | 3B | 40 (10 global + 30 SWA@512) | 2048 | GQA-8: 8 KV heads × 128d → KV width 1024 | 1 dense + 39 sparse; 256 experts ×8 + 1 shared | 262,144 |
| Laguna XS 2.1 | 33.44B | 3B | 40 (10 global + 30 SWA@512) | 2048 | GQA-8 (same); **KV cache natively FP8** | same as XS.2 | 262,144 |
| Laguna M.1 | 225.80B | 23B | 70 — **all-global attention** (3 dense + 67 sparse) | 4096 | GQA-8: 64 Q / 8 KV heads × 128d, softplus attn output gating | 256 experts ×16 + 1 shared | 262,144 |
| Laguna S 2.1 | 117.56B | 8B | 48 (12 global + 36 SWA@512) | 3072 | GQA-8 (48 Q on global / 72 Q on sliding, 8 KV × 128d) | 1 dense + 47 sparse; 256 experts ×10 + 1 shared | 1,048,576 (262K free tier) |

All four share the `laguna` model type (LagunaForCausalLM, Transformers ≥ v5.7), vocab 100,352, Muon optimizer, RoPE + YaRN, interleaved thinking. Sources: HF model cards & configs [1](https://huggingface.co/poolside/Laguna-M.1) [2](https://huggingface.co/poolside/Laguna-S-2.1) [3](https://huggingface.co/poolside/Laguna-XS-2.1) [4](https://huggingface.co/poolside/Laguna-XS.2); release coverage [5](https://www.marktechpost.com/2026/07/21/poolside-releases-laguna-s-2-1/) [6](https://kie.ai/blog/what-is-laguna-s-2-1) [7](https://dealroom.co/news/140333-poolside-releases-laguna-s-2-1-a-118b-open-weight-coding-model-pitched-a/).

Remaining HF repos are variants of the same four architectures and add no new synthesis points: FP8 / NVFP4 / INT4 / GGUF / MLX quants, DFlash speculative draft models (5-layer Llama-style), the two `-base` pre-trained checkpoints, and a `Laguna-tiny-per-element` config-fixture.

## 2. Changes applied

- **New synthesis points** (table 28 → 31, strictly sorted):
  - `{ size: 33, layers: 40, hidden: 2048 }` — Laguna XS.2 / XS 2.1 33B-A3B
  - `{ size: 118, layers: 48, hidden: 3072 }` — Laguna S 2.1 118B-A8B (1M ctx)
  - `{ size: 225, layers: 70, hidden: 4096 }` — Laguna M.1 225B-A23B (all-global attention)
- **Attention:** no new bucket needed — every Laguna is GQA-8 at 128-dim heads (KV width 1024), already the auto-estimate default. Notably Laguna XS is 2048-hidden like North Mini Code (GQA-4) and North Micro (GQA-8); the layer-count condition (`2048H && ≥49L → GQA-4`) keeps XS at 40 layers correctly on GQA-8, so the North disambiguation rule was reused unchanged.
- **New quantization option `NVFP4 (~0.56 bytes)`** (4-bit floats + FP8 scales per 16-element group + per-tensor scale ≈ 4.5 bpw) in Model Precision + `quantizationBytes.nvfp4 = 0.56`. Sanity: S 2.1 at NVFP4 → 118 × 0.56 × 1.05 ≈ **69.4 GB**, matching the published ~71 GB 4-bit footprint; M.1 → ~132 GB (1× B200 192 GB). Blackwell-native FP4, same slot in the ladder as MXFP4 (~0.53) for gpt-oss/Kimi K3.
- Doc strings (formulas / reference buckets / sources) updated **EN + ID**; README + README-id updated (family list + quantization mentions).

## 3. Notes & caveats

- **Sliding-window KV conservatism:** XS / XS 2.1 / S 2.1 use a 3:1 SWA(512):global interleave, so only their global layers' KV truly scales with context (e.g. S 2.1 at 1M bf16: real ≈ 48 GB vs our conservative 192 GB full-KV estimate). This matches the calculator's existing treatment of gpt-oss (window 128), Command A/A+, North, and Gemma sliding layers — full-context KV costing. XS 2.1 also ships **FP8 KV cache** natively, which the KV Cache dropdown already models.
- **Laguna M.1 is all-global attention** (sliding_window: 0), so its estimate is exact: 70L × KV width 1024 → 70 GB KV at 256K bf16.

## 4. Verification

- `referenceConfigs.auto` now 31 strictly-sorted points; all 3 new anchors resolve to their exact (layers, hidden) and attention type (GQA-8; XS correctly *not* GQA-4 despite 2048H).
- All 28 previous anchor resolutions regression-tested unchanged (incl. gpt-oss `gqa_8_64d`, North Mini GQA-4, DeepSeek/Kimi MLA).
- Curve sweep 0.3B → 3000B: no NaN/nonsense; Node syntax check passes.
- NVFP4 weight estimates match published 4-bit footprints (see §2).
