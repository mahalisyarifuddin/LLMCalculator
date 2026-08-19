# Model-Synthesis Update: Xiaomi MiMo, StepFun, Muse Glimmer, and IBM Granite — 2026-08-19

**Scope:** Open-weight autoregressive language backbones from Xiaomi MiMo, StepFun, Muse Glimmer, and IBM Granite added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` and both READMEs. The reference table grows from 54 to **72 strictly sorted, unique anchors**.

## 1. Verified architecture data

### Xiaomi MiMo

| Model | Total / active | Layers × hidden | Attention / KV details | Context | Auto-estimate treatment |
|---|---:|---:|---|---:|---|
| MiMo-7B | 7B dense | 36 × 4096 | GQA, 8 KV × 128 | 32K–48K | Folded into the existing 7B GQA-8 region |
| MiMo-V2-Flash | 309B / A15B | 48 × 4096 | 9 global layers with 4 KV heads; 39 SWA-128 layers with 8 KV heads; QK/V dims 192/128 | 256K | New 309B `mimo_swa` anchor |
| MiMo-V2.5 | 310B / A15B | 48 × 4096 | Same language backbone; adds 729M ViT, 261M audio encoder, and 329M MTP | 1M | New 310B `mimo_swa` anchor |
| MiMo-V2.5-Pro | 1.02T / A42B | 70 × 6144 | 10 global + 60 SWA-128; 8 KV heads; QK/V dims 192/128 | 1M | New 1020B `mimo_swa_pro` anchor |

MiMo hybrid KV cache uses the asymmetric cached width directly: each KV head stores **192 K + 128 V elements**, rather than assuming equal 128-d K/V heads.

### StepFun

| Model | Total / active | Layers × hidden | Attention / KV details | Context | Auto-estimate treatment |
|---|---:|---:|---|---:|---|
| Step3-VL-10B | 10B VLM | 36 × 4096 text | Qwen3-8B GQA-8 text backbone + 1.8B perception encoder | 64K | New 10B anchor |
| Step-3 | 321B VLM / A38B (316B LLM) | 61 × 7168 | MFA: 64 query heads share one 256-d K and one 256-d V head | 64K | New 321B `mfa` anchor |
| Step-3.5-Flash | 196.81B / A11B | 45 × 4096 | 12 full + 33 SWA-512 layers; 8 KV × 128; 3 dense + 42 MoE | 256K | New 196.8B `step_swa` anchor |
| Step-3.7-Flash | 198B VLM / A11B | 45 × 4096 text | Same 196B language backbone + 1.8B perception encoder | 256K | New 198B `step_swa` anchor |

Step-3 MFA cache is calculated as `layers × context × (256 K + 256 V) × bytes`. Step 3.5/3.7 retain the full prefix only in their 12 global layers; the other 33 layers retain at most 512 tokens.

### Muse Glimmer

| Model | Total | Layers × hidden | Attention / KV details | Context | Auto-estimate treatment |
|---|---:|---:|---|---:|---|
| Muse Glimmer 30B | ~29.6B including ~1.8B perception encoder | 52 × 6656 | 13 full + 39 SWA-2048 layers; GQA with 2 KV × 128 | 128K | New 29.6B `muse_swa` anchor |

Its cache formula retains the full prefix in one of every four layers and only the latest 2,048 tokens in the remaining layers.

### IBM Granite

| Family / model | Parameters | Layers × hidden | Attention | Auto-estimate treatment |
|---|---:|---:|---|---|
| Granite 4.0 350M dense | 0.35B | 28 × 1024 | 4 KV × 64 | New 0.35B `gqa_4_64d` anchor |
| Granite 3.0/3.1 1B-A400M | 1.3B / A0.4B | 24 × 1024 | 8 KV × 64 | New 1.3B `gqa_8_64d` anchor |
| Granite 4.0 1B dense | 1.6B actual | 40 × 2048 | 4 KV × 128 | New 1.6B GQA-4 anchor |
| Granite-SWASH-2B | 2.144B dense | 24 × 2560 | 7 full + 17 SWA-128; 4 KV × 128; sinks | New 2.14B `granite_swa_2b` anchor |
| Granite 3.0–3.3 2B dense | 2.5B actual | 40 × 2048 | 8 KV × 64 | New 2.5B `gqa_8_64d` anchor |
| Granite-SWASH-3B-A600M | 3.02B / A0.598B | 28 × 1280 | 8 full + 20 SWA-128; 4 KV × 64; sinks | New 3.02B `granite_swa_3b` anchor |
| Granite 3.0/3.1 3B-A800M | 3.3B / A0.8B | 32 × 1536 | 8 KV × 64 | New 3.3B `gqa_8_64d` anchor |
| Granite 3.x / 4.1 8B | 8.1B | 40 × 4096 | 8 KV × 128 | New 8.1B anchor |
| Granite Code 20B | 20B | 52 × 6144 | MQA, 1 KV × 128 | New 20B `mqa_128d` anchor |
| Granite Code 34B | 34B | 88 × 6144 | MQA, 1 KV × 128 | New 34B `mqa_128d` anchor |

Also documented in the synthesis:

- Granite Code 3B/8B, Granite 4.0 Micro/H-Micro/H-Tiny/H-Small, and Granite 4.1 3B/8B/30B.
- Several advertised Granite sizes overlap existing same-size representatives. For example, Granite 4.1 30B overlaps North Mini/Qwen3-30B-A3B, and Granite 4.0 H-Small 32B overlaps Qwen 32B. Duplicate numeric anchors are intentionally not allowed.
- Hybrid Granite 4.0 H models use 4 attention layers plus Mamba2 layers. Their Mamba recurrent state is not a token-growing KV cache; all-attention siblings/general anchors remain the conservative auto-estimate at overlapping sizes.

## 2. New cache buckets

The update adds these architecture-aware cache paths:

- `mimo_swa`: 9 full + 39 SWA-128 layers, different GA/SWA KV-head counts, asymmetric 192/128 K/V dimensions.
- `mimo_swa_pro`: 10 full + 60 SWA-128 layers, 8 KV heads, asymmetric 192/128 K/V dimensions.
- `step_swa`: 12 full + 33 SWA-512 layers, GQA-8.
- `muse_swa`: 13 full + 39 SWA-2048 layers, GQA-2.
- `granite_swa_2b` / `granite_swa_3b`: exact Granite SWASH full/SWA layer splits and 128/64-d head widths.
- `mfa`: Step-3 shared 256-d key plus 256-d value.
- `gqa_4_64d`, `gqa_8_64d`, and `mqa_128d`: compact Granite head layouts.

For all hybrid SWA paths:

```text
KV elements = full_layers × context × full_KV_width
            + SWA_layers × min(context, window) × SWA_KV_width
KV GiB      = KV elements × bytes_per_element / 1024³
```

## 3. Quantization labels

- MXFP4 documentation now includes MiMo-V2.5-Pro’s official **expert-only MXFP4** checkpoint. Non-expert layers remain at higher precision, so the calculator’s 0.53 byte/parameter option is still an idealized whole-model estimate.
- NVFP4 documentation now includes the official Step-3.7-Flash NVFP4 checkpoint alongside Poolside Laguna.

## 4. Scope decisions

- Included multimodal models when they expose an autoregressive language backbone whose weights and text KV cache are relevant to this calculator (MiMo-V2.5, Step VLMs, Muse Glimmer).
- Excluded standalone image/video/audio generators, embeddings, rerankers, time-series models, and third-party fine-tunes.
- Granite Vision/Speech task heads are documented through their shared Granite language bases rather than receiving separate duplicate-size anchors.
- Proprietary/API-only models without downloadable architecture configs are not used as anchors.

## 5. Verification

- **72** anchors; all sizes are finite, unique, and strictly increasing.
- Every new exact anchor resolves to its expected `(layers, hidden, attention bucket)`.
- Direct formula checks pass for MiMo V2/V2.5-Pro, Step 3.5, Step-3 MFA, Muse Glimmer, both Granite SWASH variants, and Granite Code MQA.
- **22 prior exact anchors** regression-tested unchanged; a 0.3B→3000B curve sweep returns finite, non-negative cache values.
- Representative results at native maximum context:
  - MiMo-V2-Flash, 256K, BF16 KV: **5.6488 GiB**.
  - MiMo-V2.5-Pro, 1M, FP8 KV: **25.0183 GiB**.
  - Step-3.5-Flash, 256K, BF16 KV: **12.0645 GiB**.
  - Muse Glimmer, 128K, BF16 KV: **1.7012 GiB**.
  - Step-3, 64K, BF16 KV: **3.8125 GiB**.
- Extracted JavaScript passes `node --check`; HTML diff passes `git diff --check`.

## 6. Primary sources

- Xiaomi MiMo-7B report: https://arxiv.org/abs/2505.07608
- MiMo-V2-Flash report: https://arxiv.org/abs/2601.02780
- MiMo-V2.5 model/config: https://huggingface.co/XiaomiMiMo/MiMo-V2.5
- MiMo-V2.5-Pro model/config: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- Step-3 report and checkpoint: https://arxiv.org/abs/2507.19427 and https://huggingface.co/stepfun-ai/step3
- Step-3.5-Flash report/checkpoint: https://arxiv.org/abs/2602.10604 and https://huggingface.co/stepfun-ai/Step-3.5-Flash
- Step-3.7-Flash checkpoint: https://huggingface.co/stepfun-ai/Step-3.7-Flash
- Step3-VL-10B checkpoint: https://huggingface.co/stepfun-ai/Step3-VL-10B
- Muse Glimmer checkpoint/config: https://huggingface.co/meta-models/Muse-Glimmer-30B
- Granite Code report: https://arxiv.org/abs/2405.04324
- IBM Granite official checkpoints/configs: https://huggingface.co/ibm-granite
