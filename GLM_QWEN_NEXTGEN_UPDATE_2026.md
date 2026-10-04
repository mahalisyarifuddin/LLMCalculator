# Model-Synthesis Update: Zhipu GLM-4.5 → GLM-5.3 and Qwen3-Next / Qwen3.5 — 2026-10-04

**Scope:** Newer-generation open-weight language backbones from Z.ai / Zhipu (GLM MoE line) and Alibaba (Qwen3-Next and Qwen3.5) added as interpolation anchors in the multi-generational auto-estimate synthesis in `LLMCalculator.html`, plus both READMEs. The reference table grows from 72 to **81 strictly sorted, unique anchors**.

Two new KV-cache shapes are introduced: `qwen_gdn` / `qwen_gdn_4kv` (hybrid Gated DeltaNet + Gated Attention) and `mla_dsa` (MLA latent cache + DeepSeek Sparse Attention indexer).

## 1. Verified architecture data

### Z.ai / Zhipu GLM

| Model | Total / active | Layers × hidden | Attention / KV details | Context | Auto-estimate treatment |
|---|---:|---:|---|---:|---|
| GLM-4.5-Air | 106B / A12B | 46 × 4096 | GQA, 96 Q / 8 KV × 128, partial RoPE 0.5; 128 experts ×8 + 1 shared | 131K | New 106B GQA-8 anchor |
| GLM-4.5 / 4.6 / 4.7 | 355–358B / A32B | 92 × 5120 | GQA, 96 Q / 8 KV × 128; 160 experts ×8 + 1 shared | 200K | New 357B GQA-8 anchor |
| GLM-4.7-Flash | 31.2B / A3B | 47 × 2048 | **MLA** (`kv_lora_rank` 512 + `qk_rope_head_dim` 64), 64 experts ×4 + 1 shared | 202K | New 31.2B `mla` anchor |
| GLM-5 / 5.1 / 5.2 / 5.3 | ~744B / A40B | 78 × 6144 | **MLA + DSA**: 512 + 64 latent cache per layer, 128-d sparse indexer (`index_topk` 2048) computed once per four layers ("IndexShare"); 256 experts ×8 + 1 shared, 3 dense blocks | 1M | New 744B `mla_dsa` anchor |

GLM-4.7-Flash is a `glm4_moe_lite` checkpoint — despite its small size it stores an MLA latent cache (576 elements/layer/token), not a GQA cache, which is why it carries a per-anchor override instead of inheriting the 2048-hidden GQA-4 rule.

GLM-5.x sparsity (top-2048 token selection) reduces attention **FLOPs**, not the stored cache, so the full MLA latent cache is still costed for every layer; only the indexer is amortised 1-in-4.

### Alibaba Qwen3-Next / Qwen3.5

| Model | Total / active | Layers × hidden | Attention / KV details | Context | Auto-estimate treatment |
|---|---:|---:|---|---:|---|
| Qwen3-Next-80B-A3B | 80B (81B weights) / A3B | 48 × 2048 | 12 × (3 Gated DeltaNet → 1 Gated Attention); 16 Q / 2 KV, head dim 256 | 262K → 1M | New 81B `qwen_gdn` anchor |
| Qwen3.5-27B | 27B dense | 64 × 5120 | 16 × (3 Gated DeltaNet → 1 Gated Attention); 24 Q / 4 KV, head dim 256 | 262K | New 27.5B `qwen_gdn_4kv` anchor |
| Qwen3.5-35B-A3B | 35B / A3B | 40 × 2048 | 10 × (3 GDN → 1 Gated Attention); 16 Q / 2 KV × 256; 256 experts ×8 + 1 shared | 262K → 1M | New 35.5B `qwen_gdn` anchor |
| Qwen3.5-122B-A10B | 122B / A10B | 48 × 3072 | 12 × (3 GDN → 1 Gated Attention); 32 Q / 2 KV × 256; 256 experts ×8 + 1 shared | 262K → 1M | New 122B `qwen_gdn` anchor |
| Qwen3.5-397B-A17B | 397B / A17B | 60 × 4096 | 15 × (3 GDN → 1 Gated Attention); 32 Q / 2 KV × 256; 512 experts ×10 + 1 shared | 262K → 1M | New 397B `qwen_gdn` anchor |

Gated DeltaNet layers are linear-attention layers with a **fixed-size recurrent state**: their memory does not grow with context. Only the 1-in-4 softmax ("Gated Attention") layers hold a per-token KV cache.

## 2. New KV formulas

```
KV (qwen_gdn / qwen_gdn_4kv) = round(layers / 4) × context × kv_heads × (256 K + 256 V) × bytes
                               kv_heads = 2 (MoE sizes) or 4 (Qwen3.5-27B dense)

KV (mla_dsa)                 = layers × context × 576 × bytes
                               + round(layers / 4) × context × 128 × bytes      (DSA indexer)
```

Worked examples at 32K context, FP16 KV:

| Anchor | Computation | KV |
|---|---|---:|
| GLM-4.7-Flash 31.2B | 47 × 32768 × 576 × 2 | 1.65 GiB |
| Qwen3.5-35B-A3B | 10 × 32768 × 1024 × 2 | 0.63 GiB |
| Qwen3-Next-80B-A3B | 12 × 32768 × 1024 × 2 | 0.75 GiB |
| Qwen3.5-397B-A17B | 15 × 32768 × 1024 × 2 | 0.94 GiB |
| GLM-4.5 / 4.6 / 4.7 357B | 2 × 92 × 1024 × 32768 × 2 | 11.50 GiB |
| GLM-5.2 744B | 78 × 32768 × 576 × 2 + 20 × 32768 × 128 × 2 | 2.90 GiB |

The contrast is the point of this update: a 397B Qwen3.5 checkpoint costs **less KV at 32K than a 35B GQA-8 model**, while a 357B GQA-8 GLM-4.6 costs 12× more than the similarly sized Qwen3.5.

## 3. Anchor placement notes

Anchors must stay strictly sorted and unique, so two entries are nudged off their nominal size:

- **27.5B** — Qwen3.5-27B dense sits next to the existing 27.0B Gemma 2 27B anchor (46L × 4608H).
- **35.5B** — Qwen3.5-35B-A3B sits next to the existing 35.0B Command R / Aya 23 anchor (40L × 8192H).

Both offsets are smaller than the interpolation step in that region and are documented inline in `referenceConfigs`.

`81.0B` is used for Qwen3-Next-80B-A3B (its published safetensors size) so it does not collide with the 80.0B Hunyuan-A13B anchor.

## 4. Verification

- All 81 anchors verified strictly ascending and unique.
- Script block passes `node --check`.
- Sweep of 0.1B → 2800B in 0.1B steps: every interpolated point yields positive layers/hidden and a finite, non-negative KV estimate (0 bad points).

## 5. Sources

- Z.ai / Zhipu `zai-org` Hugging Face configs: `GLM-4.5-Air` (`glm4_moe`, 46L × 4096H, 96 Q / 8 KV), `GLM-4.6` / `GLM-4.7` (92L × 5120H, 160 experts), `GLM-4.7-Flash` (`glm4_moe_lite`, 47L × 2048H, `kv_lora_rank` 512 / `qk_rope_head_dim` 64), `GLM-5.2` (`glm_moe_dsa`, 78L × 6144H, `index_topk` 2048, `index_topk_freq` 4).
- GLM-4.6 / GLM-4.7-Flash / GLM-5 → 5.3 release notes and Artificial Analysis model pages (parameter counts, context windows, licensing).
- Qwen Hugging Face model cards: `Qwen3-Next-80B-A3B-Instruct/Thinking`, `Qwen3.5-397B-A17B`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`, `Qwen3.5-27B` (hidden layout, Gated DeltaNet / Gated Attention head counts and head dims, expert configuration, context lengths).
