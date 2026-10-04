# Model-Synthesis Update: Zhipu GLM-4.5 → GLM-5.3, Qwen3-Next / Qwen3.5 / Qwen3.6, and InclusionAI Ling 2.0 → 3.0 — 2026-10-04

**Scope:** Newer-generation open-weight language backbones from Z.ai / Zhipu (GLM MoE line) and Alibaba (Qwen3-Next and Qwen3.5) added as interpolation anchors in the multi-generational auto-estimate synthesis in `LLMCalculator.html`, plus both READMEs. The reference table grows from 72 to **90 strictly sorted, unique anchors**.

Four new KV-cache shapes are introduced: `qwen_gdn` / `qwen_gdn_4kv` (hybrid Gated DeltaNet + Gated Attention), `mla_dsa` (MLA latent cache + DeepSeek Sparse Attention indexer), and `ling_kda_mla` / `ling_lightning_mla` (hybrid linear attention + one gated-MLA layer per layer group).

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

### Alibaba Qwen3.5 small sizes and Qwen3.6

| Model | Params | Layers × hidden | Gated Attention heads | Auto-estimate treatment |
|---|---:|---:|---|---|
| Qwen3.5-0.8B | 0.8B dense | 24 × 1024 | 8 Q / 2 KV × 256 | New 0.8B `qwen_gdn` anchor |
| Qwen3.5-2B | 2B dense | 24 × 2048 | 8 Q / 2 KV × 256 | New 2.0B `qwen_gdn` anchor |
| Qwen3.5-4B | 4B dense | 32 × 2560 | 16 Q / 4 KV × 256 | New 4.2B `qwen_gdn_4kv` anchor (offset from the 4.0B Gemma 3 / Qwen3 anchor) |
| Qwen3.5-9B | 9B dense | 32 × 4096 | 16 Q / 4 KV × 256 | New 9.0B `qwen_gdn_4kv` anchor |
| Qwen3.6-27B | 27B dense | 64 × 5120 | 24 Q / 4 KV × 256 | Identical geometry to Qwen3.5-27B — folded into the 27.5B anchor |
| Qwen3.6-35B-A3B | 35B / A3B | 40 × 2048 | 16 Q / 2 KV × 256 | Identical geometry to Qwen3.5-35B-A3B — folded into the 35.5B anchor |

### InclusionAI Ling / Ring (previously documented but unanchored)

| Model | Total / active | Layers × hidden | Attention / KV details | Auto-estimate treatment |
|---|---:|---:|---|---|
| Ling-mini-2.0 | 16.4B / A1.4B | 20 × 2048 | GQA, 16 Q / 4 KV × 128, half RoPE | New 16.4B `gqa_4` anchor |
| Ling-flash-2.0 | 100–104B / A6B | 32 × 4096 | GQA, 32 Q / 4 KV × 128 | New 100B `gqa_4` anchor |
| Ling-1T | 1T / A50B | 80 × 8192 | GQA, 64 Q / 8 KV × 128 (no MLA), 4 dense + 76 sparse layers | New 1005B GQA-8 anchor |
| Ling 2.5 / 2.6-1T | 1T / A63B | 80 × 8192 | `bailing_hybrid`, `layer_group_size` 8 → 70 lightning-attention + 10 gated-MLA (512 + 64) | New 1010B `ling_lightning_mla` anchor |
| Ling-3.0-flash | 124B / A5.1B | 42 × 2560 | `bailing_hybrid`, `layer_group_size` 6 → 35 Kimi Delta Attention + 7 gated-MLA (512 + 64) | New 124B `ling_kda_mla` anchor |

Ring-mini-linear-2.0 (16B) and Ring-flash-linear-2.0 (104B) share the Ling sizes with 1:4 / 1:7 linear-to-softmax stacks; they are noted inline rather than given separate anchors, because the table cannot hold two different attention shapes at one size.

Ling-1T and Ling 2.6 are deliberately two anchors five billion apart: both are 1T-class, but the 2.0-generation Ling-1T pays full GQA-8 KV on all 80 layers, while the hybrid-linear 2.5/2.6 generation caches only 10 layers. Their published KV figure — 11.2 KiB/token in BF16 — matches the formula below exactly (10 × 576 × 2 bytes = 11.25 KiB).

## 2. New KV formulas

```
KV (qwen_gdn / qwen_gdn_4kv) = round(layers / 4) × context × kv_heads × (256 K + 256 V) × bytes
                               kv_heads = 2 (MoE sizes) or 4 (Qwen3.5-27B dense)

KV (mla_dsa)                 = layers × context × 576 × bytes
                               + round(layers / 4) × context × 128 × bytes      (DSA indexer)

KV (ling_kda_mla)            = round(layers / 6) × context × 576 × bytes        (Ling 3.0)
KV (ling_lightning_mla)      = round(layers / 8) × context × 576 × bytes        (Ling 2.5 / 2.6)
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
| Ling-3.0-flash 124B | 7 × 32768 × 576 × 2 | 0.25 GiB |
| Ling-1T (2.0, GQA-8) | 2 × 80 × 1024 × 32768 × 2 | 10.00 GiB |
| Ling 2.6-1T (hybrid) | 10 × 32768 × 576 × 2 | 0.35 GiB |

The contrast is the point of this update: a 397B Qwen3.5 checkpoint costs **less KV at 32K than a 35B GQA-8 model**, while a 357B GQA-8 GLM-4.6 costs 12× more than the similarly sized Qwen3.5.

## 3. Anchor placement notes

Anchors must stay strictly sorted and unique, so two entries are nudged off their nominal size:

- **27.5B** — Qwen3.5-27B dense sits next to the existing 27.0B Gemma 2 27B anchor (46L × 4608H).
- **35.5B** — Qwen3.5-35B-A3B sits next to the existing 35.0B Command R / Aya 23 anchor (40L × 8192H).

Both offsets are smaller than the interpolation step in that region and are documented inline in `referenceConfigs`.

`81.0B` is used for Qwen3-Next-80B-A3B (its published safetensors size) so it does not collide with the 80.0B Hunyuan-A13B anchor.

## 4. Verification

- All 90 anchors verified strictly ascending and unique.
- Script block passes `node --check`.
- Sweep of 0.1B → 2800B in 0.1B steps: every interpolated point yields positive layers/hidden and a finite, non-negative KV estimate (0 bad points).

## 5. Sources

- Z.ai / Zhipu `zai-org` Hugging Face configs: `GLM-4.5-Air` (`glm4_moe`, 46L × 4096H, 96 Q / 8 KV), `GLM-4.6` / `GLM-4.7` (92L × 5120H, 160 experts), `GLM-4.7-Flash` (`glm4_moe_lite`, 47L × 2048H, `kv_lora_rank` 512 / `qk_rope_head_dim` 64), `GLM-5.2` (`glm_moe_dsa`, 78L × 6144H, `index_topk` 2048, `index_topk_freq` 4).
- GLM-4.6 / GLM-4.7-Flash / GLM-5 → 5.3 release notes and Artificial Analysis model pages (parameter counts, context windows, licensing).
- Qwen Hugging Face model cards: `Qwen3-Next-80B-A3B-Instruct/Thinking`, `Qwen3.5-397B-A17B`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`, `Qwen3.5-27B`, `Qwen3.5-9B`, `Qwen3.5-4B`, `Qwen3.5-2B`, `Qwen3.5-0.8B`, `Qwen3.6-27B`, `Qwen3.6-35B-A3B` (hidden layout, Gated DeltaNet / Gated Attention head counts and head dims, expert configuration, context lengths).
- InclusionAI Hugging Face configs: `Ling-1T` (`bailing_moe`, 80L × 8192H, 64 Q / 8 KV × 128), `Ling-2.6-1T` (`bailing_hybrid`, `layer_group_size` 8, `kv_lora_rank` 512 / `qk_rope_head_dim` 64), `Ling-3.0-flash` (`bailing_hybrid`, 42L × 2560H, `layer_group_size` 6); NVIDIA NeMo AutoModel Ling 2.0 coverage page (mini 20L × 2048H, flash 32L × 4096H); Ring-linear 2.0 report arXiv:2510.19338.
