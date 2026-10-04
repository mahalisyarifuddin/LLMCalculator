# Model-Synthesis Update: Zhipu GLM-4.5 → GLM-5.3, Qwen3-Next → Qwen3.8, InclusionAI Ling 2.0 → 3.0, DeepSeek V4.1-Flash and Hy4-preview — 2026-10-04

**Scope:** Newer-generation open-weight language backbones from Z.ai / Zhipu (GLM MoE line) and Alibaba (Qwen3-Next and Qwen3.5) added as interpolation anchors in the multi-generational auto-estimate synthesis in `LLMCalculator.html`, plus both READMEs. The reference table grows from 72 to **98 strictly sorted, unique anchors** (sections 1–5 cover the first 90; section 6 covers the 2026 Q3 generation sweep; section 7 adds the Gemma 3 / Gemma 4 interleaved-attention anchors).

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

- All 98 anchors verified strictly ascending and unique.
- Script block passes `node --check`.
- Sweep of 0.1B → 2800B in 0.1B steps: every interpolated point yields positive layers/hidden and a finite, non-negative KV estimate (0 bad points).

## 5. Sources

- Z.ai / Zhipu `zai-org` Hugging Face configs: `GLM-4.5-Air` (`glm4_moe`, 46L × 4096H, 96 Q / 8 KV), `GLM-4.6` / `GLM-4.7` (92L × 5120H, 160 experts), `GLM-4.7-Flash` (`glm4_moe_lite`, 47L × 2048H, `kv_lora_rank` 512 / `qk_rope_head_dim` 64), `GLM-5.2` (`glm_moe_dsa`, 78L × 6144H, `index_topk` 2048, `index_topk_freq` 4).
- GLM-4.6 / GLM-4.7-Flash / GLM-5 → 5.3 release notes and Artificial Analysis model pages (parameter counts, context windows, licensing).
- Qwen Hugging Face model cards: `Qwen3-Next-80B-A3B-Instruct/Thinking`, `Qwen3.5-397B-A17B`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`, `Qwen3.5-27B`, `Qwen3.5-9B`, `Qwen3.5-4B`, `Qwen3.5-2B`, `Qwen3.5-0.8B`, `Qwen3.6-27B`, `Qwen3.6-35B-A3B` (hidden layout, Gated DeltaNet / Gated Attention head counts and head dims, expert configuration, context lengths).
- InclusionAI Hugging Face configs: `Ling-1T` (`bailing_moe`, 80L × 8192H, 64 Q / 8 KV × 128), `Ling-2.6-1T` (`bailing_hybrid`, `layer_group_size` 8, `kv_lora_rank` 512 / `qk_rope_head_dim` 64), `Ling-3.0-flash` (`bailing_hybrid`, 42L × 2560H, `layer_group_size` 6); NVIDIA NeMo AutoModel Ling 2.0 coverage page (mini 20L × 2048H, flash 32L × 4096H); Ring-linear 2.0 report arXiv:2510.19338.


---

## 6. 2026 Q3 generation sweep — Qwen 3.7 / 3.8, DeepSeek V4.1-Flash, Hy4-preview

This pass walked every tracked family looking for a newer open-weight generation than the one already anchored. Four anchors were added, several existing anchors absorbed new models as fold-ins, and a number of recent releases were deliberately **not** anchored (section 6.4).

### 6.1 New anchors

| Size anchor | Model | Layers × hidden | Attention / KV details | Context | `attn` |
|---:|---|---:|---|---:|---|
| **180.0** | Qwen3.8-Flash-Next (Aug 26 2026) | 48 × 2560 | `qwen4_exp_text`: 180B resident = 125B-A6B LM + 51B n-gram table + 4B MTP; `full_attention_interval` 4 (3 Gated DeltaNet : 1 gated full attention), 24 Q / 2 KV × 256; **Qwen Sparse Attention** indexer (`indexer_kv_heads` 1, `indexer_head_dim` 128, `indexer_compress_ratio` 4, budget 2048); 512 experts top-10 + 1 shared | 262K → 1M | `qwen_qsa` (new) |
| **552.0** | DeepSeek-V4.1-Flash (Sep 10 2026, MIT) | 40 × 5120 | **Causal encoder-decoder**: 20 encoder + 20 decoder layers; CSA2 Full / Reindex / Reuse with cross-layer KV sharing + hierarchical sparse indexer; 384 routed experts top-6 + 1 shared; 8B active prefill / 16B decode; +196B Engram memory and a 32-layer DeepSeek-ViT tower (≈748–763B on disk) | 1M | `ced_csa2` (new) |
| **770.0** | Hy4-preview (Aug 28 2026, Apache 2.0) | 78 × 6144 | 770B / A49B, 1 dense + 77 MoE blocks (256 experts ×8 + 1 shared, FFN 18432), 64 heads; **Gated DSA** with IndexCache (q compression 2048, KV compression 512 + RoPE 64, 32 × 128-d indexer, top-k 2048), 4 iHC residual streams, +10B MTP | 1M | `mla_dsa` (reused) |
| **2400.0** | Qwen3.8-Max (Aug 12–13 2026) | 92 × 8192 | 2.4T / A95B, `qwen35moe`; 23 × (3 Gated DeltaNet → 1 Gated Attention), 64 Q / **4 KV** × 256 (RoPE 64, θ=1e7); GDN 16 QK / 128 V × 128; 512 experts top-10 + 1 shared | 262K → 1.01M | `qwen_gdn_4kv` (reused) |

### 6.2 New KV formulas

```js
// qwen_qsa — Qwen3.8-Flash-Next
fullLayers = max(1, round(layers / 4));
elements   = fullLayers * (context * 2 * (256 + 256) + (context / 4) * 128);

// ced_csa2 — DeepSeek-V4.1-Flash
elements   = layers * context * 44.5;   // 890 B/token at FP4 over 40 layers
```

`ced_csa2` is calibrated directly against the published figure: the model card quotes a **~890 bytes/token** global KV cache at FP4 (0.5 B/element) ⇒ 1780 elements/token over 40 layers ⇒ 44.5 elements per layer per token. The formula scales with layer count so interpolation across neighbouring anchors stays smooth, and the calculator reproduces 890 B/token exactly at FP4.

Worked examples (fp16, 32K context):

| Anchor | KV @ 32K fp16 | KV @ 1M fp8 |
|---|---:|---:|
| 180B Qwen3.8-Flash-Next | 0.77 GiB | 12.4 GiB |
| 552B DeepSeek-V4.1-Flash | 0.11 GiB | 1.7 GiB |
| 770B Hy4-preview | 2.90 GiB | 46.4 GiB |
| 2.4T Qwen3.8-Max | 2.88 GiB | 46.0 GiB |

### 6.3 Fold-ins (no new anchor needed)

- **Qwen3.8-27B** (Aug 14 2026, Apache 2.0) — 27.3B, 65 blocks × 5120, 24 Q / 4 KV, native vision, 262K → 1M ⇒ folded into the existing **27.5** anchor.
- **IBM Granite 4.2** (Aug 25 2026) — dense 3B / 8B / 30B, 131K context ⇒ folded into **3.0 / 8.1 / 30.0**.
- **Meta Muse Glimmer 30B** (Aug 10 2026) — 30B multimodal ⇒ existing **30.0** anchor (the 29.6 Glimmer geometry anchor already covers its SWA layout).
- **gpt-oss 2026 refresh** (Jun 9 2026) — gpt-oss-8b dense, 20b-a3b, 120b-a12b at 256K context reuse the same 2880-hidden / 64-dim GQA-8 family shape ⇒ existing anchors.
- **MiMo-V2.6-Flash / V2.6-Pro** ⇒ existing **309.0 / 1020.0**; **MiniCPM5-2B** ⇒ **3.0**; **Cohere Command A+** (open weights Sep 22 2026) ⇒ existing **218.0**; Mistral Medium 3.5 / Small 4 / Large 3 / Devstral 2 ⇒ already anchored.

### 6.4 Checked and deliberately not anchored

| Model | Why not |
|---|---|
| **Qwen 3.7** (3.7-Max May 20 2026, 3.7-Flash / Plus) | Generation skipped for open weights — API-only. Recorded as a documented gap rather than guessed geometry. |
| **StepFun Step 5 Preview** (600B-A27B) | Weights not released at the time of writing (announced for Oct 15 2026). |
| **Kimi K2.8-Preview / K3.x** | Architecture undisclosed; K3 (2.8T, 93L × 7168H) remains the newest documented Kimi anchor. |
| **GLM-5.3-Flash / GLM-5.3-FlashX** (open weights Sep 18 2026) | Published with a 1M context but no layer/hidden geometry; GLM-5.3 shares the 744B `mla_dsa` anchor. |
| **Meta Muse Spark 1.2 / 1.3** | Closed weights (open release promised ~Q4 2026). |
| **MiniMax H3 / Hailuo 3.0** | A video-generation model (dense 33B H3-Omni, 50 × 5376), not a language backbone — out of this calculator's scope. |
| **DiffusionGemma 26B-A4B** | Text-diffusion decoder; its decoding cost model is not the autoregressive KV cache this tool estimates. |
| **"Llama 5"** | Does not exist; the pages claiming a 600B / 5M-context April 2026 release are low quality. |

A condensed version of this table ships in the app's reference-data panel (EN and ID) so the gaps are visible in-product, not only in this document.

### 6.5 Sources

- `Qwen/Qwen3.8-Flash-Next` `config.json` (`qwen4_exp_text`: 48L × 2560H, `full_attention_interval` 4, 24 Q / 2 KV × 256, `indexer_budget` 2048 / `indexer_head_dim` 128 / `indexer_kv_heads` 1 / `indexer_compress_ratio` 4, `hc_count` 4, vocab 248,320, Qwen Community License 1.0).
- Unsloth `Qwen3.8-2.4T-A95B-GGUF` model card and the MaxText `qwen35moe` support PR (92 layers × 8192 hidden, 23 × (3 GDN → 1 gated attention), 64 Q / 4 KV × 256, 512 experts top-10 + 1 shared).
- Qwen3.8-27B release notes (27.3B, 65 × 5120, 24 Q / 4 KV, Apache 2.0).
- DeepSeek-V4.1-Flash model card and technical report (552B backbone, 20 + 20 causal encoder-decoder, 5120 hidden, vocab 129,280, CSA2 Full/Reindex/Reuse, 890 B/token FP4 global KV, 196B Engram, DSpark draft head).
- `github.com/Tencent-Hunyuan/Hy4-preview` repository and model card (78 layers, 6144 hidden, 770B/A49B, Gated DSA + IndexCache, KV compression 512 + RoPE 64, indexer 32 × 128, top-k 2048, 256 experts ×8 + shared, Apache 2.0).
- Release trackers and roundups for the not-anchored set: Qwen 3.7 API-only coverage, StepFun Step 5 Preview announcement, GLM-5.3-Flash/FlashX release notes, IBM Granite 4.2 announcement, Meta Muse Spark / Glimmer notes, MiniMax Hailuo 3.0 (H3) model card, DiffusionGemma announcement.


---

## 7. Gemma 3 / Gemma 4 — interleaved sliding-window anchors

Gemma was the last tracked family whose newer generations were still being interpolated from the Gemma 2 27B anchor (46 × 4608), which badly mis-sizes both the geometry and the KV cache. Gemma 3 and Gemma 4 interleave **five sliding-window layers per global layer** (`sliding_window_pattern` 6), so most of the stack never grows past a 1,024-token window.

### 7.1 New anchors

| Size anchor | Model | Layers × hidden | Attention / KV details | Context | `attn` |
|---:|---|---:|---|---:|---|
| **12.0** | Gemma 3 12B | 48 × 3840 | 8 KV × 256-d, window 1024, 40 local + 8 global | 128K | `gemma3_swa` (new) |
| **25.2** | Gemma 4 26B-A4B (25.2B total / 3.8B active) | 30 × 2816 | 25 local + 5 global; local 8 KV × 256-d, global 2 KV × 512-d unified K=V + p-RoPE; 128 experts top-8 + 1 shared | 256K | `gemma4_swa` (new) |
| **27.2** | Gemma 3 27B | 62 × 5376 | 32 Q / 16 KV × 128-d, window 1024, 52 local + 10 global, FFN 21504, vocab 262,208 (+0.4B SigLIP tower) | 128K | `gemma3_swa` |
| **30.7** | Gemma 4 31B dense (30.7B) | 60 × 5376 | 50 local + 10 global; local 16 KV × 256-d, global 4 KV × 512-d unified K=V + proportional RoPE; FFN 21504 (+0.55B vision tower), Apache 2.0 | 256K | `gemma4_swa` |

The sizes 12.0 / 25.2 / 27.2 / 30.7 sit in free gaps (10.0 → 14.0, 24.0 → 26.0, 27.0 → 27.5, 30.0 → 31.2), so no existing anchor had to move. The Gemma 2 27B anchor at 27.0 stays as-is.

### 7.2 KV formulas

```js
globalLayers = max(1, round(layers / 6));
localLayers  = layers - globalLayers;
localTokens  = min(context, 1024);

// gemma3_swa — 27B is 16 KV × 128-d, 12B is 8 KV × 256-d: 4096 K+V elements/token/layer either way
elements = (localLayers * localTokens + globalLayers * context) * 4096;

// gemma4_swa — wider local heads, narrow unified-K=V global heads
kv       = hidden >= 5376 ? 16 : 8;
elements = localLayers * localTokens * kv * (256 + 256)
         + globalLayers * context * (kv / 4) * 512;
```

Worked examples:

| Anchor | KV @ 32K fp16 | KV @ 256K bf16 |
|---|---:|---:|
| Gemma 3 12B | 2.31 GiB | 16.31 GiB |
| Gemma 3 27B | 2.91 GiB | 20.41 GiB |
| Gemma 4 26B-A4B | 0.51 GiB | 2.70 GiB |
| Gemma 4 31B | 2.03 GiB | 10.78 GiB |

The Gemma 4 31B figure decomposes into 50 local layers × 16 MiB (a full 1,024-token window at bf16) plus 10 global layers × 1 GiB at 256K — the per-layer breakdown published in third-party memory analyses of the release. Gemma 4 31B therefore costs roughly **half** the KV of Gemma 3 27B at the same context despite being larger and supporting twice the window, which is exactly the effect these anchors exist to capture.

### 7.3 Not anchored

- **Gemma 4 12B** — announced and released after the 31B / 26B-A4B pair; layer/hidden geometry not published at the time of writing, so it is listed in the app's "checked, not anchored" line rather than guessed.
- **Gemma 4 E2B / E4B** — per-layer-embedding (PLE) models whose effective parameter count differs from their resident size; the existing 2B–4B anchors already cover their decoder shape.
- **DiffusionGemma 26B-A4B** — text diffusion, not autoregressive KV decoding.

### 7.4 Sources

- Gemma 4 Technical Report (arXiv:2607.02770) — family composition (E2B, E4B, 12B, 31B dense; 26B-A4B MoE), MTP drafter, training setup.
- Hugging Face `transformers` `Gemma4TextConfig` documentation and the Gemma 4 release blog — interleaved local/global attention, dual RoPE (standard for sliding layers, pruned/proportional for global layers), per-layer embeddings, 256K context for 31B / 26B-A4B.
- Third-party architecture breakdowns of the Gemma 4 configs — `num_hidden_layers` 60 / 30, `num_key_value_heads` 16 / 8, `head_dim` 256, `num_global_key_value_heads` 4 / 2, `global_head_dim` 512, `sliding_window` 1024 with a 5-local : 1-global `layer_types` pattern, 128 experts top-8 + 1 shared, and the per-layer KV memory breakdown used to validate the formula above.
- `google/gemma-3-27b-it` `config.json` — 62 layers, hidden 5376, 32 Q / 16 KV × 128, `sliding_window` 1024, `sliding_window_pattern` 6, FFN 21504, vocab 262,208.
