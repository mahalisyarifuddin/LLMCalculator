# Auto-Estimate Anchor Audit + DeepSeek V4 — 2026-08-19

**Scope:** Verify every `referenceConfigs.auto` row in `LLMCalculator.html` against official `config.json` / technical reports, fix the rows that were factually wrong, and add **DeepSeek V4-Flash** and **DeepSeek V4-Pro**.

Table 44 → **50** strictly-sorted anchors. Offline-only constraint unchanged.

---

## 1. Method

Each existing anchor was checked against the model's official Hugging Face `config.json` or technical report (not third-party aggregator pages, which repeatedly disagreed with the configs). A row is **exact** when `(size, layers, hidden, attn)` matches a shipped checkpoint. A row is a **synthesis average** when the comment lists several families at nearby sizes — those are allowed to stay averaged, but comments that attributed the *wrong family's* dims to a size were corrected.

KV-critical fields (layers + attention bucket) were treated as higher priority than hidden size, because hidden size only affects KV when the bucket is MHA (`kvWidth = hidden`) or a heuristic classifier (`hidden ≤ 1536` → GQA-4, `hidden === 7168 && layers ≥ 61` → MLA, `hidden === 2880` → 64-dim GQA-8).

---

## 2. Verdicts

### 2.1 Exact rows that were already correct (kept)

| Size | Layers | Hidden | Attn | Official source |
|---|---|---|---|---|
| 0.5 | 24 | 896 | GQA-4 heuristic | Qwen2.5-0.5B (24L×896H, 2 KV) |
| 6.1 | 56 | 1024 | `gqa_2` | Trinity Nano |
| 7.0 | 32 | 4096 | GQA-8 | Llama 3 8B / Mistral 7B |
| 8.0 | 34 | 4096 | GQA-8 | Ministral-3 8B |
| 17.0 | 48 | 5120 | GQA-8 | Llama 4 Scout/Maverick active |
| 21.0 | 24 | 2880 | `gqa_8_64d` | gpt-oss-20b |
| 24.0 | 40 | 5120 | GQA-8 | Mistral Small 3.x 24B |
| 26.0 | 32 | 2048 | `gqa_4` | Trinity Mini |
| 30.0 | 49 | 2048 | `gqa_4` | North Mini Code; also matches Qwen3-30B-A3B (48L×2048H, 4 KV) |
| 33.0 | 40 | 2048 | GQA-8 | Laguna XS.2 / XS 2.1 |
| 35.0 | 40 | 8192 | GQA-8 | Command R / Aya Expanse 32B (R/Aya-23 are MHA; shared row stays GQA-8) |
| 47.0 | 32 | 4096 | GQA-8 | Mixtral 8x7B |
| 70.0 | 80 | 8192 | GQA-8 | Llama 2/3 70B |
| 111.0 | 64 | 12288 | GQA-8 | Command A |
| 117.0 | 36 | 2880 | `gqa_8_64d` | gpt-oss-120b |
| 118.0 | 48 | 3072 | GQA-8 | Laguna S 2.1 |
| 119.0 | 36 | 4096 | `mla_mistral` | Mistral Small 4 (MLA 256+64) |
| 123.0 | 88 | 12288 | GQA-8 | Devstral 2 / Medium 3.5 |
| 141.0 | 56 | 6400 | GQA-8 | Mixtral 8x22B |
| 218.0 | 32 | 4096 | GQA-8 | Command A+ |
| 225.0 | 70 | 4096 | GQA-8 | Laguna M.1 |
| 230.0 | 62 | 3072 | GQA-8 | MiniMax M2→M2.7 |
| 399.0 | 60 | 3072 | GQA-8 | Trinity Large |
| 405.0 | 126 | 16384 | GQA-8 | Llama 3.1 405B |
| 428.0 | 60 | 6144 | `gqa_4` | MiniMax M3 |
| 456.0 | 80 | 6144 | GQA-8 cons. | MiniMax Text-01 / M1 |
| 671.0 | 61 | 7168 | `mla` | DeepSeek-V3 / V3.2 / R1 |
| 675.0 | 61 | 7168 | `mla` | Mistral Large 3 |
| 2800.0 | 93 | 7168 | `mla` | Kimi K3 |

### 2.2 Rows that were wrong — fixed

| Size | Old | Official | Error | Fix |
|---|---|---|---|---|
| **0.6** | 28L × **1152H**, GQA-4 heuristic | Qwen3-0.6B: 28L × **1024H**, **8 KV** [1](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Hidden +2.5%; heuristic under-counted KV 2× (512 vs 1024) | `{ 0.6, 28, 1024, gqa_8 }` |
| **4.0** | 34L × **3072H** | Gemma 3 4B **34L×2560H** [2](https://huggingface.co/google/gemma-3-4b-it); Qwen3-4B **36L×2560H** [3](https://huggingface.co/Qwen/Qwen3-4B) | Hidden 20% high (3072 is Gemma-1-7B-class, not 4B) | `{ 4.0, 34, 2560 }` |
| **32.0** | 62L × **6144H** “Qwen 32B + Kimi 32B active” | Qwen2.5-32B / Qwen3-32B: **64L × 5120H**, 8 KV [4](https://huggingface.co/Qwen/Qwen3-32B-AWQ) | Mixing Kimi-*active* (1T model, 7168H MLA) into a 32B *total* bucket inflated hidden and under-counted layers | `{ 32.0, 64, 5120 }` |
| **104.0** | 64L × **10240H** MHA | Command R+ `c4ai-command-r-plus`: 64L × **12288H**, 96-head MHA [5](https://huggingface.co/blog/Andyrasika/memory-consumption-estimation) | MHA `kvWidth = hidden` → **17% KV underestimate** (80 GB vs 96 GB @ 32K bf16) | `{ 104.0, 64, 12288, mha }` |
| **130.0** | 75L × **10240H** “GLM-130B, Qwen 1.5 110B” | GLM-130B: **70L × 12288H** [6](https://keg.cs.tsinghua.edu.cn/jietang/publications/ICLR23-GLM-130B.pdf) | Averaging with Qwen-110B hid GLM's real 12288 width | `{ 130.0, 70, 12288 }` |
| **250.0** | 64L × 6144H “Qwen3-235B + Inkling-Small” | **Neither model**: Qwen3-235B is 94L×4096H GQA-4 [7](https://huggingface.co/Qwen/Qwen3-235B-A22B); Inkling-Small is 42L×4096H GQA-8 [8](https://huggingface.co/thinkingmachines/Inkling-Small) | Composite invented a model that does not exist | **Removed.** Replaced by exact 235 / 276 rows |
| **1000.0** | 67L × 7168H MLA “Inkling + Ling-1T + Kimi K2” | Kimi K2/K2.5: **61L × 7168H MLA** [9](https://huggingface.co/moonshotai/Kimi-K2.5); Inkling 975B: **66L × 6144H GQA**, *not* MLA [10](https://huggingface.co/thinkingmachines/Inkling) | Inkling was mis-tagged MLA (3.5× KV under-count). 67L matched no shipped 1T checkpoint | Split: `{ 975, 66, 6144 }` Inkling GQA-8; `{ 1000, 61, 7168, mla }` Kimi K2/K2.5 |

### 2.3 Missing models that distorted interpolation — added

| Size | Layers | Hidden | Attn | Why it belongs |
|---|---|---|---|---|
| **1.7** | 28 | 2048 | GQA-8 | Qwen3-1.7B official [11](https://huggingface.co/Qwen/Qwen3-1.7B). Was interpolated as ~38L×1756H GQA-4 |
| **235** | 94 | 4096 | `gqa_4` | Qwen3-235B-A22B (94L, 4 KV). Old 250-row used 64L GQA-8 |
| **236** | 60 | 5120 | `mla` | DeepSeek-V2 236B-A21B [12](https://arxiv.org/html/2405.04434v4). Was interpolated as MiniMax/Laguna GQA-8 (~62L×3072H) — ~3.5× KV over-count vs MLA 576 |
| **276** | 42 | 4096 | GQA-8 | Inkling-Small 276B-A12B official |
| **284** | 43 | 4096 | `csa_hca_flash` | **DeepSeek-V4-Flash** 284B-A13B |
| **1600** | 61 | 7168 | `csa_hca_pro` | **DeepSeek-V4-Pro** 1.6T-A49B |

### 2.4 Synthesis averages left in place (honest comments only)

These rows are *supposed* to be multi-family blends. Dims are plausible; comments no longer claim a specific official config they do not match.

- 1.5 / 2.2 / 3.0 / 6.5 / 8.5 / 14.0 / 27.0 / 39.0 — multi-family averages.
- 27.0 stays Gemma 2 27B (46L×4608H). Gemma 3 27B is 62L×5376H and Gemma 4 31B is 60L×5376H / 16 KV×256d; not folded in (would need a new `gqa_16_256d` bucket).
- 35.0 stays GQA-8 even though Command R / Aya 23 35B are MHA — the row is shared with Aya Expanse/Vision 32B (GQA-8). Conservative for R, exact for Expanse.
- 130.0 is now GLM-130B-exact; MHA vs GQA-8 is still the default GQA-8 (no `mha` override) because 12288-wide MHA at 130B would dominate nearby interpolations. Flagged below.

---

## 3. DeepSeek V4

Released 2026-04-24 (preview), GA Pro 2026-08-13. Technical report [arXiv:2606.19348](https://arxiv.org/html/2606.19348v1). Official configs: [`DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [`DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base).

| | V4-Flash | V4-Pro |
|---|---|---|
| Total / active | 284B / 13B | 1.6T / 49B |
| Layers × hidden | 43 × 4096 | 61 × 7168 |
| Experts | 256 routed + 1 shared, top-6 | 384 routed + 1 shared, top-6 |
| Context | 1,048,576 | 1,048,576 |
| Attention | 2× SWA + interleaved CSA (m=4) / HCA (m'=128) | 2× HCA + interleaved CSA / HCA |
| KV vs V3.2 @ 1M | ~7–10% | 10% (paper); 9.62 GiB bf16 (vLLM) |

V4 **replaces MLA** with hybrid Compressed Sparse Attention + Heavily Compressed Attention. Backbone is **shared K=V MQA**, `head_dim = 512` (no 2× K/V). New buckets:

- `csa_hca_flash` — 2 SWA + 21 CSA + 20 HCA
- `csa_hca_pro` — 30 CSA + 31 HCA

Stored-KV (bf16, matches [vLLM 2026-04-24](https://vllm.ai/blog/2026-04-24-deepseek-v4) and the LocalLLaMA reconstruction):

```
CSA layer: (SWA-128 + ctx/4) × 512 + (ctx/4) × 128 indexer
HCA layer: (SWA-128 + ctx/128) × 512
SWA layer:  SWA-128 × 512
```

Spot-check @ 1M bf16: **V4-Pro 9.625 GiB** (published 9.62), **V4-Flash 6.724 GiB** (published ~6.72). Costing V4-Pro as MLA would have been 68.6 GiB — a 7× overestimate.

V4-Pro is dimension-identical to V3 (61L×7168H); the per-anchor `attn` override is what distinguishes CSA/HCA from MLA. Do **not** add a 13B / 49B *active*-param row — that would paint ordinary 13B/50B dense models with V4 attention.

Native weights are FP4 experts + FP8 elsewhere. The existing FP8 / MXFP4 / NVFP4 / INT4 options already cover serving; no new quant slot.

---

## 4. Attention / KV impact of the fixes

| Scenario (bf16) | Before | After |
|---|---|---|
| Qwen3-0.6B any ctx | GQA-4 (512) | GQA-8 (1024) — 2× more honest |
| Command R+ 32K | 80 GB (10240-wide MHA) | **96 GB** (12288-wide MHA) |
| Qwen3-235B 32K | ~8 GB (64L GQA-8) | **5.88 GB** (94L GQA-4) |
| DeepSeek-V2 128K | ~15 GB (GQA-8 @ ~62L) | **8.44 GB** (MLA 60×576) |
| Inkling 975B 32K | 2.4 GB (MLA 67×576) | **8.25 GB** (GQA-8 66×1024) — was 3.4× optimistic |
| DeepSeek-V4-Pro 1M | would have been MLA 68.6 GB | **9.63 GB** CSA/HCA |
| DeepSeek-V4-Flash 1M | would have been GQA-8 ~21 GB | **6.72 GB** CSA/HCA |
| Llama 3 8B 8K | 1.00 GB | 1.00 GB (unchanged) |
| DeepSeek-V3 1M | 68.625 GB | 68.625 GB (unchanged) |

---

## 5. Intentionally not added

- **Ling-1T** — public cards confirm 1T-A50B MoE but no audited `config.json` for (layers, hidden, attn). Early third-party notes said 80L×8192H GQA; not promoted to an anchor until a config is checked.
- **Gemma 4 31B** (60L×5376H, 16 KV × 256d → KV width 4096) — needs a new `gqa_16_256d` bucket. Documented; not in this pass.
- **GLM-4 9B** (40L×4096H, **2 KV**) — would be GQA-2; the 8.5B row is a GQA-8 average with Qwen3-8B / Gemma 2 9B.
- **DeepSeek-V2-Lite 16B** (27L×2048H MLA) — stays a comment on the 14B synthesis row; an exact 16B MLA point would leak MLA into the 14–17B dense band.
- **V4 *active* sizes (13B / 49B)** — see §3.

Sliding-window layers (gpt-oss, Command A, North, Laguna, Inkling, Gemma) are still costed at full-context KV. That remains a documented conservative bias, not a bug.

---

## 6. Collateral fixes

- Trailing duplicated `</script></html>` junk after the first document close removed.
- Corrupted Bahasa Indonesia architecture string (`Aya Exp��35B…` duplicated clause) restored to match the English line.
- ID architecture / formulas / sources updated in lockstep with English.

---

## 7. Verification

- 50 strictly-sorted unique sizes; every new/fixed exact row resolves to its official `(layers, hidden, attn)`.
- Unchanged exact rows (7B, 21B, 26B, 70B, 119B, 405B, 671B, 2800B, …) regression-tested identical.
- Boundaries: 8.2B stays GQA-8 (no Small-4 MLA bleed); 118.5B flips to `mla_mistral`; 280B flips to V4-Flash; 987.5B flips to Kimi MLA; 1300B flips to V4-Pro.
- Curve sweep 0.3B → 3000B: no NaN / negative KV / non-positive dims.
- `node --check` on the extracted `<script>` passes.
- V4 KV matches published 9.62 / 6.72 GiB @ 1M bf16 to three decimals.

---

## 8. Sources

1. [Qwen3-0.6B-Base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base) — 28L, 1024H, 8 KV
2. [Gemma 3 4B-it text_config](https://huggingface.co/google/gemma-3-4b-it/discussions/14) — 34L, 2560H
3. [Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B) — 36L, 2560H, 8 KV
4. [Qwen3-32B-AWQ config](https://huggingface.co/Qwen/Qwen3-32B-AWQ) — 64L, 5120H, 8 KV
5. [Command R+ AutoConfig dump](https://huggingface.co/blog/Andyrasika/memory-consumption-estimation) — 64L, 12288H, 96 heads
6. [GLM-130B ICLR 2023](https://keg.cs.tsinghua.edu.cn/jietang/publications/ICLR23-GLM-130B.pdf) — hidden 12288; 70L
7. [Qwen3-235B-A22B config](https://huggingface.co/Qwen/Qwen3-235B-A22B) — 94L, 4096H, 4 KV
8. [Inkling-Small config](https://huggingface.co/thinkingmachines/Inkling-Small) — 42L, 4096H, 8 KV
9. [Kimi K2.5 config](https://huggingface.co/moonshotai/Kimi-K2.5) — 61L, 7168H, MLA
10. [Inkling config](https://huggingface.co/thinkingmachines/Inkling) — 66L, 6144H, hybrid GQA
11. [Qwen3-1.7B config](https://huggingface.co/Qwen/Qwen3-1.7B) — 28L, 2048H, 8 KV
12. [DeepSeek-V2 paper](https://arxiv.org/html/2405.04434v4) — 60L, 5120H, MLA, 236B-A21B
13. [DeepSeek-V4 paper](https://arxiv.org/html/2606.19348v1) — Flash 43L×4096 / Pro 61L×7168; CSA m=4, HCA m'=128
14. [DeepSeek-V4 HF docs](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4) — shared K=V, head_dim 512
15. [vLLM DeepSeek V4 KV math](https://vllm.ai/blog/2026-04-24-deepseek-v4) — 9.62 GiB @ 1M bf16 Pro
16. [Qwen3 blog architecture table](https://qwenlm.github.io/blog/qwen3/) — dense + 30B-A3B / 235B-A22B layer/KV-head counts
17. [aiwiki DeepSeek V4-Pro](https://aiwiki.ai/wiki/deepseek_v4_pro) — shipped config table, 1.6T-A49B, 1M ctx
