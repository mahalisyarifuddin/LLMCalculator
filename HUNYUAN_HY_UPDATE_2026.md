# Model-Synthesis Update: Tencent Hunyuan / Hy — 2026-08-19

**Scope:** The open-weight Tencent Hunyuan / Hy text-LLM line added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs. Table 50 → **54** strictly-sorted anchors.

## 1. Verified architecture data

| Model | Weights | Total / active | Layers | Hidden | Attention | Notes |
|---|---|---|---|---|---|---|
| Hunyuan-0.5B (Pretrain/Instruct) | open (HF) | 0.5B dense | 24 | 1024 | GQA-8 (16 Q / 8 KV × 128) | folds into the 0.6 Qwen3 row (same 1024H / 8 KV) |
| Hunyuan-1.8B (+ HY-MT1.5-1.8B) | open | 1.8B dense | **32** | **2048** | **GQA-4** (16 Q / **4 KV** × 128) | new `{ 1.8, 32, 2048, gqa_4 }` |
| Hunyuan-4B | open | 4B dense | 36 | 3072 | GQA-8 | nearby the 4.0 Gemma/Qwen3 row (34L×2560H); documented, not a new size |
| Hunyuan-7B (+ HY-MT1.5-7B) | open | 7B dense | 32 | 4096 | GQA-8 | folds into the 7B Llama/Mistral row |
| Hunyuan-A13B (Pretrain/Instruct/FP8/GPTQ) | open | **80B / A13B** | **32** | **4096** | GQA-8 (32 Q / 8 KV) | new `{ 80, 32, 4096 }`; 64 experts ×8 + 1 shared; `use_cla: false`; 256K ctx |
| Hy3 / Hy3-preview / Hy3-FP8 | Apache 2.0 | **295B / A21B** (+ 3.8B MTP) | **80** | **4096** | GQA-8 (64 Q / 8 KV × 128) | new `{ 295, 80, 4096 }`; 192 experts ×8 + 1 shared; 256K ctx |
| Hunyuan-Large (A52B Pretrain/Instruct/FP8) | open | **389B / A52B** | **64** | **6400** | GQA-8 **+ CLA-2** (80 Q / 8 KV) | new `{ 389, 64, 6400, gqa_8_cla2 }`; 16 routed ×1 + 1 shared; 256K ctx |

Sources: Hunyuan-0.5B/1.8B/4B/7B `config.json` [1](https://huggingface.co/tencent/Hunyuan-0.5B-Instruct) [2](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct) [3](https://huggingface.co/tencent/Hunyuan-4B-Instruct) [4](https://huggingface.co/tencent/Hunyuan-7B-Instruct); A13B config (32L, 4096H, 8 KV, 64 experts) [5](https://huggingface.co/tencent/Hunyuan-A13B-Instruct); Hunyuan-Large A52B-Pretrain config (64L, 6400H, `use_cla: true`, `cla_share_factor: 2`) [6](https://huggingface.co/tencent/Tencent-Hunyuan-Large/blob/main/Hunyuan-A52B-Pretrain/config.json) + paper arXiv:2411.02265 [7](https://arxiv.org/html/2411.02265v3); Hy3 official table + `config.json` (80L, 4096H, 8 KV, 192 experts) [8](https://huggingface.co/tencent/Hy3).

## 2. Enabling change: `gqa_8_cla2`

Hunyuan-Large is the only open Hunyuan checkpoint with Cross-Layer Attention (`use_cla: true`, share factor 2): every two decoder blocks share one KV cache, so stored KV layers = `ceil(L/2)` = 32. A new bucket **`gqa_8_cla2`** costs GQA-8 width (1024) over those shared layers.

- Hunyuan-Large @ 256K bf16: `2 × 32 × 1024 × 262144 × 2 / 1 GiB` = **32 GB** (would have been 64 GB under plain GQA-8).
- Display name: `GQA (8-head, CLA-2)`.
- Dense 0.5B–7B and A13B / Hy3 ship `use_cla: false` and stay on ordinary GQA-8 (or GQA-4 for 1.8B).

## 3. Not added (documented rationale)

- **Hunyuan-T1, Hunyuan TurboS, Hunyuan 2.0 Think/Instruct** — API / Yuanbao only; no public weights or `config.json`.
- **HY-MT / HY-MT1.5 / HY-MT2 translation checkpoints** — 1.8B and 7B fine-tunes of the dense bases already covered.
- **HunyuanImage / HunyuanVideo / Hunyuan3D / Hy-Embodied** — not text LLMs; out of scope for a KV-based LLM estimator.
- **Hunyuan-4B as its own size** — 4.0 is already Gemma 3 / Qwen3 4B (34L×2560H). Hunyuan-4B’s 36L×3072H GQA-8 is noted on that row; hidden size does not change GQA-8 KV.

## 4. Verification

- 54 strictly-sorted unique sizes; the 4 new-family anchors resolve to their official `(layers, hidden, attn)`; Hunyuan-7B / 0.5B fold into existing exact GQA-8 rows as designed.
- Boundaries: 1.75B stays on Qwen3 GQA-8 (no 1.8B GQA-4 bleed below the midpoint); 289.5B flips Hy3 GQA-8 vs V4-Flash CSA/HCA; 394B flips Hunyuan-Large CLA-2 vs Trinity GQA-8.
- Prior exact rows (7B, 21B, 70B, 119B, 284B, 405B, 671B, 1600B, 2800B, …) regression-tested unchanged.
- Curve sweep 0.3B→3000B clean; Node syntax check passes.

## 5. Changes applied

- New anchors: `{ 1.8, 32, 2048, gqa_4 }`, `{ 80, 32, 4096 }`, `{ 295, 80, 4096 }`, `{ 389, 64, 6400, gqa_8_cla2 }`.
- New attention bucket `gqa_8_cla2` (KV layers = `ceil(L/2)`, width 1024).
- Doc strings (formulas / reference buckets / sources) updated **EN + ID**; README family lists updated.
