# Model-Synthesis Update: Arcee Trinity · Mistral Open Weights — 2026-08-19

**Scope:** Two families added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs:

1. **Arcee Trinity** (entire series): Nano, Mini, Large (+ Base/Preview/Thinking checkpoints, W4A16/FP8/NVFP4/GGUF/MLX variants).
2. **Mistral open-weight models** (entire HF lineup): Mistral 7B, Mixtral 8x7B/8x22B, NeMo 12B, Small 3.x 24B, Devstral 2 123B, Medium 3.5 128B, Small 4 119B-A6B, Ministral 3 (3B/8B/14B), Large 3 675B-A41B.
3. Enabling change: **per-anchor `attn` overrides** in the synthesis table — exact KV-width buckets that (layers, hidden) alone cannot disambiguate.

All changes keep the app offline-only (no URL hash/storage/network) per the file-header constraint.

---

## 1. Arcee Trinity (all `afmoe`, OpenMDW-1.1, 1-in-4 global attention, Muon, 10–17T tokens)

| Model | Total (safetensors) | Active | Layers | Hidden | KV heads × 128d | Experts | Ctx |
|---|---|---|---|---|---|---|---|
| Trinity Nano (Base/Preview) | 6.12B | ~1B | 56 (2 dense) | 1024 | **2 → GQA-2 (256)** | 128 ×8 + 1 shared | 128K, SWA 2048 |
| Trinity Mini (Base/-) | 26.12B | 3B | 32 (2 dense) | 2048 | **4 → GQA-4 (512)** | 128 ×8 + 1 shared | 128K, SWA 2048 |
| Trinity Large (TrueBase/Base/Preview/Thinking) | 398.6B | ~13B | 60 (6 dense) | 3072 | **8 → GQA-8 (1024)** | 256 ×4 + 1 shared | 512K (config 262K), SWA 4096 |

Sources: HF model cards + config.json + safetensors counts [1](https://huggingface.co/arcee-ai/Trinity-Nano-Preview) [2](https://huggingface.co/arcee-ai/Trinity-Mini) [3](https://huggingface.co/arcee-ai/Trinity-Large-Preview) [4](https://huggingface.co/arcee-ai/Trinity-Large-Base).

**Anchors added:** `{ 6.1, 56, 1024, gqa_2 }`, `{ 26, 32, 2048, gqa_4 }`, `{ 399, 60, 3072 }`. New attention bucket **`gqa_2`** (KV width 256).

## 2. Mistral open weights

| Model | Weights | Total | Layers | Hidden | Attention | Notes |
|---|---|---|---|---|---|---|
| Mistral 7B v0.1–0.3 (+ Mathstral) | Apache 2.0 | 7.25B | 32 | 4096 | GQA-8 (8 KV × 128) | folds into the 7B Llama anchor (identical dims) |
| Mixtral 8x7B | Apache 2.0 | 46.7B (A12.9B) | 32 | 4096 | GQA-8 | new anchor `{ 47, 32, 4096 }` |
| Mistral NeMo 12B | Apache 2.0 | 12.2B | 40 | 5120 | GQA-8 | folds into the 14B anchor (identical 40L/5120H) |
| Mistral Small 3.x 24B (+ Magistral Small, Devstral Small/Small-2, Voxtral Small-24B) | Apache 2.0 | 23.6B | 40 | 5120 | GQA-8 | new anchor `{ 24, 40, 5120 }` |
| Ministral 3 (3B/8B/14B, Base/Instruct/Reasoning ×9) | Apache 2.0 | 3.9/8.5/14.5B | 26/34/40 | 3072/4096/5120 | GQA-8 | 262K ctx; 8B → new anchor `{ 8, 34, 4096 }`; 3B folds into 3B bucket, 14B into 14B anchor (identical dims) |
| Devstral 2 (123B) | Mistral licence (weights on HF, GGUF/Ollama day-one) | 123B | 88 | 12288 | GQA-8 (96 Q / 8 KV) | new anchor `{ 123, 88, 12288 }` (shared with Medium 3.5) |
| Mistral Medium 3.5 (128B) | Mistral licence | 128B | 88 | 12288 | GQA-8 | multimodal, FP8, shares the 123 anchor |
| **Mistral Small 4** (119B-2603) | Apache 2.0 | 119B (A~6B) | 36 | 4096 | **MLA (kv_lora 256 + rope 64 = 320)** | new anchor `{ 119, 36, 4096, mla_mistral }`; MoE 128 ×4 + 1 shared, 1M ctx, multimodal, FP8 |
| **Mistral Large 3** (675B-2512) | Apache 2.0 | 675B (A41B) | 61 | 7168 | **MLA (512 + 64 = 576)** | new anchor `{ 675, 61, 7168, mla }`; MoE 128 ×4 + 1 shared (3 dense), multimodal, 256K ctx |

Sources: configs / params.json / HF org listing [5](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) [6](https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512) [7](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) [8](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) [9](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512) [10](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) [11](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/). Fun fact: Mistral Large 3's (61L, 7168H, MLA 576) is dimension-for-dimension DeepSeek V3/R1's block — it slots onto the existing MLA bucket exactly.

**Not added (documented rationale):** Codestral 22B (MNPL non-production licence; 32L/6144H noted in the 24B row comment), Codestral 25.01+ / Mistral Large 1/2/2.5 / Magistral Medium / Medium 3.1 (closed or API-only — no public weights), Mamba-Codestral-7B (Mamba linear attention — no KV cache; out of scope for a KV-based estimator), Voxtral speech models (audio in/out; the 3B/24B text backbones are already covered via Ministral-3/Small), Pixtral 12B (NeMo-base VLM), Mathstral/Shieldstral (fine-tunes of covered bases).

## 3. Enabling change: per-anchor `attn` overrides

The four 2048-hidden models now split three ways (Trinity Mini GQA-4, North Mini GQA-4, North Micro/Laguna XS GQA-8), and 4096H/36L is GQA-8 at ~8B (Ministral-3-8B interpolation) but MLA at 119B (Small 4) — (layers, hidden) can no longer disambiguate. Rows now carry an optional `attn` field; `estimateArchitecture` attributes the nearest anchor's override along with the interpolated (layers, hidden), and `getAutoAttentionType` honours `arch.attn` before the heuristics (the 7168H-MLA / 2880H-64d / ≤1536H-GQA-4 heuristics remain as backstop for the manual Advanced-override path).

Side benefit: **Command R+ 104B is now MHA-exact** (its row carries `attn: 'mha'`, KV width = hidden 10240 instead of the GQA-8 1024 default) — fixing the underestimate documented in the 2026-08-19 Cohere update. Command R 35B stays on GQA-8 because its anchor is shared with Aya Expanse/Vision 32B (GQA-8).

New buckets: `gqa_2` (KV width 256, Trinity Nano), `mla_mistral` (MLA width 320, Mistral Small 4). Display names: "GQA (2-head)", "MLA (Mistral Small 4)".

## 4. Verification

- `referenceConfigs.auto`: 31 → **41** strictly-sorted points; all 10 new-family anchors resolve to their exact (layers, hidden) and attention bucket; all 31 previous anchors regression-tested unchanged.
- Edge cases: 8.2B in the 4096H/36L interpolation zone stays GQA-8 (no MLA bleed); the Laguna S ↔ Small 4 boundary flips attention at the nearest-anchor midpoint (118.5B) as designed; manual Advanced overrides still use the heuristic path; curve sweep 0.3B→3000B clean; Node syntax check passes.
- KV spot checks: Large 3 @256K bf16 = 17.2 GB (61 × 576, exact); Small 4 @1M bf16 = 22.5 GB (36 × 320, exact); Trinity Nano @128K bf16 = 7.0 GB (2 × 56 × 256); Command R+ @32K bf16 = 80 GB (MHA-exact, was ~8 GB under GQA-8).
- Doc strings (formulas / reference buckets / sources) updated **EN + ID**; README + README-id family lists updated.
