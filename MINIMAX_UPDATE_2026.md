# Model-Synthesis Update: MiniMax — 2026-08-19

**Scope:** The MiniMax open-weight LLM series added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs. Table 41 → **44** strictly-sorted anchors.

## 1. Verified architecture data

| Model | Weights | Total / active | Layers | Hidden | Attention | Notes |
|---|---|---|---|---|---|---|
| MiniMax-Text-01 (2025-01) | open (HF) | 456B / A45.9B | 80 | 6144 | hybrid **lightning (linear) + softmax** attention; full-attn layers GQA | 1M–4M ctx; hidden per transformers MiniMax docs |
| MiniMax-M1-40k/80k (2025-06, Apache 2.0) | open | 456B / A45.9B | 80 | 6144 | same hybrid arch as Text-01 | reasoning line |
| MiniMax-M2 (2025-10) → M2.1 (2025-12) → M2.5 (2026-02) → M2.7 (2026-04) | open (MiniMax licence, FP8 checkpoints) | 230B / A~10B | **62** | **3072** | **GQA-8**: 48 Q / 8 KV × 128d → KV width 1024 | 256 experts ×8; interleaved thinking; M2.7 also ships as nvidia NVFP4 |
| MiniMax-M3 (2026-06, + M3-MXFP8 build) | open | 428B / A~23B | **60** (3 dense + 57 MoE) | **6144** | **GQA-4**: 64 Q / **4 KV** × 128d → KV width 512, + **MiniMax Sparse Attention** (top-16 of 128-token blocks) | 128 experts ×4 + 1 shared, 1M ctx, multimodal (image/video), MTP modules |

Sources: M2 dims via transformers MiniMaxM2Config defaults (62L/3072H/48Q/8KV) [1](https://huggingface.co/docs/transformers/model_doc/minimax_m2) and nvidia/MiniMax-M2.7-NVFP4 (62L, 3072H, 256 experts ×8) [2](https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4); M3 config.json from MiniMaxAI/MiniMax-M3 [3](https://huggingface.co/MiniMaxAI/MiniMax-M3) (text_config: 60L, 6144H, 4 KV heads, sparse_attention_config, 1M ctx); org listing confirming M2.1/M2.5/M2.7 and Text-01/M1 lineage [4](https://huggingface.co/MiniMaxAI); M3 overview [5](https://www.morphllm.com/minimax-m3).

## 2. Changes applied

- **New anchors:** `{ 230, 62, 3072 }` (M2/M2.1/M2.5/M2.7 — GQA-8 exact under the default classifier), `{ 428, 60, 6144, attn: 'gqa_4' }` (M3 — hidden 6144 would otherwise default to GQA-8; the per-anchor override makes the 4-KV-head bucket exact), `{ 456, 80, 6144 }` (Text-01 / M1).
- Reference buckets (EN + ID): the ~141B–399B MoE line extended to **~141B–456B (56–80L, 3072–6400H)** to carry Mixtral 8x22B, the M2 family, Trinity Large, M3 and Text-01/M1; sources updated; README family lists updated.

## 3. Notes & caveats

- **MSA sparsifies compute, not stored KV** — M3 still writes a full KV cache (sparse index selects blocks at attention time), so the KV estimate 2 × 60 × 512 × ctx × bytes is the honest memory figure (120 GB @ 1M bf16).
- **Text-01 / M1 hybrid lightning attention:** most layers are linear-attention (≈constant KV state); costing all 80 layers at GQA-8 width is deliberately conservative, same treatment as Kimi K3's KDA/MLA hybrid.
- **Excluded (documented):** MiniMax-H3 (video generator) and Music3 (music generator) — not LLMs; VL-01 (vision wrapper over Text-01); SynLogic 7B/32B (Qwen2.5 fine-tunes, base covered); the rumored 2.7T "M3 Pro" (not released as of 2026-08-19). M3's MXFP8 build ≈ FP8 (1.0 B/param option already covers it).

## 4. Verification

- 44 strictly-sorted anchors; the 3 new MiniMax anchors resolve exactly (M2 GQA-8 exact, M3 GQA-4 via override, Text-01 conservative GQA-8); all 41 prior anchors regression-tested unchanged.
- Nearest-anchor attention boundaries verified (410B → GQA-8 on the Llama-405 side, 420B → GQA-4 on the M3 side, 450B → GQA-8 on the M1 side); curve sweep 0.3B→3000B clean; Node syntax check passes.
