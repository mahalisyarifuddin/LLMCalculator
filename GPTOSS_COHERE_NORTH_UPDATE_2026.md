# Model-Synthesis Update: GPT-OSS · Cohere Command / Aya / North — 2026-08-19

**Scope:** Four families added to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs:

1. **GPT-OSS** (OpenAI) — 20b & 120b, both **MXFP4-native** → new MXFP4 quantization option (~0.53 B/param).
2. **Cohere Command** — the entire open-weight line: Command R, Command R+, Command R7B, Command A (incl. A Reasoning / A Vision / A Translate), Command A+.
3. **Cohere Aya** — Aya 101, Aya 23, Aya Expanse, Aya Vision (+ Tiny Aya, 2026).
4. **Cohere North** — North Mini Code, North Micro / North Micro Vision.

All changes keep the app offline-only (no URL hash/storage/network) per the file-header constraint.

---

## 1. GPT-OSS (MXFP4-native)

**Facts (OpenAI, released 2025-08-05; still the only two variants as of mid-2026):**

| Model | Total / active | Layers | Hidden | Attention | Native quant | Context |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 21B / 3.6B (32 experts ×4) | 24 | 2880 | GQA-8, 64 Q / 8 KV heads × **64-dim** → KV width 512 | MXFP4 (MoE experts; attn/embeddings BF16) | 131,072 |
| gpt-oss-120b | 117B / 5.1B (128 experts ×4) | 36 | 2880 | GQA-8, 64 Q / 8 KV × 64-dim → KV width 512 | MXFP4 | 131,072 |

Both alternate sliding (window 128) and full attention; RoPE + YaRN to 128K. Official 120b MXFP4 artifact ≈ 60.8 GB (≈4.16 bits/param) [1](https://huggingface.co/openai/gpt-oss-120b) [2](https://huggingface.co/docs/transformers/en/model_doc/gpt_oss) [3](https://aiwiki.ai/wiki/gpt_oss).

**Changes applied:**

- New synthesis points: `{ size: 21, layers: 24, hidden: 2880 }` and `{ size: 117, layers: 36, hidden: 2880 }`.
- New attention bucket `gqa_8_64d`: hidden **2880** → GQA-8 with 64-dim heads → KV width **512** (the default GQA-8 assumes 8×128=1024 and would double gpt-oss KV). Display name "GQA (8-head × 64d)".
- **New quantization option `MXFP4 (~0.53 bytes)`** (4-bit float + 8-bit block scale per 32-element group ≈ 4.25 bpw) in Model Precision + `quantizationBytes.mxfp4 = 0.53`. Sanity: 117B × 0.53 = 62 GB ≈ official 60.8 GB; 21B × 0.53 ≈ 11.7 GB ≈ official ~12–13 GB safetensors. Kimi K3 (also MXFP4 weights) benefits from the same option.

## 2. Cohere Command (entire series)

| Model | Release | Params | Layers | Hidden | Attention |
|---|---|---|---|---|---|
| Command 52B / Command Light | 2022–2023 | 52B / ~6B | closed weights — no public architecture (documented in the family comment only) | | |
| Command R | 2024-03 | 35B | 40 | 8192 | MHA (64 KV heads) |
| Command R+ (04-2024 & 08-2024) | 2024 | 104B | 64 | 10240 | MHA (64 KV heads) |
| Command R7B | 2024-12 | 7B | 32 | 4096 | GQA-8 (32 Q / 8 KV × 128) |
| Command A (+ A Reasoning / A Vision / A Translate) | 2025-03 → 2026 | 111B dense (A Vision 112B) | 64 | 12288 | GQA-8 (96 Q / 8 KV × 128), 3:1 sliding(4096)/global interleave |
| Command A+ | 2026-05 | **218B total / 25B active MoE** (128 experts ×8 + 4 shared) | 32 | 4096 | GQA-8 (128 Q / 8 KV × 128), 3:1 sliding/global |

Sources: Command R 40L/8192H/64-KV MHA via llama.cpp GGUF metadata [4](https://github.com/ggml-org/llama.cpp/issues/6112); Command R+ 64L/10240H from its HF repo layout [5](https://huggingface.co/CohereForAI/c4ai-command-r-plus); Command A config (Cohere2, 64L, 12288H, 8 KV heads) [6](https://huggingface.co/unsloth/c4ai-command-a-03-2025); Command A+ cohere2_moe config (32L, 4096H, 128 experts) + 218B-A25B sizing [7](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16) [8](https://cohere.com/blog/command-a-plus).

**Changes applied:**

- New synthesis points: `{ 104, 64, 10240 }` (R+), `{ 111, 64, 12288 }` (A), `{ 218, 32, 4096 }` (A+), `{ 35, 40, 8192 }` (R, shared with Aya 35B/32B below). Command R7B folds into the existing 8.5B bucket comment (it is exactly 32L/4096H like the Llama-7B point).
- Note: Command R / R+ are **MHA**, which the auto-estimate conservatively treats as GQA-8 (its generic default for 2B–405B); row comments flag them as MHA.

## 3. Cohere Aya (entire series)

| Model | Release | Params | Layers | Hidden | Attention |
|---|---|---|---|---|---|
| Aya 101 | 2024-02 | 13B | mT5-xxl **encoder-decoder** (24+24 blocks, 4096) | | documented in the 14B bucket comment only |
| Aya 23 8B / 35B | 2024-05 | 8B / 35B | 32 / 40 | 4096 / 8192 | GQA-8 / MHA (per Aya 23 report table) |
| Aya Expanse 8B / 32B | 2024-10 | 8B / 32B | 32 / 40 | 4096 / 8192 | GQA-8 (64 Q / 8 KV × 128 on 32B) |
| Aya Vision 8B / 32B | 2025-03 | 8B / 32B | = Expanse backbones (8B from Command R7B, 32B from Expanse 32B) | | GQA-8 |
| Tiny Aya (Global/Earth/Fire/Water) | 2026 | 3.35B | compact multilingual, 70+ languages | | documented in the 3B bucket comment only |

Sources: Aya 23 architecture table [9](https://cohere.com/research/aya/aya-23-technical-report.pdf); Aya Expanse 32B config (40L, 8192H, 8 KV heads) [10](https://huggingface.co/Andrewwwwww/aya-expanse-32b); family overview incl. Aya Vision bases and Tiny Aya [11](https://docs.cohere.com/docs/models) [12](https://aiwiki.ai/wiki/aya_expanse).

**Changes applied:**

- Aya 23/Expanse/Vision 8B join the 8.5B bucket comment; Expanse/Vision 32B + Aya 23 35B + Command R 35B share the new `{ 35, 40, 8192 }` point; Aya 101 and Tiny Aya are documented in the 14B / 3B bucket comments (encoder-decoder and no published block config respectively).

## 4. Cohere North (entire series)

| Model | Release | Params | Layers | Hidden | Attention |
|---|---|---|---|---|---|
| North Mini Code 1.0 | 2026-06 | 30B total / 3B active MoE (128 experts ×8), 256K ctx | 49 | 2048 | GQA-4 (32 Q / 4 KV × 128 → KV width 512), 3:1 sliding(4096)/global |
| North Micro (LM inside North Micro Vision Instruct) | 2026-08 | 2B (2.4B with 400M SigLIP2-based vision encoder) | 28 | 2048 | GQA-8 (16 Q / 8 KV × 128) |

Sources: cohere2_moe config for North Mini Code (49L, 2048H, 4 KV heads, vocab 262K, max_position 500K) [13](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) [14](https://www.marktechpost.com/2026/06/11/meet-north-mini-code-coheres-30b-open-weight-mixture-of-experts-model-with-3b-active-parameters-for-agentic-coding/); cohere_compass text config for North Micro (28L, 2048H, 8 KV heads) [15](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) [16](https://benchgen.com/models/cohere-labs/north-micro-vision-instruct).

**Changes applied:**

- New synthesis point `{ 30, 49, 2048 }` (North Mini Code).
- **GQA-4 classification extended**: `hidden ≤ 1536` **or** `(hidden === 2048 && layers ≥ 49)` → GQA-4. The layer condition disambiguates the two 2048-hidden North models (Mini Code = GQA-4; Micro = GQA-8), and keeps every interpolated 2B-class estimate on GQA-8 as before.
- North Micro / Micro Vision join the 2.2B bucket comment.

## Verification

- `referenceConfigs.auto` now 28 strictly-sorted points (was 21); all 7 new anchors resolve to their exact (layers, hidden) and auto-attention type.
- All 21 pre-existing anchor resolutions regression-tested unchanged.
- Curve sweep 0.3B → 3000B: no NaN/negative outputs.
- MXFP4 weight estimates match official artifact sizes (see §1).
- Conservative simplification retained: sliding-window layers (gpt-oss alternating window-128; Command A/A+/North 3:1 pattern) are costed at full-context KV, consistent with how Gemma's sliding layers were already treated.
- Doc strings (formulas / reference buckets / sources) updated in **English + Bahasa Indonesia**; README + README-id updated.
