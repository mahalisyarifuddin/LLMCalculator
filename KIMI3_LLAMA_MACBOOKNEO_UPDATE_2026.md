# Model-Synthesis Update: Kimi K3 · Llama Gen 1–4 · MacBook Neo — 2026-08-19

**Scope:** Three updates to the multi-generational auto-estimate synthesis in `LLMCalculator.html` + READMEs:

1. Kimi family range extended **V1–K2.5 → V1–K3**.
2. **Llama Gen 1–4** integrated into the merged architecture calculation.
3. **MacBook Neo** placed in the hardware presets (Apple/Mobile tier).

All changes keep the app offline-only (no URL hash/storage/network) per the file-header constraint.

---

## 1. Kimi: V1–K2.5 → V1–K3

**Kimi K3 facts (Moonshot AI, announced 2026-07-16, weights 2026-07-27):**

- 2.8T total / **104B active** parameters (MoE, 896 routed experts, 16 active/token + shared experts) [1](https://moclaw.ai/blog/what-is-kimi-k3) [2](https://huggingface.co/moonshotai/Kimi-K3)
- **93 layers**: 69 Kimi Delta Attention (KDA, linear/recurrent) + 24 Gated MLA + Attention Residuals [2](https://huggingface.co/moonshotai/Kimi-K3) [3](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
- **Attention hidden dim = 7168** (96 heads); 1M-token context; MXFP4 weights [2](https://huggingface.co/moonshotai/Kimi-K3)

**Change applied:**

- New top reference point in `referenceConfigs`: `{ size: 2800, layers: 93, hidden: 7168 }` (was capped at 1T/67L). The curve now runs 0.5B → 2.8T.
- MLA classification broadened from `61≤layers≤67 && hidden===7168` to **`hidden === 7168 && layers ≥ 61`**, which covers DeepSeek V3/R1 (61L), Kimi K2/K2.5 / Inkling / Ling-1T (67L) **and Kimi K3 (93L)**. Dense 70B (8192H), 130B (10240H) and Llama 405B (16384H) still correctly fall to GQA-8 — the prior "narrow band" fix is preserved.
- Doc strings ("Kimi V1–K2.5 Technical Reports" → "V1–K3", bucket text "Kimi (…K2.5 1T, K3 2.8T)") updated EN + ID.

## 2. Llama Gen 1–4 into the merged calculation

Confirmed reference configurations:

| Family | Size | Layers | Hidden | Attention |
|---|---|---|---|---|
| Llama 1/2 7B · Llama 3 8B | 7–8B | 32 | 4096 | GQA-8 (Llama 3) / MHA (Llama 1/2) [4](https://ar5iv.labs.arxiv.org/html/2407.21783) |
| Llama 1/2 13B | 13B | 40 | 5120 | MHA |
| Llama 1 65B · Llama 2/3 70B | 65–70B | 80 | 8192 | GQA-8 (8 KV heads) [5](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/apidocs/bridge/bridge.models.llama.llama_provider.html) |
| Llama 3.1 405B | 405B | 126 | 16384 | GQA-8 (128 Q / 8 KV heads) [4](https://ar5iv.labs.arxiv.org/html/2407.21783) [6](https://apxml.com/models/llama-3-1-405b) |
| Llama 4 Scout / Maverick | 17B active (109B / 400B total) | 48 | 5120 | GQA-8 (40 Q / 8 KV heads, iRoPE) [7](https://huggingface.co/docs/transformers/en/model_doc/llama4) |

**Change applied:**

- New / updated `referenceConfigs` points:
  - `{ size: 7.0, layers: 32, hidden: 4096 }` — Llama 1/2 7B & Llama 3 8B
  - `{ size: 14.0, layers: 40, hidden: 5120 }` — Llama 1/2 13B + Qwen 14B (replaces the old 15B/38L/3584H point; 5120H matches Llama 13B and Qwen 14B, fixing a hidden-size that was too small)
  - `{ size: 17.0, layers: 48, hidden: 5120 }` — Llama 4 Scout/Maverick 17B active
  - `{ size: 405.0, layers: 126, hidden: 16384 }` — Llama 3.1 405B dense
  - 70B point comment now cites Llama 1 65B / Llama 2/3 70B.
- Attention summary updated: **GQA-8 covers 2B–405B** (was "2B–276B"); **MLA for 671B+ MoE** (was "300B+ MoE") — because Llama 4 Maverick (400B MoE) is GQA-8, not MLA.
- Family lists (code comment, EN/ID reference text, README bullets) now include **Llama (Gen 1–4)**.

## 3. MacBook Neo in the presets

**Where it fits:** the **entry-level Apple/Mobile preset** (8 GB unified, `soc` = Apple Silicon 25% Metal reservation).

**Why:** MacBook Neo (Apple A18 Pro, announced March 2026, $599) is the current budget Mac — the first with an iPhone-class A-series chip instead of an M-series chip [8](https://support.apple.com/en-us/126322) [9](https://www.macworld.com/article/3081612/macbook-neo-a18-pro-review.html). Specs: 6-core CPU (2P+4E), **5-core GPU**, **8 GB unified memory** (no RAM upgrade option), 60 GB/s bandwidth, 256/512 GB SSD [8](https://support.apple.com/en-us/126322) [10](https://www.macworld.com/article/2854313/macbook-neo-design-processor-specs-release.html). Geekbench 6 Metal ≈31K — slightly below an M1 MacBook Air (~33K) [11](https://www.theverge.com/tech/891741/apple-macbook-neo-a18-pro-review). The current MacBook Air starts at 16 GB (M5, $1,099), so the 8 GB "Air" slot no longer reflects a real product [10](https://www.macworld.com/article/2854313/macbook-neo-design-processor-specs-release.html).

**Change applied:**

- The existing `macbookneo` preset (8 GB, Q4_K_M, 8K context, `soc`) — previously mislabeled "MacBook Air 8GB" — is now labeled **"MacBook Neo 8GB"** and documented as such.
- GPU-type option relabeled **"Apple Silicon (M-series)" → "Apple Silicon (M/A-series)"** so the A18 Pro is accurately covered (same unified-memory/Metal model).
- Apple/Mobile tier text + sources updated EN/ID ("MacBook Neo 8GB" and "Apple A18 Pro MacBook Neo (8GB unified, $599)").

## Verification

- `node --check` on the extracted `<script>` — syntax OK.
- Interpolation/attention smoke test: 7B→32L×4096H GQA-8 · 14B→40L×5120H GQA-8 · 17B→48L×5120H GQA-8 · 405B→126L×16384H GQA-8 · 671B→61L×7168H MLA · 1000B→67L×7168H MLA · 2800B→93L×7168H MLA. Reference sizes remain ascending (21 points, 0.5B–2.8T).

## References

1. [What Is Kimi K3? (moclaw.ai)](https://moclaw.ai/blog/what-is-kimi-k3)
2. [moonshotai/Kimi-K3 model card (Hugging Face)](https://huggingface.co/moonshotai/Kimi-K3)
3. [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
4. [The Llama 3 Herd of Models (arXiv 2407.21783)](https://ar5iv.labs.arxiv.org/html/2407.21783)
5. [Llama 2 70B config — NVIDIA Megatron Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/apidocs/bridge/bridge.models.llama.llama_provider.html)
6. [Llama 3.1 405B specs (apxml.com)](https://apxml.com/models/llama-3-1-405b)
7. [Llama4 config — Hugging Face transformers docs](https://huggingface.co/docs/transformers/en/model_doc/llama4)
8. [MacBook Neo – Tech Specs (Apple Support)](https://support.apple.com/en-us/126322)
9. [MacBook Neo review (Macworld)](https://www.macworld.com/article/3081612/macbook-neo-a18-pro-review.html)
10. [MacBook Neo vs MacBook Air guide (Macworld)](https://www.macworld.com/article/2854313/macbook-neo-design-processor-specs-release.html)
11. [MacBook Neo review (The Verge)](https://www.theverge.com/tech/891741/apple-macbook-neo-a18-pro-review)
