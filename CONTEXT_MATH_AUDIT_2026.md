# Context and Memory Math Audit — 2026-08-19

## Scope

This audit reviews the calculator's weight-memory arithmetic, architecture-aware KV formulas, and maximum-fit solver. It does not turn the interpolated model into an exact checkpoint catalogue or model a particular inference server.

## Findings

### 1. Standard MHA/GQA/MQA KV formula — verified

For one active sequence:

```text
KV bytes = 2 × layers × context_tokens × kv_heads × head_dim × bytes_per_element
```

The factor of two is for K and V. In MHA, `kv_heads × head_dim` equals hidden width. In GQA and MQA it is the smaller shared-KV width. The implementation follows this formula.

Reference checks at 8,192 tokens and FP16:

| Architecture | Expected KV | Calculator formula |
|---|---:|---:|
| Llama-class 8B: 32L, GQA-8 × 128d | 1.000 GiB | 1.000 GiB |
| Llama-class 70B: 80L, GQA-8 × 128d | 2.500 GiB | 2.500 GiB |
| Command R+ class: 64L, 12,288-wide MHA | 24.000 GiB | 24.000 GiB |
| MQA-1 × 128d | `2 × L × 128 × context × bytes` | Same |

### 2. MLA formula — verified for compact serving caches

DeepSeek-style MLA stores the compressed KV latent plus the separate RoPE key component:

```text
KV elements = layers × context × (kv_lora_rank + qk_rope_head_dim)
```

For DeepSeek V2/V3/R1, this is `512 + 64 = 576` elements per token per layer. At 61 layers, 8,192 tokens, and FP16, the result is 0.5361328125 GiB. The implementation matches the compact MLA serving representation.

A runtime that expands latent tensors instead of retaining the compact serving form can consume more memory. That is runtime-specific behavior outside this calculator.

### 3. Specialized formulas — internally consistent, architecture-dependent

- **MFA (Step-3):** caches one 256-d K and one 256-d V per layer.
- **Hybrid SWA/global:** global layers retain the selected context; sliding layers retain `min(context, window)`. Width values include both K and V.
- **CLA-2:** uses `ceil(layers / 2)` KV sets with the normal K+V factor.
- **CSA/HCA:** applies the documented compressed-token ratios and local SWA branch assumptions.

These formulas are appropriate for the calculator's architecture anchors. They remain estimates when architecture interpolation lands between anchors or when a runtime uses a different cache layout, alignment, mixed precision, or extra index buffers.

### 4. Weight units were inconsistent — fixed

KV bytes were converted to binary GiB with division by `2^30`, but model weights previously treated one billion bytes as one displayed GB without the equivalent conversion. This mixed decimal parameter counts with binary memory capacity.

The corrected weight formula is:

```text
weight GiB
= params_in_billions × 10^9 × bytes_per_param × 1.05 ÷ 2^30
```

Therefore one billion bytes equals `0.9313225746 GiB`. The fixed 5% factor remains an explicit estimate for metadata, scales, and alignment.

The UI retains conventional GPU capacity labels such as “24 GB,” while calculations consistently use binary GiB internally, matching how GPU capacities and runtime memory are commonly reported.

### 5. Five-pass fixed-point solver could return a mismatched pair — fixed

The former solver alternated between parameter count and interpolated architecture only five times. Its final architecture could be one iteration newer than the parameter count used to derive it. Abrupt attention-anchor changes could also produce oscillation or skip a higher fitting region.

The replacement solver:

1. computes the weight-only upper bound;
2. scans downward across the interpolated architecture curve to find the highest fitting region;
3. refines that region with binary search;
4. recomputes weight and KV memory from the same final parameter/architecture pair.

The descending scan is intentional because nearest-anchor attention overrides can make KV cost discontinuous. A simple global binary search would incorrectly assume monotonic memory use.

Advanced architecture overrides do not need this search because layers and hidden width are fixed; their maximum parameter count is solved directly after subtracting KV memory.

### 6. Displayed MB/token unit was binary — label fixed

The metric computes:

```text
KV GiB × 1024 ÷ tokens
```

That is MiB/token, not decimal MB/token. The label now says **MiB/token** in both languages.

## Remaining intentional approximations

- The architecture is interpolated from open-weight reference anchors; it is not an exact checkpoint lookup.
- Quantized bytes per parameter are format-level averages. Tensor mixes, block scales, metadata, padding, and runtime conversion can differ.
- “KV precision same as model” is a user-selected hypothetical; many runtimes default to a separate KV type.
- Runtime workspaces, graph buffers, temporary activations, allocator fragmentation, prompt batching, and backend-specific reservations are represented only through the configurable system-overhead allowance.
- Hybrid attention, state-space layers, sparse attention, and novel cache indexers are anchor-specific estimates.
- CPU offload, parallel sequences, shared server pools, and subagent concurrency remain deliberately excluded.
- Model training context limits and long-context quality are not inferred from memory fit.

## UX result

Because a normal request and one agent inference step use the same KV formula, the redundant mode control was removed. The single logarithmic slider now has the dual label:

> Context Window / Active Working Context

Its value remains the token quantity passed to all architecture-aware KV formulas.

## Primary references

- [llama.cpp completion context documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md)
- [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434)
- [Hugging Face DeepSeek V3 configuration](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v3)
- [KV cache architecture calculations](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)
