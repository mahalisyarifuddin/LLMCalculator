## 2025-02-21 - GQA Memory Ratio Heuristic
**Mode:** Medic
**Learning:** Modern LLMs (Llama 3, Qwen 2.5, etc.) use Grouped Query Attention (GQA) which significantly reduces KV cache memory. A robust heuristic for the KV head ratio is `min(1.0, 1024 / hidden_size)`. This matches Llama 3 8B (0.25), 70B (0.125), and 405B (0.0625) perfectly.
**Action:** Use this heuristic to avoid underestimating maximum parameter counts when large context windows are used.

## 2025-02-21 - Dynamic DOM Access Patterns
**Mode:** Palette
**Learning:** The codebase uses a `Proxy` object for `elements` which dynamically calls `document.getElementById(id)`. This means new UI elements do not need manual registration in a central mapping object.
**Action:** Simply ensure new elements have unique IDs and access them via `elements.id`.
