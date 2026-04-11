## 2025-02-21 - GQA Memory Ratio Heuristic
**Mode:** Medic
**Learning:** Modern LLMs (Llama 3, Qwen 2.5, etc.) use Grouped Query Attention (GQA) which significantly reduces KV cache memory. A robust heuristic for the KV head ratio is `min(1.0, 1024 / hidden_size)`. This matches Llama 3 8B (0.25), 70B (0.125), and 405B (0.0625) perfectly.
**Action:** Use this heuristic to avoid underestimating maximum parameter counts when large context windows are used.

## 2025-02-21 - Dynamic DOM Access Patterns
**Mode:** Palette
**Learning:** The codebase uses a `Proxy` object for `elements` which dynamically calls `document.getElementById(id)`. This means new UI elements do not need manual registration in a central mapping object.
**Action:** Simply ensure new elements have unique IDs and access them via `elements.id`.

## 2025-05-15 - Precise Slider Label Alignment
**Mode:** Palette
**Learning:** In single-file HTML tools with custom ranges (e.g., 4-256GB), default Flexbox 'space-between' for labels leads to misalignment with slider thumbs. Absolute positioning using calculated percentage offsets `((value - min) / (max - min) * 100)` ensures visual precision across different scales.
**Action:** Use absolute positioning and inline `left` styles for slider labels to maintain professional UI standards.

## 2025-05-16 - Lifecycle Order for URL State Persistence
**Mode:** Palette
**Learning:** For deep-linking to work reliably in single-file tools, the initialization order must be: 1. setup DOM listeners, 2. load URL hash into internal state, 3. apply localization and initial calculation. Reversing 2 and 3 can lead to default values overwriting URL-provided parameters during the first render.
**Action:** Always call state restoration methods after event registration but before the initial UI render/calculation pass.
