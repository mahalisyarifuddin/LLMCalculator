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

## 2026-10-07 - Platform-Specific Controls Survive Language Switches
**Mode:** Palette
**Learning:** `lang()` bulk-writes `textContent` for every element whose id matches a `text.en` key, then calls `updateVramDisplay()`. Any label that depends on the GPU type (e.g., the RTX Spark slider becoming "Desktop / Driver Headroom", the carveout group, the `#overheadHint` text) must therefore be set inside `updateVramDisplay()`, never only in an `onchange` handler. Otherwise a language switch silently reverts it. Keys with no matching element id (e.g., `carveoutNone`, `appleHint`) are safe to add without being auto-applied.
**Action:** Put GPU-type-dependent labels, visibility toggles and hints in `updateVramDisplay()`. Add a new GPU type only when its memory math differs (RTX Spark's carveout + clamped-shared budget). Hardware that shares the same math belongs under an existing type.

## 2026-10-08 - Presets Are Inputs, Not Products
**Mode:** Palette
**Learning:** A preset only sets memory, GPU type, precision, context and overhead. Products with the same memory and memory model give byte-identical results: the old set had three exact duplicate pairs (RTX 4060 = RTX 5050 Laptop, RTX 3060 = Arc B580, RTX 5070 Ti = Arc A770). Unified-memory platforms are the opposite. At the same 128 GB, the DGX Spark, Ryzen AI Max+ (Linux), RTX Spark and Apple rules span ~36B parameters, so they need separate buttons.
**Action:** Use one button per memory tier per memory model. Name the two most common cards on the label and list same-memory equivalents in a `title` tooltip (product names only, so no translation is needed). Pick tiers from ownership data (Steam VRAM tiers) plus LLM-buyer tiers (24/32 GB). In tests, assert that each button's "…GB" label equals the memory it applies.
