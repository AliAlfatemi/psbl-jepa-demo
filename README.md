# PSBL-JEPA demo

This is the companion demo for **When Non-Collapse Is Not Enough: Stress-Testing Predictive and Geometric
Regularizers for Physical JEPAs**. The authors are Ali Alfatemi, Yuanhong Wu and Nikoloz Khmaladze, who
contributed equally, for Deep Learning, NYU, Fall 2026.

## Release modes

Check `assets/MANIFEST.json` to see which mode a build is.

**Public static demo** (`release_mode: public-static`):
- Shows 12 validation clips, precomputed predictions for them, and the paper's aggregate charts and tables.
- Ships no model weights, no training-split features or labels, no probe parameters and no runtime.

**Private full demo** (`release_mode: private-full`):
- Runs the frozen encoders in the browser with onnxruntime-web.
- Uses the same frozen-evaluation protocol as the paper, with the epoch-30 seed-0 checkpoint shown for
  interactive illustration:
  - a linear head refit with the validation-selected settings;
  - optional distance-weighted kNN over standardized training features.
- It contains trained weights and training-derived artifacts. **Do not deploy it publicly before course and IP
  clearance.**

## What the numbers mean

**Three-seed aggregates.** The paper's tables are aggregates over three seeds (MSE on training-z-scored labels).
The clip predictions on the page are illustrations from one checkpoint, in raw physical units. Twelve clips do
not estimate performance.

**Runtime parity** (full mode only) is the largest difference between the browser's ONNX output and PyTorch's
output on the same 8-bit input. It stayed below 10⁻⁶ on all 12 clips in our automated tests; every
run displays its own value.

**Quantization impact** is the largest difference between the PyTorch embeddings of the 8-bit and the
full-precision input. It is at most 4.1·10⁻⁵ over the 12 clips (`clips.json`, `max_abs_diff_quantization`).

**Clip selection.** The clips are deterministic and documented in `assets/clips/SELECTION_MANIFEST.json`:
1. Sort the 24 validation trajectories by (α, ζ).
2. Take 12 evenly spaced trajectories.
3. Use the middle window of each.

## Running locally

```bash
python -m http.server 8000   # then open http://localhost:8000
```

No analytics, no remote fonts and no third-party requests. A Content Security Policy is set with a `<meta>` tag.

GitHub Pages cannot send custom headers, so `X-Content-Type-Options` and `frame-ancestors` are not enforced
there.

## Attribution and licenses

See `NOTICE.md`.
- The data come from The Well `active_matter`, CC BY 4.0 (Ohana et al., 2024).
- No license is granted for the project's own code, weights or derived artifacts.
