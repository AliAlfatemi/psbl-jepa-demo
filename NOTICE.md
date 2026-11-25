# Notice

## Dataset

The validation clips, thumbnails and embedding maps are derived from **The Well**, `active_matter`.

- Source: Ohana et al., 2024, *The Well: a large-scale collection of diverse physics simulations for machine
  learning*, Polymathic AI.
- Dataset card: https://huggingface.co/datasets/polymathic-ai/active_matter
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Modifications: cropped/processed into context windows and quantized to 8 bits per channel for browser delivery.
  The 2-D maps are projections of model embeddings of validation windows.
- Only the **validation** split is represented by individual samples. No test-split sample is included.

## Baseline model code

The encoder architecture is the provided convolutional JEPA of Qu et al. (2026), *Representation Learning for
Spatiotemporal Physical Systems*.

## Third-party software (private full demo only)

- onnxruntime-web 1.20.1: MIT License, © Microsoft Corporation (`vendor/LICENSE-onnxruntime`)
- coi-serviceworker 0.1.7: MIT License (`vendor/LICENSE-coi-serviceworker`)

## Release status

This is a course project (Deep Learning, NYU, Fall 2026) under evaluation. **No license is granted** for the
project's own code, trained model weights, derived features or other artifacts, pending course and intellectual
property clearance.

The public static demo ships no model weights or training-derived artifacts.
