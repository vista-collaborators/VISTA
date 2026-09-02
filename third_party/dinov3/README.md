# Filtered DINOv3 Runtime

This directory contains a minimal subset of DINOv3 needed by VISTA to construct
backbone models such as `dinov3_vitl16`.

Included components:

- `dinov3/hub/backbones.py`
- `dinov3/models/vision_transformer.py`
- required `dinov3/layers/` and `dinov3/utils/` modules

Excluded components:

- upstream training and evaluation pipelines
- notebooks
- caches
- logs
- model weights

The DINOv3 code is distributed under its own license. See
`licenses/DINOV3_LICENSE.md`.
