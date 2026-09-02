# DINOv3 Runtime In VISTA

This package contains the DINOv3 backbone modules used internally by VISTA.
Users should normally interact with VISTA through `vista.model`,
`run_finetuning.py`, and `test_task_model.py`.

Model weights are not included. Put local backbone weights under `weights/`
and pass them with `--dinov3_weights`.

The DINOv3 code is distributed under its own license; see `LICENSE.md`.
