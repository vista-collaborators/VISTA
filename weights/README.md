# VISTA Weights

This repository intentionally does not include model weights.

Expected local layout:

```text
weights/
  dinov3/
    dinov3_vitl16_oph_backbone.pth
  tasks/
    injection_best_model.pth
    diagnosis_best_model.pth
    biomarker_IRF_best_model.pth
```

Pass actual paths at runtime:

```bash
python run_finetuning.py -c cfgs/finetune_injection.yaml \
  --dinov3_weights weights/dinov3/dinov3_vitl16_oph_backbone.pth
```

The DINOv3 backbone implementation used by VISTA is included under
`vista/dinov3`; only model weights need to be provided locally.
