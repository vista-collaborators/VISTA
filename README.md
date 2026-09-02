# VISTA

**Vision-based Intelligent System for Treatment of Anti-VEGF**

VISTA is a minimal public-ready codebase for fine-tuning continued-pretraining
DINOv3 OCT image weights on three downstream ophthalmology tasks:

- Anti-VEGF injection decision: binary classification.
- Disease diagnosis: default classes `nAMD`, `DME`, `RVO`.
- Biomarker recognition: one binary classifier per biomarker, for example `IRF`.

This repository is structured after the lightweight style of Retina4IRD: configs
live in `cfgs/`, runnable scripts live in `scripts/`, reusable code lives in
`src/`, and weights are documented but not included.

## Repository Layout

```text
.
├── cfgs/
│   ├── finetune_injection.yaml
│   ├── finetune_diagnosis.yaml
│   ├── finetune_biomarker.yaml
│   ├── test_injection.yaml
│   ├── test_diagnosis.yaml
│   └── test_biomarker.yaml
├── run_finetuning.py
├── test_task_model.py
├── scripts/
├── src/vista/
├── third_party/dinov3/
├── tools/
└── weights/
```

## Environment

Create a fresh environment:

```bash
conda create -n vista python=3.10 -y
conda activate vista
pip install -r requirements.txt
pip install -e .
```

This repository vendors the filtered DINOv3 backbone runtime under
`third_party/dinov3`, so users do not need to clone DINOv3 separately for VISTA
training or testing.

If you want to compare against a full upstream DINOv3 checkout instead of the
vendored runtime, use:

```bash
bash scripts/install_dinov3.sh
```

The normal VISTA workflow does not require this optional step.

The vendored DINOv3 code is distributed under its own license; see
`third_party/dinov3/licenses/DINOV3_LICENSE.md`.

## Weights

No weights are included. Config files intentionally leave weight fields blank:

```yaml
dinov3_weights: ""
checkpoint: ""
```

The provided configs use the relative vendored path:

```yaml
dinov3_hub_repo: third_party/dinov3
```

At runtime, place your local continued-pretraining DINOv3 backbone under
`weights/dinov3/` or pass another relative path:

```bash
python run_finetuning.py -c cfgs/finetune_injection.yaml \
  --dinov3_weights weights/dinov3/dinov3_vitl16_oph_backbone.pth
```

If your continued-pretraining checkpoint is a DINOv3 teacher checkpoint, export a
raw backbone state dict first:

```bash
python tools/export_dinov3_backbone.py \
  --input weights/dinov3/teacher_checkpoint.pth \
  --output weights/dinov3/dinov3_vitl16_oph_backbone.pth \
  --verify \
  --dinov3_hub_repo third_party/dinov3
```

## Dataset

The minimum dataset layout is:

```text
minimum_dataset/
  metadata.xlsx
  A01/5.jpg A01/7.jpg A01/9.jpg A01/11.jpg A01/13.jpg
  A02/5.jpg A02/7.jpg A02/9.jpg A02/11.jpg A02/13.jpg
```

Default metadata columns:

| Column | Meaning |
| --- | --- |
| `OCT_pathway` | OCT slice folder relative to dataset root |
| `anti-VEGF injection` | injection label, `1` means need injection, `0/2` means no injection |
| `Diagnosis` | disease label, default `nAMD/DME/RVO` |
| `IRF`, `SRF`, `SHRM`, `DRT`, `NV`, `PED` | biomarker binary labels |

Prepare splits from the minimum zip:

```bash
DATA_ZIP="data/minimum_dataset.zip" bash scripts/prepare_minimum_dataset.sh
```

This creates:

```text
data/minimum_dataset/
  metadata.xlsx
  A01/
  A02/
  A03/
  splits/
    train.csv
    val.csv
    test.csv
```

For larger datasets you can provide your own `train.csv`, `val.csv`, and
`test.csv` directly. The code also accepts `.xlsx` tables.

Validate the split files without constructing DINOv3:

```bash
python run_finetuning.py -c cfgs/finetune_injection.yaml --dry_run
python run_finetuning.py -c cfgs/finetune_diagnosis.yaml --dry_run
python run_finetuning.py -c cfgs/finetune_biomarker.yaml --biomarker IRF --dry_run
```

## Fresh Clone Checklist

From a fresh clone, another user only needs to add local data and local weights:

```text
data/
  minimum_dataset/
    metadata.xlsx
    A01/
    A02/
    splits/
      train.csv
      val.csv
      test.csv
weights/
  dinov3/
    dinov3_vitl16_oph_backbone.pth
  tasks/
    injection_best_model.pth
    diagnosis_best_model.pth
    biomarker_IRF_best_model.pth
```

The DINOv3 backbone code needed by VISTA is already present in
`third_party/dinov3`.

## Fine-Tuning

Injection decision:

```bash
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_injection.sh
```

Disease diagnosis:

```bash
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_diagnosis.sh
```

Biomarker recognition, for example `SRF`:

```bash
BIOMARKER=SRF \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_biomarker.sh
```

`adaptation` supports:

- `lp`: freeze DINOv3 and train the slice-fusion module plus classifier head.
- `finetune`: train DINOv3, fusion module, and classifier head.

Each experiment writes:

```text
outputs/<task>/<adaptation>/exp_YYYYmmdd_HHMMSS/
  run_config.json
  run_summary.json
  models/best_model.pth
  logs/training_history.csv
  reports/best_val_metrics.json
  reports/val_predictions.csv
  reports/test_predictions.csv
```

## Testing A Saved Model

Set `CHECKPOINT` to a downstream `best_model.pth`:

```bash
CHECKPOINT=outputs/injection/finetune/exp_xxx/models/best_model.pth \
bash scripts/test_model.sh
```

Use a different test config for diagnosis or biomarkers:

```bash
CONFIG=cfgs/test_diagnosis.yaml CHECKPOINT=weights/tasks/diagnosis_best_model.pth bash scripts/test_model.sh
CONFIG=cfgs/test_biomarker.yaml CHECKPOINT=weights/tasks/biomarker_IRF_best_model.pth BIOMARKER=IRF bash scripts/test_model.sh
```

Or call the test script directly:

```bash
python test_task_model.py -c cfgs/test_biomarker.yaml \
  --checkpoint weights/tasks/biomarker_IRF_best_model.pth \
  --biomarker IRF \
  --table data/minimum_dataset/splits/test.csv
```

If a saved checkpoint contains the full downstream model state, `dinov3_weights`
may remain empty at test time; the DINOv3 code is still needed to instantiate
the architecture before loading the saved state.

## Notes

- This is a minimum public-ready codebase. Large experimental outputs, Slurm logs,
  private absolute paths, and model weights are intentionally excluded.
- The minimum example dataset is for structure validation only; use a real train,
  validation, and test split for meaningful metrics.
- If your metadata stores explicit slice paths, pass `--slice_paths_col slice_paths`
  and separate paths with `|`, `;`, or `,`.
