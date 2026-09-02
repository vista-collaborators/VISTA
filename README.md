# OPH DINOv3 Downstream Tasks

Minimal code for fine-tuning continued-pretraining DINOv3 OCT image weights on
three downstream ophthalmology tasks:

- Anti-VEGF injection decision: binary classification.
- Disease diagnosis: default classes `nAMD`, `DME`, `RVO`.
- Biomarker recognition: one binary classifier per biomarker, for example `IRF`.

This repository is structured after the lightweight style of Retina4IRD: configs
live in `cfgs/`, runnable scripts live in `scripts/`, reusable code lives in
`src/`, examples live in `examples/`, and weights are documented but not included.

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
├── src/oph_dinov3_tasks/
├── tools/
├── examples/
└── weights/
```

## Environment

Create a fresh environment:

```bash
conda create -n oph-dinov3 python=3.10 -y
conda activate oph-dinov3
pip install -r requirements.txt
pip install -e .
```

DINOv3 itself is not vendored here. Clone or place a local DINOv3 repository and
pass it with `--dinov3_hub_repo`:

```bash
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
```

## Weights

No weights are included. Config files intentionally leave these fields blank:

```yaml
dinov3_hub_repo: ""
dinov3_weights: ""
checkpoint: ""
```

At runtime, pass your local continued-pretraining DINOv3 backbone:

```bash
python run_finetuning.py -c cfgs/finetune_injection.yaml \
  --dinov3_hub_repo external/dinov3 \
  --dinov3_weights weights/dinov3/dinov3_vitl16_oph_backbone.pth
```

If your continued-pretraining checkpoint is a DINOv3 teacher checkpoint, export a
raw backbone state dict first:

```bash
python tools/export_dinov3_backbone.py \
  --input /path/to/teacher_checkpoint.pth \
  --output weights/dinov3/dinov3_vitl16_oph_backbone.pth \
  --verify \
  --dinov3_hub_repo external/dinov3
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
DATA_ZIP="/path/to/minimum dataset.zip" bash scripts/prepare_minimum_dataset.sh
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

## Fine-Tuning

Injection decision:

```bash
DINOV3_REPO=external/dinov3 \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_injection.sh
```

Disease diagnosis:

```bash
DINOV3_REPO=external/dinov3 \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_diagnosis.sh
```

Biomarker recognition, for example `SRF`:

```bash
BIOMARKER=SRF \
DINOV3_REPO=external/dinov3 \
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
DINOV3_REPO=external/dinov3 \
bash scripts/test_model.sh
```

Use a different test config for diagnosis or biomarkers:

```bash
CONFIG=cfgs/test_diagnosis.yaml CHECKPOINT=/path/to/best_model.pth bash scripts/test_model.sh
CONFIG=cfgs/test_biomarker.yaml CHECKPOINT=/path/to/best_model.pth BIOMARKER=IRF bash scripts/test_model.sh
```

Or call the test script directly:

```bash
python test_task_model.py -c cfgs/test_biomarker.yaml \
  --checkpoint /path/to/best_model.pth \
  --biomarker IRF \
  --table data/minimum_dataset/splits/test.csv \
  --dinov3_hub_repo external/dinov3
```

If a saved checkpoint contains the full downstream model state, `dinov3_weights`
may remain empty at test time; the DINOv3 repository path is still needed to
instantiate the architecture before loading the saved state.

## Notes

- This is a minimum public-ready codebase. Large experimental outputs, Slurm logs,
  private absolute paths, and model weights are intentionally excluded.
- The minimum example dataset is for structure validation only; use a real train,
  validation, and test split for meaningful metrics.
- If your metadata stores explicit slice paths, pass `--slice_paths_col slice_paths`
  and separate paths with `|`, `;`, or `,`.
