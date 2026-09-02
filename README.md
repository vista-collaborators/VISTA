# VISTA

**Vision-based Intelligent System for Treatment of Anti-VEGF**

VISTA is a minimal codebase for fine-tuning and testing DINOv3-based OCT image
models on three ophthalmology tasks:

- Anti-VEGF injection decision
- Disease diagnosis
- Biomarker recognition

The DINOv3 backbone implementation required by VISTA is included inside
`vista/dinov3/`. Model weights and datasets are not included.

## Structure

```text
.
├── cfgs/                 # training and testing configs
├── scripts/              # runnable shell scripts
├── tools/                # dataset and weight preparation helpers
├── vista/                # VISTA package, including vista/dinov3
├── weights/              # local weights placeholder
├── run_finetuning.py
└── test_task_model.py
```

## Installation

```bash
conda create -n vista python=3.10 -y
conda activate vista
pip install -r requirements.txt
pip install -e .
```

## Data

Expected minimum layout:

```text
data/minimum_dataset/
  metadata.xlsx
  A01/5.jpg
  A01/7.jpg
  A01/9.jpg
  A01/11.jpg
  A01/13.jpg
  splits/
    train.csv
    val.csv
    test.csv
```

Required metadata columns:

```text
OCT_pathway
anti-VEGF injection
Diagnosis
IRF, SRF, SHRM, DRT, NV, PED
```

Prepare splits from a zip file:

```bash
DATA_ZIP="data/minimum_dataset.zip" bash scripts/prepare_minimum_dataset.sh
```

## Weights

Weights are intentionally not released in this repository. Put local weights
under `weights/`, for example:

```text
weights/
  dinov3/dinov3_vitl16_oph_backbone.pth
  tasks/injection_best_model.pth
  tasks/diagnosis_best_model.pth
  tasks/biomarker_IRF_best_model.pth
```

Config files keep weight fields empty by default:

```yaml
dinov3_model: dinov3_vitl16
dinov3_weights: ""
checkpoint: ""
```

## Validate Setup

These commands check data/config parsing without training or writing outputs:

```bash
python run_finetuning.py -c cfgs/finetune_injection.yaml --dry_run
python run_finetuning.py -c cfgs/finetune_diagnosis.yaml --dry_run
python run_finetuning.py -c cfgs/finetune_biomarker.yaml --biomarker IRF --dry_run
```

## Fine-Tuning

```bash
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_injection.sh

DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_diagnosis.sh

BIOMARKER=IRF \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_biomarker.sh
```

## Testing

```bash
CHECKPOINT=weights/tasks/injection_best_model.pth \
CONFIG=cfgs/test_injection.yaml \
bash scripts/test_model.sh

CHECKPOINT=weights/tasks/diagnosis_best_model.pth \
CONFIG=cfgs/test_diagnosis.yaml \
bash scripts/test_model.sh

CHECKPOINT=weights/tasks/biomarker_IRF_best_model.pth \
CONFIG=cfgs/test_biomarker.yaml \
BIOMARKER=IRF \
bash scripts/test_model.sh
```

## Notes

- All paths in configs are relative to the repository root.
- Training outputs are written to `outputs/`, which is ignored by git.
- DINOv3 code is included under `vista/dinov3/`; see
  `vista/dinov3/LICENSE.md` for its license.
