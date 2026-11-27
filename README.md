# VISTA

**Vision-based Intelligent System for Treatment of Anti-VEGF**

VISTA supports fine-tuning and evaluation of DINOv3-based OCT image models for three tasks:

- Anti-VEGF injection decision support
- Disease diagnosis
- Biomarker detection

## Repository Structure

```text
.
├── cfgs/
├── data/
│   └── example/
├── scripts/
├── tools/
├── vista/
│   └── dinov3/
├── weights/
├── run_finetuning.py
└── test_task_model.py
```

## Environment Setup

```bash
conda create -n vista python=3.10 -y
conda activate vista
pip install -r requirements.txt
pip install -e .
```

## Data Structure

```text
data/example/
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

`metadata.xlsx` and the split CSV files must include the following columns:

```text
OCT_pathway
anti-VEGF injection
Diagnosis
IRF, SRF, SHRM, DRT, NV, PED
```

## Model Weight Paths

```text
weights/
  dinov3/dinov3_vitl16_oph_backbone.pth
  tasks/injection_best_model.pth
  tasks/diagnosis_best_model.pth
  tasks/biomarker_IRF_best_model.pth
```

## Fine-tuning

```bash
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_injection.sh

DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_diagnosis.sh

BIOMARKER=IRF \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_biomarker.sh
```

## Evaluation

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

## Configuration Files

```text
cfgs/finetune_injection.yaml
cfgs/finetune_diagnosis.yaml
cfgs/finetune_biomarker.yaml
cfgs/test_injection.yaml
cfgs/test_diagnosis.yaml
cfgs/test_biomarker.yaml
```

All paths are relative to the repository root.
