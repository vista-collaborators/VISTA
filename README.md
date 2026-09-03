# VISTA

**Vision-based Intelligent System for Treatment of Anti-VEGF**

VISTA 用于基于 DINOv3 的 OCT 图像模型微调与测试，支持三项任务：

- 抗 VEGF 打针决策
- 疾病诊断
- 生物标志物识别

## 目录结构

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

## 环境配置

```bash
conda create -n vista python=3.10 -y
conda activate vista
pip install -r requirements.txt
pip install -e .
```

## 数据结构

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

`metadata.xlsx` 和 split CSV 需要包含：

```text
OCT_pathway
anti-VEGF injection
Diagnosis
IRF, SRF, SHRM, DRT, NV, PED
```

## 权重路径

```text
weights/
  dinov3/dinov3_vitl16_oph_backbone.pth
  tasks/injection_best_model.pth
  tasks/diagnosis_best_model.pth
  tasks/biomarker_IRF_best_model.pth
```

## 微调

```bash
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_injection.sh

DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_diagnosis.sh

BIOMARKER=IRF \
DINOV3_WEIGHTS=weights/dinov3/dinov3_vitl16_oph_backbone.pth \
bash scripts/finetune_biomarker.sh
```

## 测试

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

## 配置文件

```text
cfgs/finetune_injection.yaml
cfgs/finetune_diagnosis.yaml
cfgs/finetune_biomarker.yaml
cfgs/test_injection.yaml
cfgs/test_diagnosis.yaml
cfgs/test_biomarker.yaml
```

所有路径均相对于仓库根目录。
