set -euo pipefail

DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-weights/dinov3/dinov3_vitl16_oph_backbone.pth}"
BIOMARKER="${BIOMARKER:-IRF}"

python run_finetuning.py -c cfgs/finetune_biomarker.yaml \
  --biomarker "${BIOMARKER}" \
  --dinov3_weights "${DINOV3_WEIGHTS}"
