#!/usr/bin/env bash
set -euo pipefail

DINOV3_REPO="${DINOV3_REPO:-}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-}"

python run_finetuning.py -c cfgs/finetune_diagnosis.yaml \
  --dinov3_hub_repo "${DINOV3_REPO}" \
  --dinov3_weights "${DINOV3_WEIGHTS}"
