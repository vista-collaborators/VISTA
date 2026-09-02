#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
CONFIG="${CONFIG:-cfgs/test_injection.yaml}"
DINOV3_REPO="${DINOV3_REPO:-third_party/dinov3}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-}"
BIOMARKER="${BIOMARKER:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Please set CHECKPOINT=weights/tasks/<task>_best_model.pth"
  exit 1
fi

extra_args=()
if [[ -n "${BIOMARKER}" ]]; then
  extra_args+=(--biomarker "${BIOMARKER}")
fi

python test_task_model.py -c "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dinov3_hub_repo "${DINOV3_REPO}" \
  --dinov3_weights "${DINOV3_WEIGHTS}" \
  "${extra_args[@]}"
