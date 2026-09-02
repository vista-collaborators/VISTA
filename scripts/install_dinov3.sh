#!/usr/bin/env bash
set -euo pipefail

DINOV3_REPO_URL="${DINOV3_REPO_URL:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_DIR="${DINOV3_DIR:-third_party/dinov3_upstream}"

if [[ ! -d "${DINOV3_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${DINOV3_DIR}")"
  git clone "${DINOV3_REPO_URL}" "${DINOV3_DIR}"
fi

python -m pip install -e "${DINOV3_DIR}"

echo "DINOv3 installed from ${DINOV3_DIR}"
