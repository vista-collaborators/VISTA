#!/usr/bin/env bash
set -euo pipefail

DATA_ZIP="${DATA_ZIP:-}"
EXTRACT_DIR="${EXTRACT_DIR:-data/minimum_dataset}"

if [[ -z "${DATA_ZIP}" ]]; then
  echo "Please set DATA_ZIP=/path/to/minimum_dataset.zip"
  exit 1
fi

python tools/prepare_dataset_splits.py \
  --zip_path "${DATA_ZIP}" \
  --extract_dir "${EXTRACT_DIR}" \
  --metadata "${EXTRACT_DIR}/metadata.xlsx" \
  --output_dir "${EXTRACT_DIR}/splits" \
  --image_col OCT_pathway \
  --add_image_path_alias
