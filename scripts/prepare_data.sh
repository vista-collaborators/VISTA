set -euo pipefail

DATA_ZIP="${DATA_ZIP:-}"
EXTRACT_DIR="${EXTRACT_DIR:-data/example}"

if [[ -z "${DATA_ZIP}" ]]; then
  echo "Please set DATA_ZIP=data/example.zip"
  exit 1
fi

python tools/prepare_dataset_splits.py \
  --zip_path "${DATA_ZIP}" \
  --extract_dir "${EXTRACT_DIR}" \
  --metadata "${EXTRACT_DIR}/metadata.xlsx" \
  --output_dir "${EXTRACT_DIR}/splits" \
  --image_col OCT_pathway \
  --add_image_path_alias
