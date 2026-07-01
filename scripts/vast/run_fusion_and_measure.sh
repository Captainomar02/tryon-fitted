#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/sam3d-clad}"
INPUT_DIR="${1:-${INPUT_DIR:-/workspace/input}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/workspace/output}}"
TARGET_HEIGHT_CM="${3:-${TARGET_HEIGHT_CM:-}}"
MEASURE_PRESET="${MEASURE_PRESET:-all}"

if [[ -z "${TARGET_HEIGHT_CM}" ]]; then
  echo "Usage: scripts/vast/run_fusion_and_measure.sh INPUT_DIR OUTPUT_DIR TARGET_HEIGHT_CM"
  echo "Example: scripts/vast/run_fusion_and_measure.sh /workspace/input /workspace/output 178"
  exit 2
fi

cd "${APP_DIR}"
mkdir -p "${OUTPUT_DIR}"

CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-./checkpoints/sam-3d-body-dinov3}"
CHECKPOINT_PATH="${SAM3D_CHECKPOINT_PATH:-${CHECKPOINT_DIR}/model.ckpt}"
MHR_PATH="${SAM3D_MHR_PATH:-${CHECKPOINT_DIR}/assets/mhr_model.pt}"

if [[ ! -f "${CHECKPOINT_PATH}" || ! -f "${MHR_PATH}" ]]; then
  scripts/vast/download_checkpoints.sh
fi

python run_front_side_fusion.py \
  --target-height "${TARGET_HEIGHT_CM}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"

python scripts/measure_mhr_params.py \
  --params "${OUTPUT_DIR}/front_fused_all_body_params_scaled.json" \
  --out-json "${OUTPUT_DIR}/body_measurements.json" \
  --render "${OUTPUT_DIR}/body_measurements.png" \
  --preset "${MEASURE_PRESET}"

echo "Pipeline complete."
echo "Measurements: ${OUTPUT_DIR}/body_measurements.json"
