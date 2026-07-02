#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_DIR="${APP_DIR:-${REPO_DIR}}"
INPUT_DIR="${INPUT_DIR:-${APP_DIR}/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${APP_DIR}/output}"
TARGET_HEIGHT_CM="${TARGET_HEIGHT_CM:-}"
MEASURE_PRESET="${MEASURE_PRESET:-all}"

case "$#" in
  0)
    ;;
  1)
    TARGET_HEIGHT_CM="$1"
    ;;
  3)
    INPUT_DIR="$1"
    OUTPUT_DIR="$2"
    TARGET_HEIGHT_CM="$3"
    ;;
  *)
    echo "Usage: scripts/vast/run_fusion_and_measure.sh [TARGET_HEIGHT_CM]"
    echo "   or: scripts/vast/run_fusion_and_measure.sh INPUT_DIR OUTPUT_DIR TARGET_HEIGHT_CM"
    echo "Example: scripts/vast/run_fusion_and_measure.sh 178"
    echo "Example: scripts/vast/run_fusion_and_measure.sh ./input ./output 178"
    exit 2
    ;;
esac

if [[ -z "${TARGET_HEIGHT_CM}" ]]; then
  echo "Usage: scripts/vast/run_fusion_and_measure.sh [TARGET_HEIGHT_CM]"
  echo "   or: scripts/vast/run_fusion_and_measure.sh INPUT_DIR OUTPUT_DIR TARGET_HEIGHT_CM"
  echo "Example: scripts/vast/run_fusion_and_measure.sh 178"
  echo "Default input/output: <repo>/input and <repo>/output"
  exit 2
fi

cd "${APP_DIR}"
mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

export SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-${APP_DIR}/checkpoints/sam-3d-body-dinov3}"
CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR}"
CHECKPOINT_PATH="${SAM3D_CHECKPOINT_PATH:-${CHECKPOINT_DIR}/model.ckpt}"
MHR_PATH="${SAM3D_MHR_PATH:-${CHECKPOINT_DIR}/assets/mhr_model.pt}"

if [[ ! -f "${CHECKPOINT_PATH}" || ! -f "${MHR_PATH}" ]]; then
  scripts/vast/download_checkpoints.sh
fi

scripts/vast/download_mhr_assets.sh

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
