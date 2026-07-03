#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/tryon-fitted}"
APP_REPO_URL="${APP_REPO_URL:-https://github.com/Captainomar02/tryon-fitted.git}"
APP_REF="${APP_REF:-main}"
SAM3D_PREFETCH_RUNTIME_MODELS="${SAM3D_PREFETCH_RUNTIME_MODELS:-1}"

if [[ ! -d "${APP_DIR}/.git" ]]; then
  rm -rf "${APP_DIR}"
  git clone --branch "${APP_REF}" "${APP_REPO_URL}" "${APP_DIR}"
else
  git -C "${APP_DIR}" fetch origin "${APP_REF}"
  git -C "${APP_DIR}" checkout "${APP_REF}"
  git -C "${APP_DIR}" pull --ff-only origin "${APP_REF}"
fi

cd "${APP_DIR}"

mkdir -p "${APP_DIR}/input" "${APP_DIR}/output" "${APP_DIR}/checkpoints"
python -m pip install -e "./clad-body[mhr,render]" --no-build-isolation --no-deps
scripts/vast/download_checkpoints.sh
scripts/vast/download_mhr_assets.sh
scripts/vast/setup_sam2.sh

if [[ "${SAM3D_PREFETCH_RUNTIME_MODELS}" == "1" ]]; then
  python scripts/vast/prefetch_runtime_models.py
fi

echo "SAM 3D Body + CLAD Body + SAM2 container is ready."
echo "Put front.* and side.* in ${APP_DIR}/input, then run:"
echo "  scripts/vast/run_fusion_and_measure.sh ${APP_DIR}/input ${APP_DIR}/output 178"
