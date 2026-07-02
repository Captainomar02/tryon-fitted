#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/sam3d-clad}"
APP_REPO_URL="${APP_REPO_URL:-https://github.com/Captainomar02/tryon-fitted.git}"
APP_REF="${APP_REF:-main}"

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

echo "SAM 3D Body + CLAD Body container is ready."
echo "Put front.* and side.* in ${APP_DIR}/input, then run:"
echo "  scripts/vast/run_fusion_and_measure.sh ${APP_DIR}/input ${APP_DIR}/output 178"
