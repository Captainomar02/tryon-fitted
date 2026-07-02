#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/sam3d-clad}"
IMAGE_APP_DIR="${IMAGE_APP_DIR:-/opt/workspace-internal/sam3d-clad}"

if [[ ! -d "${APP_DIR}/sam_3d_body" ]]; then
  mkdir -p "${APP_DIR}"
  cp -a "${IMAGE_APP_DIR}/." "${APP_DIR}/"
fi

cd "${APP_DIR}"

mkdir -p "${APP_DIR}/input" "${APP_DIR}/output" "${APP_DIR}/checkpoints"
python -m pip install -e "./clad-body[mhr,render]" --no-build-isolation --no-deps
scripts/vast/download_checkpoints.sh

echo "SAM 3D Body + CLAD Body container is ready."
echo "Put front.* and side.* in ${APP_DIR}/input, then run:"
echo "  scripts/vast/run_fusion_and_measure.sh ${APP_DIR}/input ${APP_DIR}/output 178"
