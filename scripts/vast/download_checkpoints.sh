#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/sam3d-clad}"
SAM3D_MODEL_REPO="${SAM3D_MODEL_REPO:-facebook/sam-3d-body-dinov3}"
SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-${APP_DIR}/checkpoints/sam-3d-body-dinov3}"

mkdir -p "${SAM3D_CHECKPOINT_DIR}"

if [[ -f "${SAM3D_CHECKPOINT_DIR}/model.ckpt" && -f "${SAM3D_CHECKPOINT_DIR}/assets/mhr_model.pt" ]]; then
  echo "SAM 3D Body checkpoints already present in ${SAM3D_CHECKPOINT_DIR}"
  exit 0
fi

echo "Downloading ${SAM3D_MODEL_REPO} to ${SAM3D_CHECKPOINT_DIR}"
echo "If this fails, make sure your Hugging Face account has model access and HF_TOKEN is set."

hf download "${SAM3D_MODEL_REPO}" \
  --local-dir "${SAM3D_CHECKPOINT_DIR}"

echo "Checkpoint download complete."
