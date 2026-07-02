#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_DIR="${APP_DIR:-${REPO_DIR}}"
SAM3D_MODEL_REPO="${SAM3D_MODEL_REPO:-facebook/sam-3d-body-dinov3}"
SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-${APP_DIR}/checkpoints/sam-3d-body-dinov3}"

mkdir -p "${SAM3D_CHECKPOINT_DIR}"

if [[ -f "${SAM3D_CHECKPOINT_DIR}/model.ckpt" && -f "${SAM3D_CHECKPOINT_DIR}/assets/mhr_model.pt" ]]; then
  echo "SAM 3D Body checkpoints already present in ${SAM3D_CHECKPOINT_DIR}"
  exit 0
fi

echo "Downloading ${SAM3D_MODEL_REPO} to ${SAM3D_CHECKPOINT_DIR}"
echo "If this fails, make sure your Hugging Face account has model access and HF_TOKEN is set."

if command -v hf >/dev/null 2>&1; then
  hf download "${SAM3D_MODEL_REPO}" \
    --local-dir "${SAM3D_CHECKPOINT_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${SAM3D_MODEL_REPO}" \
    --local-dir "${SAM3D_CHECKPOINT_DIR}" \
    --local-dir-use-symlinks False
else
  python - "${SAM3D_MODEL_REPO}" "${SAM3D_CHECKPOINT_DIR}" <<'PYDOWNLOAD'
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
)
PYDOWNLOAD
fi

if [[ ! -f "${SAM3D_CHECKPOINT_DIR}/model.ckpt" || ! -f "${SAM3D_CHECKPOINT_DIR}/assets/mhr_model.pt" ]]; then
  echo "Checkpoint download finished, but required files were not found:" >&2
  echo "  ${SAM3D_CHECKPOINT_DIR}/model.ckpt" >&2
  echo "  ${SAM3D_CHECKPOINT_DIR}/assets/mhr_model.pt" >&2
  exit 1
fi

echo "Checkpoint download complete."
