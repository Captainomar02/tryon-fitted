#!/usr/bin/env bash
# Install the minimal SAM2 runtime source and the checkpoint used by fusion.
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/tryon-fitted}"
SAM2_DIR="${SAM2_DIR:-${SAM3D_SEGMENTOR_PATH:-${APP_DIR}/external/sam2}}"
SAM2_REF="${SAM2_REF:-main}"
CHECKPOINT="${SAM2_CHECKPOINT_PATH:-${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt}"

mkdir -p "$(dirname "${SAM2_DIR}")"

echo "[sam2] Checking out the SAM2 runtime source..."
if [[ ! -f "${SAM2_DIR}/sam2/build_sam.py" ]]; then
  rm -rf "${SAM2_DIR}"
  git clone --depth 1 --filter=blob:none --sparse --branch "${SAM2_REF}" \
    https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
fi
# setup.py reads the project README while generating editable-package metadata.
git -C "${SAM2_DIR}" sparse-checkout set sam2 setup.py pyproject.toml README.md

mkdir -p "$(dirname "${CHECKPOINT}")"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "[sam2] Downloading sam2.1_hiera_large checkpoint..."
  curl --fail --location --retry 3 \
    --output "${CHECKPOINT}" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

# SAM2 declares Torch as a build requirement.  The fusion runtime already has a
# compatible Torch installation; avoiding build isolation prevents pip from
# needlessly downloading and building a second copy during startup.
python -m pip install --no-cache-dir --no-deps --no-build-isolation -e "${SAM2_DIR}"
echo "[sam2] Ready: ${SAM2_DIR}"
