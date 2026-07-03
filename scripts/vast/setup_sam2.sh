#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_DIR="${APP_DIR:-${REPO_DIR}}"
SAM2_DIR="${SAM2_DIR:-${APP_DIR}/external/sam2}"
SAM2_REPO_URL="${SAM2_REPO_URL:-https://github.com/facebookresearch/sam2.git}"
SAM2_REF="${SAM2_REF:-main}"
SAM2_CHECKPOINT_URL="${SAM2_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"
SAM2_CHECKPOINT_PATH="${SAM2_CHECKPOINT_PATH:-${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt}"
SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"

mkdir -p "$(dirname "${SAM2_DIR}")"

if [[ ! -d "${SAM2_DIR}/.git" ]]; then
  if [[ -d "${SAM2_DIR}" ]]; then
    echo "Using existing non-git SAM2 directory: ${SAM2_DIR}"
  else
    echo "Cloning SAM2 ${SAM2_REF} into ${SAM2_DIR}"
    git clone --depth 1 --branch "${SAM2_REF}" "${SAM2_REPO_URL}" "${SAM2_DIR}"
  fi
else
  echo "Updating SAM2 checkout in ${SAM2_DIR}"
  git -C "${SAM2_DIR}" fetch --depth 1 origin "${SAM2_REF}"
  git -C "${SAM2_DIR}" checkout "${SAM2_REF}"
  git -C "${SAM2_DIR}" pull --ff-only origin "${SAM2_REF}"
fi

if [[ ! -f "${SAM2_DIR}/setup.py" ]]; then
  echo "SAM2 setup.py not found at ${SAM2_DIR}/setup.py" >&2
  exit 1
fi

mkdir -p "$(dirname "${SAM2_CHECKPOINT_PATH}")"
if [[ ! -f "${SAM2_CHECKPOINT_PATH}" ]]; then
  echo "Downloading SAM2 checkpoint to ${SAM2_CHECKPOINT_PATH}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${SAM2_CHECKPOINT_PATH}" "${SAM2_CHECKPOINT_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${SAM2_CHECKPOINT_PATH}" "${SAM2_CHECKPOINT_URL}"
  else
    python - "${SAM2_CHECKPOINT_URL}" "${SAM2_CHECKPOINT_PATH}" <<'PYDOWNLOAD'
import sys
import urllib.request
url, out = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, out)
PYDOWNLOAD
  fi
else
  echo "SAM2 checkpoint already present at ${SAM2_CHECKPOINT_PATH}"
fi

if [[ ! -s "${SAM2_CHECKPOINT_PATH}" ]]; then
  echo "SAM2 checkpoint is missing or empty: ${SAM2_CHECKPOINT_PATH}" >&2
  exit 1
fi

export SAM2_BUILD_CUDA
python -m pip install -e "${SAM2_DIR}" --no-build-isolation

cat <<EOF
SAM2 is ready.
  SAM3D_SEGMENTOR=sam2
  SAM3D_SEGMENTOR_PATH=${SAM2_DIR}
EOF
