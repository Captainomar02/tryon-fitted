#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_DIR="${MHR_APP_DIR:-${REPO_DIR}}"
MHR_ASSETS_ROOT="${MHR_ASSETS_ROOT:-${APP_DIR}/checkpoints/mhr-assets}"
MHR_ASSETS_DIR="${MHR_ASSETS_DIR:-${MHR_ASSETS_ROOT}/assets}"
MHR_ASSETS_URL="${MHR_ASSETS_URL:-https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip}"
MHR_ASSETS_ZIP="${MHR_ASSETS_ZIP:-/tmp/mhr-assets.zip}"

required_files=(
  "${MHR_ASSETS_DIR}/lod1.fbx"
  "${MHR_ASSETS_DIR}/compact_v6_1.model"
  "${MHR_ASSETS_DIR}/corrective_blendshapes_lod1.npz"
)

all_present=1
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    all_present=0
    break
  fi
done

if [[ "${all_present}" == "1" ]]; then
  echo "MHR assets already present in ${MHR_ASSETS_DIR}"
  exit 0
fi

mkdir -p "${MHR_ASSETS_ROOT}"

echo "Downloading MHR assets to ${MHR_ASSETS_ZIP}"
curl -L "${MHR_ASSETS_URL}" -o "${MHR_ASSETS_ZIP}"

echo "Extracting required MHR LOD1 assets to ${MHR_ASSETS_ROOT}"
unzip -o "${MHR_ASSETS_ZIP}"   assets/lod1.fbx   assets/compact_v6_1.model   assets/corrective_blendshapes_lod1.npz   assets/LICENSE.txt   -d "${MHR_ASSETS_ROOT}"

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required MHR asset after extraction: ${path}" >&2
    exit 1
  fi
done

echo "MHR asset download complete."
