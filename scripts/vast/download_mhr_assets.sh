#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_DIR="${MHR_APP_DIR:-${REPO_DIR}}"
MHR_ASSETS_ROOT="${MHR_ASSETS_ROOT:-${APP_DIR}/checkpoints/mhr-assets}"
MHR_ASSETS_DIR="${MHR_ASSETS_DIR:-${MHR_ASSETS_ROOT}/assets}"
MHR_ASSETS_URL="${MHR_ASSETS_URL:-https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip}"
MHR_ASSETS_ZIP="${MHR_ASSETS_ZIP:-/tmp/mhr-assets.zip}"

link_mhr_package_assets() {
  local package_assets_dir
  package_assets_dir="$(python - <<'PY'
from pathlib import Path
import mhr
print(Path(mhr.__file__).resolve().parents[1] / "assets")
PY
)"
  ln -sfn "${MHR_ASSETS_DIR}" "${package_assets_dir}"
  echo "Linked MHR package assets: ${package_assets_dir} -> ${MHR_ASSETS_DIR}"
}

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
  link_mhr_package_assets
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

# The upstream mhr package defaults to <site-packages>/assets when
# MHR.from_files() is called without a folder. Link our downloaded assets
# there so the vendored upstream clad-body works without local loader patches.
link_mhr_package_assets

echo "MHR asset download complete."
