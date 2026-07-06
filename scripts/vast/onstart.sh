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

ensure_mhr_runtime_deps() {
  if python - <<'PY_CHECK'
import torch
import torchvision
import torchaudio
import pymomentum.geometry  # noqa: F401
import mhr  # noqa: F401
def version_tuple(value):
    return tuple(int(part) for part in value.split("+")[0].split(".")[:3])
if version_tuple(torch.__version__) < (2, 8, 0):
    raise SystemExit(1)
if version_tuple(torchvision.__version__) < (0, 23, 0):
    raise SystemExit(1)
if version_tuple(torchaudio.__version__) < (2, 8, 0):
    raise SystemExit(1)
PY_CHECK
  then
    echo "MHR runtime deps already installed."
    return
  fi

  echo "Installing missing MHR runtime deps into $(python -c 'import sys; print(sys.executable)')..."
  python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    pymomentum-cpu==0.1.108.post0 \
    mhr==1.0.1

  python - <<'PY_CHECK'
import torch
import torchvision
import torchaudio
import pymomentum.geometry  # noqa: F401
import mhr  # noqa: F401
print(f"MHR runtime deps ok: torch {torch.__version__}, torchvision {torchvision.__version__}, torchaudio {torchaudio.__version__}")
PY_CHECK
}

ensure_mhr_runtime_deps
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
