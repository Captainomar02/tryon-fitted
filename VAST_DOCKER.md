# Vast.ai Docker Setup

This repository builds an environment-only Docker image for Vast.ai. The image
contains CUDA, Python, PyTorch, system packages, and Python dependencies. It does
not contain this repository's source code, user images, generated outputs, or SAM
checkpoints.

On each rental, clone this repo into `/workspace/sam3d-clad`. That keeps the
current code, `input/`, `output/`, and `checkpoints/` together in one folder you
can push, pull, or replace independently of the Docker image.

## Build

From this repo:

```bash
docker build -t sam3d-clad:latest .
```

For GitHub Container Registry:

```bash
docker tag sam3d-clad:latest ghcr.io/captainomar02/sam3d-clad:latest
docker push ghcr.io/captainomar02/sam3d-clad:latest
```

This repo also includes a GitHub Actions workflow at
`.github/workflows/docker-ghcr.yml`. After you push to `main`/`master`, or run
the workflow manually, it publishes:

```text
ghcr.io/captainomar02/sam3d-clad:latest
```

## Vast.ai Template

Use this Docker image:

```text
ghcr.io/captainomar02/sam3d-clad:latest
```

Recommended launch mode while editing:

```text
Jupyter + SSH
```

Add account-level or template environment variables:

```text
HF_TOKEN=your_huggingface_token
APP_DIR=/workspace/sam3d-clad
APP_REPO_URL=https://github.com/Captainomar02/tryon-fitted.git
APP_REF=main
```

Use this on-start command:

```bash
set -euo pipefail
APP_DIR="${APP_DIR:-/workspace/sam3d-clad}"
APP_REPO_URL="${APP_REPO_URL:-https://github.com/Captainomar02/tryon-fitted.git}"
APP_REF="${APP_REF:-main}"
if [[ ! -d "${APP_DIR}/.git" ]]; then
  rm -rf "${APP_DIR}"
  git clone --branch "${APP_REF}" "${APP_REPO_URL}" "${APP_DIR}"
fi
"${APP_DIR}/scripts/vast/onstart.sh"
```

Your Hugging Face account must already have access to:

```text
facebook/sam-3d-body-dinov3
```

## Run The Full Pipeline

Upload two images:

```text
/workspace/sam3d-clad/input/front.jpg
/workspace/sam3d-clad/input/side.jpg
```

Then run:

```bash
cd /workspace/sam3d-clad
scripts/vast/run_fusion_and_measure.sh 178
```

Outputs:

```text
/workspace/sam3d-clad/output/front_fused_all_body_params_scaled.json
/workspace/sam3d-clad/output/body_measurements.json
/workspace/sam3d-clad/output/body_measurements.png
/workspace/sam3d-clad/output/front_raw.jpg
/workspace/sam3d-clad/output/side_raw.jpg
```

## Useful Overrides

```bash
SAM3D_MODEL_REPO=facebook/sam-3d-body-vith
SAM3D_CHECKPOINT_DIR=/workspace/sam3d-clad/checkpoints/sam-3d-body-dinov3
SAM3D_CHECKPOINT_PATH=/workspace/sam3d-clad/checkpoints/sam-3d-body-dinov3/model.ckpt
SAM3D_MHR_PATH=/workspace/sam3d-clad/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
SAM3D_DETECTOR=rtdetr
SAM3D_FOV=moge2
MEASURE_PRESET=all
```
