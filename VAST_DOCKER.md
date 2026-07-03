# Vast.ai Docker Setup

This repository builds an environment-only Docker image for Vast.ai. The image
contains CUDA, Python, PyTorch, system packages, and Python dependencies. It does
not contain this repository's source code, user images, generated outputs, SAM
3D checkpoints, or the SAM2 checkpoint. The on-start script downloads/clones those
runtime assets into the repo folder.

On each rental, clone this repo into `/workspace/tryon-fitted`. That keeps the
current code, `input/`, `output/`, and `checkpoints/` together in one folder you
can push, pull, or replace independently of the Docker image.

## Build

From this repo:

```bash
docker build -t tryon-fitted:latest .
```

For GitHub Container Registry:

```bash
docker tag tryon-fitted:latest ghcr.io/captainomar02/tryon-fitted:latest
docker push ghcr.io/captainomar02/tryon-fitted:latest
```

This repo also includes a GitHub Actions workflow at
`.github/workflows/docker-ghcr.yml`. After you push to `main`/`master`, or run
the workflow manually, it publishes:

```text
ghcr.io/captainomar02/tryon-fitted:latest
```

## Vast.ai Template

Use this Docker image:

```text
ghcr.io/captainomar02/tryon-fitted:latest
```

Recommended launch mode while editing:

```text
Jupyter + SSH
```

Add account-level or template environment variables:

```text
HF_TOKEN=your_huggingface_token
APP_DIR=/workspace/tryon-fitted
APP_REPO_URL=https://github.com/Captainomar02/tryon-fitted.git
APP_REF=main
SAM3D_PREFETCH_RUNTIME_MODELS=1
SAM3D_SEGMENTOR=sam2
SAM3D_SEGMENTOR_PATH=/workspace/tryon-fitted/external/sam2
FUSION_SIDE_SDF_CHEST_MODE=apex_lobe
FUSION_SIDE_SDF_CHEST_LOBE_GAIN=2.4
```

Use this on-start command:

```bash
set -euo pipefail
APP_DIR="${APP_DIR:-/workspace/tryon-fitted}"
APP_REPO_URL="${APP_REPO_URL:-https://github.com/Captainomar02/tryon-fitted.git}"
APP_REF="${APP_REF:-main}"
if [[ ! -d "${APP_DIR}/.git" ]]; then
  rm -rf "${APP_DIR}"
  git clone --branch "${APP_REF}" "${APP_REPO_URL}" "${APP_DIR}"
fi
"${APP_DIR}/scripts/vast/onstart.sh"
```

Keep `HF_TOKEN` as a Vast.ai secret or template environment variable. Do not
commit it to the repository or bake it into the Docker image.

Your Hugging Face account must already have access to:

```text
facebook/sam-3d-body-dinov3
```

## Run The Full Pipeline

Upload two images:

```text
/workspace/tryon-fitted/input/front.jpg
/workspace/tryon-fitted/input/side.jpg
```

Then run:

```bash
cd /workspace/tryon-fitted
scripts/vast/run_fusion_and_measure.sh 178
```

Outputs:

```text
/workspace/tryon-fitted/output/front_fused_all_body_params_scaled.json
/workspace/tryon-fitted/output/body_measurements.json
/workspace/tryon-fitted/output/body_measurements.png
/workspace/tryon-fitted/output/front_raw.jpg
/workspace/tryon-fitted/output/side_raw.jpg
/workspace/tryon-fitted/output/side_mask.png
/workspace/tryon-fitted/output/side_sdf_profile_edited.jpg
/workspace/tryon-fitted/output/side_sdf_profile_edited.obj
```

## Useful Overrides

```bash
SAM3D_MODEL_REPO=facebook/sam-3d-body-vith
SAM3D_CHECKPOINT_DIR=/workspace/tryon-fitted/checkpoints/sam-3d-body-dinov3
SAM3D_CHECKPOINT_PATH=/workspace/tryon-fitted/checkpoints/sam-3d-body-dinov3/model.ckpt
SAM3D_MHR_PATH=/workspace/tryon-fitted/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
SAM3D_DETECTOR=rtdetr
SAM3D_FOV=moge2
SAM3D_PREFETCH_RUNTIME_MODELS=1
SAM3D_PREFETCH_DINOV3=1
SAM3D_SEGMENTOR=sam2
SAM3D_SEGMENTOR_PATH=/workspace/tryon-fitted/external/sam2
SAM2_DIR=/workspace/tryon-fitted/external/sam2
SAM2_REF=main
SAM2_BUILD_CUDA=0
FUSION_SIDE_SDF_CHEST_MODE=apex_lobe
FUSION_SIDE_SDF_CHEST_LOBE_GAIN=2.4
FUSION_SIDE_SDF_PROFILE_STRENGTH=0.65
FUSION_SIDE_SDF_PROFILE_MAX_PUSH_CM=7.0
MEASURE_PRESET=all
```

## SAM2 Segmentation

The Vast startup script runs `scripts/vast/setup_sam2.sh`. That script clones
`facebookresearch/sam2` into `external/sam2`, downloads
`sam2.1_hiera_large.pt`, and installs SAM2 editable into the active Python
environment. `scripts/vast/run_fusion_and_measure.sh` exports SAM2 as the
default segmentor before calling `run_front_side_fusion.py`.

If you need to refresh SAM2 manually:

```bash
cd /workspace/tryon-fitted
scripts/vast/setup_sam2.sh
```
