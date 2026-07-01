# Vast.ai Docker Setup

This repository is set up to build one Docker image containing both:

- SAM 3D Body inference and front/side fusion
- CLAD Body MHR measurement extraction

The image does not include SAM checkpoints. They are downloaded on Vast.ai into
`/workspace/sam3d-clad/checkpoints` so the image stays smaller and the model
license/access flow remains separate.

The built image stores the source at `/opt/workspace-internal/sam3d-clad`.
The Vast on-start script copies it to `/workspace/sam3d-clad` for editing and
persistent outputs.

## Build

From this repo:

```bash
docker build -t sam3d-clad:latest .
```

For GitHub Container Registry:

```bash
docker tag sam3d-clad:latest ghcr.io/YOUR_USERNAME/sam3d-clad:latest
docker push ghcr.io/YOUR_USERNAME/sam3d-clad:latest
```

This repo also includes a GitHub Actions workflow at
`.github/workflows/docker-ghcr.yml`. After you push to `main`/`master`, or run
the workflow manually, it publishes:

```text
ghcr.io/YOUR_USERNAME/sam3d-clad:latest
```

## Vast.ai Template

Use this Docker image:

```text
ghcr.io/YOUR_USERNAME/sam3d-clad:latest
```

Recommended launch mode while editing:

```text
Jupyter + SSH
```

Add account-level or template environment variables:

```text
HF_TOKEN=your_huggingface_token
APP_DIR=/workspace/sam3d-clad
```

Use this on-start script:

```bash
/workspace/sam3d-clad/scripts/vast/onstart.sh
```

Your Hugging Face account must already have access to:

```text
facebook/sam-3d-body-dinov3
```

## Run The Full Pipeline

Upload two images:

```text
/workspace/input/front.jpg
/workspace/input/side.jpg
```

Then run:

```bash
cd /workspace/sam3d-clad
scripts/vast/run_fusion_and_measure.sh /workspace/input /workspace/output 178
```

Outputs:

```text
/workspace/output/front_fused_all_body_params_scaled.json
/workspace/output/body_measurements.json
/workspace/output/body_measurements.png
/workspace/output/front_raw.jpg
/workspace/output/side_raw.jpg
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
