# SAM 3D Body + CLAD Body on Vast.ai

This repository packages SAM 3D Body and CLAD Body in one Docker image for
front/side body reconstruction and measurement extraction on Vast.ai.

The Docker image contains:

- SAM 3D Body code
- CLAD Body measurement code vendored in `clad-body/`
- Python/CUDA runtime dependencies
- Vast.ai startup and pipeline scripts

Large model checkpoints are not committed. They are downloaded on the Vast.ai
instance with `scripts/vast/download_checkpoints.sh`.

## Vast.ai

See [VAST_DOCKER.md](VAST_DOCKER.md) for build, push, template, and run
instructions.

Quick local build:

```bash
docker build -t sam3d-clad:latest .
```

Quick run on a Vast.ai instance after the image is started:

```bash
cd /workspace/sam3d-clad
scripts/vast/run_fusion_and_measure.sh /workspace/input /workspace/output 178
```

Expected input files:

```text
/workspace/input/front.jpg
/workspace/input/side.jpg
```

Main outputs:

```text
/workspace/output/front_fused_all_body_params_scaled.json
/workspace/output/body_measurements.json
/workspace/output/body_measurements.png
```
