# tryon-fitted runtime layout

This checkout is intended to be self-contained after cloning onto a new rental instance.

## Folders

- `input/` - put the two source photos here.
- `output/` - fusion renders, params, measurements, and generated previews are written here.
- `checkpoints/sam-3d-body-dinov3/` - SAM 3D Body model files live here.
- `external/sam2/` - SAM2 is cloned here by the Vast setup script for side/front person masks.

The fusion script expects these input names:

```text
input/front.png  or input/front.jpg  or input/front.jpeg  or input/front.webp
input/side.png   or input/side.jpg   or input/side.jpeg   or input/side.webp
```

## First run on a new instance

From the repo root:

```bash
python -m pip install -r requirements-vast.txt
python -m pip install -e "./clad-body[mhr,render]" --no-build-isolation --no-deps
scripts/vast/download_checkpoints.sh
scripts/vast/download_mhr_assets.sh
scripts/vast/setup_sam2.sh
```

The checkpoint download requires Hugging Face access to `facebook/sam-3d-body-dinov3`.
Set `HF_TOKEN` in the instance environment first if the model is gated for your
account. Do not commit tokens into this repo.

On Vast.ai, the recommended on-start command is:

```bash
APP_DIR="${APP_DIR:-/workspace/tryon-fitted}"
APP_REPO_URL="${APP_REPO_URL:-https://github.com/Captainomar02/tryon-fitted.git}"
APP_REF="${APP_REF:-main}"
if [[ ! -d "${APP_DIR}/.git" ]]; then
  rm -rf "${APP_DIR}"
  git clone --branch "${APP_REF}" "${APP_REPO_URL}" "${APP_DIR}"
fi
"${APP_DIR}/scripts/vast/onstart.sh"
```

## Run fusion and measurements

```bash
scripts/vast/run_fusion_and_measure.sh ./input ./output 178
```

Replace `178` with the subject height in centimeters.

The main fused params file will be:

```text
output/front_fused_all_body_params_scaled.json
```

The measurement helper writes:

```text
output/body_measurements.json
output/body_measurements.png
```


## Current Fusion Behavior

`run_front_side_fusion.py` now uses SAM2 masks by default:

```text
SAM3D_SEGMENTOR=sam2
SAM3D_SEGMENTOR_PATH=external/sam2
```

The side mesh is first measured by CLAD untouched to locate the bust and hip height anchors.
Then only the side-profile chest and butt regions are corrected against the side mask before
front/side fusion. The chest correction defaults to the newer `apex_lobe` mode, while the
older row-edge SDF method is still available:

```bash
FUSION_SIDE_SDF_CHEST_MODE=apex_lobe   # default
FUSION_SIDE_SDF_CHEST_MODE=row_sdf     # old chest behavior
FUSION_SIDE_SDF_CHEST_LOBE_GAIN=2.4    # apex_lobe chest gain
FUSION_SIDE_SDF_PROFILE_STRENGTH=0.65
FUSION_SIDE_SDF_PROFILE_MAX_PUSH_CM=7.0
```

The fused geometry is passed into CLAD as the measurement/render mesh unless
`--no-fused-vertex-override` is used.

## Keeping large files in the repo

Checkpoints and images are large, so this repo includes `.gitattributes` rules for Git LFS.
If you want the actual input images, output images, and checkpoints to come back when you clone on the next rental, commit them with Git LFS enabled.
