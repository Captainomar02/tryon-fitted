# sam3d-clad runtime layout

This checkout is intended to be self-contained after cloning onto a new rental instance.

## Folders

- `input/` - put the two source photos here.
- `output/` - fusion renders, params, measurements, and generated previews are written here.
- `checkpoints/sam-3d-body-dinov3/` - SAM 3D Body model files live here.

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
```

The checkpoint download requires Hugging Face access to `facebook/sam-3d-body-dinov3`.
Set `HF_TOKEN` first if the model is gated for your account.

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

## Keeping large files in the repo

Checkpoints and images are large, so this repo includes `.gitattributes` rules for Git LFS.
If you want the actual input images, output images, and checkpoints to come back when you clone on the next rental, commit them with Git LFS enabled.
