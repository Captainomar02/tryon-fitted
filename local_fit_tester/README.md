# Local Fit Tester

This is a small local fit-mode harness for continuous manual testing.

It uses:

- `sam-3d-body` to generate body params from `front + side + height`
- `clad-body` to extract body measurements
- `tryon-3d/body/fit_report.py` to score garment variants and choose the best size

At the end, it can print either:

- a JSON object with the recommended size and extracted body measurements
- only the recommended size label

It can also generate the Clad render image and include its path in the JSON output.

## Inputs

- `--front`: front image path
- `--side`: side image path
- `--height-cm`: person height in centimeters
- `--variants-json`: garment variants measurement file

The variants JSON can use the same structure as the app's fit pipeline. A sample file is included as `variants.sample.json`.

## Run

```bash
python /home/mizore/sam-3d-body/local_fit_tester/recommend_size.py \
  --front /path/to/front.jpg \
  --side /path/to/side.jpg \
  --height-cm 178 \
  --variants-json /home/mizore/sam-3d-body/local_fit_tester/variants.sample.json \
  --render
```

Default output is JSON:

```json
{
  "body_measurements_cm": {
    "arm_length": 61.2,
    "body_height": 178.0,
    "chest_circumference": 97.5
  },
  "recommended_size": "M",
  "render_path": "/home/mizore/sam-3d-body/local_fit_tester/runs/fit_20260606_184200/clad_render.png"
}
```

If you want only the final size label:

```bash
python /home/mizore/sam-3d-body/local_fit_tester/recommend_size.py \
  --front /path/to/front.jpg \
  --side /path/to/side.jpg \
  --height-cm 178 \
  --variants-json /home/mizore/sam-3d-body/local_fit_tester/variants.sample.json \
  --output-format size
```

Example size-only stdout:

```text
M
```

## Notes

- Progress logs are written to stderr so stdout stays clean.
- Add `--keep-run-dir` if you want to inspect generated artifacts like `fit_report.json`.
- `--output-format json` is the default.
- Add `--render` to keep the Clad render image and include its path in the JSON output.
- Default conda envs:
  - `sam_3d_body` for SAM-3D
  - `sam_3d_body_unified` for Clad
