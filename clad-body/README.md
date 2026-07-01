# clad-body

Vendored measurement package used by this SAM 3D Body Docker image.

It loads SAM 3D Body MHR parameter JSON files and extracts body measurements
such as height, bust/chest, waist, hip, limb circumferences, inseam, sleeve
length, shirt length, and related garment measurements.

In this repository it is installed inside Docker with:

```bash
pip install -e ./clad-body[mhr,render]
```

The main pipeline entry point is:

```bash
scripts/vast/run_fusion_and_measure.sh /workspace/input /workspace/output 178
```

Expected input image names are `front.*` and `side.*`.
