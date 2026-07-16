#!/usr/bin/env bash
# Minimal Vast.ai bootstrap for the front/side body-measurement command.
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/tryon-fitted}"
CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-${APP_DIR}/checkpoints/sam-3d-body-dinov3}"
MHR_ASSETS_DIR="${MHR_ASSETS_DIR:-${APP_DIR}/checkpoints/mhr-assets/assets}"
SAM2_DIR="${SAM2_DIR:-${APP_DIR}/external/sam2}"
SAM2_REF="${SAM2_REF:-main}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

export HF_HOME MHR_ASSETS_DIR SAM3D_SEGMENTOR="${SAM3D_SEGMENTOR:-sam2}"
export SAM3D_SEGMENTOR_PATH="${SAM3D_SEGMENTOR_PATH:-${SAM2_DIR}}"

mkdir -p "${CHECKPOINT_DIR}" "${MHR_ASSETS_DIR}" "${SAM2_DIR%/*}" "${HF_HOME}"

echo '[bootstrap] Downloading the required SAM-3D checkpoint files...'
python - "${CHECKPOINT_DIR}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    local_dir=sys.argv[1],
    allow_patterns=["model.ckpt", "model_config.yaml", "assets/mhr_model.pt"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY

echo '[bootstrap] Downloading the required MHR LOD1 assets...'
if [[ ! -s "${MHR_ASSETS_DIR}/corrective_blendshapes_lod1.npz" ]]; then
  archive="$(mktemp)"
  trap 'rm -f "${archive}"' EXIT
  curl --fail --location --retry 3 \
    --output "${archive}" \
    https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -o "${archive}" \
    assets/lod1.fbx assets/compact_v6_1.model assets/corrective_blendshapes_lod1.npz assets/LICENSE.txt \
    -d "${APP_DIR}/checkpoints/mhr-assets"
  rm -f "${archive}"
  trap - EXIT
fi

echo '[bootstrap] Building the exact MHR arm/hand skinning cache...'
if [[ ! -s "${MHR_ASSETS_DIR}/arm_skinning_masks.npz" ]]; then
  MHR_FBX_PYTHON="${MHR_FBX_PYTHON:-/opt/mhr-fbx-venv/bin/python}"
  if [[ ! -x "${MHR_FBX_PYTHON}" ]]; then
    # Compatibility fallback for an existing image built before this helper
    # environment was added to the Dockerfile.
    MHR_FBX_ENV="${APP_DIR}/.cache/mhr-fbx-venv"
    if [[ ! -x "${MHR_FBX_ENV}/bin/python" ]]; then
      python -m venv "${MHR_FBX_ENV}"
      "${MHR_FBX_ENV}/bin/python" -m pip install --no-cache-dir numpy==2.3.3 ufbx==0.0.5
    fi
    MHR_FBX_PYTHON="${MHR_FBX_ENV}/bin/python"
  fi
  "${MHR_FBX_PYTHON}" - "${MHR_ASSETS_DIR}/lod1.fbx" "${MHR_ASSETS_DIR}/arm_skinning_masks.npz" <<'PY'
import gc
import os
import sys

# ufbx's native scene destructor is unstable during cyclic GC in this image.
# Disable cyclic GC and exit directly after the fully-written cache is closed.
gc.disable()
import numpy as np
import ufbx

fbx_path, output_path = sys.argv[1:]
scene = ufbx.load_file(fbx_path)
mesh = scene.meshes[0]
vertex_count = mesh.num_vertices
tokens = ("uparm", "lowarm", "wrist", "pinky", "ring", "middle", "index", "thumb")
wrist_tokens = ("wrist", "pinky", "ring", "middle", "index", "thumb")

def score(prefix, selected_tokens):
    values = np.zeros(vertex_count, dtype=np.float32)
    for cluster in mesh.skin_deformers[0].clusters:
        name = str(cluster.bone_node.name).lower()
        if name.startswith(prefix) and any(token in name for token in selected_tokens):
            indices = np.asarray(list(cluster.vertices), dtype=np.int64)
            weights = np.asarray(list(cluster.weights), dtype=np.float32)
            values[indices] += weights
    return values

groups = {
    "l_upper": score("l_", ("uparm",)), "l_lower": score("l_", ("lowarm",)),
    "l_wrist": score("l_", wrist_tokens), "r_upper": score("r_", ("uparm",)),
    "r_lower": score("r_", ("lowarm",)), "r_wrist": score("r_", wrist_tokens),
}
l_score, r_score = score("l_", tokens), score("r_", tokens)
np.savez_compressed(
    output_path, l_mask=l_score >= 0.05, r_mask=r_score >= 0.05,
    l_score=l_score, r_score=r_score, **groups,
)
print(f"[bootstrap] Cached MHR arm/hand masks for {vertex_count} vertices.", flush=True)
os._exit(0)
PY
fi

# MHR.from_files() defaults to an assets directory beside the installed mhr package.
MHR_PACKAGE_ASSETS="$(python - <<'PY'
from pathlib import Path
import mhr
print(Path(mhr.__file__).resolve().parents[1] / 'assets')
PY
)"
rm -rf "${MHR_PACKAGE_ASSETS}"
ln -s "${MHR_ASSETS_DIR}" "${MHR_PACKAGE_ASSETS}"

echo '[bootstrap] Checking out only SAM2 runtime source and its required checkpoint...'
if [[ ! -f "${SAM2_DIR}/sam2/build_sam.py" ]]; then
  rm -rf "${SAM2_DIR}"
  git clone --depth 1 --filter=blob:none --sparse --branch "${SAM2_REF}" \
    https://github.com/facebookresearch/sam2.git "${SAM2_DIR}"
  git -C "${SAM2_DIR}" sparse-checkout set sam2 setup.py pyproject.toml
fi
mkdir -p "${SAM2_DIR}/checkpoints"
if [[ ! -s "${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt" ]]; then
  curl --fail --location --retry 3 \
    --output "${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
python -m pip install --no-cache-dir --no-deps -e "${SAM2_DIR}"

echo '[bootstrap] Prefetching the exact RT-DETR and MoGe2 runtime models...'
python - <<'PY'
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from moge.model.v2 import MoGeModel

RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r50vd_coco_o365')
RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r50vd_coco_o365')
MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')
print('[bootstrap] Required runtime models are ready.')
PY

echo '[bootstrap] Prefetching the DINOv3 Torch Hub source used by SAM-3D...'
python - <<'PY'
import torch

# SAM-3D loads this exact backbone with pretrained=False, then applies its own
# checkpoint.  Loading it once here caches the GitHub source in TORCH_HOME and
# verifies transitive Hub dependencies (including termcolor) before use.
torch.hub.load(
    'facebookresearch/dinov3',
    'dinov3_vith16plus',
    source='github',
    pretrained=False,
)
print('[bootstrap] DINOv3 Torch Hub source is ready.')
PY

echo '[bootstrap] Verifying the fusion runtime imports...'
python - <<'PY'
import cv2
import numpy
import torch
import trimesh
from sam_3d_body import load_sam_3d_body
from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from tools.build_sam import HumanSegmentor

print('[bootstrap] Fusion runtime imports are ready.')
PY
