#!/usr/bin/env python
"""Warm runtime model caches used by run_front_side_fusion.py."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download


def prefetch_hf(repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Prefetching Hugging Face model: {repo_id}")
    snapshot_download(repo_id=repo_id, token=token)


def prefetch_dinov3() -> None:
    import torch

    backbone = os.environ.get("SAM3D_DINOV3_BACKBONE", "dinov3_vith16plus")
    drop_path = float(os.environ.get("SAM3D_DINOV3_DROP_PATH", "0.1"))

    print(f"Prefetching torch hub model code: facebookresearch/dinov3:{backbone}")
    torch.hub.load(
        "facebookresearch/dinov3",
        backbone,
        source="github",
        pretrained=False,
        drop_path=drop_path,
    )


def main() -> None:
    detector = os.environ.get("SAM3D_DETECTOR", "rtdetr")
    fov = os.environ.get("SAM3D_FOV", "moge2")

    if detector == "rtdetr":
        prefetch_hf("PekingU/rtdetr_r50vd_coco_o365")

    if fov == "moge2":
        prefetch_hf(os.environ.get("SAM3D_FOV_PATH") or "Ruicheng/moge-2-vitl-normal")

    if os.environ.get("SAM3D_PREFETCH_DINOV3", "1") == "1":
        prefetch_dinov3()

    print("Runtime model prefetch complete.")


if __name__ == "__main__":
    main()
