# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch
from contextlib import nullcontext
import numpy as np
from PIL import Image


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("########### Using human segmentor: SAM2...")
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2
        elif name == "sam3":
            print("########### Using human segmentor: SAM3...")
            self.sam = load_sam3(device, **kwargs)
            self.sam_func = run_sam3
        else:
            raise NotImplementedError
    
    def run_sam(self, img, boxes, **kwargs):
        return self.sam_func(self.sam, img, boxes, device=self.device)
        

def load_sam2(device, path):
    if not path:
        raise FileNotFoundError(
            "SAM2 segmentor path is empty. Set SAM3D_SEGMENTOR_PATH or pass "
            "--segmentor-path, usually external/sam2."
        )

    checkpoint = os.path.join(path, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"SAM2 repo not found at {path}. Run scripts/vast/setup_sam2.sh "
            "or set SAM3D_SEGMENTOR_PATH to an existing SAM2 checkout."
        )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint}. Run scripts/vast/setup_sam2.sh "
            "or set SAM2_CHECKPOINT_PATH/SAM3D_SEGMENTOR_PATH correctly."
        )

    import sys
    if path not in sys.path:
        sys.path.append(path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.eval()

    return predictor


def load_sam3(device, path):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    model = build_sam3_image_model()
    predictor = Sam3Processor(model)
    return predictor


def run_sam2(sam_predictor, img, boxes, device="cuda"):
    device_type = torch.device(device).type
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda" and torch.cuda.is_available()
        else nullcontext()
    )
    with autocast_ctx:
        sam_predictor.set_image(img)
        all_masks, all_scores = [], []
        for i in range(boxes.shape[0]):
            # First prediction: bbox only
            masks, scores, logits = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[[i]],
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            mask_1 = masks[0]
            score_1 = scores[0]
            all_masks.append(mask_1)
            all_scores.append(score_1)

            # cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)[:-4]}_mask_{i}.jpg"), (mask_1 * 255).astype(np.uint8))
        all_masks = np.stack(all_masks)
        all_scores = np.stack(all_scores)

    return all_masks, all_scores


def run_sam3(sam_predictor, img, boxes, device="cuda"):
    # switch bgr to rgb 
    img = img[:, :, ::-1].copy()
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    inference_state = sam_predictor.set_image(img)
    # Prompt the model with text
    output = sam_predictor.set_text_prompt(state=inference_state, prompt="person")

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    score_threshold = 0.5
    confident_idx = scores > score_threshold
    masks = masks[confident_idx].float().squeeze(1).cpu().numpy()
    scores = scores[confident_idx].cpu().numpy()

    return masks, scores