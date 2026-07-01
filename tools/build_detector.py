# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", path=""):
        self.device = device
        self.name = name

        if name == "vitdet":
            print("########### Using human detector: ViTDet...")
            self.detector = load_detectron2_vitdet(path=path)
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            self.detector_func = run_detectron2_vitdet

        elif name == "rtdetr":
            print("########### Using human detector: RT-DETR...")
            self.detector = load_rtdetr(device=self.device)
            self.detector_func = run_rtdetr

        elif name == "sam3":
            print("########### Using human detector: SAM3...")
            self.detector = load_sam3(device=self.device, path=path)
            self.detector_func = run_sam3

        else:
            raise NotImplementedError(f"Unsupported detector: {name}")

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)


def load_detectron2_vitdet(path=""):
    """
    Load the official ViTDet detector.

    If path == "", use the official public checkpoint URL.
    If path != "", expect model_final_f05665.pkl inside that folder.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))

    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )

    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)
    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]
    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )

    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )

    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    sorted_indices = np.lexsort(
        (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    )
    boxes = boxes[sorted_indices]
    return boxes


def load_rtdetr(device="cuda"):
    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

    model_name = "PekingU/rtdetr_r50vd_coco_o365"
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def run_rtdetr(detector_bundle, img, bbox_thr=0.8, **kwargs):
    processor = detector_bundle["processor"]
    model = detector_bundle["model"]
    device = detector_bundle["device"]

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.shape[:2]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=bbox_thr, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()
    labels = results["labels"].detach().cpu().numpy()

    keep = labels == 0  # COCO class 0 = person
    boxes = boxes[keep]

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return boxes.astype(np.float32)


def load_sam3(device="cuda", path=""):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    detector = build_sam3_image_model()
    detector = detector.to(device)
    detector.eval()

    processor = Sam3Processor(detector)
    return {"detector": detector, "processor": processor, "device": device}


def run_sam3(detector_bundle, img, det_cat_id: int = 0, bbox_thr: float = 0.5, **kwargs):
    detector = detector_bundle["detector"]
    processor = detector_bundle["processor"]

    img = img[:, :, ::-1].copy()
    img = Image.fromarray(img.astype("uint8"), "RGB")

    inference_state = processor.set_image(img)
    output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    confident_idx = scores > bbox_thr
    boxes = boxes[confident_idx].cpu().numpy()

    scale = 1.2
    enlarged_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        new_x1 = max(cx - w / 2, 0)
        new_y1 = max(cy - h / 2, 0)
        new_x2 = cx + w / 2
        new_y2 = cy + h / 2
        enlarged_boxes.append([new_x1, new_y1, new_x2, new_y2])

    return np.array(enlarged_boxes, dtype=np.float32)