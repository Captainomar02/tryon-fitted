import glob
import json
import os
import argparse
import shutil
import sys

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together


def find_image(input_dir, stem):
    matches = []
    ext_priority = ("png", "jpg", "jpeg", "webp")
    for ext in ext_priority:
        matches.extend(glob.glob(os.path.join(input_dir, f"{stem}.{ext}")))
        matches.extend(glob.glob(os.path.join(input_dir, f"{stem.upper()}.{ext}")))
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find image named {stem}.* in {input_dir}")
    if len(matches) > 1:
        # Keep behavior deterministic when multiple formats exist for the same image.
        unique_matches = sorted(set(matches))
        priority_map = {f".{ext}": i for i, ext in enumerate(ext_priority)}
        unique_matches.sort(
            key=lambda p: (
                priority_map.get(os.path.splitext(p)[1].lower(), len(ext_priority)),
                p.lower(),
            )
        )
        chosen = unique_matches[0]
        print(
            f"[fusion] Found multiple matches for {stem}; "
            f"using {chosen} and ignoring {unique_matches[1:]}"
        )
        return chosen
    return matches[0]


def convert_result_to_save_dict(result):
    save_dict = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            save_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (int, float, bool, np.generic)):
            save_dict[k] = np.array(v)
        elif v is None:
            continue
    return save_dict


def save_mesh_obj(vertices, faces, obj_path):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape [N,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape [M,3], got {faces.shape}")

    with open(obj_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return obj_path


def save_result_json(result, output_json_path):
    save_dict = convert_result_to_save_dict(result)

    json_dict = {}
    for k, v in save_dict.items():
        if isinstance(v, np.ndarray):
            json_dict[k] = v.tolist()
        elif isinstance(v, np.generic):
            json_dict[k] = v.item()
        else:
            json_dict[k] = v

    with open(output_json_path, "w") as f:
        json.dump(json_dict, f)

    return output_json_path


def clear_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        return

    keep_names = {".gitkeep", "README.md"}
    for name in os.listdir(output_dir):
        if name in keep_names:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def render_and_save(image_path, outputs, faces, output_dir, out_name):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rend_img = visualize_sample_together(img, outputs, faces)
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, rend_img.astype(np.uint8))
    return out_path


def normalize_vector(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def rotation_matrix_from_vectors(a, b, eps=1e-8):
    a = normalize_vector(a.astype(np.float64), eps=eps)
    b = normalize_vector(b.astype(np.float64), eps=eps)

    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)

    if s < eps:
        if c > 0:
            return np.eye(3, dtype=np.float64)

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = normalize_vector(np.cross(a, helper), eps=eps)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s ** 2))
    return R


def estimate_up_direction_from_mesh(vertices):
    v = vertices.astype(np.float64)
    y = v[:, 1]

    low_thresh = np.quantile(y, 0.08)
    high_thresh = np.quantile(y, 0.92)

    low_pts = v[y <= low_thresh]
    high_pts = v[y >= high_thresh]

    if low_pts.shape[0] < 10 or high_pts.shape[0] < 10:
        low_center = v[np.argmin(y)]
        high_center = v[np.argmax(y)]
    else:
        low_center = low_pts.mean(axis=0)
        high_center = high_pts.mean(axis=0)

    # SAM-3D body vertices use image-like vertical polarity here: smaller Y is
    # closer to the head and larger Y is closer to the feet. Physical "up" is
    # therefore low-Y minus high-Y.
    up_dir = low_center - high_center
    return normalize_vector(up_dir)


def compute_height_along_axis(vertices, axis):
    axis = normalize_vector(axis.astype(np.float64))
    proj = vertices.astype(np.float64) @ axis
    return float(proj.max() - proj.min())


def center_and_orient_mesh(vertices, target_up=np.array([0.0, 1.0, 0.0], dtype=np.float64)):
    v = vertices.astype(np.float64)
    centroid = v.mean(axis=0, keepdims=True)

    up_dir = estimate_up_direction_from_mesh(v)
    height_before = compute_height_along_axis(v - centroid, up_dir)

    R = rotation_matrix_from_vectors(up_dir, target_up)

    v_centered = v - centroid
    v_oriented = v_centered @ R.T

    height_after = compute_height_along_axis(v_oriented, target_up)

    return {
        "vertices_original": v.astype(np.float32),
        "vertices_centered": v_centered.astype(np.float32),
        "vertices_oriented": v_oriented.astype(np.float32),
        "centroid": centroid.astype(np.float64),
        "estimated_up_direction": up_dir.astype(np.float64),
        "rotation_matrix": R.astype(np.float64),
        "height_before": np.array(height_before, dtype=np.float64),
        "height_after": np.array(height_after, dtype=np.float64),
    }


def kabsch_align_vertices(front_vertices, side_vertices):
    """
    Align side mesh into front mesh frame using corresponding vertex indices.
    """
    vf = front_vertices.astype(np.float64)
    vs = side_vertices.astype(np.float64)

    cf = vf.mean(axis=0, keepdims=True)
    cs = vs.mean(axis=0, keepdims=True)

    X = vf - cf
    Y = vs - cs

    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    vs_aligned = (Y @ R.T) + cf
    return vs_aligned.astype(np.float32), R.astype(np.float64), cf.astype(np.float64), cs.astype(np.float64)


def fuse_front_xy_with_side_z(front_vertices, side_vertices_aligned):
    """
    Final fusion rule:
    - keep front x, y
    - replace z with aligned side z
    """
    if front_vertices.shape != side_vertices_aligned.shape:
        raise ValueError(
            f"Shape mismatch: front {front_vertices.shape} vs side_aligned {side_vertices_aligned.shape}"
        )

    fused = front_vertices.astype(np.float32).copy()
    fused[:, 2] = side_vertices_aligned[:, 2].astype(np.float32)
    return fused


def _smooth_band_weight(height_pct, center, half_width):
    distance = np.abs(height_pct - float(center))
    weight = np.clip(1.0 - (distance / float(half_width)), 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


SIDE_SDF_BAND_HALF_WIDTH = 0.02953125
SIDE_SDF_BAND_FEATHER_WIDTH = SIDE_SDF_BAND_HALF_WIDTH
SIDE_SDF_BAND_CORE_WEIGHT = 1.0
SIDE_SDF_OUTER_SMOOTH_WIDTH = SIDE_SDF_BAND_HALF_WIDTH * 1.45
SIDE_SDF_CONTAINMENT_PASSES = 5
SIDE_SDF_CONTAINMENT_GAIN = 1.0
SIDE_SDF_CONTAINMENT_MARGIN_PX = 2.0
SIDE_SDF_CONTAINMENT_MAX_STEP_PX = 24.0
SIDE_SDF_DISPLACEMENT_SMOOTH_ITERS = 16
SIDE_SDF_DISPLACEMENT_SMOOTH_ALPHA = 0.36
SIDE_SDF_DISPLACEMENT_SMOOTH_BLEND = 0.60
SIDE_SDF_EDGE_HEIGHT_SMOOTH_BINS = 220
SIDE_SDF_EDGE_HEIGHT_SMOOTH_RADIUS_BINS = 9
SIDE_SDF_EDGE_HEIGHT_SMOOTH_BLEND = 0.58
SIDE_SDF_ROW_EDGE_LOW_QUANTILE = 0.01
SIDE_SDF_ROW_EDGE_HIGH_QUANTILE = 0.99
SIDE_SDF_SOLVE_DATA_WEIGHT = 180.0
SIDE_SDF_SOLVE_PIN_WEIGHT = 0.85
SIDE_SDF_SOLVE_SMOOTH_LAMBDA = 1.2
SIDE_SDF_TARGET_CORE_HALF_WIDTH = SIDE_SDF_BAND_HALF_WIDTH + SIDE_SDF_BAND_FEATHER_WIDTH
SIDE_SDF_TARGET_FEATHER_WIDTH = SIDE_SDF_BAND_HALF_WIDTH * 3.0
SIDE_SDF_SOLVE_OUTSIDE_SMOOTH_WIDTH = SIDE_SDF_BAND_HALF_WIDTH * 2.5
SIDE_SDF_SOLVE_SIDE_PIN_SCALE = 0.35
SIDE_SDF_SOLVE_MIN_DIAGONAL = 1e-6
SIDE_SDF_RESIDUAL_SOLVE_PASSES = 3
SIDE_SDF_RESIDUAL_SOLVE_GAIN = 0.95
SIDE_SDF_ROW_UNDERFIT_PEAK_PRESERVE = True
SIDE_SDF_SOLVE_EDGE_DATA_FLOOR = 0.20
SIDE_SDF_SOLVE_EDGE_DATA_POWER = 2.0


def _flat_band_with_feather(height_pct, center, half_width, feather_width):
    distance = np.abs(height_pct - float(center))
    weight = np.zeros_like(distance, dtype=np.float64)
    weight[distance <= float(half_width)] = SIDE_SDF_BAND_CORE_WEIGHT

    if feather_width > 0.0:
        feather = (distance > float(half_width)) & (distance <= float(half_width + feather_width))
        t = (distance[feather] - float(half_width)) / float(feather_width)
        smooth = t * t * (3.0 - 2.0 * t)
        weight[feather] = SIDE_SDF_BAND_CORE_WEIGHT * (1.0 - smooth)

    return weight


def _outside_band_smooth_weight(height_pct, center, half_width, feather_width, outer_width):
    distance = np.abs(height_pct - float(center))
    start = float(half_width + feather_width)
    weight = np.zeros_like(distance, dtype=np.float64)
    if outer_width <= 0.0:
        return weight

    outer = (distance > start) & (distance <= start + float(outer_width))
    t = (distance[outer] - start) / float(outer_width)
    smooth = t * t * (3.0 - 2.0 * t)
    weight[outer] = 1.0 - smooth
    return weight


def _band_extent(vertices, height_pct, low_pct, high_pct):
    mask = (height_pct >= low_pct) & (height_pct <= high_pct)
    band = vertices[mask]
    if band.shape[0] < 10:
        return 0.0, 0.0
    x_extent = float(band[:, 0].max() - band[:, 0].min())
    z_extent = float(band[:, 2].max() - band[:, 2].min())
    return x_extent, z_extent


def enhance_profile_depth_from_front_width(vertices, strength=0.35, max_scale=1.22):
    """
    Conservative depth correction for cases where the MHR prior flattens very
    curvy profiles. It uses front-view lateral width as a sanity bound for the
    side-derived depth, then smoothly expands depth in bust and hip/glute bands.
    """
    if strength <= 0.0:
        return vertices.astype(np.float32), {
            "enabled": np.array(False),
            "strength": np.array(float(strength), dtype=np.float64),
            "bust_scale": np.array(1.0, dtype=np.float64),
            "hip_scale": np.array(1.0, dtype=np.float64),
        }

    v = vertices.astype(np.float64).copy()
    y_min = float(v[:, 1].min())
    height = float(v[:, 1].max() - y_min)
    if height <= 1e-8:
        return vertices.astype(np.float32), {
            "enabled": np.array(False),
            "strength": np.array(float(strength), dtype=np.float64),
            "bust_scale": np.array(1.0, dtype=np.float64),
            "hip_scale": np.array(1.0, dtype=np.float64),
        }

    pct = (v[:, 1] - y_min) / height
    corrections = [
        {
            "name": "bust",
            "low": 0.66,
            "high": 0.79,
            "center": 0.725,
            "half_width": 0.075,
            "min_depth_to_width": 0.72,
        },
        {
            "name": "hip",
            "low": 0.42,
            "high": 0.58,
            "center": 0.50,
            "half_width": 0.09,
            "min_depth_to_width": 0.70,
        },
    ]

    scale_by_name = {}
    total_weight = np.zeros(v.shape[0], dtype=np.float64)
    for correction in corrections:
        x_extent, z_extent = _band_extent(v, pct, correction["low"], correction["high"])
        if z_extent <= 1e-8 or x_extent <= 1e-8:
            band_scale = 1.0
        else:
            target_depth = x_extent * correction["min_depth_to_width"]
            raw_scale = target_depth / z_extent
            band_scale = 1.0 + float(strength) * max(0.0, raw_scale - 1.0)
            band_scale = float(np.clip(band_scale, 1.0, max_scale))

        scale_by_name[correction["name"]] = band_scale
        total_weight += _smooth_band_weight(
            pct,
            correction["center"],
            correction["half_width"],
        ) * (band_scale - 1.0)

    if np.max(total_weight) > 0.0:
        # Scale depth around each horizontal slice center so global alignment is
        # preserved while local torso thickness can increase.
        unique_bins = np.floor(pct * 200).astype(np.int32)
        z_center = np.zeros(v.shape[0], dtype=np.float64)
        for bin_id in np.unique(unique_bins):
            bin_mask = unique_bins == bin_id
            z_center[bin_mask] = np.median(v[bin_mask, 2])
        v[:, 2] = z_center + (v[:, 2] - z_center) * (1.0 + total_weight)

    return v.astype(np.float32), {
        "enabled": np.array(True),
        "strength": np.array(float(strength), dtype=np.float64),
        "max_scale": np.array(float(max_scale), dtype=np.float64),
        "bust_scale": np.array(float(scale_by_name.get("bust", 1.0)), dtype=np.float64),
        "hip_scale": np.array(float(scale_by_name.get("hip", 1.0)), dtype=np.float64),
    }


def sam_upright_vertices_to_clad_canonical(vertices):
    """
    Convert fusion-space vertices (X lateral, Y up, Z profile depth) to the
    CLAD/MHR convention (X lateral, Z up, +Y front/back profile axis).
    """
    src = vertices.astype(np.float64)
    dst = np.zeros_like(src, dtype=np.float64)
    dst[:, 0] = src[:, 0]
    dst[:, 1] = -src[:, 2]
    dst[:, 2] = src[:, 1]
    dst[:, 2] -= dst[:, 2].min()
    center_xy = (dst[:, :2].max(axis=0) + dst[:, :2].min(axis=0)) * 0.5
    dst[:, 0] -= center_xy[0]
    dst[:, 1] -= center_xy[1]
    return dst.astype(np.float32)


def load_binary_mask(mask_path, expected_shape=None):
    if not mask_path:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if expected_shape is not None and mask.shape[:2] != tuple(expected_shape[:2]):
        mask = cv2.resize(
            mask,
            (int(expected_shape[1]), int(expected_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 127


def result_mask_to_binary(result, expected_shape=None):
    mask = result.get("mask") if isinstance(result, dict) else None
    if mask is None:
        return None
    mask = _to_numpy(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    if expected_shape is not None and mask.shape[:2] != tuple(expected_shape[:2]):
        mask = cv2.resize(
            mask,
            (int(expected_shape[1]), int(expected_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0.5


def mask_to_sdf(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(255 - mask_u8, cv2.DIST_L2, 5)
    return outside.astype(np.float32) - inside.astype(np.float32)


def sample_image_bilinear(image, xy):
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0].astype(np.float64), 0.0, w - 1.0)
    y = np.clip(xy[:, 1].astype(np.float64), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def save_mask_and_sdf(mask, output_dir, stem):
    if mask is None:
        return None, None, None
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_path = os.path.join(output_dir, f"{stem}_mask.png")
    cv2.imwrite(mask_path, mask_u8)

    sdf = mask_to_sdf(mask)
    sdf_path = os.path.join(output_dir, f"{stem}_sdf.npy")
    np.save(sdf_path, sdf)

    sdf_vis = np.clip((sdf / 32.0) * 127.0 + 128.0, 0, 255).astype(np.uint8)
    sdf_vis_path = os.path.join(output_dir, f"{stem}_sdf.png")
    cv2.imwrite(sdf_vis_path, sdf_vis)
    return mask_path, sdf_path, sdf_vis_path


def save_side_anchor_debug_mask(
    side_mask,
    side_image_bgr,
    vertices,
    side_result,
    anchor_pcts,
    output_dir,
    out_name="side_anchor_debug_mask.png",
):
    if side_mask is None:
        return None

    focal_length = side_result.get("focal_length")
    cam_t = side_result.get("pred_cam_t")
    if focal_length is None or cam_t is None:
        return None

    h, w = side_mask.shape[:2]
    mask_u8 = side_mask.astype(np.uint8) * 255
    overlay = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    if side_image_bgr is not None:
        image = side_image_bgr
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        tinted = image.copy()
        tinted[side_mask] = (
            0.45 * tinted[side_mask].astype(np.float32) +
            0.55 * np.array([40.0, 190.0, 80.0], dtype=np.float32)
        ).astype(np.uint8)
        overlay = tinted

    v = vertices.astype(np.float64)
    proj, depth = project_vertices_to_image(v, cam_t, float(focal_length), (h, w))
    in_image = (
        (depth > 1e-5) &
        (proj[:, 0] >= 0) & (proj[:, 0] < w) &
        (proj[:, 1] >= 0) & (proj[:, 1] < h)
    )

    up_dir = estimate_up_direction_from_mesh(v)
    height_coord = v @ up_dir
    height_span = float(height_coord.max() - height_coord.min())
    if height_span <= 1e-8:
        return None
    pct = (height_coord - float(height_coord.min())) / height_span

    def pct_to_row(target_pct):
        window = 0.006
        row = np.nan
        for _ in range(5):
            near = in_image & (np.abs(pct - float(target_pct)) <= window)
            if near.sum() >= 6:
                row = float(np.median(proj[near, 1]))
                break
            window *= 1.8
        return row

    def draw_anchor(name, center, half_width, color, label_y_offset):
        center_row = pct_to_row(center)
        low_row = pct_to_row(max(0.0, center - half_width))
        high_row = pct_to_row(min(1.0, center + half_width))

        band_rows = [r for r in (low_row, high_row) if np.isfinite(r)]
        if len(band_rows) == 2:
            y0 = int(np.clip(round(min(band_rows)), 0, h - 1))
            y1 = int(np.clip(round(max(band_rows)), 0, h - 1))
            band = overlay.copy()
            cv2.rectangle(band, (0, y0), (w - 1, y1), color, thickness=-1)
            overlay[:] = cv2.addWeighted(band, 0.18, overlay, 0.82, 0.0)
            cv2.line(overlay, (0, y0), (w - 1, y0), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (0, y1), (w - 1, y1), color, 1, cv2.LINE_AA)

        if np.isfinite(center_row):
            y = int(np.clip(round(center_row), 0, h - 1))
            cv2.line(overlay, (0, y), (w - 1, y), color, 3, cv2.LINE_AA)
            label = f"{name} {center * 100.0:.2f}%"
            text_y = int(np.clip(y + label_y_offset, 16, h - 8))
            cv2.putText(
                overlay,
                label,
                (8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    chest_center = required_anchor_pct(anchor_pcts, "chest")
    butt_center = required_anchor_pct(anchor_pcts, "butt")
    draw_anchor("bust", chest_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, (0, 80, 255), -8)
    draw_anchor("butt", butt_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, (255, 80, 0), 20)

    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, overlay)
    return out_path


def save_side_projection_alignment_debug(
    side_mask,
    side_image_bgr,
    vertices,
    side_result,
    output_dir,
    out_name="side_projection_alignment_debug.png",
):
    if side_mask is None or side_image_bgr is None:
        return None

    focal_length = side_result.get("focal_length") if isinstance(side_result, dict) else None
    cam_t = side_result.get("pred_cam_t") if isinstance(side_result, dict) else None
    if focal_length is None or cam_t is None:
        return None

    h, w = side_mask.shape[:2]
    image = side_image_bgr
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    overlay = image.copy()
    mask_u8 = side_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

    bbox = side_result.get("bbox") if isinstance(side_result, dict) else None
    if bbox is not None:
        try:
            x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]
            cv2.rectangle(
                overlay,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    v = vertices.astype(np.float64)
    proj, depth = project_vertices_to_image(v, cam_t, float(focal_length), (h, w))
    in_image = (
        (depth > 1e-5) &
        (proj[:, 0] >= 0) & (proj[:, 0] < w) &
        (proj[:, 1] >= 0) & (proj[:, 1] < h)
    )

    points = np.rint(proj[in_image]).astype(np.int32)
    # Draw a sparse projected vertex cloud in cyan. Dense enough to reveal offset,
    # sparse enough to keep the mask contour readable.
    if points.shape[0] > 0:
        step = max(1, points.shape[0] // 3500)
        sampled = points[::step]
        overlay[sampled[:, 1], sampled[:, 0]] = (255, 220, 0)

    rows = np.clip(np.rint(proj[:, 1]).astype(np.int32), 0, h - 1)
    torso_core, _ = compute_torso_core_mask(v, side_result)
    row_left = np.full(h, np.nan, dtype=np.float64)
    row_right = np.full(h, np.nan, dtype=np.float64)
    for y in range(h):
        near = in_image & torso_core & (np.abs(rows - y) <= 3)
        if near.any():
            row_left[y] = float(np.quantile(proj[near, 0], SIDE_SDF_ROW_EDGE_LOW_QUANTILE))
            row_right[y] = float(np.quantile(proj[near, 0], SIDE_SDF_ROW_EDGE_HIGH_QUANTILE))

    for y in range(0, h, 2):
        if np.isfinite(row_left[y]):
            cv2.circle(overlay, (int(round(row_left[y])), y), 1, (255, 255, 0), -1, cv2.LINE_AA)
        if np.isfinite(row_right[y]):
            cv2.circle(overlay, (int(round(row_right[y])), y), 1, (0, 255, 255), -1, cv2.LINE_AA)

    kps = side_result.get("pred_keypoints_2d") if isinstance(side_result, dict) else None
    if kps is not None:
        try:
            kps = _to_numpy(kps).astype(np.float64)
            for x, y in kps[:, :2]:
                if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
                    cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 165, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    legend = [
        ("red: mask", (0, 0, 255)),
        ("magenta: bbox", (255, 0, 255)),
        ("cyan/yellow: mesh row edges", (255, 255, 0)),
        ("orange: 2D keypoints", (0, 165, 255)),
    ]
    for i, (text, color) in enumerate(legend):
        cv2.putText(overlay, text, (12, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, overlay)
    return out_path


def project_vertices_to_image(vertices, cam_t, focal_length, image_shape):
    v = vertices.astype(np.float64)
    cam_t = np.asarray(cam_t, dtype=np.float64).reshape(3)
    pts = v + cam_t[None, :]
    z = pts[:, 2].copy()
    z_safe = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6 + (z == 0) * 1e-6, z)
    h, w = image_shape[:2]
    u = float(focal_length) * (pts[:, 0] / z_safe) + (w / 2.0)
    vv = float(focal_length) * (pts[:, 1] / z_safe) + (h / 2.0)
    return np.column_stack([u, vv]), z


def row_bounds_from_mask(mask, row_radius=4):
    h, _ = mask.shape[:2]
    left = np.full(h, np.nan, dtype=np.float64)
    right = np.full(h, np.nan, dtype=np.float64)
    for y in range(h):
        y0 = max(0, y - row_radius)
        y1 = min(h, y + row_radius + 1)
        xs = np.nonzero(mask[y0:y1].any(axis=0))[0]
        if xs.size > 0:
            left[y] = float(xs.min())
            right[y] = float(xs.max())
    return left, right
def required_anchor_pct(anchor_pcts, key):
    try:
        value = float((anchor_pcts or {})[key])
    except Exception as e:
        raise ValueError(f"Missing required CLAD anchor: {key}") from e
    if not (0.0 < value < 1.0):
        raise ValueError(f"Invalid required CLAD anchor {key}: {value}")
    return value


def infer_side_anterior_sign(side_result, image_shape):
    """Return -1 if the person faces image-left, +1 if image-right."""
    keypoints = side_result.get("pred_keypoints_2d") if isinstance(side_result, dict) else None
    if keypoints is not None:
        try:
            kps = _to_numpy(keypoints).astype(np.float64)
            if kps.ndim == 2 and kps.shape[0] >= 70 and kps.shape[1] >= 2:
                nose_x = float(kps[0, 0])
                torso_ids = [5, 6, 9, 10, 69]
                torso_xs = [float(kps[i, 0]) for i in torso_ids if np.isfinite(kps[i, 0])]
                if torso_xs and np.isfinite(nose_x):
                    torso_x = float(np.median(torso_xs))
                    if abs(nose_x - torso_x) > max(3.0, image_shape[1] * 0.01):
                        return -1 if nose_x < torso_x else 1, "nose_vs_torso_keypoints"
        except Exception:
            pass
    return -1, "default_facing_left"


def smooth_rows(values, valid, sigma_px):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.size == 0:
        return values
    if sigma_px <= 0:
        return np.where(valid, values, 0.0)
    k = int(max(3, round(float(sigma_px) * 6.0)))
    if k % 2 == 0:
        k += 1
    weights = valid.astype(np.float64)
    weighted = np.where(valid, values, 0.0).reshape(-1, 1)
    weights_2d = weights.reshape(-1, 1)
    smooth_num = cv2.GaussianBlur(weighted, (1, k), float(sigma_px), borderType=cv2.BORDER_REPLICATE).reshape(-1)
    smooth_den = cv2.GaussianBlur(weights_2d, (1, k), float(sigma_px), borderType=cv2.BORDER_REPLICATE).reshape(-1)
    out = np.zeros_like(values, dtype=np.float64)
    good = smooth_den > 1e-6
    out[good] = smooth_num[good] / smooth_den[good]
    return out


def compute_torso_core_mask(vertices, side_result):
    """Approximate torso/core vertices using 3D shoulder/hip left-right axis."""
    default = np.ones(vertices.shape[0], dtype=bool)
    keypoints = side_result.get("pred_keypoints_3d") if isinstance(side_result, dict) else None
    if keypoints is None:
        return default, {
            "enabled": np.array(False),
            "reason": np.array("missing_keypoints_3d", dtype=object),
            "threshold_m": np.array(0.0, dtype=np.float64),
        }
    try:
        kps = _to_numpy(keypoints).astype(np.float64)
        if kps.ndim != 2 or kps.shape[0] < 11 or kps.shape[1] < 3:
            raise ValueError("bad keypoint shape")
        left_shoulder, right_shoulder = kps[5], kps[6]
        left_hip, right_hip = kps[9], kps[10]
        pairs = []
        for a, b in ((left_shoulder, right_shoulder), (left_hip, right_hip)):
            axis = a - b
            norm = float(np.linalg.norm(axis))
            if norm > 1e-5:
                pairs.append((axis / norm, norm))
        if not pairs:
            raise ValueError("bad shoulder/hip width")
        lateral_axis = normalize_vector(np.mean([p[0] for p in pairs], axis=0))
        width = float(np.median([p[1] for p in pairs]))
        center = np.mean([left_shoulder, right_shoulder, left_hip, right_hip], axis=0)
        lateral = (vertices.astype(np.float64) - center[None, :]) @ lateral_axis
        threshold = max(0.10, min(0.32, width * 0.85))
        mask = np.abs(lateral) <= threshold
        if mask.sum() < vertices.shape[0] * 0.20:
            threshold = max(threshold, np.quantile(np.abs(lateral), 0.55))
            mask = np.abs(lateral) <= threshold
        return mask, {
            "enabled": np.array(True),
            "reason": np.array("ok", dtype=object),
            "threshold_m": np.array(float(threshold), dtype=np.float64),
            "vertex_count": np.array(int(mask.sum()), dtype=np.int64),
        }
    except Exception as e:
        return default, {
            "enabled": np.array(False),
            "reason": np.array(f"failed: {e}", dtype=object),
            "threshold_m": np.array(0.0, dtype=np.float64),
        }


def save_side_sdf_row_debug(
    side_mask,
    side_image_bgr,
    vertices,
    side_result,
    anchor_pcts,
    output_dir,
    out_name="side_sdf_profile_row_debug.png",
    row_radius=4,
):
    """Save a row-wise side-profile residual diagnostic for the edited mesh."""
    if side_mask is None or side_image_bgr is None:
        return None

    focal_length = side_result.get("focal_length") if isinstance(side_result, dict) else None
    cam_t = side_result.get("pred_cam_t") if isinstance(side_result, dict) else None
    if focal_length is None or cam_t is None:
        return None

    h, w = side_mask.shape[:2]
    image = side_image_bgr
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    overlay = image.copy()
    mask_tint = overlay.copy()
    mask_tint[side_mask] = (
        0.45 * mask_tint[side_mask].astype(np.float32) +
        0.55 * np.array([40.0, 190.0, 80.0], dtype=np.float32)
    ).astype(np.uint8)
    overlay = cv2.addWeighted(mask_tint, 0.55, overlay, 0.45, 0.0)

    mask_u8 = side_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

    v = vertices.astype(np.float64)
    proj, depth = project_vertices_to_image(v, cam_t, float(focal_length), (h, w))
    in_image = (
        (depth > 1e-5) &
        (proj[:, 0] >= 0) & (proj[:, 0] < w) &
        (proj[:, 1] >= 0) & (proj[:, 1] < h)
    )
    if not np.any(in_image):
        return None

    up_dir = estimate_up_direction_from_mesh(v)
    height_coord = v @ up_dir
    height_min = float(height_coord.min())
    height_span = float(height_coord.max() - height_min)
    if height_span <= 1e-8:
        return None
    pct = (height_coord - height_min) / height_span

    try:
        chest_center = required_anchor_pct(anchor_pcts, "chest")
        butt_center = required_anchor_pct(anchor_pcts, "butt")
    except Exception:
        return None

    torso_core, _ = compute_torso_core_mask(v, side_result)
    debug_band_half_width = SIDE_SDF_TARGET_CORE_HALF_WIDTH
    chest_height_weight = (np.abs(pct - chest_center) <= debug_band_half_width).astype(np.float64)
    butt_height_weight = (np.abs(pct - butt_center) <= debug_band_half_width).astype(np.float64)
    selected = in_image & torso_core & ((chest_height_weight > 1e-4) | (butt_height_weight > 1e-4))
    if not selected.any():
        return None

    rows = np.clip(np.rint(proj[:, 1]).astype(np.int32), 0, h - 1)
    row_left = np.full(h, np.nan, dtype=np.float64)
    row_right = np.full(h, np.nan, dtype=np.float64)
    row_min = np.full(h, np.nan, dtype=np.float64)
    row_max = np.full(h, np.nan, dtype=np.float64)
    row_center = np.full(h, np.nan, dtype=np.float64)
    row_half = np.full(h, np.nan, dtype=np.float64)
    for y in range(h):
        near = selected & (np.abs(rows - y) <= int(row_radius))
        if near.any():
            xs = proj[near, 0]
            row_left[y] = float(np.quantile(xs, SIDE_SDF_ROW_EDGE_LOW_QUANTILE))
            row_right[y] = float(np.quantile(xs, SIDE_SDF_ROW_EDGE_HIGH_QUANTILE))
            row_min[y] = float(np.min(xs))
            row_max[y] = float(np.max(xs))
            row_center[y] = 0.5 * (row_left[y] + row_right[y])
            row_half[y] = max(1.0, 0.5 * (row_right[y] - row_left[y]))

    mask_left, mask_right = row_bounds_from_mask(side_mask, row_radius=row_radius)
    sdf = mask_to_sdf(side_mask)
    anterior_sign, _ = infer_side_anterior_sign(side_result, (h, w))
    posterior_sign = -int(anterior_sign)

    row_center_at_vertex = row_center[rows]
    row_half_at_vertex = row_half[rows]
    valid_profile_row = np.isfinite(row_center_at_vertex) & np.isfinite(row_half_at_vertex) & (row_half_at_vertex > 1e-6)
    signed_side_all = np.zeros(v.shape[0], dtype=np.float64)
    signed_side_all[valid_profile_row] = (
        proj[valid_profile_row, 0] - row_center_at_vertex[valid_profile_row]
    ) / row_half_at_vertex[valid_profile_row]
    chest_side_mask = in_image & torso_core & valid_profile_row & (chest_height_weight > 1e-4) & ((anterior_sign * signed_side_all) > 0.02)
    butt_side_mask = in_image & torso_core & valid_profile_row & (butt_height_weight > 1e-4) & ((posterior_sign * signed_side_all) > 0.02)

    points = np.rint(proj[in_image]).astype(np.int32)
    if points.shape[0] > 0:
        step = max(1, points.shape[0] // 3000)
        sampled = points[::step]
        overlay[sampled[:, 1], sampled[:, 0]] = (255, 230, 0)

    chart_w = 260
    canvas = np.full((h, w + chart_w, 3), 245, dtype=np.uint8)
    canvas[:, :w] = overlay
    chart_x0 = w + 12
    chart_center = w + chart_w // 2
    cv2.line(canvas, (chart_center, 0), (chart_center, h - 1), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "row residual px", (chart_x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "right=needs outward", (chart_x0, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

    def band_rows(center):
        near = in_image & (np.abs(pct - center) <= debug_band_half_width)
        if near.sum() < 4:
            return None
        ys = proj[near, 1]
        return int(np.clip(np.nanmin(ys), 0, h - 1)), int(np.clip(np.nanmax(ys), 0, h - 1))

    for center, color in ((chest_center, (0, 120, 255)), (butt_center, (255, 120, 0))):
        br = band_rows(center)
        if br is None:
            continue
        y0, y1 = br
        band = canvas.copy()
        cv2.rectangle(band, (0, y0), (w + chart_w - 1, y1), color, -1)
        canvas[:] = cv2.addWeighted(band, 0.08, canvas, 0.92, 0.0)
        cv2.line(canvas, (0, y0), (w + chart_w - 1, y0), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, y1), (w + chart_w - 1, y1), color, 1, cv2.LINE_AA)

    def draw_region(region_mask, side_sign, base_color):
        for y in range(0, h, 2):
            near = region_mask & (np.abs(rows - y) <= int(row_radius))
            if not near.any():
                continue
            if not (np.isfinite(mask_left[y]) and np.isfinite(mask_right[y]) and np.isfinite(row_left[y]) and np.isfinite(row_right[y])):
                continue
            if side_sign < 0:
                mesh_edge = row_left[y]
                raw_edge = row_min[y]
                mask_edge = mask_left[y]
            else:
                mesh_edge = row_right[y]
                raw_edge = row_max[y]
                mask_edge = mask_right[y]
            if not (np.isfinite(mesh_edge) and np.isfinite(mask_edge)):
                continue

            signed_gap = (mask_edge - mesh_edge) * float(side_sign)
            sample_xy = np.array([[mesh_edge, float(y)]], dtype=np.float64)
            sdf_at_edge = float(sample_image_bilinear(sdf, sample_xy)[0])
            abs_gap = abs(signed_gap)
            if abs_gap <= 2.0:
                color = (60, 210, 80)
            elif signed_gap > 0.0:
                color = (255, 0, 255)
            else:
                color = (255, 120, 30)
            thickness = int(np.clip(round(abs_gap / 8.0) + 1, 1, 5))

            x0 = int(np.clip(round(mesh_edge), 0, w - 1))
            x1 = int(np.clip(round(mask_edge), 0, w - 1))
            cv2.line(canvas, (x0, y), (x1, y), color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, (x0, y), 1, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
            if np.isfinite(raw_edge):
                xr = int(np.clip(round(raw_edge), 0, w - 1))
                cv2.circle(canvas, (xr, y), 1, (180, 180, 180), -1, cv2.LINE_AA)

            bar = int(np.clip(round(signed_gap * 2.0), -110, 110))
            chart_color = color if abs(sdf_at_edge) > 2.0 else (60, 210, 80)
            cv2.line(canvas, (chart_center, y), (chart_center + bar, y), chart_color, thickness, cv2.LINE_AA)

    draw_region(chest_side_mask, int(anterior_sign), (0, 120, 255))
    draw_region(butt_side_mask, int(posterior_sign), (255, 120, 0))

    legend_y = max(66, min(h - 92, 66))
    legend = [
        ("red dot/contour: mask edge", (0, 0, 255)),
        ("cyan dot: solver mesh row edge", (255, 255, 0)),
        ("gray dot: raw mesh min/max edge", (180, 180, 180)),
        ("magenta bar: underfit, mask farther out", (255, 0, 255)),
        ("orange bar: overshoot, mesh outside mask", (255, 120, 30)),
        ("green bar: within 2 px", (60, 210, 80)),
    ]
    for i, (text, color) in enumerate(legend):
        y = legend_y + i * 19
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, canvas)
    return out_path


def deform_side_mesh_to_mask_profile(
    vertices,
    side_result,
    side_mask,
    image_shape,
    anchor_pcts=None,
    strength=0.65,
    max_push_cm=15.0,
    row_radius=6,
    faces=None,
):
    """Smoothly move chest/front and butt/back side-profile bands toward the mask.

    Chest and butt share one row-edge SDF/profile algorithm so there is a
    single source of truth for side-mask correction.
    """
    profile_method = "row_edge_sdf"
    meta = {
        "enabled": np.array(False),
        "reason": np.array("disabled_or_no_mask", dtype=object),
        "profile_method": np.array(profile_method, dtype=object),
        "strength": np.array(float(strength), dtype=np.float64),
        "max_push_cm": np.array(float(max_push_cm), dtype=np.float64),
        "mean_abs_push_cm": np.array(0.0, dtype=np.float64),
        "max_abs_push_cm": np.array(0.0, dtype=np.float64),
        "mean_selected_sdf_px": np.array(0.0, dtype=np.float64),
        "moved_vertex_count": np.array(0, dtype=np.int64),
    }
    if strength <= 0.0:
        meta["reason"] = np.array("strength_zero", dtype=object)
        return vertices.astype(np.float32), meta
    if side_mask is None:
        return vertices.astype(np.float32), meta

    focal_length = side_result.get("focal_length")
    cam_t = side_result.get("pred_cam_t")
    if focal_length is None or cam_t is None:
        meta["reason"] = np.array("missing_camera", dtype=object)
        return vertices.astype(np.float32), meta

    v = vertices.astype(np.float64).copy()
    proj, depth = project_vertices_to_image(v, cam_t, float(focal_length), image_shape)
    sdf = mask_to_sdf(side_mask)
    sdf_values = sample_image_bilinear(sdf, proj)
    h, w = image_shape[:2]
    in_front = depth > 1e-5
    in_image = (
        (proj[:, 0] >= 0) & (proj[:, 0] < w) &
        (proj[:, 1] >= 0) & (proj[:, 1] < h) &
        in_front
    )

    up_dir = estimate_up_direction_from_mesh(v)
    height_coord = v @ up_dir
    height_min = float(height_coord.min())
    height_span = float(height_coord.max() - height_min)
    if height_span <= 1e-8:
        meta["reason"] = np.array("bad_height", dtype=object)
        return v.astype(np.float32), meta
    pct = (height_coord - height_min) / height_span

    anchor_pcts = anchor_pcts or {}
    chest_center = required_anchor_pct(anchor_pcts, "chest")
    butt_center = required_anchor_pct(anchor_pcts, "butt")
    chest_outer_smooth_width = SIDE_SDF_OUTER_SMOOTH_WIDTH * 2.0
    butt_outer_smooth_width = SIDE_SDF_OUTER_SMOOTH_WIDTH * 2.0
    chest_outer_weight = _outside_band_smooth_weight(
        pct, chest_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, 0.0, chest_outer_smooth_width
    )
    butt_outer_weight = _outside_band_smooth_weight(
        pct, butt_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, 0.0, butt_outer_smooth_width
    )
    chest_target_weight = _flat_band_with_feather(
        pct, chest_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, SIDE_SDF_TARGET_FEATHER_WIDTH
    )
    butt_target_weight = _flat_band_with_feather(
        pct, butt_center, SIDE_SDF_TARGET_CORE_HALF_WIDTH, SIDE_SDF_TARGET_FEATHER_WIDTH
    )
    torso_core, torso_core_meta = compute_torso_core_mask(v, side_result)
    selected = in_image & torso_core & ((chest_target_weight > 1e-4) | (butt_target_weight > 1e-4))
    if not selected.any():
        meta["reason"] = np.array("no_selected_vertices", dtype=object)
        return v.astype(np.float32), meta

    anterior_sign, anterior_source = infer_side_anterior_sign(side_result, image_shape)
    posterior_sign = -int(anterior_sign)
    mask_left, mask_right = row_bounds_from_mask(side_mask, row_radius=row_radius)

    max_push_m = max(0.0, float(max_push_cm)) / 100.0
    profile_fit_half_width = SIDE_SDF_TARGET_CORE_HALF_WIDTH + SIDE_SDF_TARGET_FEATHER_WIDTH
    chest_height_weight = chest_target_weight
    butt_height_weight = butt_target_weight
    chest_support_weight = _flat_band_with_feather(
        pct,
        chest_center,
        SIDE_SDF_TARGET_CORE_HALF_WIDTH,
        SIDE_SDF_SOLVE_OUTSIDE_SMOOTH_WIDTH,
    )
    butt_support_weight = _flat_band_with_feather(
        pct,
        butt_center,
        SIDE_SDF_TARGET_CORE_HALF_WIDTH,
        SIDE_SDF_SOLVE_OUTSIDE_SMOOTH_WIDTH,
    )

    def build_neighbor_lists(mesh_faces, vertex_count):
        if mesh_faces is None:
            return None
        try:
            f = np.asarray(mesh_faces, dtype=np.int64).reshape(-1, 3)
        except Exception:
            return None
        if f.size == 0:
            return None
        valid_faces = np.all((f >= 0) & (f < vertex_count), axis=1)
        f = f[valid_faces]
        if f.size == 0:
            return None
        neighbors = [set() for _ in range(vertex_count)]
        for a, b, c in f:
            neighbors[a].add(b); neighbors[a].add(c)
            neighbors[b].add(a); neighbors[b].add(c)
            neighbors[c].add(a); neighbors[c].add(b)
        return [np.fromiter(n, dtype=np.int64) if n else np.empty(0, dtype=np.int64) for n in neighbors]

    mesh_neighbors = build_neighbor_lists(faces, v.shape[0])
    solve_reason = "ok"

    def solve_weighted_displacement(target_dx, height_weight, support_weight, side_strength):
        nonlocal solve_reason
        target_dx = np.asarray(target_dx, dtype=np.float64)
        target_mask = np.abs(target_dx) > 1e-8
        if target_mask.sum() < 8 or mesh_neighbors is None:
            solve_reason = "fallback_no_targets_or_faces"
            return np.clip(target_dx, -max_push_m, max_push_m), target_mask.copy()

        edge_data_weight = (
            float(SIDE_SDF_SOLVE_EDGE_DATA_FLOOR) +
            (1.0 - float(SIDE_SDF_SOLVE_EDGE_DATA_FLOOR)) *
            np.clip(np.asarray(side_strength, dtype=np.float64), 0.0, 1.0) ** float(SIDE_SDF_SOLVE_EDGE_DATA_POWER)
        )
        data_weight = (
            float(SIDE_SDF_SOLVE_DATA_WEIGHT) *
            np.asarray(height_weight, dtype=np.float64) *
            edge_data_weight *
            target_mask.astype(np.float64)
        )
        support_weight = np.asarray(support_weight, dtype=np.float64)
        pin_weight = float(SIDE_SDF_SOLVE_PIN_WEIGHT) * (
            (1.0 - support_weight) ** 2.0 +
            float(SIDE_SDF_SOLVE_SIDE_PIN_SCALE) * support_weight *
            (1.0 - np.asarray(side_strength, dtype=np.float64)) ** 2.0
        )
        if float(data_weight.sum()) <= 1e-8:
            solve_reason = "fallback_zero_data_weight"
            return np.clip(target_dx, -max_push_m, max_push_m), target_mask.copy()
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except Exception as e:
            solve_reason = f"fallback_missing_scipy:{e}"
            return np.clip(target_dx, -max_push_m, max_push_m), target_mask.copy()

        n = v.shape[0]
        rows_l = []
        cols_l = []
        data_l = []
        degree = np.zeros(n, dtype=np.float64)
        for i, nbrs in enumerate(mesh_neighbors):
            for j in nbrs:
                if i >= int(j):
                    continue
                j = int(j)
                degree[i] += 1.0
                degree[j] += 1.0
                rows_l.extend([i, j])
                cols_l.extend([j, i])
                data_l.extend([-1.0, -1.0])
        rows_l.extend(range(n))
        cols_l.extend(range(n))
        data_l.extend(degree.tolist())
        laplacian = sparse.csr_matrix((data_l, (rows_l, cols_l)), shape=(n, n))
        diagonal = data_weight + pin_weight + float(SIDE_SDF_SOLVE_MIN_DIAGONAL)
        system = sparse.diags(diagonal, format="csr") + float(SIDE_SDF_SOLVE_SMOOTH_LAMBDA) * laplacian
        rhs = data_weight * target_dx
        try:
            solved = np.asarray(spsolve(system, rhs), dtype=np.float64)
        except Exception as e:
            solve_reason = f"fallback_solve_failed:{e}"
            return np.clip(target_dx, -max_push_m, max_push_m), target_mask.copy()
        if not np.isfinite(solved).all():
            solve_reason = "fallback_nonfinite_solution"
            return np.clip(target_dx, -max_push_m, max_push_m), target_mask.copy()
        changed = np.abs(solved - target_dx) > 1e-7
        return np.clip(solved, -max_push_m, max_push_m), changed

    def compute_profile_state(vertices_cur):
        proj_cur, depth_cur = project_vertices_to_image(vertices_cur, cam_t, float(focal_length), image_shape)
        in_image_cur = (
            (depth_cur > 1e-5) &
            (proj_cur[:, 0] >= 0) & (proj_cur[:, 0] < w) &
            (proj_cur[:, 1] >= 0) & (proj_cur[:, 1] < h)
        )
        rows_cur = np.clip(np.rint(proj_cur[:, 1]).astype(np.int32), 0, h - 1)
        selected_cur = in_image_cur & torso_core & ((chest_target_weight > 1e-4) | (butt_target_weight > 1e-4))

        row_center_cur = np.full(h, np.nan, dtype=np.float64)
        row_half_cur = np.full(h, np.nan, dtype=np.float64)
        row_left_cur = np.full(h, np.nan, dtype=np.float64)
        row_right_cur = np.full(h, np.nan, dtype=np.float64)
        for y in range(h):
            near = selected_cur & (np.abs(rows_cur - y) <= row_radius)
            if near.any():
                xs = proj_cur[near, 0]
                row_left_cur[y] = float(np.quantile(xs, SIDE_SDF_ROW_EDGE_LOW_QUANTILE))
                row_right_cur[y] = float(np.quantile(xs, SIDE_SDF_ROW_EDGE_HIGH_QUANTILE))
                row_center_cur[y] = 0.5 * (row_left_cur[y] + row_right_cur[y])
                row_half_cur[y] = max(1.0, 0.5 * (row_right_cur[y] - row_left_cur[y]))

        row_center_at_vertex = row_center_cur[rows_cur]
        row_half_at_vertex = row_half_cur[rows_cur]
        valid_profile_row_cur = (
            np.isfinite(row_center_at_vertex) &
            np.isfinite(row_half_at_vertex) &
            (row_half_at_vertex > 1e-6)
        )
        signed_side_all_cur = np.zeros(v.shape[0], dtype=np.float64)
        signed_side_all_cur[valid_profile_row_cur] = (
            proj_cur[valid_profile_row_cur, 0] - row_center_at_vertex[valid_profile_row_cur]
        ) / row_half_at_vertex[valid_profile_row_cur]
        chest_side_mask_cur = in_image_cur & torso_core & valid_profile_row_cur & ((anterior_sign * signed_side_all_cur) > 0.02)
        butt_side_mask_cur = in_image_cur & torso_core & valid_profile_row_cur & ((posterior_sign * signed_side_all_cur) > 0.02)
        return {
            "proj": proj_cur,
            "depth": depth_cur,
            "in_image": in_image_cur,
            "rows": rows_cur,
            "row_left": row_left_cur,
            "row_right": row_right_cur,
            "valid_profile_row": valid_profile_row_cur,
            "signed_side_all": signed_side_all_cur,
            "chest_side_mask": chest_side_mask_cur,
            "butt_side_mask": butt_side_mask_cur,
            "selected": selected_cur,
        }

    def target_shift_by_row(profile_state, side_sign):
        shift = np.zeros(h, dtype=np.float64)
        valid = np.zeros(h, dtype=bool)
        row_left_cur = profile_state["row_left"]
        row_right_cur = profile_state["row_right"]
        for y in range(h):
            if not (
                np.isfinite(mask_left[y]) and np.isfinite(mask_right[y]) and
                np.isfinite(row_left_cur[y]) and np.isfinite(row_right_cur[y])
            ):
                continue
            if side_sign < 0:
                shift[y] = mask_left[y] - row_left_cur[y]
            else:
                shift[y] = mask_right[y] - row_right_cur[y]
            valid[y] = abs(shift[y]) > 1e-6

        raw_shift = shift.copy()
        sigma_px = max(float(row_radius) * 2.5, 10.0)
        smoothed = smooth_rows(raw_shift, valid, sigma_px)
        sign_flip = (
            valid &
            (np.abs(raw_shift) > 1e-6) &
            (np.abs(smoothed) > 1e-6) &
            (np.sign(raw_shift) != np.sign(smoothed))
        )
        smoothed[sign_flip] = raw_shift[sign_flip]
        if SIDE_SDF_ROW_UNDERFIT_PEAK_PRESERVE:
            raw_underfit = raw_shift * float(side_sign)
            smoothed_underfit = smoothed * float(side_sign)
            underfit_peak = (
                valid &
                (raw_underfit > 1e-6) &
                (raw_underfit > smoothed_underfit)
            )
            smoothed[underfit_peak] = raw_shift[underfit_peak]
        return smoothed, valid, raw_shift, sign_flip

    def build_target_dx(profile_state, shift_row, valid_row, side_strength, height_weight, side_mask):
        target_dx = np.zeros(v.shape[0], dtype=np.float64)
        target_px = np.zeros(v.shape[0], dtype=np.float64)
        max_push_px_cur = np.maximum(
            1.0,
            max_push_m * float(focal_length) / np.maximum(profile_state["depth"], 1e-5),
        )
        rows_cur = profile_state["rows"]
        selected_cur = profile_state["selected"]
        for idx in np.nonzero(selected_cur)[0]:
            y = rows_cur[idx]
            if height_weight[idx] <= 1e-4 or side_strength[idx] <= 1e-4 or not valid_row[y] or not side_mask[idx]:
                continue
            raw_px = shift_row[y] * float(strength)
            raw_px = float(np.clip(raw_px, -max_push_px_cur[idx], max_push_px_cur[idx]))
            target_px[idx] = raw_px
            target_dx[idx] = raw_px * profile_state["depth"][idx] / float(focal_length)
        return target_dx, target_px

    def signed_row_gap_stats(shift_row, valid_row, side_sign):
        valid_shift = valid_row & np.isfinite(shift_row)
        if not np.any(valid_shift):
            return {
                "valid_count": 0,
                "mean_abs_px": 0.0,
                "p95_abs_px": 0.0,
                "max_abs_px": 0.0,
                "mean_underfit_px": 0.0,
                "p95_underfit_px": 0.0,
                "max_underfit_px": 0.0,
                "mean_overfit_px": 0.0,
                "max_overfit_px": 0.0,
            }
        signed_gap = shift_row[valid_shift] * float(side_sign)
        underfit = signed_gap[signed_gap > 1e-6]
        overfit = -signed_gap[signed_gap < -1e-6]
        abs_gap = np.abs(signed_gap)
        return {
            "valid_count": int(valid_shift.sum()),
            "mean_abs_px": float(np.mean(abs_gap)),
            "p95_abs_px": float(np.percentile(abs_gap, 95.0)),
            "max_abs_px": float(np.max(abs_gap)),
            "mean_underfit_px": float(np.mean(underfit)) if underfit.size else 0.0,
            "p95_underfit_px": float(np.percentile(underfit, 95.0)) if underfit.size else 0.0,
            "max_underfit_px": float(np.max(underfit)) if underfit.size else 0.0,
            "mean_overfit_px": float(np.mean(overfit)) if overfit.size else 0.0,
            "max_overfit_px": float(np.max(overfit)) if overfit.size else 0.0,
        }

    def displacement_stats(dx_values):
        active = np.abs(dx_values) > 1e-8
        if not np.any(active):
            return {"count": 0, "mean_abs_cm": 0.0, "p95_abs_cm": 0.0, "max_abs_cm": 0.0}
        abs_cm = np.abs(dx_values[active]) * 100.0
        return {
            "count": int(active.sum()),
            "mean_abs_cm": float(np.mean(abs_cm)),
            "p95_abs_cm": float(np.percentile(abs_cm, 95.0)),
            "max_abs_cm": float(np.max(abs_cm)),
        }

    def pixel_target_stats(px_values):
        active = np.abs(px_values) > 1e-8
        if not np.any(active):
            return {"count": 0, "mean_abs_px": 0.0, "p95_abs_px": 0.0, "max_abs_px": 0.0}
        abs_px = np.abs(px_values[active])
        return {
            "count": int(active.sum()),
            "mean_abs_px": float(np.mean(abs_px)),
            "p95_abs_px": float(np.percentile(abs_px, 95.0)),
            "max_abs_px": float(np.max(abs_px)),
        }

    dx_chest = np.zeros(v.shape[0], dtype=np.float64)
    dx_butt = np.zeros(v.shape[0], dtype=np.float64)
    chest_solve_changed = np.zeros(v.shape[0], dtype=bool)
    butt_solve_changed = np.zeros(v.shape[0], dtype=bool)
    target_dx_chest = np.zeros(v.shape[0], dtype=np.float64)
    target_dx_butt = np.zeros(v.shape[0], dtype=np.float64)
    target_px_chest = np.zeros(v.shape[0], dtype=np.float64)
    target_px_butt = np.zeros(v.shape[0], dtype=np.float64)
    solved_dx_chest_initial = np.zeros(v.shape[0], dtype=np.float64)
    solved_dx_butt_initial = np.zeros(v.shape[0], dtype=np.float64)
    chest_shift_row = np.zeros(h, dtype=np.float64)
    butt_shift_row = np.zeros(h, dtype=np.float64)
    chest_raw_shift_row = np.zeros(h, dtype=np.float64)
    butt_raw_shift_row = np.zeros(h, dtype=np.float64)
    chest_valid_row = np.zeros(h, dtype=bool)
    butt_valid_row = np.zeros(h, dtype=bool)
    chest_shift_sign_flip_row = np.zeros(h, dtype=bool)
    butt_shift_sign_flip_row = np.zeros(h, dtype=bool)
    residual_solve_pass_count = 0
    residual_solve_gains = []
    residual_chest_raw_mean_underfit_px_by_pass = []
    residual_chest_raw_max_underfit_px_by_pass = []
    residual_chest_smoothed_mean_underfit_px_by_pass = []
    residual_chest_smoothed_max_underfit_px_by_pass = []
    residual_butt_raw_mean_underfit_px_by_pass = []
    residual_butt_raw_max_underfit_px_by_pass = []
    residual_butt_smoothed_mean_underfit_px_by_pass = []
    residual_butt_smoothed_max_underfit_px_by_pass = []
    residual_chest_applied_mean_abs_cm_by_pass = []
    residual_chest_applied_max_abs_cm_by_pass = []
    residual_butt_applied_mean_abs_cm_by_pass = []
    residual_butt_applied_max_abs_cm_by_pass = []

    for pass_idx in range(max(1, int(SIDE_SDF_RESIDUAL_SOLVE_PASSES))):
        profile_state = compute_profile_state(v)
        chest_side_mask_pass = profile_state["chest_side_mask"]
        butt_side_mask_pass = profile_state["butt_side_mask"]
        signed_side_all = profile_state["signed_side_all"]
        chest_side_strength_pass = np.clip((anterior_sign * signed_side_all - 0.02) / 0.55, 0.0, 1.0)
        butt_side_strength_pass = np.clip((posterior_sign * signed_side_all - 0.02) / 0.55, 0.0, 1.0)

        chest_shift_row_pass, chest_valid_row_pass, chest_raw_shift_row_pass, chest_shift_sign_flip_row_pass = target_shift_by_row(
            profile_state,
            anterior_sign,
        )
        butt_shift_row_pass, butt_valid_row_pass, butt_raw_shift_row_pass, butt_shift_sign_flip_row_pass = target_shift_by_row(
            profile_state,
            posterior_sign,
        )

        chest_raw_pass_stats = signed_row_gap_stats(chest_raw_shift_row_pass, chest_valid_row_pass, anterior_sign)
        chest_smoothed_pass_stats = signed_row_gap_stats(chest_shift_row_pass, chest_valid_row_pass, anterior_sign)
        butt_raw_pass_stats = signed_row_gap_stats(butt_raw_shift_row_pass, butt_valid_row_pass, posterior_sign)
        butt_smoothed_pass_stats = signed_row_gap_stats(butt_shift_row_pass, butt_valid_row_pass, posterior_sign)
        residual_chest_raw_mean_underfit_px_by_pass.append(chest_raw_pass_stats["mean_underfit_px"])
        residual_chest_raw_max_underfit_px_by_pass.append(chest_raw_pass_stats["max_underfit_px"])
        residual_chest_smoothed_mean_underfit_px_by_pass.append(chest_smoothed_pass_stats["mean_underfit_px"])
        residual_chest_smoothed_max_underfit_px_by_pass.append(chest_smoothed_pass_stats["max_underfit_px"])
        residual_butt_raw_mean_underfit_px_by_pass.append(butt_raw_pass_stats["mean_underfit_px"])
        residual_butt_raw_max_underfit_px_by_pass.append(butt_raw_pass_stats["max_underfit_px"])
        residual_butt_smoothed_mean_underfit_px_by_pass.append(butt_smoothed_pass_stats["mean_underfit_px"])
        residual_butt_smoothed_max_underfit_px_by_pass.append(butt_smoothed_pass_stats["max_underfit_px"])

        target_dx_chest_pass, target_px_chest_pass = build_target_dx(
            profile_state,
            chest_shift_row_pass,
            chest_valid_row_pass,
            chest_side_strength_pass,
            chest_height_weight,
            chest_side_mask_pass,
        )
        target_dx_butt_pass, target_px_butt_pass = build_target_dx(
            profile_state,
            butt_shift_row_pass,
            butt_valid_row_pass,
            butt_side_strength_pass,
            butt_height_weight,
            butt_side_mask_pass,
        )

        pass_gain = float(SIDE_SDF_RESIDUAL_SOLVE_GAIN)
        target_dx_chest_pass *= pass_gain
        target_dx_butt_pass *= pass_gain
        target_px_chest_pass *= pass_gain
        target_px_butt_pass *= pass_gain

        if pass_idx == 0:
            chest_shift_row = chest_shift_row_pass.copy()
            butt_shift_row = butt_shift_row_pass.copy()
            chest_raw_shift_row = chest_raw_shift_row_pass.copy()
            butt_raw_shift_row = butt_raw_shift_row_pass.copy()
            chest_valid_row = chest_valid_row_pass.copy()
            butt_valid_row = butt_valid_row_pass.copy()
            chest_shift_sign_flip_row = chest_shift_sign_flip_row_pass.copy()
            butt_shift_sign_flip_row = butt_shift_sign_flip_row_pass.copy()
            target_dx_chest = target_dx_chest_pass.copy()
            target_dx_butt = target_dx_butt_pass.copy()
            target_px_chest = target_px_chest_pass.copy()
            target_px_butt = target_px_butt_pass.copy()

        dx_chest_pass, chest_solve_changed_pass = solve_weighted_displacement(
            target_dx_chest_pass,
            chest_height_weight,
            chest_support_weight,
            chest_side_strength_pass,
        )
        dx_butt_pass, butt_solve_changed_pass = solve_weighted_displacement(
            target_dx_butt_pass,
            butt_height_weight,
            butt_support_weight,
            butt_side_strength_pass,
        )
        if pass_idx == 0:
            solved_dx_chest_initial = dx_chest_pass.copy()
            solved_dx_butt_initial = dx_butt_pass.copy()

        requested_step = dx_chest_pass + dx_butt_pass
        total_before = dx_chest + dx_butt
        total_after = np.clip(total_before + requested_step, -max_push_m, max_push_m)
        applied_step = total_after - total_before
        step_scale = np.ones(v.shape[0], dtype=np.float64)
        nonzero_step = np.abs(requested_step) > 1e-8
        step_scale[nonzero_step] = applied_step[nonzero_step] / requested_step[nonzero_step]
        dx_chest_step = dx_chest_pass * step_scale
        dx_butt_step = dx_butt_pass * step_scale

        v[:, 0] += applied_step
        dx_chest += dx_chest_step
        dx_butt += dx_butt_step
        chest_step_stats = displacement_stats(dx_chest_step)
        butt_step_stats = displacement_stats(dx_butt_step)
        residual_chest_applied_mean_abs_cm_by_pass.append(chest_step_stats["mean_abs_cm"])
        residual_chest_applied_max_abs_cm_by_pass.append(chest_step_stats["max_abs_cm"])
        residual_butt_applied_mean_abs_cm_by_pass.append(butt_step_stats["mean_abs_cm"])
        residual_butt_applied_max_abs_cm_by_pass.append(butt_step_stats["max_abs_cm"])
        chest_solve_changed |= chest_solve_changed_pass | (np.abs(dx_chest_step) > 1e-7)
        butt_solve_changed |= butt_solve_changed_pass | (np.abs(dx_butt_step) > 1e-7)
        residual_solve_pass_count += 1
        residual_solve_gains.append(pass_gain)

    final_pre_containment_state = compute_profile_state(v)
    chest_side_mask = final_pre_containment_state["chest_side_mask"]
    butt_side_mask = final_pre_containment_state["butt_side_mask"]
    signed_side_all = final_pre_containment_state["signed_side_all"]
    chest_side_strength = np.clip((anterior_sign * signed_side_all - 0.02) / 0.55, 0.0, 1.0)
    butt_side_strength = np.clip((posterior_sign * signed_side_all - 0.02) / 0.55, 0.0, 1.0)
    dx = dx_chest + dx_butt

    containment_dx_chest = np.zeros(v.shape[0], dtype=np.float64)
    containment_dx_butt = np.zeros(v.shape[0], dtype=np.float64)
    containment_active_chest = np.zeros(v.shape[0], dtype=bool)
    containment_active_butt = np.zeros(v.shape[0], dtype=bool)

    def containment_du_for_side(proj_cur, depth_cur, sdf_cur, side_region_mask, region_weight, side_sign):
        corr_px = np.zeros(v.shape[0], dtype=np.float64)
        rows_cur = np.clip(np.rint(proj_cur[:, 1]).astype(np.int32), 0, h - 1)
        in_image_cur = (
            (depth_cur > 1e-5) &
            (proj_cur[:, 0] >= 0) & (proj_cur[:, 0] < w) &
            (proj_cur[:, 1] >= 0) & (proj_cur[:, 1] < h)
        )
        if side_sign < 0:
            target = mask_left[rows_cur] + SIDE_SDF_CONTAINMENT_MARGIN_PX
            spill = target - proj_cur[:, 0]
            outside_side = proj_cur[:, 0] < target
        else:
            target = mask_right[rows_cur] - SIDE_SDF_CONTAINMENT_MARGIN_PX
            spill = target - proj_cur[:, 0]
            outside_side = proj_cur[:, 0] > target

        valid_target = np.isfinite(target)
        active = (
            side_region_mask &
            (region_weight > 1e-4) &
            in_image_cur &
            valid_target &
            outside_side &
            ((sdf_cur > 0.25) | (np.abs(spill) > SIDE_SDF_CONTAINMENT_MARGIN_PX))
        )
        if active.any():
            raw = spill[active] * SIDE_SDF_CONTAINMENT_GAIN * region_weight[active]
            corr_px[active] = np.clip(
                raw,
                -SIDE_SDF_CONTAINMENT_MAX_STEP_PX,
                SIDE_SDF_CONTAINMENT_MAX_STEP_PX,
            )
        return corr_px, active

    def smooth_displacement_over_mesh(dx_region, side_region_mask, region_weight, outer_weight):
        if mesh_neighbors is None or SIDE_SDF_DISPLACEMENT_SMOOTH_ITERS <= 0:
            return dx_region.copy(), np.zeros(v.shape[0], dtype=bool)
        active = side_region_mask & (
            (region_weight > 1e-4) |
            (outer_weight > 1e-4) |
            (np.abs(dx_region) > 1e-6)
        )
        source = active & (np.abs(dx_region) > 1e-6)
        if source.sum() < 8:
            return dx_region.copy(), np.zeros(v.shape[0], dtype=bool)

        smoothed = dx_region.astype(np.float64).copy()
        active_idx = np.nonzero(active)[0]
        for _ in range(int(SIDE_SDF_DISPLACEMENT_SMOOTH_ITERS)):
            nxt = smoothed.copy()
            for idx in active_idx:
                nbr = mesh_neighbors[idx]
                if nbr.size < 3:
                    continue
                nbr = nbr[active[nbr]]
                if nbr.size < 3:
                    continue
                local_weight = max(float(region_weight[idx]), float(outer_weight[idx]) * 0.5)
                local_alpha = float(SIDE_SDF_DISPLACEMENT_SMOOTH_ALPHA) * np.clip(0.35 + local_weight, 0.35, 1.0)
                nxt[idx] = (1.0 - local_alpha) * smoothed[idx] + local_alpha * float(np.mean(smoothed[nbr]))
            smoothed = nxt

        blended = dx_region.copy()
        changed = active & (np.abs(smoothed - dx_region) > 1e-7)
        blend = float(SIDE_SDF_DISPLACEMENT_SMOOTH_BLEND)
        blended[active] = (1.0 - blend) * dx_region[active] + blend * smoothed[active]
        return blended, changed

    def smooth_displacement_over_height(dx_region, side_region_mask, region_weight, outer_weight):
        active = side_region_mask & (
            (region_weight > 1e-4) |
            (outer_weight > 1e-4) |
            (np.abs(dx_region) > 1e-6)
        )
        if active.sum() < 8:
            return dx_region.copy(), np.zeros(v.shape[0], dtype=bool)

        transition = np.clip(1.0 - np.abs((region_weight * 2.0) - 1.0), 0.0, 1.0)
        edge_weight = np.maximum(transition, outer_weight)
        edge = active & (edge_weight > 1e-4)
        source = active & (np.abs(dx_region) > 1e-6)
        if edge.sum() < 4 or source.sum() < 8:
            return dx_region.copy(), np.zeros(v.shape[0], dtype=bool)

        bin_count = int(SIDE_SDF_EDGE_HEIGHT_SMOOTH_BINS)
        bins = np.clip(np.floor(pct * bin_count).astype(np.int32), 0, bin_count - 1)
        radius = int(SIDE_SDF_EDGE_HEIGHT_SMOOTH_RADIUS_BINS)
        smoothed = dx_region.astype(np.float64).copy()

        for bin_id in np.unique(bins[edge]):
            active_bin = edge & (bins == bin_id)
            source_bin = source & (np.abs(bins - bin_id) <= radius)
            if active_bin.sum() < 1 or source_bin.sum() < 6:
                continue
            target = float(np.median(dx_region[source_bin]))
            blend = float(SIDE_SDF_EDGE_HEIGHT_SMOOTH_BLEND) * edge_weight[active_bin]
            smoothed[active_bin] = (1.0 - blend) * dx_region[active_bin] + blend * target

        changed = edge & (np.abs(smoothed - dx_region) > 1e-7)
        return smoothed, changed

    containment_weight_chest = np.maximum(chest_target_weight, chest_outer_weight * 0.5)
    containment_weight_butt = np.maximum(butt_target_weight, butt_outer_weight * 0.5)
    for _ in range(SIDE_SDF_CONTAINMENT_PASSES):
        proj_cur, depth_cur = project_vertices_to_image(v, cam_t, float(focal_length), image_shape)
        sdf_cur = sample_image_bilinear(sdf, proj_cur)
        du_contain_chest, active_chest = containment_du_for_side(
            proj_cur,
            depth_cur,
            sdf_cur,
            chest_side_mask,
            containment_weight_chest,
            anterior_sign,
        )
        du_contain_butt, active_butt = containment_du_for_side(
            proj_cur,
            depth_cur,
            sdf_cur,
            butt_side_mask,
            containment_weight_butt,
            posterior_sign,
        )
        du_contain = du_contain_chest + du_contain_butt
        if not np.any(np.abs(du_contain) > 1e-6):
            break

        dx_step_chest = du_contain_chest * depth_cur / float(focal_length)
        dx_step_butt = du_contain_butt * depth_cur / float(focal_length)
        dx_step = dx_step_chest + dx_step_butt
        total_before = dx_chest + dx_butt + containment_dx_chest + containment_dx_butt
        dx_step = np.clip(total_before + dx_step, -max_push_m, max_push_m) - total_before

        requested_step = dx_step_chest + dx_step_butt
        scale = np.ones(v.shape[0], dtype=np.float64)
        requested_abs = np.abs(requested_step)
        nonzero = requested_abs > 1e-8
        scale[nonzero] = dx_step[nonzero] / requested_step[nonzero]
        dx_step_chest *= scale
        dx_step_butt *= scale

        v[:, 0] += dx_step
        containment_dx_chest += dx_step_chest
        containment_dx_butt += dx_step_butt
        containment_active_chest |= active_chest & (np.abs(dx_step_chest) > 1e-8)
        containment_active_butt |= active_butt & (np.abs(dx_step_butt) > 1e-8)

    dx_chest = dx_chest + containment_dx_chest
    dx_butt = dx_butt + containment_dx_butt
    dx = dx_chest + dx_butt

    pre_smooth_dx_chest = dx_chest.copy()
    pre_smooth_dx_butt = dx_butt.copy()
    dx_chest_smoothed, chest_smooth_changed = smooth_displacement_over_mesh(
        dx_chest,
        chest_side_mask,
        containment_weight_chest,
        chest_outer_weight,
    )
    dx_chest_smoothed, chest_edge_smooth_changed = smooth_displacement_over_height(
        dx_chest_smoothed,
        chest_side_mask,
        containment_weight_chest,
        chest_outer_weight,
    )
    chest_smooth_changed = chest_smooth_changed | chest_edge_smooth_changed
    dx_butt_smoothed, butt_smooth_changed = smooth_displacement_over_mesh(
        dx_butt,
        butt_side_mask,
        containment_weight_butt,
        butt_outer_weight,
    )
    dx_butt_smoothed, butt_edge_smooth_changed = smooth_displacement_over_height(
        dx_butt_smoothed,
        butt_side_mask,
        containment_weight_butt,
        butt_outer_weight,
    )
    butt_smooth_changed = butt_smooth_changed | butt_edge_smooth_changed
    smooth_delta_chest = dx_chest_smoothed - dx_chest
    smooth_delta_butt = dx_butt_smoothed - dx_butt
    smooth_delta = smooth_delta_chest + smooth_delta_butt
    if np.any(np.abs(smooth_delta) > 1e-8):
        total_before = dx_chest + dx_butt
        total_after = np.clip(total_before + smooth_delta, -max_push_m, max_push_m)
        smooth_delta = total_after - total_before
        requested = smooth_delta_chest + smooth_delta_butt
        scale = np.ones(v.shape[0], dtype=np.float64)
        requested_abs = np.abs(requested)
        nonzero = requested_abs > 1e-8
        scale[nonzero] = smooth_delta[nonzero] / requested[nonzero]
        smooth_delta_chest *= scale
        smooth_delta_butt *= scale
        v[:, 0] += smooth_delta
        dx_chest = dx_chest + smooth_delta_chest
        dx_butt = dx_butt + smooth_delta_butt

        # Smoothing improves surface quality, then this pulls any newly exposed
        # silhouette spill back inside the mask contour.
        for _ in range(SIDE_SDF_CONTAINMENT_PASSES):
            proj_cur, depth_cur = project_vertices_to_image(v, cam_t, float(focal_length), image_shape)
            sdf_cur = sample_image_bilinear(sdf, proj_cur)
            du_contain_chest, active_chest = containment_du_for_side(
                proj_cur,
                depth_cur,
                sdf_cur,
                chest_side_mask,
                containment_weight_chest,
                anterior_sign,
            )
            du_contain_butt, active_butt = containment_du_for_side(
                proj_cur,
                depth_cur,
                sdf_cur,
                butt_side_mask,
                containment_weight_butt,
                posterior_sign,
            )
            du_contain = du_contain_chest + du_contain_butt
            if not np.any(np.abs(du_contain) > 1e-6):
                break

            dx_step_chest = du_contain_chest * depth_cur / float(focal_length)
            dx_step_butt = du_contain_butt * depth_cur / float(focal_length)
            dx_step = dx_step_chest + dx_step_butt
            total_before = dx_chest + dx_butt
            dx_step = np.clip(total_before + dx_step, -max_push_m, max_push_m) - total_before

            requested_step = dx_step_chest + dx_step_butt
            scale = np.ones(v.shape[0], dtype=np.float64)
            requested_abs = np.abs(requested_step)
            nonzero = requested_abs > 1e-8
            scale[nonzero] = dx_step[nonzero] / requested_step[nonzero]
            dx_step_chest *= scale
            dx_step_butt *= scale

            v[:, 0] += dx_step
            dx_chest += dx_step_chest
            dx_butt += dx_step_butt
            containment_dx_chest += dx_step_chest
            containment_dx_butt += dx_step_butt
            containment_active_chest |= active_chest & (np.abs(dx_step_chest) > 1e-8)
            containment_active_butt |= active_butt & (np.abs(dx_step_butt) > 1e-8)

    displacement_smooth_delta_chest = dx_chest - pre_smooth_dx_chest
    displacement_smooth_delta_butt = dx_butt - pre_smooth_dx_butt
    dx = dx_chest + dx_butt

    initial_chest_row_gap = signed_row_gap_stats(chest_shift_row, chest_valid_row, anterior_sign)
    initial_butt_row_gap = signed_row_gap_stats(butt_shift_row, butt_valid_row, posterior_sign)
    initial_chest_raw_row_gap = signed_row_gap_stats(chest_raw_shift_row, chest_valid_row, anterior_sign)
    initial_butt_raw_row_gap = signed_row_gap_stats(butt_raw_shift_row, butt_valid_row, posterior_sign)
    target_chest_px_stats = pixel_target_stats(target_px_chest)
    target_butt_px_stats = pixel_target_stats(target_px_butt)
    target_chest_dx_stats = displacement_stats(target_dx_chest)
    target_butt_dx_stats = displacement_stats(target_dx_butt)
    solved_chest_initial_stats = displacement_stats(solved_dx_chest_initial)
    solved_butt_initial_stats = displacement_stats(solved_dx_butt_initial)
    final_chest_dx_stats = displacement_stats(dx_chest)
    final_butt_dx_stats = displacement_stats(dx_butt)

    final_profile_state = compute_profile_state(v)
    final_chest_shift_row, final_chest_valid_row, final_chest_raw_shift_row, _ = target_shift_by_row(
        final_profile_state,
        anterior_sign,
    )
    final_butt_shift_row, final_butt_valid_row, final_butt_raw_shift_row, _ = target_shift_by_row(
        final_profile_state,
        posterior_sign,
    )
    final_chest_row_gap = signed_row_gap_stats(final_chest_shift_row, final_chest_valid_row, anterior_sign)
    final_butt_row_gap = signed_row_gap_stats(final_butt_shift_row, final_butt_valid_row, posterior_sign)
    final_chest_raw_row_gap = signed_row_gap_stats(final_chest_raw_shift_row, final_chest_valid_row, anterior_sign)
    final_butt_raw_row_gap = signed_row_gap_stats(final_butt_raw_shift_row, final_butt_valid_row, posterior_sign)

    moved = np.abs(dx) > 1e-6
    chest_moved = np.abs(dx_chest) > 1e-6
    butt_moved = np.abs(dx_butt) > 1e-6
    meta.update({
        "enabled": np.array(True),
        "reason": np.array("ok", dtype=object),
        "chest_center_pct": np.array(chest_center * 100.0, dtype=np.float64),
        "butt_center_pct": np.array(butt_center * 100.0, dtype=np.float64),
        "anterior_sign": np.array(int(anterior_sign), dtype=np.int64),
        "anterior_source": np.array(anterior_source, dtype=object),
        "torso_core_enabled": torso_core_meta.get("enabled", np.array(False)),
        "torso_core_reason": torso_core_meta.get("reason", np.array("unknown", dtype=object)),
        "torso_core_threshold_m": torso_core_meta.get("threshold_m", np.array(0.0, dtype=np.float64)),
        "torso_core_vertex_count": torso_core_meta.get("vertex_count", np.array(int(torso_core.sum()), dtype=np.int64)),
        "profile_method": np.array(profile_method, dtype=object),
        "profile_fit_half_width_pct": np.array(float(profile_fit_half_width * 100.0), dtype=np.float64),
        "target_core_half_width_pct": np.array(float(SIDE_SDF_TARGET_CORE_HALF_WIDTH * 100.0), dtype=np.float64),
        "target_feather_width_pct": np.array(float(SIDE_SDF_TARGET_FEATHER_WIDTH * 100.0), dtype=np.float64),
        "profile_fit_matches_anchor_debug_band": np.array(True, dtype=bool),
        "solve_reason": np.array(solve_reason, dtype=object),
        "solve_data_weight": np.array(float(SIDE_SDF_SOLVE_DATA_WEIGHT), dtype=np.float64),
        "solve_pin_weight": np.array(float(SIDE_SDF_SOLVE_PIN_WEIGHT), dtype=np.float64),
        "solve_smooth_lambda": np.array(float(SIDE_SDF_SOLVE_SMOOTH_LAMBDA), dtype=np.float64),
        "solve_outside_smooth_width_pct": np.array(float(SIDE_SDF_SOLVE_OUTSIDE_SMOOTH_WIDTH * 100.0), dtype=np.float64),
        "solve_side_pin_scale": np.array(float(SIDE_SDF_SOLVE_SIDE_PIN_SCALE), dtype=np.float64),
        "solve_edge_data_floor": np.array(float(SIDE_SDF_SOLVE_EDGE_DATA_FLOOR), dtype=np.float64),
        "solve_edge_data_power": np.array(float(SIDE_SDF_SOLVE_EDGE_DATA_POWER), dtype=np.float64),
        "residual_solve_pass_count": np.array(int(residual_solve_pass_count), dtype=np.int64),
        "residual_solve_configured_pass_count": np.array(int(SIDE_SDF_RESIDUAL_SOLVE_PASSES), dtype=np.int64),
        "residual_solve_gain": np.array(float(SIDE_SDF_RESIDUAL_SOLVE_GAIN), dtype=np.float64),
        "residual_solve_gains": np.asarray(residual_solve_gains, dtype=np.float64),
        "row_underfit_peak_preserve": np.array(bool(SIDE_SDF_ROW_UNDERFIT_PEAK_PRESERVE), dtype=bool),
        "target_displacement_height_scaled": np.array(False, dtype=bool),
        "residual_chest_raw_mean_underfit_px_by_pass": np.asarray(residual_chest_raw_mean_underfit_px_by_pass, dtype=np.float64),
        "residual_chest_raw_max_underfit_px_by_pass": np.asarray(residual_chest_raw_max_underfit_px_by_pass, dtype=np.float64),
        "residual_chest_smoothed_mean_underfit_px_by_pass": np.asarray(residual_chest_smoothed_mean_underfit_px_by_pass, dtype=np.float64),
        "residual_chest_smoothed_max_underfit_px_by_pass": np.asarray(residual_chest_smoothed_max_underfit_px_by_pass, dtype=np.float64),
        "residual_butt_raw_mean_underfit_px_by_pass": np.asarray(residual_butt_raw_mean_underfit_px_by_pass, dtype=np.float64),
        "residual_butt_raw_max_underfit_px_by_pass": np.asarray(residual_butt_raw_max_underfit_px_by_pass, dtype=np.float64),
        "residual_butt_smoothed_mean_underfit_px_by_pass": np.asarray(residual_butt_smoothed_mean_underfit_px_by_pass, dtype=np.float64),
        "residual_butt_smoothed_max_underfit_px_by_pass": np.asarray(residual_butt_smoothed_max_underfit_px_by_pass, dtype=np.float64),
        "residual_chest_applied_mean_abs_cm_by_pass": np.asarray(residual_chest_applied_mean_abs_cm_by_pass, dtype=np.float64),
        "residual_chest_applied_max_abs_cm_by_pass": np.asarray(residual_chest_applied_max_abs_cm_by_pass, dtype=np.float64),
        "residual_butt_applied_mean_abs_cm_by_pass": np.asarray(residual_butt_applied_mean_abs_cm_by_pass, dtype=np.float64),
        "residual_butt_applied_max_abs_cm_by_pass": np.asarray(residual_butt_applied_max_abs_cm_by_pass, dtype=np.float64),
        "row_edge_low_quantile": np.array(float(SIDE_SDF_ROW_EDGE_LOW_QUANTILE), dtype=np.float64),
        "row_edge_high_quantile": np.array(float(SIDE_SDF_ROW_EDGE_HIGH_QUANTILE), dtype=np.float64),
        "chest_solve_changed_vertex_count": np.array(int(chest_solve_changed.sum()), dtype=np.int64),
        "butt_solve_changed_vertex_count": np.array(int(butt_solve_changed.sum()), dtype=np.int64),
        "chest_valid_row_count": np.array(int(chest_valid_row.sum()), dtype=np.int64),
        "butt_valid_row_count": np.array(int(butt_valid_row.sum()), dtype=np.int64),
        "chest_raw_negative_row_count": np.array(int(np.sum(chest_valid_row & (chest_raw_shift_row < -1e-6))), dtype=np.int64),
        "chest_raw_positive_row_count": np.array(int(np.sum(chest_valid_row & (chest_raw_shift_row > 1e-6))), dtype=np.int64),
        "butt_raw_negative_row_count": np.array(int(np.sum(butt_valid_row & (butt_raw_shift_row < -1e-6))), dtype=np.int64),
        "butt_raw_positive_row_count": np.array(int(np.sum(butt_valid_row & (butt_raw_shift_row > 1e-6))), dtype=np.int64),
        "chest_smoothed_negative_row_count": np.array(int(np.sum(chest_valid_row & (chest_shift_row < -1e-6))), dtype=np.int64),
        "chest_smoothed_positive_row_count": np.array(int(np.sum(chest_valid_row & (chest_shift_row > 1e-6))), dtype=np.int64),
        "butt_smoothed_negative_row_count": np.array(int(np.sum(butt_valid_row & (butt_shift_row < -1e-6))), dtype=np.int64),
        "butt_smoothed_positive_row_count": np.array(int(np.sum(butt_valid_row & (butt_shift_row > 1e-6))), dtype=np.int64),
        "chest_shift_sign_flip_row_count": np.array(int(np.sum(chest_shift_sign_flip_row)), dtype=np.int64),
        "butt_shift_sign_flip_row_count": np.array(int(np.sum(butt_shift_sign_flip_row)), dtype=np.int64),
        "chest_moved_vertex_count": np.array(int(chest_moved.sum()), dtype=np.int64),
        "butt_moved_vertex_count": np.array(int(butt_moved.sum()), dtype=np.int64),
        "displacement_smooth_enabled": np.array(mesh_neighbors is not None, dtype=bool),
        "displacement_smooth_iterations": np.array(int(SIDE_SDF_DISPLACEMENT_SMOOTH_ITERS), dtype=np.int64),
        "displacement_smooth_alpha": np.array(float(SIDE_SDF_DISPLACEMENT_SMOOTH_ALPHA), dtype=np.float64),
        "displacement_smooth_blend": np.array(float(SIDE_SDF_DISPLACEMENT_SMOOTH_BLEND), dtype=np.float64),
        "chest_displacement_smooth_vertex_count": np.array(int(chest_smooth_changed.sum()), dtype=np.int64),
        "butt_displacement_smooth_vertex_count": np.array(int(butt_smooth_changed.sum()), dtype=np.int64),
        "chest_displacement_smooth_mean_abs_delta_cm": np.array(float(np.mean(np.abs(displacement_smooth_delta_chest[chest_smooth_changed])) * 100.0) if chest_smooth_changed.any() else 0.0, dtype=np.float64),
        "butt_displacement_smooth_mean_abs_delta_cm": np.array(float(np.mean(np.abs(displacement_smooth_delta_butt[butt_smooth_changed])) * 100.0) if butt_smooth_changed.any() else 0.0, dtype=np.float64),
        "chest_displacement_smooth_max_abs_delta_cm": np.array(float(np.max(np.abs(displacement_smooth_delta_chest))) * 100.0 if displacement_smooth_delta_chest.size else 0.0, dtype=np.float64),
        "butt_displacement_smooth_max_abs_delta_cm": np.array(float(np.max(np.abs(displacement_smooth_delta_butt))) * 100.0 if displacement_smooth_delta_butt.size else 0.0, dtype=np.float64),
        "chest_containment_moved_vertex_count": np.array(int(containment_active_chest.sum()), dtype=np.int64),
        "butt_containment_moved_vertex_count": np.array(int(containment_active_butt.sum()), dtype=np.int64),
        "chest_containment_mean_abs_push_cm": np.array(float(np.mean(np.abs(containment_dx_chest[containment_active_chest])) * 100.0) if containment_active_chest.any() else 0.0, dtype=np.float64),
        "butt_containment_mean_abs_push_cm": np.array(float(np.mean(np.abs(containment_dx_butt[containment_active_butt])) * 100.0) if containment_active_butt.any() else 0.0, dtype=np.float64),
        "chest_containment_max_abs_push_cm": np.array(float(np.max(np.abs(containment_dx_chest))) * 100.0 if containment_dx_chest.size else 0.0, dtype=np.float64),
        "butt_containment_max_abs_push_cm": np.array(float(np.max(np.abs(containment_dx_butt))) * 100.0 if containment_dx_butt.size else 0.0, dtype=np.float64),
        "chest_negative_moved_vertex_count": np.array(int(np.sum(dx_chest < -1e-6)), dtype=np.int64),
        "chest_positive_moved_vertex_count": np.array(int(np.sum(dx_chest > 1e-6)), dtype=np.int64),
        "butt_negative_moved_vertex_count": np.array(int(np.sum(dx_butt < -1e-6)), dtype=np.int64),
        "butt_positive_moved_vertex_count": np.array(int(np.sum(dx_butt > 1e-6)), dtype=np.int64),
        "chest_mean_abs_push_cm": np.array(float(np.mean(np.abs(dx_chest[chest_moved])) * 100.0) if chest_moved.any() else 0.0, dtype=np.float64),
        "butt_mean_abs_push_cm": np.array(float(np.mean(np.abs(dx_butt[butt_moved])) * 100.0) if butt_moved.any() else 0.0, dtype=np.float64),
        "chest_mean_signed_push_cm": np.array(float(np.mean(dx_chest[chest_moved]) * 100.0) if chest_moved.any() else 0.0, dtype=np.float64),
        "butt_mean_signed_push_cm": np.array(float(np.mean(dx_butt[butt_moved]) * 100.0) if butt_moved.any() else 0.0, dtype=np.float64),
        "chest_max_abs_push_cm": np.array(float(np.max(np.abs(dx_chest)) * 100.0) if dx_chest.size else 0.0, dtype=np.float64),
        "butt_max_abs_push_cm": np.array(float(np.max(np.abs(dx_butt)) * 100.0) if dx_butt.size else 0.0, dtype=np.float64),
        "mean_abs_push_cm": np.array(float(np.mean(np.abs(dx[moved])) * 100.0) if moved.any() else 0.0, dtype=np.float64),
        "max_abs_push_cm": np.array(float(np.max(np.abs(dx)) * 100.0) if dx.size else 0.0, dtype=np.float64),
        "mean_selected_sdf_px": np.array(float(np.mean(sdf_values[selected])) if selected.any() else 0.0, dtype=np.float64),
        "moved_vertex_count": np.array(int(moved.sum()), dtype=np.int64),
    })

    def add_count_float_stats(prefix, stats, value_suffix):
        meta[f"{prefix}_count"] = np.array(int(stats.get("count", 0)), dtype=np.int64)
        for key, value in stats.items():
            if key == "count":
                continue
            meta[f"{prefix}_{key.replace('_px', value_suffix).replace('_cm', value_suffix)}"] = np.array(
                float(value),
                dtype=np.float64,
            )

    def add_row_gap_stats(prefix, stats):
        meta[f"{prefix}_valid_row_count"] = np.array(int(stats.get("valid_count", 0)), dtype=np.int64)
        for key, value in stats.items():
            if key == "valid_count":
                continue
            meta[f"{prefix}_{key}"] = np.array(float(value), dtype=np.float64)

    add_row_gap_stats("chest_initial_smoothed_row_gap", initial_chest_row_gap)
    add_row_gap_stats("butt_initial_smoothed_row_gap", initial_butt_row_gap)
    add_row_gap_stats("chest_initial_raw_row_gap", initial_chest_raw_row_gap)
    add_row_gap_stats("butt_initial_raw_row_gap", initial_butt_raw_row_gap)
    add_row_gap_stats("chest_final_smoothed_row_gap", final_chest_row_gap)
    add_row_gap_stats("butt_final_smoothed_row_gap", final_butt_row_gap)
    add_row_gap_stats("chest_final_raw_row_gap", final_chest_raw_row_gap)
    add_row_gap_stats("butt_final_raw_row_gap", final_butt_raw_row_gap)
    add_count_float_stats("chest_target_vertex_px", target_chest_px_stats, "_px")
    add_count_float_stats("butt_target_vertex_px", target_butt_px_stats, "_px")
    add_count_float_stats("chest_target_vertex_dx", target_chest_dx_stats, "_cm")
    add_count_float_stats("butt_target_vertex_dx", target_butt_dx_stats, "_cm")
    add_count_float_stats("chest_solved_initial_dx", solved_chest_initial_stats, "_cm")
    add_count_float_stats("butt_solved_initial_dx", solved_butt_initial_stats, "_cm")
    add_count_float_stats("chest_final_dx", final_chest_dx_stats, "_cm")
    add_count_float_stats("butt_final_dx", final_butt_dx_stats, "_cm")

    return v.astype(np.float32), meta


def scale_mesh_to_target_height(vertices, target_height, axis=np.array([0.0, 1.0, 0.0], dtype=np.float64)):
    current_height = compute_height_along_axis(vertices, axis)
    if current_height <= 1e-8:
        raise ValueError(f"Current mesh height is too small to scale safely: {current_height}")

    scale_factor = float(target_height) / float(current_height)
    scaled_vertices = vertices.astype(np.float64) * scale_factor

    return (
        scaled_vertices.astype(np.float32),
        np.array(scale_factor, dtype=np.float64),
        np.array(current_height, dtype=np.float64),
        np.array(target_height, dtype=np.float64),
    )


def should_scale_result_key(key, value):
    """
    Scale only spatial parameters that are likely to be consumed by downstream
    measurement code. Do NOT scale pose, shape, rotation, confidence, or image-space data.
    """
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        return False

    key_l = key.lower()

    blocked_substrings = (
        "pose",
        "rot",
        "orient",
        "quat",
        "theta",
        "beta",
        "shape",
        "conf",
        "score",
        "prob",
        "mask",
        "bbox",
        "uv",
        "2d",
        "heatmap",
        "focal",
        "intrinsic",
        "extrinsic",
    )
    if any(token in key_l for token in blocked_substrings):
        return False

    spatial_substrings = (
        "vert",
        "joint",
        "keypoint",
        "landmark",
        "cam_t",
        "camera_translation",
        "translation",
        "transl",
        "pelvis",
        "root",
        "center_3d",
        "offset_3d",
    )
    if any(token in key_l for token in spatial_substrings):
        return True

    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim >= 1 and arr.shape[-1] == 3 and np.issubdtype(arr.dtype, np.number):
        return True

    return False


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def scale_result_params(result, scale_factor):
    """
    Return a copy of the original result dict with spatial 3D parameters scaled.
    Also stores scale metadata so downstream code can verify units.

    Important:
    - Never scale latent body-shape parameters (scale_params, mhr_model_params).
      Those are model-space coefficients, not world-space lengths.
    """
    scaled = {}
    applied_keys = []

    for k, v in result.items():
        key_l = k.lower()

        if key_l in {"scale_params", "mhr_model_params"}:
            if isinstance(v, torch.Tensor):
                scaled[k] = v.clone()
            elif isinstance(v, np.ndarray):
                scaled[k] = v.copy()
            else:
                scaled[k] = v
            continue

        if isinstance(v, torch.Tensor):
            if should_scale_result_key(k, v):
                scaled[k] = v * float(scale_factor)
                applied_keys.append(k)
            else:
                scaled[k] = v.clone()
        elif isinstance(v, np.ndarray):
            if should_scale_result_key(k, v):
                scaled[k] = v.astype(np.float64) * float(scale_factor)
                if np.issubdtype(v.dtype, np.floating):
                    scaled[k] = scaled[k].astype(v.dtype)
                applied_keys.append(k)
            else:
                scaled[k] = v.copy()
        else:
            scaled[k] = v

    scaled["applied_scale_factor"] = np.array(scale_factor, dtype=np.float64)
    scaled["scaled_param_keys"] = np.array(applied_keys, dtype=object)
    return scaled



def _jsonable_measurement_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable_measurement_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_measurement_value(v) for v in value]
    if value.__class__.__name__ == "Trimesh":
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def save_measurements_json(measurements, output_json_path):
    json_dict = {}
    for key, value in measurements.items():
        json_value = _jsonable_measurement_value(value)
        if json_value is not None:
            json_dict[str(key)] = json_value
    with open(output_json_path, "w") as f:
        json.dump(json_dict, f, indent=2, sort_keys=True)
    return output_json_path


def measure_untouched_side_anchor_pcts(side_result, side_vertices, faces, target_height_m, output_dir):
    """Use CLAD on the un-SDF-ed side mesh to locate bust and butt height bands."""
    meta = {
        "enabled": np.array(False),
        "reason": np.array("not_run", dtype=object),
        "bust_pct": np.array(0.0, dtype=np.float64),
        "hip_pct": np.array(0.0, dtype=np.float64),
        "bust_anchor_pct": np.array(0.0, dtype=np.float64),
        "hip_anchor_pct": np.array(0.0, dtype=np.float64),
    }
    paths = {
        "params_json": None,
        "measurements_json": None,
        "clad_obj": None,
    }

    try:
        side_oriented = center_and_orient_mesh(side_vertices)
        side_scaled, side_scale_factor, _, _ = scale_mesh_to_target_height(
            side_oriented["vertices_oriented"],
            target_height_m,
        )
        side_clad_vertices = sam_upright_vertices_to_clad_canonical(side_scaled)

        side_anchor_params = scale_result_params(
            _copy_result_dict(side_result),
            float(side_scale_factor),
        )
        side_anchor_params["pred_vertices"] = side_scaled.astype(np.float32)
        side_anchor_params["fusion_target_height"] = np.array(float(target_height_m * 100.0), dtype=np.float64)
        side_anchor_params["fusion_target_height_cm"] = np.array(float(target_height_m * 100.0), dtype=np.float64)
        side_anchor_params["fusion_prefer_vertices_for_clad"] = np.array(True)
        side_anchor_params["fusion_rule"] = np.array("side_untouched_clad_anchor_mesh", dtype=object)
        side_anchor_params["fusion_vertices_clad"] = side_clad_vertices.astype(np.float32)
        side_anchor_params["fusion_faces_clad"] = np.asarray(faces, dtype=np.int32)
        side_anchor_params["fusion_vertices_clad_coordinate_system"] = np.array(
            "x_lateral_y_profile_z_up_meters",
            dtype=object,
        )

        paths["params_json"] = os.path.join(output_dir, "side_untouched_clad_anchor_params.json")
        paths["clad_obj"] = os.path.join(output_dir, "side_untouched_clad_anchor.obj")
        save_result_json(side_anchor_params, paths["params_json"])
        save_mesh_obj(side_clad_vertices, faces, paths["clad_obj"])

        repo_root = os.path.dirname(os.path.abspath(__file__))
        clad_root = os.path.join(repo_root, "clad-body")
        if os.path.isdir(clad_root) and clad_root not in sys.path:
            sys.path.insert(0, clad_root)

        from clad_body.load import load_mhr_from_params
        from clad_body.measure import measure

        body = load_mhr_from_params(paths["params_json"])
        measurements = measure(body, only=["bust_cm", "hip_cm"])
        paths["measurements_json"] = os.path.join(output_dir, "side_untouched_clad_anchor_measurements.json")
        save_measurements_json(measurements, paths["measurements_json"])

        anchors = {}
        try:
            bust_pct = float(measurements.get("_bust_pct", 0.0))
            if 0.0 < bust_pct < 100.0:
                chest_pct = bust_pct / 100.0
                anchors["chest"] = chest_pct
                meta["bust_pct"] = np.array(bust_pct, dtype=np.float64)
                meta["bust_anchor_pct"] = np.array(chest_pct * 100.0, dtype=np.float64)
        except Exception:
            pass
        try:
            hip_pct = float(measurements.get("_hip_pct", 0.0))
            if 0.0 < hip_pct < 100.0:
                butt_pct = hip_pct / 100.0
                anchors["butt"] = butt_pct
                meta["hip_pct"] = np.array(hip_pct, dtype=np.float64)
                meta["hip_anchor_pct"] = np.array(butt_pct * 100.0, dtype=np.float64)
        except Exception:
            pass

        if not anchors:
            meta["reason"] = np.array("missing_bust_hip_pct", dtype=object)
            return {}, meta, paths

        meta["enabled"] = np.array(True)
        meta["reason"] = np.array("ok", dtype=object)
        return anchors, meta, paths
    except Exception as e:
        meta["reason"] = np.array(f"failed: {e}", dtype=object)
        print(f"[fusion] Warning: side CLAD anchor measurement failed: {e}")
        return {}, meta, paths

def _copy_result_dict(result):
    copied = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            copied[k] = v.clone()
        elif isinstance(v, np.ndarray):
            copied[k] = v.copy()
        else:
            copied[k] = v
    return copied


def build_fused_result(front_result, side_result, fused_vertices, target_height):
    """
    Build one fused params dict before scaling.
    The final clad-body JSON is produced by scaling this fused result afterward.
    """
    fused = _copy_result_dict(front_result)
    fused["pred_vertices"] = fused_vertices.astype(np.float32)

    # Fuse shape and scale parameters across front/side for one final param set.
    if "shape_params" in front_result and "shape_params" in side_result:
        front_shape = _to_numpy(front_result["shape_params"]).astype(np.float64)
        side_shape = _to_numpy(side_result["shape_params"]).astype(np.float64)
        if front_shape.shape == side_shape.shape:
            fused["shape_params"] = ((front_shape + side_shape) * 0.5).astype(np.float32)

    if "scale_params" in front_result and "scale_params" in side_result:
        front_scale = _to_numpy(front_result["scale_params"]).astype(np.float64)
        side_scale = _to_numpy(side_result["scale_params"]).astype(np.float64)
        if front_scale.shape == side_scale.shape:
            fused["scale_params"] = ((front_scale + side_scale) * 0.5).astype(np.float32)

    # For MHR params, only the scale block [136:] should be fused.
    if "mhr_model_params" in front_result and "mhr_model_params" in side_result:
        front_mhr = _to_numpy(front_result["mhr_model_params"]).astype(np.float64)
        side_mhr = _to_numpy(side_result["mhr_model_params"]).astype(np.float64)
        if front_mhr.shape == side_mhr.shape and front_mhr.shape[-1] > 136:
            fused_mhr = front_mhr.copy()
            fused_mhr[..., 136:] = (front_mhr[..., 136:] + side_mhr[..., 136:]) * 0.5
            fused["mhr_model_params"] = fused_mhr.astype(np.float32)

    # Keep target height explicitly in cm across the pipeline.
    fused["fusion_target_height"] = np.array(float(target_height), dtype=np.float64)
    fused["fusion_target_height_cm"] = np.array(float(target_height), dtype=np.float64)
    fused["fusion_rule"] = np.array("front_xy_side_z", dtype=object)

    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-height",
        type=float,
        required=True,
        help="Desired final body height in centimeters.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./input",
        help="Directory containing front/side images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to write fusion outputs.",
    )
    parser.add_argument(
        "--profile-depth-correction-strength",
        type=float,
        default=float(os.environ.get("FUSION_PROFILE_DEPTH_CORRECTION", "0.35")),
        help=(
            "Smoothly expand bust and hip/glute profile depth when side depth "
            "is implausibly flat relative to front width. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--profile-depth-correction-max-scale",
        type=float,
        default=float(os.environ.get("FUSION_PROFILE_DEPTH_MAX_SCALE", "1.22")),
        help="Maximum local depth expansion from profile correction.",
    )
    parser.add_argument(
        "--side-mask",
        type=str,
        default=os.environ.get("FUSION_SIDE_MASK", ""),
        help=(
            "Optional side-view binary or grayscale person mask. If omitted, the SAM3D "
            "segmentor mask from side inference is used."
        ),
    )
    parser.add_argument(
        "--segmentor-name",
        type=str,
        default=os.environ.get("SAM3D_SEGMENTOR", "sam2"),
        help="Human segmentation model used when SAM mask generation is enabled.",
    )
    parser.add_argument(
        "--segmentor-path",
        type=str,
        default=os.environ.get("SAM3D_SEGMENTOR_PATH", "external/sam2"),
        help="Path to human segmentation model folder. Defaults to external/sam2 for SAM2.",
    )
    parser.add_argument(
        "--no-sam-mask",
        action="store_true",
        help="Disable automatic SAM mask generation and rely only on --side-mask if provided.",
    )
    parser.add_argument(
        "--side-sdf-profile-strength",
        type=float,
        default=float(os.environ.get("FUSION_SIDE_SDF_PROFILE_STRENGTH", "1.0")),
        help="Strength for bidirectional side-mask silhouette correction in chest/butt bands. Use 0 to disable.",
    )
    parser.add_argument(
        "--side-sdf-profile-max-push-cm",
        type=float,
        default=float(os.environ.get("FUSION_SIDE_SDF_PROFILE_MAX_PUSH_CM", "35.0")),
        help="Maximum side-profile displacement per vertex, in centimeters.",
    )
    parser.add_argument(
        "--side-sdf-row-radius",
        type=int,
        default=int(os.environ.get("FUSION_SIDE_SDF_ROW_RADIUS", "6")),
        help="Vertical pixel radius used when matching side mesh rows to mask/SDF rows.",
    )
    parser.add_argument(
        "--no-fused-vertex-override",
        action="store_true",
        help=(
            "Opt out of using the fused SAM mesh as CLAD's measurement/render mesh. "
            "When set, CLAD falls back to MHR rest-pose reconstruction plus profile correction."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    checkpoint_dir = os.environ.get(
        "SAM3D_CHECKPOINT_DIR",
        "./checkpoints/sam-3d-body-dinov3",
    )
    checkpoint_path = os.environ.get(
        "SAM3D_CHECKPOINT_PATH",
        os.path.join(checkpoint_dir, "model.ckpt"),
    )
    mhr_path = os.environ.get(
        "SAM3D_MHR_PATH",
        os.path.join(checkpoint_dir, "assets", "mhr_model.pt"),
    )
    detector_name = os.environ.get("SAM3D_DETECTOR", "rtdetr")
    fov_name = os.environ.get("SAM3D_FOV", "moge2")
    detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = os.environ.get("SAM3D_FOV_PATH", "")

    os.makedirs(output_dir, exist_ok=True)
    clear_output_dir(output_dir)

    front_image_path = find_image(input_dir, "front")
    side_image_path = find_image(input_dir, "side")

    target_height_cm = float(args.target_height)
    target_height_m = target_height_cm / 100.0
    use_sam_mask = not bool(args.no_sam_mask)

    print(f"Front image: {front_image_path}")
    print(f"Side image : {side_image_path}")
    print(f"Target fused mesh height: {target_height_cm}")
    print(f"SAM mask generation      : {'enabled' if use_sam_mask else 'disabled'}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path,
        device=device,
        mhr_path=mhr_path,
    )

    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(
            name=detector_name,
            device=device,
            path=detector_path,
        )

    if use_sam_mask:
        if not args.segmentor_name:
            raise RuntimeError("SAM mask generation is enabled, but --segmentor-name is empty.")
        if args.segmentor_name == "sam2" and not args.segmentor_path:
            raise RuntimeError(
                "SAM mask generation with sam2 requires --segmentor-path or SAM3D_SEGMENTOR_PATH. "
                "Set SAM3D_SEGMENTOR_PATH or pass --no-sam-mask to disable."
            )
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(
            name=args.segmentor_name,
            device=device,
            path=args.segmentor_path,
        )

    if fov_name:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(
            name=fov_name,
            device=device,
            path=fov_path,
        )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # ---- run front ----
    print("\nRunning front image...")
    front_outputs = estimator.process_one_image(
        front_image_path,
        bbox_thr=0.8,
        use_mask=use_sam_mask,
    )
    if not (isinstance(front_outputs, list) and len(front_outputs) > 0 and isinstance(front_outputs[0], dict)):
        raise RuntimeError("Front inference did not return expected outputs[0] dict")
    front_result = front_outputs[0]

    # ---- run side ----
    print("\nRunning side image...")
    side_outputs = estimator.process_one_image(
        side_image_path,
        bbox_thr=0.8,
        use_mask=use_sam_mask,
    )
    if not (isinstance(side_outputs, list) and len(side_outputs) > 0 and isinstance(side_outputs[0], dict)):
        raise RuntimeError("Side inference did not return expected outputs[0] dict")
    side_result = side_outputs[0]

    side_image_bgr = cv2.imread(side_image_path)
    if side_image_bgr is None:
        raise FileNotFoundError(f"Could not read side image: {side_image_path}")

    side_mask = (
        load_binary_mask(args.side_mask, side_image_bgr.shape)
        if args.side_mask
        else result_mask_to_binary(side_result, side_image_bgr.shape)
    )
    side_mask_path, side_sdf_path, side_sdf_vis_path = save_mask_and_sdf(
        side_mask,
        output_dir,
        "side",
    )
    front_raw_render = render_and_save(
        front_image_path,
        [front_result],
        estimator.faces,
        output_dir,
        "front_raw.jpg",
    )
    side_raw_render = render_and_save(
        side_image_path,
        [side_result],
        estimator.faces,
        output_dir,
        "side_raw.jpg",
    )

    # ---- extract vertices ----
    front_vertices = front_result["pred_vertices"]
    side_vertices = side_result["pred_vertices"]

    if isinstance(front_vertices, torch.Tensor):
        front_vertices = front_vertices.detach().cpu().numpy()
    if isinstance(side_vertices, torch.Tensor):
        side_vertices = side_vertices.detach().cpu().numpy()

    front_vertices = front_vertices.astype(np.float32)
    side_vertices = side_vertices.astype(np.float32)

    side_anchor_pcts, side_anchor_meta, side_anchor_paths = measure_untouched_side_anchor_pcts(
        side_result,
        side_vertices,
        estimator.faces,
        target_height_m,
        output_dir,
    )
    measurement_anchor_pcts = dict(side_anchor_pcts)
    missing_anchor_names = [name for name in ("chest", "butt") if name not in measurement_anchor_pcts]
    if missing_anchor_names:
        reason = np.asarray(side_anchor_meta.get("reason", "unknown")).item()
        raise RuntimeError(
            "CLAD anchor measurement is required for side SDF/profile correction, "
            f"but missing {missing_anchor_names}. CLAD reason: {reason}"
        )

    side_anchor_reason = np.asarray(side_anchor_meta.get("reason", "unknown")).item()
    print("\nUntouched side CLAD anchors:")
    print(f"  reason       : {side_anchor_reason}")
    print(f"  bust pct     : {float(np.asarray(side_anchor_meta.get('bust_pct', 0.0)).item()):.4f} (CLAD)")
    print(f"  hip pct      : {float(np.asarray(side_anchor_meta.get('hip_pct', 0.0)).item()):.4f} (CLAD)")
    print(f"  bust anchor  : {float(np.asarray(side_anchor_meta.get('bust_anchor_pct', 0.0)).item()):.4f} (SDF)")
    print(f"  hip anchor   : {float(np.asarray(side_anchor_meta.get('hip_anchor_pct', 0.0)).item()):.4f} (SDF)")

    side_anchor_debug_mask_path = save_side_anchor_debug_mask(
        side_mask,
        side_image_bgr,
        side_vertices,
        side_result,
        measurement_anchor_pcts,
        output_dir,
    )
    side_projection_alignment_debug_path = save_side_projection_alignment_debug(
        side_mask,
        side_image_bgr,
        side_vertices,
        side_result,
        output_dir,
    )

    side_vertices, side_sdf_profile_meta = deform_side_mesh_to_mask_profile(
        side_vertices,
        side_result,
        side_mask,
        side_image_bgr.shape,
        anchor_pcts=measurement_anchor_pcts,
        strength=float(args.side_sdf_profile_strength),
        max_push_cm=float(args.side_sdf_profile_max_push_cm),
        row_radius=int(args.side_sdf_row_radius),
        faces=estimator.faces,
    )
    side_result["pred_vertices"] = side_vertices.astype(np.float32)

    side_sdf_edited_obj = None
    side_sdf_edited_render = None
    side_sdf_row_debug_path = None
    if bool(np.asarray(side_sdf_profile_meta.get("enabled", False)).item()):
        side_sdf_edited_obj = os.path.join(output_dir, "side_sdf_profile_edited.obj")
        save_mesh_obj(side_vertices, estimator.faces, side_sdf_edited_obj)
        side_sdf_edited_render = render_and_save(
            side_image_path,
            [side_result],
            estimator.faces,
            output_dir,
            "side_sdf_profile_edited.jpg",
        )
        side_sdf_row_debug_path = save_side_sdf_row_debug(
            side_mask,
            side_image_bgr,
            side_vertices,
            side_result,
            measurement_anchor_pcts,
            output_dir,
            row_radius=int(args.side_sdf_row_radius),
        )

    side_sdf_reason = np.asarray(side_sdf_profile_meta.get("reason", "unknown")).item()
    side_sdf_moved_count = int(np.asarray(side_sdf_profile_meta.get("moved_vertex_count", 0)).item())
    side_sdf_mean_push_cm = float(np.asarray(side_sdf_profile_meta.get("mean_abs_push_cm", 0.0)).item())
    side_sdf_max_push_cm = float(np.asarray(side_sdf_profile_meta.get("max_abs_push_cm", 0.0)).item())
    side_sdf_profile_method = np.asarray(side_sdf_profile_meta.get("profile_method", "row_edge_sdf")).item()
    side_sdf_chest_count = int(np.asarray(side_sdf_profile_meta.get("chest_moved_vertex_count", 0)).item())
    side_sdf_chest_mean_cm = float(np.asarray(side_sdf_profile_meta.get("chest_mean_abs_push_cm", 0.0)).item())
    side_sdf_chest_max_cm = float(np.asarray(side_sdf_profile_meta.get("chest_max_abs_push_cm", 0.0)).item())
    side_sdf_chest_contain_count = int(np.asarray(side_sdf_profile_meta.get("chest_containment_moved_vertex_count", 0)).item())
    side_sdf_chest_contain_max_cm = float(np.asarray(side_sdf_profile_meta.get("chest_containment_max_abs_push_cm", 0.0)).item())
    side_sdf_chest_smooth_count = int(np.asarray(side_sdf_profile_meta.get("chest_displacement_smooth_vertex_count", 0)).item())
    side_sdf_chest_smooth_max_cm = float(np.asarray(side_sdf_profile_meta.get("chest_displacement_smooth_max_abs_delta_cm", 0.0)).item())
    side_sdf_butt_count = int(np.asarray(side_sdf_profile_meta.get("butt_moved_vertex_count", 0)).item())
    side_sdf_butt_mean_cm = float(np.asarray(side_sdf_profile_meta.get("butt_mean_abs_push_cm", 0.0)).item())
    side_sdf_butt_max_cm = float(np.asarray(side_sdf_profile_meta.get("butt_max_abs_push_cm", 0.0)).item())
    side_sdf_butt_contain_count = int(np.asarray(side_sdf_profile_meta.get("butt_containment_moved_vertex_count", 0)).item())
    side_sdf_butt_contain_max_cm = float(np.asarray(side_sdf_profile_meta.get("butt_containment_max_abs_push_cm", 0.0)).item())
    side_sdf_butt_smooth_count = int(np.asarray(side_sdf_profile_meta.get("butt_displacement_smooth_vertex_count", 0)).item())
    side_sdf_butt_smooth_max_cm = float(np.asarray(side_sdf_profile_meta.get("butt_displacement_smooth_max_abs_delta_cm", 0.0)).item())

    print("\nSide SDF/profile correction:")
    print(f"  profile method      : {side_sdf_profile_method}")
    print(f"  reason              : {side_sdf_reason}")
    print(f"  moved vertices      : {side_sdf_moved_count}")
    print(f"  mean abs push       : {side_sdf_mean_push_cm:.4f} cm")
    print(f"  max abs push        : {side_sdf_max_push_cm:.4f} cm")
    print(f"  chest moved/max     : {side_sdf_chest_count} / {side_sdf_chest_max_cm:.4f} cm mean {side_sdf_chest_mean_cm:.4f} cm")
    print(f"  chest smooth/max    : {side_sdf_chest_smooth_count} / {side_sdf_chest_smooth_max_cm:.4f} cm")
    print(f"  chest contain/max   : {side_sdf_chest_contain_count} / {side_sdf_chest_contain_max_cm:.4f} cm")
    print(f"  butt moved/max      : {side_sdf_butt_count} / {side_sdf_butt_max_cm:.4f} cm mean {side_sdf_butt_mean_cm:.4f} cm")
    print(f"  butt smooth/max     : {side_sdf_butt_smooth_count} / {side_sdf_butt_smooth_max_cm:.4f} cm")
    print(f"  butt contain/max    : {side_sdf_butt_contain_count} / {side_sdf_butt_contain_max_cm:.4f} cm")

    # ---- orient both meshes into canonical upright space ----
    front_oriented = center_and_orient_mesh(front_vertices)
    side_oriented = center_and_orient_mesh(side_vertices)

    front_upright = front_oriented["vertices_oriented"]
    side_upright = side_oriented["vertices_oriented"]

    print("\nFront mesh:")
    print(f"  estimated up direction : {front_oriented['estimated_up_direction']}")
    print(f"  height before rotation : {float(front_oriented['height_before']):.6f}")
    print(f"  height after rotation  : {float(front_oriented['height_after']):.6f}")

    print("\nSide mesh:")
    print(f"  estimated up direction : {side_oriented['estimated_up_direction']}")
    print(f"  height before rotation : {float(side_oriented['height_before']):.6f}")
    print(f"  height after rotation  : {float(side_oriented['height_after']):.6f}")

    # ---- align side to front in canonical space ----
    side_aligned, R_align, front_centroid, side_centroid = kabsch_align_vertices(
        front_upright,
        side_upright,
    )

    align_dist = np.linalg.norm(side_aligned - front_upright, axis=1)

    print("\nAlignment stats:")
    print(f"  mean aligned distance   : {float(np.mean(align_dist)):.6f}")
    print(f"  median aligned distance : {float(np.median(align_dist)):.6f}")
    print(f"  max aligned distance    : {float(np.max(align_dist)):.6f}")

    # ---- fuse: keep front x,y ; take z from aligned side ----
    fused_vertices = fuse_front_xy_with_side_z(front_upright, side_aligned)
    fused_vertices, profile_depth_meta = enhance_profile_depth_from_front_width(
        fused_vertices,
        strength=float(args.profile_depth_correction_strength),
        max_scale=float(args.profile_depth_correction_max_scale),
    )

    dist_fused_to_front = np.linalg.norm(fused_vertices - front_upright, axis=1)
    dist_fused_to_side = np.linalg.norm(fused_vertices - side_aligned, axis=1)

    print("\nFusion stats:")
    print(f"  mean fused->front distance : {float(np.mean(dist_fused_to_front)):.6f}")
    print(f"  mean fused->side distance  : {float(np.mean(dist_fused_to_side)):.6f}")
    print(
        "  profile depth correction : "
        f"bust x{float(profile_depth_meta['bust_scale']):.3f}, "
        f"hip x{float(profile_depth_meta['hip_scale']):.3f}"
    )

    # ---- scale fused mesh to requested height ----
    target_height_cm = float(args.target_height)
    target_height_m = target_height_cm / 100.0

    fused_vertices_scaled, scale_factor, fused_height_before_scale, fused_height_after_scale = scale_mesh_to_target_height(
        fused_vertices,
        target_height_m,
    )

    fused_result_unscaled = build_fused_result(
        front_result=front_result,
        side_result=side_result,
        fused_vertices=fused_vertices,
        target_height=target_height_cm,
    )
    fused_result_scaled = scale_result_params(fused_result_unscaled, float(scale_factor))
    fused_result_scaled["pred_vertices"] = fused_vertices_scaled.astype(np.float32)
    fused_result_scaled["fusion_applied_scale_factor"] = np.array(float(scale_factor), dtype=np.float64)
    for key, value in profile_depth_meta.items():
        fused_result_scaled[f"fusion_profile_depth_{key}"] = value
    for key, value in side_sdf_profile_meta.items():
        fused_result_scaled[f"fusion_side_sdf_profile_{key}"] = value
    for key, value in side_anchor_meta.items():
        fused_result_scaled[f"fusion_side_anchor_{key}"] = value
    for key, value in side_anchor_paths.items():
        if value:
            fused_result_scaled[f"fusion_side_anchor_{key}_path"] = np.array(value, dtype=object)
    fused_result_scaled["fusion_side_anchor_source"] = np.array(
        "untouched_side_clad_mesh",
        dtype=object,
    )
    if side_mask_path:
        fused_result_scaled["fusion_side_mask_path"] = np.array(side_mask_path, dtype=object)
    if side_sdf_path:
        fused_result_scaled["fusion_side_sdf_path"] = np.array(side_sdf_path, dtype=object)
    if side_sdf_vis_path:
        fused_result_scaled["fusion_side_sdf_visualization_path"] = np.array(side_sdf_vis_path, dtype=object)
    if side_anchor_debug_mask_path:
        fused_result_scaled["fusion_side_anchor_debug_mask_path"] = np.array(side_anchor_debug_mask_path, dtype=object)
    if side_projection_alignment_debug_path:
        fused_result_scaled["fusion_side_projection_alignment_debug_path"] = np.array(side_projection_alignment_debug_path, dtype=object)
    if side_sdf_edited_obj:
        fused_result_scaled["fusion_side_sdf_edited_obj_path"] = np.array(side_sdf_edited_obj, dtype=object)
    if side_sdf_edited_render:
        fused_result_scaled["fusion_side_sdf_edited_render_path"] = np.array(side_sdf_edited_render, dtype=object)
    if side_sdf_row_debug_path:
        fused_result_scaled["fusion_side_sdf_profile_row_debug_path"] = np.array(side_sdf_row_debug_path, dtype=object)

    fused_clad_vertices = sam_upright_vertices_to_clad_canonical(fused_vertices_scaled)
    fused_clad_obj = os.path.join(output_dir, "front_fused_clad_geometry.obj")
    save_mesh_obj(fused_clad_vertices, estimator.faces, fused_clad_obj)
    fused_result_scaled["fusion_apply_profile_depth_to_mhr"] = np.array(
        bool(args.no_fused_vertex_override and float(args.profile_depth_correction_strength) > 0.0),
    )
    fused_result_scaled["fusion_vertices_clad"] = fused_clad_vertices.astype(np.float32)
    fused_result_scaled["fusion_faces_clad"] = np.asarray(estimator.faces, dtype=np.int32)
    fused_result_scaled["fusion_vertices_clad_coordinate_system"] = np.array(
        "x_lateral_y_profile_z_up_meters",
        dtype=object,
    )
    side_sdf_rule_part = (
        "side_sdf_profile_"
        if bool(np.asarray(side_sdf_profile_meta.get("enabled", False)).item())
        else ""
    )
    if args.no_fused_vertex_override:
        fused_result_scaled["fusion_prefer_vertices_for_clad"] = np.array(False)
        fused_result_scaled["fusion_rule"] = np.array(
            f"front_xy_{side_sdf_rule_part}side_z_profile_depth_mhr_restpose",
            dtype=object,
        )
    else:
        fused_result_scaled["fusion_prefer_vertices_for_clad"] = np.array(True)
        fused_result_scaled["fusion_rule"] = np.array(
            f"front_xy_{side_sdf_rule_part}side_z_profile_depth_fused_vertex_mesh",
            dtype=object,
        )

    fused_params_json = os.path.join(output_dir, "front_fused_all_body_params_scaled.json")
    save_result_json(fused_result_scaled, fused_params_json)

    print("\nScaling stats:")
    print(f"  fused height before scale : {float(fused_height_before_scale) * 100.0:.6f} cm")
    print(f"  target fused height       : {float(fused_height_after_scale) * 100.0:.6f} cm")
    print(f"  applied scale factor      : {float(scale_factor):.6f}")

    print("\nDone.")
    print(f"Saved fused params JSON     : {fused_params_json}")
    print(f"Saved fused CLAD OBJ        : {fused_clad_obj}")
    print(f"Saved front render          : {front_raw_render}")
    print(f"Saved side render           : {side_raw_render}")
    if side_mask_path:
        print(f"Saved side mask             : {side_mask_path}")
    if side_sdf_path:
        print(f"Saved side SDF              : {side_sdf_path}")
    if side_sdf_vis_path:
        print(f"Saved side SDF visualization: {side_sdf_vis_path}")
    if side_anchor_debug_mask_path:
        print(f"Saved side anchor debug mask: {side_anchor_debug_mask_path}")
    if side_projection_alignment_debug_path:
        print(f"Saved side projection debug  : {side_projection_alignment_debug_path}")
    if side_sdf_edited_obj:
        print(f"Saved side edited OBJ       : {side_sdf_edited_obj}")
    for key, value in side_anchor_paths.items():
        if value:
            print(f"Saved side anchor {key:<12}: {value}")
    if side_sdf_edited_render:
        print(f"Saved side edited render    : {side_sdf_edited_render}")
    if side_sdf_row_debug_path:
        print(f"Saved side row debug        : {side_sdf_row_debug_path}")


if __name__ == "__main__":
    main()
