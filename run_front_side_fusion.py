import glob
import json
import os
import re
import argparse
import shutil

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def sanitize_filename_component(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def save_individual_arrays(save_dict, output_dir, base_name, file_suffix):
    saved_paths = []
    for key, value in save_dict.items():
        if not isinstance(value, np.ndarray):
            continue
        safe_key = sanitize_filename_component(key)
        npy_path = os.path.join(output_dir, f"{base_name}_{file_suffix}_{safe_key}.npy")
        np.save(npy_path, value)
        saved_paths.append(npy_path)
    return saved_paths


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


def save_result_files(result, output_dir, base_name, file_suffix="all_body_params", faces=None, mesh_vertices=None):
    save_dict = convert_result_to_save_dict(result)

    npz_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.npz")
    np.savez(npz_path, **save_dict)

    json_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.json")
    json_dict = {}
    for k, v in save_dict.items():
        if isinstance(v, np.ndarray):
            try:
                json_dict[k] = v.tolist()
            except Exception:
                json_dict[k] = None
        elif isinstance(v, np.generic):
            json_dict[k] = v.item()
        else:
            json_dict[k] = v

    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    saved_paths = [npz_path, json_path]
    saved_paths.extend(save_individual_arrays(save_dict, output_dir, base_name, file_suffix))

    mesh_vertices = mesh_vertices if mesh_vertices is not None else save_dict.get("pred_vertices")
    if faces is not None and mesh_vertices is not None:
        obj_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.obj")
        try:
            saved_paths.append(save_mesh_obj(mesh_vertices, faces, obj_path))
        except Exception as e:
            print(f"[fusion] Warning: could not save OBJ for {base_name}: {e}")

    if "mhr_model_params" in result:
        mhr_val = result["mhr_model_params"]
        if isinstance(mhr_val, torch.Tensor):
            mhr_val = mhr_val.detach().cpu().numpy()
        mhr_path = os.path.join(output_dir, f"{base_name}_{file_suffix}_mhr_model_params.npy")
        np.save(mhr_path, mhr_val)
        saved_paths.append(mhr_path)

    return saved_paths


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

    for name in os.listdir(output_dir):
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

    up_dir = high_center - low_center
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


def copy_input_image(image_path, output_dir, out_name):
    out_path = os.path.join(output_dir, out_name)
    shutil.copy2(image_path, out_path)
    return out_path


def save_mesh_preview(vertices, save_path, title):
    v = vertices.astype(np.float64)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.5)

    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = np.max(maxs - mins)
    if extent < 1e-8:
        extent = 1.0
    half = extent / 2.0

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_xlabel("X")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=10, azim=-90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


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

    print(f"Front image: {front_image_path}")
    print(f"Side image : {side_image_path}")
    print(f"Target fused mesh height: {args.target_height}")

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
        use_mask=False,
    )
    if not (isinstance(front_outputs, list) and len(front_outputs) > 0 and isinstance(front_outputs[0], dict)):
        raise RuntimeError("Front inference did not return expected outputs[0] dict")
    front_result = front_outputs[0]

    # ---- run side ----
    print("\nRunning side image...")
    side_outputs = estimator.process_one_image(
        side_image_path,
        bbox_thr=0.8,
        use_mask=False,
    )
    if not (isinstance(side_outputs, list) and len(side_outputs) > 0 and isinstance(side_outputs[0], dict)):
        raise RuntimeError("Side inference did not return expected outputs[0] dict")
    side_result = side_outputs[0]

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

    dist_fused_to_front = np.linalg.norm(fused_vertices - front_upright, axis=1)
    dist_fused_to_side = np.linalg.norm(fused_vertices - side_aligned, axis=1)

    print("\nFusion stats:")
    print(f"  mean fused->front distance : {float(np.mean(dist_fused_to_front)):.6f}")
    print(f"  mean fused->side distance  : {float(np.mean(dist_fused_to_side)):.6f}")

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

    fused_params_json = os.path.join(output_dir, "front_fused_all_body_params_scaled.json")
    save_result_json(fused_result_scaled, fused_params_json)

    print("\nScaling stats:")
    print(f"  fused height before scale : {float(fused_height_before_scale) * 100.0:.6f} cm")
    print(f"  target fused height       : {float(fused_height_after_scale) * 100.0:.6f} cm")
    print(f"  applied scale factor      : {float(scale_factor):.6f}")

    print("\nDone.")
    print(f"Saved fused params JSON     : {fused_params_json}")
    print(f"Saved front render          : {front_raw_render}")
    print(f"Saved side render           : {side_raw_render}")


if __name__ == "__main__":
    main()
