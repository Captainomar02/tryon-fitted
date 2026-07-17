"""Lightweight, dependency-free quality gates for front/side body fusion.

The functions in this module deliberately operate on estimator outputs rather
than model internals so they can be unit tested without loading SAM3D.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_scale: float = 0.78
    max_scale: float = 1.28
    max_torso_p95_height_ratio: float = 0.12
    max_full_p95_height_ratio: float = 0.25
    min_mask_iou: float = 0.35
    max_mask_contour_p95_diagonal_ratio: float = 0.15


DEFAULT_THRESHOLDS = QualityThresholds()


def _as_points(value: Any, minimum: int = 1) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < minimum or not np.isfinite(points).all():
        return None
    return points[:, :3]


def validate_result(result: dict[str, Any], mask: np.ndarray | None, label: str) -> list[str]:
    """Return stable rejection codes for one estimator result."""
    errors: list[str] = []
    if _as_points(result.get("pred_vertices"), 20) is None:
        errors.append(f"{label}_invalid_vertices")
    keypoints = _as_points(result.get("pred_keypoints_3d"), 11)
    if keypoints is None:
        errors.append(f"{label}_invalid_keypoints_3d")
    elif np.linalg.norm(keypoints[5] - keypoints[6]) < 1e-5 or np.linalg.norm(keypoints[9] - keypoints[10]) < 1e-5:
        errors.append(f"{label}_degenerate_torso_keypoints")
    if mask is None or mask.ndim != 2 or np.count_nonzero(mask) < 100:
        errors.append(f"{label}_invalid_mask")
    return errors


def torso_mask_from_landmarks(vertices: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Semantic torso-only correspondence mask shared by the fixed topology.

    Shoulder/hip landmarks define a conservative trunk slab.  It intentionally
    removes the mobile limbs and head from the alignment objective.
    """
    v = np.asarray(vertices, dtype=np.float64)
    k = _as_points(keypoints, 11)
    if k is None:
        return np.zeros(len(v), dtype=bool)
    shoulders = k[[5, 6]]
    hips = k[[9, 10]]
    lateral = shoulders[0] - shoulders[1]
    width = float(np.linalg.norm(lateral))
    vertical = shoulders.mean(axis=0) - hips.mean(axis=0)
    height = float(np.linalg.norm(vertical))
    if width < 1e-5 or height < 1e-5:
        return np.zeros(len(v), dtype=bool)
    lateral /= width
    vertical /= height
    center = 0.5 * (shoulders.mean(axis=0) + hips.mean(axis=0))
    x = (v - center) @ lateral
    y = (v - center) @ vertical
    # Wider than the shoulder line, but bounded enough to reject arms.
    return (np.abs(x) <= 0.62 * width) & (y >= -0.72 * height) & (y <= 0.72 * height)


def similarity_align(source: np.ndarray, target: np.ndarray, mask: np.ndarray, trim_fraction: float = 0.12) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust Umeyama similarity alignment of same-topology vertices."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    active = np.asarray(mask, dtype=bool).copy()
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or active.sum() < 20:
        raise ValueError("insufficient_torso_correspondences")
    R = np.eye(3)
    scale = 1.0
    src_center = src[active].mean(axis=0)
    dst_center = dst[active].mean(axis=0)
    for _ in range(3):
        x = src[active]
        y = dst[active]
        src_center = x.mean(axis=0)
        dst_center = y.mean(axis=0)
        xc = x - src_center
        yc = y - dst_center
        H = xc.T @ yc
        U, singular, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        denom = float(np.sum(xc * xc))
        scale = float(singular.sum() / denom) if denom > 1e-10 else 1.0
        aligned = scale * ((src - src_center) @ R.T) + dst_center
        residual = np.linalg.norm(aligned - dst, axis=1)
        candidates = np.flatnonzero(mask)
        keep_count = max(20, int(round(len(candidates) * (1.0 - trim_fraction))))
        kept = candidates[np.argsort(residual[candidates])[:keep_count]]
        active[:] = False
        active[kept] = True
    aligned = scale * ((src - src_center) @ R.T) + dst_center
    return aligned.astype(np.float32), {
        "scale": scale,
        "rotation": R,
        "source_center": src_center,
        "target_center": dst_center,
        "inlier_count": int(active.sum()),
        "inlier_mask": active,
    }


def upright_similarity_align(source: np.ndarray, target: np.ndarray, mask: np.ndarray, trim_fraction: float = 0.12) -> tuple[np.ndarray, dict[str, Any]]:
    """Align same-topology bodies without changing their upright posture.

    Inputs have already been put in fusion space where X is lateral, Y is
    vertical, and Z is profile depth.  A full 3-D Kabsch solve can still add a
    pitch or roll at this stage.  That is particularly damaging here because
    fusion copies the aligned side mesh's Z coordinate into the result.  Fit
    only a yaw about Y, plus uniform scale and translation, so registration
    cannot manufacture a forward/backward lean.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    active = np.asarray(mask, dtype=bool).copy()
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or active.sum() < 20:
        raise ValueError("insufficient_torso_correspondences")

    R = np.eye(3, dtype=np.float64)
    scale = 1.0
    src_center = src[active].mean(axis=0)
    dst_center = dst[active].mean(axis=0)
    for _ in range(3):
        x = src[active]
        y = dst[active]
        src_center = x.mean(axis=0)
        dst_center = y.mean(axis=0)
        xc = x - src_center
        yc = y - dst_center

        # Solve the best 2-D rotation in the horizontal X/Z plane.  This is
        # the only remaining rigid degree of freedom after upright alignment.
        H = xc[:, [0, 2]].T @ yc[:, [0, 2]]
        U, singular, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = Vt.T @ U.T
        R = np.array(
            [[R2[0, 0], 0.0, R2[0, 1]], [0.0, 1.0, 0.0], [R2[1, 0], 0.0, R2[1, 1]]],
            dtype=np.float64,
        )
        rotated = xc @ R.T
        denom = float(np.sum(xc * xc))
        scale = float(np.sum(rotated * yc) / denom) if denom > 1e-10 else 1.0
        aligned = scale * ((src - src_center) @ R.T) + dst_center
        residual = np.linalg.norm(aligned - dst, axis=1)
        candidates = np.flatnonzero(mask)
        keep_count = max(20, int(round(len(candidates) * (1.0 - trim_fraction))))
        kept = candidates[np.argsort(residual[candidates])[:keep_count]]
        active[:] = False
        active[kept] = True

    aligned = scale * ((src - src_center) @ R.T) + dst_center
    return aligned.astype(np.float32), {
        "scale": scale,
        "rotation": R,
        "source_center": src_center,
        "target_center": dst_center,
        "inlier_count": int(active.sum()),
        "inlier_mask": active,
        "rotation_constraint": "yaw_only_preserve_upright",
    }


def alignment_report(front: np.ndarray, aligned_side: np.ndarray, torso_mask: np.ndarray, height: float, transform: dict[str, Any], thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    residual = np.linalg.norm(np.asarray(front) - np.asarray(aligned_side), axis=1)
    h = max(float(height), 1e-6)
    torso = residual[np.asarray(torso_mask, dtype=bool)]
    report = {
        "scale": float(transform["scale"]),
        "rotation_constraint": str(transform.get("rotation_constraint", "unconstrained_3d")),
        "torso_mean_m": float(np.mean(torso)) if torso.size else float("inf"),
        "torso_p95_m": float(np.percentile(torso, 95)) if torso.size else float("inf"),
        "full_p95_m": float(np.percentile(residual, 95)),
    }
    errors: list[str] = []
    if not thresholds.min_scale <= report["scale"] <= thresholds.max_scale:
        errors.append("front_side_scale_mismatch")
    if report["torso_p95_m"] / h > thresholds.max_torso_p95_height_ratio:
        errors.append("front_side_torso_pose_mismatch")
    if report["full_p95_m"] / h > thresholds.max_full_p95_height_ratio:
        errors.append("front_side_limb_pose_mismatch")
    report["errors"] = errors
    return report


def projected_silhouette(vertices: np.ndarray, faces: np.ndarray, cam_t: Any, focal_length: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Rasterize a conservative mesh silhouette in the estimator camera."""
    if cam_t is None or focal_length is None:
        return None
    h, w = image_shape[:2]
    v = np.asarray(vertices, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64).reshape(1, 3)
    z = v[:, 2]
    good = z > 1e-5
    if not np.any(good):
        return None
    xy = np.column_stack([float(focal_length) * v[:, 0] / np.maximum(z, 1e-5) + w / 2, float(focal_length) * v[:, 1] / np.maximum(z, 1e-5) + h / 2])
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in np.asarray(faces, dtype=np.int64):
        if np.any(tri < 0) or np.any(tri >= len(v)) or not np.all(good[tri]):
            continue
        pts = np.rint(xy[tri]).astype(np.int32)
        if np.all((pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)):
            continue
        cv2.fillConvexPoly(mask, pts, 1, lineType=cv2.LINE_AA)
    return mask.astype(bool)


def silhouette_report(predicted: np.ndarray | None, observed: np.ndarray | None, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    if predicted is None or observed is None or not np.any(predicted) or not np.any(observed):
        return {"available": False, "errors": ["silhouette_unavailable"]}
    pred = predicted.astype(bool)
    obs = observed.astype(bool)
    iou = float(np.count_nonzero(pred & obs) / max(1, np.count_nonzero(pred | obs)))
    distance = cv2.distanceTransform((~obs).astype(np.uint8), cv2.DIST_L2, 3)
    boundary = pred & ~cv2.erode(pred.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    distances = distance[boundary]
    diagonal = float(np.hypot(*obs.shape))
    p95 = float(np.percentile(distances, 95)) if distances.size else float("inf")
    errors = []
    if iou < thresholds.min_mask_iou:
        errors.append("silhouette_iou_too_low")
    if p95 / max(diagonal, 1.0) > thresholds.max_mask_contour_p95_diagonal_ratio:
        errors.append("silhouette_contour_mismatch")
    return {"available": True, "iou": iou, "contour_p95_px": p95, "errors": errors}


def make_quality_report(policy: str, errors: list[str], **metrics: Any) -> dict[str, Any]:
    return {
        "validation_policy": policy,
        "status": "passed" if not errors else ("rejected" if policy == "strict" else "untrusted"),
        "errors": sorted(set(errors)),
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "metrics": metrics,
        "retake_guidance": "Use one person, matching neutral front/side poses, a true side view, and clear full-body masks.",
    }
