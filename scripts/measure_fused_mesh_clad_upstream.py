#!/usr/bin/env python3
"""Measure a fused mesh using a clean upstream CLAD Body checkout.

The mesh remains the fused SDF surface. Linear joint landmarks come from the
front MHR parameters retained in the fusion JSON and are transformed into the
fused mesh's canonical measurement frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CLAD_ROOT = REPO_ROOT / "vendor" / "clad-body-upstream"
if not UPSTREAM_CLAD_ROOT.is_dir():
    raise RuntimeError(f"Upstream CLAD checkout is missing: {UPSTREAM_CLAD_ROOT}")
sys.path.insert(0, str(UPSTREAM_CLAD_ROOT))

from clad_body.load.mhr import MhrBody, load_mhr_from_params  # noqa: E402
from clad_body.measure import measure  # noqa: E402
from clad_body.measure._render import render_4view  # noqa: E402
import clad_body.measure._lengths as upstream_lengths  # noqa: E402


def scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item() if value.size else default
    return default if value is None else value


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, trimesh.Trimesh):
        return None
    if isinstance(value, dict):
        return {str(k): converted for k, v in value.items() if (converted := jsonable(v)) is not None}
    if isinstance(value, (list, tuple)):
        return [converted for v in value if (converted := jsonable(v)) is not None]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shaped [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shaped [M, 3], got {faces.shape}")
    if not len(vertices) or not len(faces):
        raise ValueError("Fused mesh is empty")
    if faces.min() < 0 or faces.max() >= len(vertices):
        if faces.min() == 1 and faces.max() == len(vertices):
            faces = faces - 1
        else:
            raise ValueError("Fused mesh faces do not index its vertices")
    return vertices, faces


def read_source(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        params = json.loads(path.read_text(encoding="utf-8"))
        if "fusion_vertices_clad" not in params or "fusion_faces_clad" not in params:
            raise ValueError("This baseline requires a fused JSON containing fusion_vertices_clad/fusion_faces_clad")
        vertices, faces = mesh_arrays(params["fusion_vertices_clad"], params["fusion_faces_clad"])
        return vertices, faces, params
    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [part for part in mesh.geometry.values() if isinstance(part, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(parts)
    vertices, faces = mesh_arrays(mesh.vertices, mesh.faces)
    return vertices, faces, {}


def target_height_m(params: dict[str, Any], height_cm: float | None, vertices: np.ndarray) -> float:
    if height_cm and height_cm > 0:
        return height_cm / 100.0
    for key in ("fusion_target_height_cm", "fusion_target_height"):
        value = scalar(params.get(key))
        if value is not None:
            return float(value) / 100.0
    return float(np.ptp(vertices[:, 2]))


def canonicalize_with_transform(vertices: np.ndarray, height_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    current_height = float(np.ptp(vertices[:, 2]))
    scale = height_m / current_height if current_height > 1e-8 and height_m > 1e-8 else 1.0
    canonical = vertices * scale
    z_offset = float(canonical[:, 2].min())
    canonical[:, 2] -= z_offset
    center_xy = (canonical[:, :2].min(axis=0) + canonical[:, :2].max(axis=0)) * 0.5
    canonical[:, :2] -= center_xy
    return canonical.astype(np.float32), {
        "scale": scale, "z_offset_m": z_offset, "xy_center_offset_m": center_xy.tolist()
    }


def transform_front_joints(params_path: Path, height_m: float, yaw_rotation: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild front MHR joints in the fused mesh's yaw-normalized frame."""
    front_body = load_mhr_from_params(str(params_path))
    canonical_vertices, transform = canonicalize_with_transform(np.asarray(front_body.mesh.vertices), height_m)
    scale = float(transform["scale"])
    center_xy = np.asarray(transform["xy_center_offset_m"], dtype=np.float32)
    yaw_center_xy = np.zeros(2, dtype=np.float32)
    clad_yaw = None
    if yaw_rotation is not None:
        # The saved yaw is in fusion space (X lateral, Y up, Z depth).
        # CLAD is X lateral, Y depth, Z up, so conjugate it into a Z-axis
        # rotation before touching CLAD-space MHR joints.
        c = float(yaw_rotation[0, 0])
        s = float(yaw_rotation[0, 2])
        clad_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Fusion rotates its centred front mesh and then recentres the rotated
        # bounds. Apply that same second centring step to every transferred joint.
        rotated_vertices = canonical_vertices.astype(np.float64) @ clad_yaw.T
        yaw_center_xy = (
            rotated_vertices[:, :2].min(axis=0) + rotated_vertices[:, :2].max(axis=0)
        ) * 0.5
    joints = {}
    for name, point in (front_body.joints or {}).items():
        point = np.asarray(point, dtype=np.float64) * scale
        point[2] -= float(transform["z_offset_m"])
        point[:2] -= center_xy
        if clad_yaw is not None:
            point = point @ clad_yaw.T
            point[:2] -= yaw_center_xy
        joints[name] = point.astype(np.float32)
    if not joints:
        raise RuntimeError("Upstream MHR loader did not provide front joint landmarks")
    transform["yaw_rotation_applied"] = yaw_rotation.tolist() if yaw_rotation is not None else None
    transform["clad_yaw_rotation_applied"] = clad_yaw.tolist() if clad_yaw is not None else None
    transform["yaw_post_rotation_xy_center_m"] = yaw_center_xy.tolist()
    return joints, transform

def fused_body(source: Path, height_cm: float | None) -> tuple[MhrBody, dict[str, Any], dict[str, Any]]:
    raw_vertices, faces, params = read_source(source)
    if not params:
        raise ValueError("Front MHR landmarks require the fused parameter JSON, not a standalone OBJ")
    height_m = target_height_m(params, height_cm, raw_vertices)
    vertices, fused_transform = canonicalize_with_transform(raw_vertices, height_m)
    raw_yaw = params.get("fusion_yaw_normalization_rotation_matrix")
    yaw_rotation = np.asarray(raw_yaw, dtype=np.float32) if raw_yaw is not None else None
    if yaw_rotation is not None and yaw_rotation.shape != (3, 3):
        yaw_rotation = None
    joints, front_transform = transform_front_joints(source, height_m, yaw_rotation)
    body = MhrBody(
        mesh=trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        source=f"fused_sdf_with_front_mhr_joints:{source.name}",
        obj_path=str(source),
        sam3d_params=params,
        joints=joints,
    )
    return body, fused_transform, front_transform





def install_landmark_constrained_neck_start(joints: dict[str, np.ndarray], mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Install a shirt start finder constrained by C7 and the lateral shoulder.

    The selected slice is the strongest shoulder-to-neck contraction above C7,
    not a fixed percentage of a particular person's shoulder width.
    """
    meta: dict[str, Any] = {"method": "landmark_constrained_neck_transition", "confidence": "fallback"}
    c7 = np.asarray(joints.get("c7"), dtype=np.float64) if joints.get("c7") is not None else None
    shoulders = [np.asarray(joints[name], dtype=np.float64) for name in ("l_shoulder", "r_shoulder") if joints.get(name) is not None]
    side_shoulder = max(shoulders, key=lambda point: point[0]) if shoulders else None
    body_height = float(np.ptp(np.asarray(mesh.vertices)[:, 2]))

    def fallback(slicer, c7_z: float) -> np.ndarray | None:
        # Conservative low-confidence fallback: stay close to C7 rather than
        # returning the outer shoulder cap.
        z = float(c7_z + 0.015 * body_height) if body_height > 0 else float(c7_z)
        contours = slicer.contours_at_z(z)
        if not contours:
            return None
        points = np.vstack([points for points, _, _ in contours])
        c7_x = float(c7[0]) if c7 is not None else 0.0
        target_x = c7_x + 0.045 * body_height
        right = points[points[:, 0] > c7_x]
        if not len(right):
            return None
        point = right[np.argmin(np.abs(right[:, 0] - target_x))]
        return np.array([float(point[0]), float(point[1]), z], dtype=np.float64)

    def neck_start(slicer, c7_z: float) -> np.ndarray | None:
        if c7 is None or side_shoulder is None or body_height <= 0:
            meta.update({"confidence": "fallback", "reason": "missing_c7_or_shoulder"})
            return fallback(slicer, c7_z)

        # Search only above C7 and no higher than the lower neck. The range is
        # height-scaled and bounded by the actual C7-to-shoulder separation.
        shoulder_drop = max(float(c7[2] - side_shoulder[2]), 0.0)
        scan_height = min(0.045 * body_height, max(0.020 * body_height, shoulder_drop * 0.80))
        z_values = np.linspace(float(c7_z) + 0.003 * body_height, float(c7_z) + scan_height, 17)
        candidates = []
        for z in z_values:
            contours = slicer.contours_at_z(float(z))
            if not contours:
                continue
            points = np.vstack([points for points, _, _ in contours])
            right = points[points[:, 0] > float(c7[0])]
            if not len(right):
                continue
            edge_x = float(right[:, 0].max())
            # The anterior point on the lateral neck boundary, rather than a
            # posterior point at the same X.
            outer = right[right[:, 0] >= edge_x - 0.010 * body_height]
            point = outer[np.argmin(outer[:, 1])]
            candidates.append(np.array([edge_x, float(point[1]), float(z)], dtype=np.float64))

        if len(candidates) < 5:
            meta.update({"confidence": "fallback", "reason": "insufficient_neck_slices"})
            return fallback(slicer, c7_z)
        candidates = np.asarray(candidates)
        lateral_slope = np.gradient(candidates[:, 0], candidates[:, 2])
        # The end of the strongest inward contraction is the neck base.
        transition = int(np.argmax(-lateral_slope))
        transition = min(transition + 1, len(candidates) - 1)
        selected = candidates[transition]

        shoulder_span = max(float(side_shoulder[0] - c7[0]), 1e-6)
        valid = (
            selected[2] > c7[2]
            and selected[0] > c7[0]
            and selected[0] < side_shoulder[0] + 0.25 * shoulder_span
            and float(-lateral_slope[transition - 1]) > 0.02
        )
        if not valid:
            meta.update({"confidence": "fallback", "reason": "landmark_validation_failed"})
            return fallback(slicer, c7_z)
        meta.update({
            "confidence": "high",
            "reason": "largest_lateral_contraction_between_c7_and_shoulder",
            "start_point_m": selected.tolist(),
            "c7_m": c7.tolist(),
            "lateral_shoulder_m": side_shoulder.tolist(),
            "scan_height_m": float(scan_height),
        })
        return selected

    upstream_lengths.find_side_neck_point = neck_start
    return meta




def _arm_group_surface_anchor(vertices: np.ndarray, mask: np.ndarray, center: np.ndarray, side_sign: float) -> np.ndarray:
    """Choose the lateral skin point from a topology-labelled arm region."""
    points = vertices[mask]
    score = (
        1.5 * side_sign * (points[:, 0] - center[0])
        - 0.75 * np.abs(points[:, 2] - center[2])
        - 0.10 * np.abs(points[:, 1] - center[1])
    )
    return points[int(np.argmax(score))].astype(np.float64)


def apply_surface_sleeve_length(measurements: dict[str, Any], body: MhrBody) -> None:
    """Measure/render sleeve from topology-labelled fused arm skin regions."""
    if "sleeve_length_cm" not in measurements or not body.joints:
        return
    from clad_body.measure.mhr import find_acromion
    from measure_fused_mesh_clad import mhr_arm_skinning_groups

    vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    groups = mhr_arm_skinning_groups(len(vertices))
    if groups is None:
        measurements["_sleeve_length_source"] = "upstream_joint_chain_fallback"
        return

    traces = []
    for prefix, side_name, side_sign in (("l", "left", 1.0), ("r", "right", -1.0)):
        shoulder = body.joints.get(f"{prefix}_shoulder")
        if shoulder is None:
            continue
        elbow_mask = (groups[f"{prefix}_upper"] > 0.05) & (groups[f"{prefix}_lower"] > 0.05)
        wrist_mask = (groups[f"{prefix}_lower"] > 0.05) & (groups[f"{prefix}_wrist"] > 0.05)
        if np.count_nonzero(elbow_mask) < 8 or np.count_nonzero(wrist_mask) < 8:
            continue
        elbow_center = np.median(vertices[elbow_mask], axis=0)
        wrist_center = np.median(vertices[wrist_mask], axis=0)
        acromion = find_acromion(vertices, np.asarray(shoulder), side=side_name)
        elbow_skin = _arm_group_surface_anchor(vertices, elbow_mask, elbow_center, side_sign)
        wrist_skin = _arm_group_surface_anchor(vertices, wrist_mask, wrist_center, side_sign)
        trace = np.asarray([acromion, elbow_skin, wrist_skin], dtype=np.float64)
        length_cm = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum() * 100.0)
        traces.append((prefix, length_cm, trace))

    if not traces:
        measurements["_sleeve_length_source"] = "upstream_joint_chain_fallback"
        return
    mean_cm = float(np.mean([item[1] for item in traces]))
    shown = min(traces, key=lambda item: abs(item[1] - mean_cm))
    measurements["sleeve_length_cm"] = mean_cm
    measurements["_sleeve_length_source"] = "mhr_skinning_topology_outer_arm_surface_chain"
    measurements["_sleeve_rendered_side"] = shown[0]
    polylines = measurements.get("_linear_polylines")
    if isinstance(polylines, dict):
        polylines["sleeve_length"] = shown[2].astype(np.float32)
    debug_joints = measurements.get("_debug_joints")
    if isinstance(debug_joints, dict):
        prefix = shown[0]
        debug_joints[f"{prefix}_shoulder"] = shown[2][0].astype(np.float32)
        debug_joints[f"{prefix}_elbow"] = shown[2][1].astype(np.float32)
        debug_joints[f"{prefix}_wrist"] = shown[2][2].astype(np.float32)



def apply_distinct_stomach_measurement(measurements: dict[str, Any], body: MhrBody) -> None:
    """Keep stomach above the hip level so both contours remain meaningful."""
    if "stomach_cm" not in measurements:
        return
    from clad_body.measure._slicer import torso_circumference_at_z

    hip_z = float(measurements.get("_hip_z", 0.0))
    waist_z = float(measurements.get("_waist_z", 0.0))
    vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    height = float(np.ptp(vertices[:, 2]))
    lower = hip_z + 0.020 * height
    upper = waist_z - 0.010 * height
    if lower >= upper:
        measurements["_stomach_measurement_source"] = "upstream_hip_level_fallback"
        return

    best_z, best_front_y = None, float("inf")
    for z in np.arange(lower, upper, 0.002):
        band = vertices[(np.abs(vertices[:, 2] - z) < 0.002) & (np.abs(vertices[:, 0]) < 0.35)]
        if len(band) < 3:
            continue
        front_y = float(band[:, 1].min())
        if front_y < best_front_y:
            best_z, best_front_y = float(z), front_y
    if best_z is None:
        measurements["_stomach_measurement_source"] = "upstream_hip_level_fallback"
        return
    circumference = torso_circumference_at_z(body.mesh, best_z)
    if circumference <= 0.30:
        measurements["_stomach_measurement_source"] = "upstream_hip_level_fallback"
        return
    measurements["stomach_cm"] = float(circumference * 100.0)
    measurements["_stomach_z"] = best_z
    measurements["_stomach_pct"] = best_z / height * 100.0
    measurements["_stomach_measurement_source"] = "above_hip_anterior_protrusion"





def apply_surface_crotch_length(measurements: dict[str, Any], body: MhrBody) -> None:
    """Trace rises over the mesh surface through an actual perineum point."""
    import networkx as nx

    front = measurements.get("_crotch_front_pts")
    back = measurements.get("_crotch_back_pts")
    if front is None or back is None:
        return
    front = np.asarray(front, dtype=np.float64).copy()
    back = np.asarray(back, dtype=np.float64).copy()
    if len(front) < 2 or len(back) < 2:
        return
    interior = (front[-1] + back[-1]) * 0.5
    vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    height = float(np.ptp(vertices[:, 2]))
    candidates = vertices[
        (np.abs(vertices[:, 0] - interior[0]) < 0.05 * height)
        & (vertices[:, 2] < interior[2])
        & (vertices[:, 2] > interior[2] - 0.12 * height)
    ]
    if len(candidates) == 0:
        measurements["_crotch_length_source"] = "upstream_interior_midpoint_fallback"
        return
    perineum = candidates[np.argmin(np.linalg.norm(candidates - interior, axis=1))]
    if float(np.linalg.norm(perineum - interior)) > 0.12 * height:
        measurements["_crotch_length_source"] = "upstream_interior_midpoint_fallback"
        return

    try:
        graph = body.mesh.vertex_adjacency_graph
        perineum_index = int(np.argmin(np.linalg.norm(vertices - perineum, axis=1)))

        def surface_tail(start: np.ndarray) -> np.ndarray:
            start_index = int(np.argmin(np.linalg.norm(vertices - start, axis=1)))
            path = nx.shortest_path(graph, start_index, perineum_index, weight="weight")
            return vertices[np.asarray(path, dtype=np.int64)]

        # Drop the upstream interior endpoint, then continue over the actual
        # surface from the final valid front/back samples to the perineum.
        front_tail = surface_tail(front[-2])
        back_tail = surface_tail(back[-2])
        if len(front_tail) < 2 or len(back_tail) < 2:
            raise RuntimeError("surface path too short")
        front = np.vstack([front[:-1], front_tail[1:]])
        back = np.vstack([back[:-1], back_tail[1:]])
    except Exception:
        front[-1] = perineum
        back[-1] = perineum
        measurements["_crotch_length_source"] = "perineum_surface_anchor_direct_fallback"
    else:
        measurements["_crotch_length_source"] = "perineum_surface_geodesic_trace"

    front_cm = float(np.linalg.norm(np.diff(front, axis=0), axis=1).sum() * 100.0)
    back_cm = float(np.linalg.norm(np.diff(back, axis=0), axis=1).sum() * 100.0)
    measurements.update({
        "front_rise_cm": front_cm,
        "back_rise_cm": back_cm,
        "crotch_length_cm": front_cm + back_cm,
        "_crotch_front_pts": front.astype(np.float32),
        "_crotch_back_pts": back.astype(np.float32),
        "_crotch_perineum_point_m": perineum.astype(np.float32),
    })
    polylines = measurements.get("_linear_polylines")
    if isinstance(polylines, dict):
        polylines["front_rise"] = front.astype(np.float32)
        polylines["back_rise"] = back.astype(np.float32)
        polylines["crotch_length"] = np.concatenate([front, back[::-1]]).astype(np.float32)





def _quality(confidence: str, *reasons: str) -> dict[str, Any]:
    return {"confidence": confidence, "reasons": [reason for reason in reasons if reason]}



def _profile_peak_index(
    zs: np.ndarray,
    lateral_extent: np.ndarray,
    low: float,
    high: float,
    *,
    choose: str,
) -> tuple[int | None, str | None]:
    """Select a non-boundary lateral prominence in an anatomical height band."""
    in_band = np.flatnonzero((zs >= low) & (zs <= high) & np.isfinite(lateral_extent))
    if len(in_band) < 7:
        return None, "insufficient_side_profile"
    smooth = np.convolve(lateral_extent, np.ones(5, dtype=np.float64) / 5.0, mode="same")
    candidates = [
        int(i) for i in in_band[2:-2]
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
    ]
    if not candidates:
        return None, "no_lateral_prominence"
    index = max(candidates, key=lambda i: zs[i]) if choose == "highest" else min(candidates, key=lambda i: zs[i])
    if index <= in_band[0] + 2 or index >= in_band[-1] - 2:
        return None, "anchor_at_search_boundary"
    return index, None


def infer_iso_waist_from_side_profiles(
    zs: np.ndarray,
    left_extent: np.ndarray,
    right_extent: np.ndarray,
    height: float,
) -> tuple[dict[str, float] | None, list[str]]:
    """Infer ISO 8559 waist level from bilateral lateral torso profiles."""
    if height <= 0 or len(zs) != len(left_extent) or len(zs) != len(right_extent):
        return None, ["invalid_side_profile"]
    # These are anatomy-bounded search windows, not a fixed-percent waist.
    iliac_low, iliac_high = 0.48 * height, 0.60 * height
    rib_low, rib_high = 0.63 * height, 0.75 * height
    anchors: dict[str, float] = {}
    reasons: list[str] = []
    for side, extent in (("left", left_extent), ("right", right_extent)):
        iliac_idx, error = _profile_peak_index(zs, extent, iliac_low, iliac_high, choose="highest")
        if error:
            reasons.append(f"{side}_iliac_{error}")
            continue
        rib_idx, error = _profile_peak_index(zs, extent, rib_low, rib_high, choose="lowest")
        if error:
            reasons.append(f"{side}_rib_{error}")
            continue
        iliac_z, rib_z = float(zs[iliac_idx]), float(zs[rib_idx])
        if rib_z <= iliac_z or not (0.06 * height <= rib_z - iliac_z <= 0.24 * height):
            reasons.append(f"{side}_invalid_anchor_ordering")
            continue
        anchors[f"{side}_iliac_z"] = iliac_z
        anchors[f"{side}_rib_z"] = rib_z
        anchors[f"{side}_waist_z"] = (iliac_z + rib_z) / 2.0
    if len(anchors) != 6:
        return None, reasons or ["anatomical_anchors_not_found"]
    bilateral_gap = abs(anchors["left_waist_z"] - anchors["right_waist_z"])
    if bilateral_gap > 0.015 * height:
        return None, ["bilateral_disagreement"]
    anchors["waist_z"] = (anchors["left_waist_z"] + anchors["right_waist_z"]) / 2.0
    return anchors, []


def find_iso_waist_landmarks(torso_mesh: trimesh.Trimesh, height: float) -> tuple[dict[str, Any] | None, list[str]]:
    """Locate the ISO lower-rib/iliac-crest side anchors on a torso mesh."""
    from clad_body.measure._slicer import MeshSlicer

    slicer = MeshSlicer(torso_mesh)
    zs = np.arange(0.47 * height, 0.76 * height, 0.002)
    left_extent = np.full(len(zs), np.nan, dtype=np.float64)
    right_extent = np.full(len(zs), np.nan, dtype=np.float64)
    left_points: dict[int, np.ndarray] = {}
    right_points: dict[int, np.ndarray] = {}
    for i, z in enumerate(zs):
        contours = slicer.contours_at_z(float(z))
        if len(contours) != 1:
            continue
        points = np.asarray(contours[0][0], dtype=np.float64)
        if len(points) < 8:
            continue
        left_i, right_i = int(np.argmax(points[:, 0])), int(np.argmin(points[:, 0]))
        left_points[i] = np.asarray([points[left_i, 0], points[left_i, 1], z], dtype=np.float64)
        right_points[i] = np.asarray([points[right_i, 0], points[right_i, 1], z], dtype=np.float64)
        left_extent[i] = abs(points[left_i, 0])
        right_extent[i] = abs(points[right_i, 0])

    inferred, reasons = infer_iso_waist_from_side_profiles(zs, left_extent, right_extent, height)
    if inferred is None:
        return None, reasons
    try:
        anchors = {
            "left": {
                "iliac_crest": left_points[int(np.argmin(abs(zs - inferred["left_iliac_z"])) )],
                "lowest_rib": left_points[int(np.argmin(abs(zs - inferred["left_rib_z"])) )],
            },
            "right": {
                "iliac_crest": right_points[int(np.argmin(abs(zs - inferred["right_iliac_z"])) )],
                "lowest_rib": right_points[int(np.argmin(abs(zs - inferred["right_rib_z"])) )],
            },
            "waist_z": inferred["waist_z"],
        }
    except KeyError:
        return None, ["anatomical_anchor_contour_missing"]
    return anchors, []



def truncate_polyline_at_fraction(points: np.ndarray, fraction: float) -> tuple[float, np.ndarray] | tuple[float, None]:
    """Return the leading fraction of a polyline, including an interpolated endpoint."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or not 0.0 < fraction <= 1.0:
        return 0.0, None
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total <= 0:
        return 0.0, None
    target = total * fraction
    cumulative = np.cumsum(segment_lengths)
    segment = int(np.searchsorted(cumulative, target, side="left"))
    before = float(cumulative[segment - 1]) if segment else 0.0
    local = (target - before) / float(segment_lengths[segment])
    endpoint = points[segment] + local * (points[segment + 1] - points[segment])
    return target * 100.0, np.vstack([points[:segment + 1], endpoint]).astype(np.float32)
def _clear_waist_dependent_measurements(measurements: dict[str, Any]) -> None:
    """Remove provisional upstream values that relied on a fixed-percent waist."""
    for key in (
        "waist_cm", "_waist_z", "_waist_pct", "stomach_cm", "_stomach_z", "_stomach_pct", "_stomach_measurement_source",
        "side_neck_to_waist_cm", "shirt_length_cm", "_side_neck_to_waist_pts",
        "_shirt_length_pts", "_shirt_length_source", "_shirt_length_deprecated_alias",
    ):
        measurements.pop(key, None)
    polylines = measurements.get("_linear_polylines")
    if isinstance(polylines, dict):
        polylines.pop("shirt_length", None)
def apply_production_core_measurements(measurements: dict[str, Any], body: MhrBody) -> trimesh.Trimesh | None:
    """Publish validated torso, perineum, and side-neck-to-waist measurements."""
    from measure_fused_mesh_clad import arm_excluded_torso_mesh
    from clad_body.measure._slicer import MeshSlicer
    from clad_body.measure._lengths import measure_shirt_length

    quality: dict[str, Any] = {}
    torso_mesh, torso_meta = arm_excluded_torso_mesh(body.mesh)
    height = float(np.ptp(np.asarray(body.mesh.vertices)[:, 2]))
    if torso_mesh is None or height <= 0:
        for name in ("bust", "waist", "stomach", "hip", "inseam", "side_neck_to_waist"):
            quality[name] = _quality("low", "arm_excluded_torso_unavailable")
        measurements["_measurement_quality"] = quality
        return None

    slicer = MeshSlicer(torso_mesh)

    def select_max(name: str, low: float, high: float, extent: float) -> tuple[float, float, dict[str, Any]]:
        zs = np.arange(low, high, 0.002)
        circs = np.asarray([
            slicer.circumference_at_z(float(z), max_x_extent=extent, combine_fragments=True)
            for z in zs
        ])
        valid = circs > 0.30
        if not valid.any():
            return 0.0, 0.0, _quality("low", "no_valid_torso_contour")
        index = int(np.argmax(np.where(valid, circs, -1.0)))
        z = float(zs[index])
        fragments = len(slicer.contours_at_z(z))
        edge = index <= 2 or index >= len(zs) - 3
        confidence = "high" if fragments == 1 and not edge else "medium"
        reasons = []
        if fragments != 1:
            reasons.append("fragmented_slice")
        if edge:
            reasons.append("selected_search_boundary")
        return float(circs[index]), z, _quality(confidence, *reasons)

    bust_m, bust_z, bust_q = select_max("bust", 0.68 * height, 0.76 * height, 0.85)
    hip_m, hip_z, hip_q = select_max("hip", 0.46 * height, 0.54 * height, 0.95)
    if bust_m:
        measurements.update({"bust_cm": bust_m * 100.0, "_bust_z": bust_z, "_bust_pct": bust_z / height * 100.0})
    if hip_m:
        measurements.update({"hip_cm": hip_m * 100.0, "_hip_z": hip_z, "_hip_pct": hip_z / height * 100.0})

    _clear_waist_dependent_measurements(measurements)
    waist_landmarks, waist_reasons = find_iso_waist_landmarks(torso_mesh, height)
    waist_z: float | None = None
    if waist_landmarks is None:
        waist_q = _quality("low", "anatomical_anchors_not_found", *waist_reasons)
        measurements["_waist_landmarks"] = {"available": False, "reasons": waist_q["reasons"]}
    else:
        candidate_z = float(waist_landmarks["waist_z"])
        contours = slicer.contours_at_z(candidate_z)
        waist_m = float(slicer.circumference_at_z(candidate_z, max_x_extent=0.85, combine_fragments=True))
        if len(contours) != 1 or waist_m <= 0.30:
            waist_q = _quality("low", "fragmented_contour" if len(contours) != 1 else "invalid_waist_contour")
            measurements["_waist_landmarks"] = {"available": False, "reasons": waist_q["reasons"]}
        else:
            waist_z = candidate_z
            measurements.update({
                "waist_cm": waist_m * 100.0,
                "_waist_z": waist_z,
                "_waist_pct": waist_z / height * 100.0,
                "_waist_landmarks": {
                    "available": True,
                    "definition": "iso_8559_1_midpoint_lowest_rib_to_highest_iliac_crest",
                    "left": waist_landmarks["left"],
                    "right": waist_landmarks["right"],
                },
            })
            waist_q = _quality("high", "bilateral_iso_rib_iliac_midpoint")

    stomach_q = _quality("low", "waist_unavailable")
    if waist_z is not None and hip_z > 0 and waist_z > hip_z:
        lo, hi = hip_z + 0.020 * height, waist_z - 0.010 * height
        best_z, best_y = 0.0, float("inf")
        for z in np.arange(lo, hi, 0.002):
            contours = slicer.contours_at_z(float(z))
            if len(contours) != 1:
                continue
            pts = contours[0][0]
            y = float(pts[:, 1].min())
            if y < best_y:
                best_z, best_y = float(z), y
        if best_z:
            stomach_m = float(slicer.circumference_at_z(best_z, max_x_extent=0.85, combine_fragments=True))
            if stomach_m > 0.30:
                measurements.update({"stomach_cm": stomach_m * 100.0, "_stomach_z": best_z, "_stomach_pct": best_z / height * 100.0})
                stomach_q = _quality("high" if abs(best_z - hip_z) > 0.010 * height else "medium", "above_hip_anterior_protrusion")
    perineum = measurements.get("_crotch_perineum_point_m")
    if perineum is not None:
        perineum = np.asarray(perineum, dtype=np.float64)
        perineum_z = float(perineum[2])
        measurements.update({"_perineum_z": perineum_z, "_inseam_z": perineum_z, "_inseam_pct": perineum_z / height * 100.0, "inseam_cm": perineum_z * 100.0})
        crotch_reason = str(measurements.get("_crotch_length_source", ""))
        inseam_q = _quality("high" if "geodesic" in crotch_reason else "medium", crotch_reason)
    else:
        inseam_q = _quality("low", "perineum_surface_not_found")

    for key in ("side_neck_to_waist_cm", "shirt_length_cm", "_side_neck_to_waist_pts", "_shirt_length_pts", "_shirt_length_source", "_shirt_length_deprecated_alias"):
        measurements.pop(key, None)
    polylines = measurements.get("_linear_polylines")
    if isinstance(polylines, dict):
        polylines.pop("shirt_length", None)
    # Regular-fit T-shirt target: 90% of the side-neck/HPS-to-crotch-level
    # front surface trace. It deliberately has no waist or hip dependency.
    hps_to_crotch_cm, hps_to_crotch_pts = (0.0, None)
    crotch_z = float(measurements.get("_inseam_z", 0.0))
    if crotch_z > 0:
        hps_to_crotch_cm, hps_to_crotch_pts = measure_shirt_length(
            body.joints or {}, body.mesh, crotch_z, measurements=measurements,
            end_offset=0.0,
        )
    regular_tshirt_cm, regular_tshirt_pts = truncate_polyline_at_fraction(hps_to_crotch_pts, 0.90) if hps_to_crotch_pts is not None else (0.0, None)
    if regular_tshirt_pts is not None and regular_tshirt_cm > 0:
        measurements["hps_to_crotch_cm"] = float(hps_to_crotch_cm)
        measurements["regular_tshirt_length_cm"] = float(regular_tshirt_cm)
        measurements["shirt_length_cm"] = float(regular_tshirt_cm)
        measurements["_hps_to_crotch_pts"] = hps_to_crotch_pts
        measurements["_regular_tshirt_length_pts"] = regular_tshirt_pts
        measurements["_shirt_length_pts"] = regular_tshirt_pts
        measurements["_shirt_length_source"] = "regular_fit_90pct_hps_to_crotch_surface_trace"
        measurements["_shirt_length_definition"] = "0.90 * hps_to_crotch_surface_length"
        polylines = measurements.get("_linear_polylines")
        if isinstance(polylines, dict):
            polylines["shirt_length"] = regular_tshirt_pts
        start_conf = str(measurements.get("_shirt_start", {}).get("confidence", "low"))
        side_q = _quality(start_conf, "regular_fit_90pct_hps_to_crotch_surface_trace")
    else:
        side_q = _quality("low", "hps_to_crotch_trace_failed")
    quality.update({"bust": bust_q, "waist": waist_q, "hip": hip_q, "stomach": stomach_q, "inseam": inseam_q, "side_neck_to_waist": _quality("low", "replaced_by_regular_tshirt_length"), "regular_tshirt_length": side_q})
    measurements["_measurement_quality"] = quality
    measurements["_torso_arm_exclusion"] = torso_meta
    return torso_mesh



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a fused mesh through the fresh upstream CLAD Body checkout.")
    parser.add_argument("--mesh", "--params", dest="mesh", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--render", default="")
    parser.add_argument("--height-cm", type=float, default=0.0)
    parser.add_argument("--preset", default="all")
    parser.add_argument("--only", default="")
    parser.add_argument("--device", default=None, help="Accepted for runner compatibility; upstream MHR measurement is CPU-based.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.mesh).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    render_path = Path(args.render).expanduser().resolve() if args.render else None
    only = [key.strip() for key in args.only.split(",") if key.strip()] or None
    body, fused_transform, front_transform = fused_body(source, args.height_cm or None)
    shirt_start_meta = install_landmark_constrained_neck_start(body.joints or {}, body.mesh)
    measurements = measure(body, preset=None if only else args.preset, only=only, render_path=None)
    apply_surface_sleeve_length(measurements, body)
    apply_distinct_stomach_measurement(measurements, body)
    apply_surface_crotch_length(measurements, body)
    torso_mesh = apply_production_core_measurements(measurements, body)
    measurements.setdefault("_shirt_length_source", "regular_fit_90pct_hps_to_crotch_surface_trace")
    measurements["_shirt_start"] = shirt_start_meta
    measurements.update({
        "_measurement_engine": "datar-psa/clad-body@a2140a7",
        "_measurement_mesh_source": "fused_sdf_mesh",
        "_joint_landmark_source": "front_mhr_params_transformed_to_fused_frame",
        "_joint_transform_source": "front_mhr_params_scaled_centered_and_fusion_yaw" if front_transform.get("yaw_rotation_applied") is not None else "front_mhr_params_scaled_centered",
        "_fused_mesh_transform": fused_transform,
        "_front_mhr_joint_transform": front_transform,
        "_measurement_mesh_vertices": len(body.mesh.vertices),
        "_measurement_mesh_faces": len(body.mesh.faces),
        "_fusion_rule": scalar(body.sam3d_params.get("fusion_rule"), ""),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(measurements), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_4view(body.mesh, measurements, str(render_path), title=source.stem, model_label="Fused SDF / upstream CLAD", torso_mesh=torso_mesh)
    print(f"Saved measurements: {out_json}")
    if render_path is not None:
        print(f"Saved render      : {render_path}")
    print(f"Measured mesh     : {body.source} ({len(body.mesh.vertices)} verts, {len(body.mesh.faces)} faces)")


if __name__ == "__main__":
    main()
