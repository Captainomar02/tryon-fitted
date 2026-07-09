#!/usr/bin/env python
"""Measure a SAM 3D Body MHR parameter JSON with CLAD Body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from clad_body.load import load_mhr_from_params
from clad_body.load.mhr import MhrBody
from clad_body.measure import measure
from clad_body.measure._circumferences import REGIONS, find_measurement
from clad_body.measure._render import extract_measurement_contours, render_4view
from clad_body.measure._slicer import MeshSlicer, torso_circumference_at_z


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, trimesh.Trimesh):
        return None
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            json_item = to_jsonable(item)
            if json_item is not None:
                result[str(key)] = json_item
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            json_item = to_jsonable(item)
            if json_item is not None:
                result.append(json_item)
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _truthy_param(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def _scalar_param(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item() if value.size else default
    return default if value is None else value


def _scaled_body_to_height(body: MhrBody, target_height_m: float) -> MhrBody:
    verts = np.asarray(body.mesh.vertices, dtype=np.float32).copy()
    current_height = float(verts[:, 2].max() - verts[:, 2].min())
    if current_height <= 1e-8 or target_height_m <= 1e-8:
        return body

    scale = float(target_height_m) / current_height
    verts *= scale
    verts[:, 2] -= float(verts[:, 2].min())

    joints = None
    if body.joints:
        joints = {name: np.asarray(pt, dtype=np.float32) * scale for name, pt in body.joints.items()}
        # Mesh feet define the true floor. Joint Z values are scaled in the same frame.
        for name in joints:
            joints[name] = joints[name].astype(np.float32)

    return MhrBody(
        mesh=trimesh.Trimesh(vertices=verts, faces=np.asarray(body.mesh.faces), process=False),
        source=f"scaled:{body.source}",
        obj_path=body.obj_path,
        sam3d_params=body.sam3d_params,
        joints=joints,
    )


def _fusion_target_height_m(params: dict, fused_vertices: np.ndarray) -> float:
    for key in ("fusion_target_height_cm", "fusion_target_height"):
        value = _scalar_param(params.get(key))
        if value is not None:
            try:
                return float(value) / 100.0
            except (TypeError, ValueError):
                pass
    return float(fused_vertices[:, 2].max() - fused_vertices[:, 2].min())


def _smooth_band_weight(height_pct: np.ndarray, center: float, half_width: float) -> np.ndarray:
    distance = np.abs(height_pct - float(center))
    weight = np.clip(1.0 - (distance / float(half_width)), 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def _float_param(params: dict, key: str, default: float = 0.0) -> float:
    value = _scalar_param(params.get(key), default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _apply_profile_depth_to_body(body: MhrBody, params: dict) -> MhrBody:
    if not _truthy_param(params.get("fusion_apply_profile_depth_to_mhr", False)):
        return body

    verts = np.asarray(body.mesh.vertices, dtype=np.float32).copy()
    height = float(verts[:, 2].max() - verts[:, 2].min())
    if height <= 1e-8:
        return body

    arm_mask = _arm_region_mask(body.mesh, body.joints)
    torso_mask = ~arm_mask if arm_mask.any() else np.ones(verts.shape[0], dtype=bool)
    z_min = float(verts[:, 2].min())
    pct = (verts[:, 2] - z_min) / height
    corrections = [
        (0.725, 0.075, _float_param(params, "fusion_profile_depth_bust_scale", 1.0)),
        (0.50, 0.09, _float_param(params, "fusion_profile_depth_hip_scale", 1.0)),
    ]
    total_weight = np.zeros(verts.shape[0], dtype=np.float64)
    for center, half_width, scale in corrections:
        if scale > 1.0:
            total_weight += _smooth_band_weight(pct, center, half_width) * (float(scale) - 1.0)
    total_weight[~torso_mask] = 0.0
    if np.max(total_weight) <= 0.0:
        return body

    bins = np.clip(np.floor(pct * 200).astype(np.int32), 0, 199)
    y_center = np.zeros(verts.shape[0], dtype=np.float64)
    for bin_id in np.unique(bins[torso_mask]):
        bin_mask = torso_mask & (bins == bin_id)
        if np.any(bin_mask):
            y_center[bin_mask] = np.median(verts[bin_mask, 1])
    active = total_weight > 0.0
    verts[active, 1] = y_center[active] + (verts[active, 1] - y_center[active]) * (1.0 + total_weight[active])
    verts[:, 2] -= float(verts[:, 2].min())

    return MhrBody(
        mesh=trimesh.Trimesh(vertices=verts, faces=np.asarray(body.mesh.faces), process=False),
        source=f"profile_depth_mhr_restpose:{body.source}",
        obj_path=body.obj_path,
        sam3d_params=body.sam3d_params,
        joints=body.joints,
    )

def _vertex_adjacency(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    neighbors = [set() for _ in range(n_vertices)]
    for a, b, c in np.asarray(faces, dtype=np.int32):
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return [list(items) for items in neighbors]


def _arm_region_mask(mesh: trimesh.Trimesh, joints: dict | None) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    if not joints or "l_shoulder" not in joints or "r_shoulder" not in joints:
        return np.zeros(vertices.shape[0], dtype=bool)

    left = np.asarray(joints["l_shoulder"], dtype=np.float64)
    right = np.asarray(joints["r_shoulder"], dtype=np.float64)
    axis = left - right
    axis[2] = 0.0
    width = float(np.linalg.norm(axis))
    if width <= 1e-8:
        return np.zeros(vertices.shape[0], dtype=bool)

    axis /= width
    lateral = vertices.astype(np.float64) @ axis
    low = min(float(left @ axis), float(right @ axis))
    high = max(float(left @ axis), float(right @ axis))
    outside_shoulders = (lateral < low) | (lateral > high)

    arm_landmarks = [left, right]
    for name in ("l_elbow", "r_elbow", "l_wrist", "r_wrist"):
        if name in joints:
            arm_landmarks.append(np.asarray(joints[name], dtype=np.float64))
    arm_z = [float(pt[2]) for pt in arm_landmarks]
    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    z_low = max(float(vertices[:, 2].min()), min(arm_z) - 0.16 * height)
    z_high = min(float(vertices[:, 2].max()), max(arm_z) + 0.06 * height)

    candidate = outside_shoulders & (vertices[:, 2] >= z_low) & (vertices[:, 2] <= z_high)
    if not candidate.any():
        return np.zeros(vertices.shape[0], dtype=bool)

    seeds = np.zeros(vertices.shape[0], dtype=bool)
    radius = max(0.07, width * 0.35)
    for pt in arm_landmarks:
        dist = np.linalg.norm(vertices.astype(np.float64) - pt[None, :], axis=1)
        seeds |= candidate & (dist <= radius)
    # Wrist joints are inside the hand/forearm chain; keep a slightly larger
    # radius there so fingers below the wrist are included via topology growth.
    for name in ("l_wrist", "r_wrist"):
        if name in joints:
            pt = np.asarray(joints[name], dtype=np.float64)
            dist = np.linalg.norm(vertices.astype(np.float64) - pt[None, :], axis=1)
            seeds |= candidate & (dist <= max(0.11, width * 0.45))

    seed_idx = np.flatnonzero(seeds)
    if seed_idx.size == 0:
        return candidate

    adjacency = _vertex_adjacency(np.asarray(mesh.faces, dtype=np.int32), vertices.shape[0])
    arm = np.zeros(vertices.shape[0], dtype=bool)
    stack = [int(i) for i in seed_idx]
    arm[seed_idx] = True
    while stack:
        cur = stack.pop()
        for nxt in adjacency[cur]:
            if not candidate[nxt] or arm[nxt]:
                continue
            arm[nxt] = True
            stack.append(int(nxt))
    return arm


def _fused_transfer_mask(mesh: trimesh.Trimesh, joints: dict | None) -> np.ndarray:
    """Copy fused geometry everywhere except connected CLAD A-pose arms/hands."""
    return ~_arm_region_mask(mesh, joints)


def _fused_transfer_weights(mesh: trimesh.Trimesh, joints: dict | None, preserve_rings=6, blend_rings=12) -> tuple[np.ndarray, np.ndarray]:
    """Blend fused torso into CLAD A-pose without a hard arm seam."""
    vertices = np.asarray(mesh.vertices)
    arm = _arm_region_mask(mesh, joints)
    if not arm.any():
        return np.ones(vertices.shape[0], dtype=np.float32), arm

    adjacency = _vertex_adjacency(np.asarray(mesh.faces, dtype=np.int32), vertices.shape[0])
    max_dist = int(preserve_rings) + int(blend_rings)
    dist = np.full(vertices.shape[0], -1, dtype=np.int32)
    frontier = [int(i) for i in np.flatnonzero(arm)]
    dist[frontier] = 0
    head = 0
    while head < len(frontier):
        cur = frontier[head]
        head += 1
        if dist[cur] >= max_dist:
            continue
        for nxt in adjacency[cur]:
            nxt = int(nxt)
            if dist[nxt] >= 0:
                continue
            dist[nxt] = dist[cur] + 1
            frontier.append(nxt)

    weights = np.ones(vertices.shape[0], dtype=np.float32)
    reached = dist >= 0
    preserve = reached & (dist <= int(preserve_rings))
    transition = reached & (dist > int(preserve_rings)) & (dist <= max_dist)
    weights[preserve] = 0.0
    if int(blend_rings) > 0:
        weights[transition] = (
            (dist[transition].astype(np.float32) - float(preserve_rings)) / float(blend_rings)
        )
    return np.clip(weights, 0.0, 1.0), arm


def _normalize_fused_vertices_for_measurement(params: dict, fused_vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(fused_vertices, dtype=np.float32).copy()
    current_height = float(vertices[:, 2].max() - vertices[:, 2].min())
    target_height = _fusion_target_height_m(params, vertices)
    if current_height > 1e-8 and target_height > 1e-8:
        vertices *= float(target_height) / current_height
    vertices[:, 2] -= float(vertices[:, 2].min())
    center_xy = (vertices[:, :2].max(axis=0) + vertices[:, :2].min(axis=0)) * 0.5
    vertices[:, 0] -= float(center_xy[0])
    vertices[:, 1] -= float(center_xy[1])
    return vertices.astype(np.float32)


def _topology_transferred_joints(body: MhrBody, fused_vertices: np.ndarray, nearest_count: int = 8) -> dict | None:
    if not body.joints:
        return None

    rest_vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    fused_vertices = np.asarray(fused_vertices, dtype=np.float64)
    if rest_vertices.shape != fused_vertices.shape or rest_vertices.ndim != 2:
        return body.joints

    k = max(1, min(int(nearest_count), rest_vertices.shape[0]))
    joints = {}
    for name, point in body.joints.items():
        point = np.asarray(point, dtype=np.float64)
        dist2 = np.sum((rest_vertices - point[None, :]) ** 2, axis=1)
        idx = np.argpartition(dist2, k - 1)[:k]
        dist = np.sqrt(np.maximum(dist2[idx], 0.0))
        if float(dist.min()) < 1e-8:
            weights = (dist == dist.min()).astype(np.float64)
        else:
            weights = 1.0 / np.maximum(dist, 1e-6)
        weights /= float(weights.sum())
        joints[name] = np.sum(fused_vertices[idx] * weights[:, None], axis=0).astype(np.float32)
    return joints


def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected fused vertices with shape [N, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected fused faces with shape [M, 3], got {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Fused measurement mesh is empty")
    if faces.min() < 0 or faces.max() >= len(vertices):
        # Some OBJ-style exports are accidentally serialized 1-indexed.
        if faces.min() == 1 and faces.max() == len(vertices):
            faces = faces - 1
        else:
            raise ValueError(
                f"Fused face indices out of bounds for {len(vertices)} vertices: "
                f"{int(faces.min())}..{int(faces.max())}"
            )
    return vertices, faces


def _fusion_mesh_arrays_from_params(params: dict, params_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = params.get("fusion_vertices_clad")
    faces = params.get("fusion_faces_clad")
    if vertices is not None and faces is not None:
        return _validate_mesh_arrays(vertices, faces)

    for key in (
        "fusion_clad_obj_path",
        "fusion_mesh_path",
        "fusion_sdf_mesh_path",
        "fusion_side_sdf_edited_obj_path",
    ):
        raw_path = _scalar_param(params.get(key))
        if not raw_path:
            continue
        mesh_path = Path(str(raw_path)).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = params_path.parent / mesh_path
        if not mesh_path.exists():
            continue
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                continue
            mesh = trimesh.util.concatenate(meshes)
        return _validate_mesh_arrays(mesh.vertices, mesh.faces)

    return None


def _source_joints_for_fused_mesh(body: MhrBody, fused_vertices: np.ndarray) -> dict | None:
    transferred = _topology_transferred_joints(body, fused_vertices)
    if transferred:
        return transferred
    if not body.joints:
        return None
    return {name: np.asarray(point, dtype=np.float32).copy() for name, point in body.joints.items()}


def load_body_for_measurement(params_path: Path) -> MhrBody:
    """Build the CLAD measurement body from the fused SDF mesh.

    CLAD's MHR loader is still used for SAM3D parameters and landmarks, but
    the measurement mesh itself comes directly from fusion. This avoids the
    previous topology gate where non-identical fused meshes silently fell back
    to the generated MHR rest-pose body.
    """
    body = load_mhr_from_params(str(params_path))

    with params_path.open() as f:
        params = json.load(f)

    mesh_arrays = _fusion_mesh_arrays_from_params(params, params_path)
    vertices = mesh_arrays[0] if mesh_arrays is not None else None
    fused_vertices_hint = (
        np.asarray(vertices, dtype=np.float32)
        if vertices is not None
        else np.asarray(body.mesh.vertices, dtype=np.float32)
    )
    body = _scaled_body_to_height(body, _fusion_target_height_m(params, fused_vertices_hint))

    if mesh_arrays is None or _truthy_param(params.get("fusion_disable_full_fused_clad_measurement", False)):
        return _apply_profile_depth_to_body(body, params)

    raw_vertices, fused_faces = mesh_arrays
    fused_vertices = _normalize_fused_vertices_for_measurement(params, raw_vertices)
    mesh = trimesh.Trimesh(vertices=fused_vertices, faces=fused_faces, process=False)
    return MhrBody(
        mesh=mesh,
        source=f"fusion_full_sdf_mesh:{params_path.name}",
        obj_path=body.obj_path,
        sam3d_params=body.sam3d_params,
        joints=_source_joints_for_fused_mesh(body, fused_vertices),
    )

def joint_anchored_upperarm_measurement(mesh: trimesh.Trimesh, joints: dict | None, height: float) -> tuple[float, float, float] | None:
    if not joints or height <= 0.0:
        return None

    from clad_body.measure._slicer import _perpendicular_limb_contour

    side_results = []
    for side in ("l", "r"):
        shoulder = joints.get(f"{side}_shoulder")
        elbow = joints.get(f"{side}_elbow")
        if shoulder is None or elbow is None:
            continue
        shoulder = np.asarray(shoulder, dtype=np.float64)
        elbow = np.asarray(elbow, dtype=np.float64)
        axis = elbow - shoulder
        length = float(np.linalg.norm(axis))
        if length <= 1e-4:
            continue
        axis /= length

        candidates = []
        for t in np.linspace(0.38, 0.72, 8):
            center = shoulder + (elbow - shoulder) * float(t)
            pts = _perpendicular_limb_contour(mesh, center, axis, max_dist=0.16)
            if pts is None or len(pts) < 3:
                continue
            closed = np.vstack([pts, pts[:1]])
            circ = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
            if 0.12 <= circ <= 0.60:
                candidates.append((circ, float(center[2])))
        if candidates:
            side_results.append(max(candidates, key=lambda item: item[0]))

    if len(side_results) < 2:
        return None
    circ = float(np.mean([item[0] for item in side_results]))
    z = float(np.mean([item[1] for item in side_results]))
    return circ * 100.0, z, z / height * 100.0


def build_torso_mesh_for_measurement(body: MhrBody, params_path: Path | None = None) -> trimesh.Trimesh | None:
    if not body.joints:
        return None
    arm_mask = _arm_region_mask(body.mesh, body.joints)
    if not arm_mask.any():
        return None

    vertices = np.asarray(body.mesh.vertices, dtype=np.float32)
    faces = np.asarray(body.mesh.faces, dtype=np.int32)
    if params_path is not None:
        try:
            with params_path.open() as f:
                params = json.load(f)
            fused_vertices_value = params.get("fusion_vertices_clad")
            fused_faces_value = params.get("fusion_faces_clad")
            if fused_vertices_value is not None and fused_faces_value is not None:
                fused_vertices = np.asarray(fused_vertices_value, dtype=np.float32).copy()
                fused_faces = np.asarray(fused_faces_value, dtype=np.int32)
                if fused_vertices.shape == vertices.shape and fused_faces.shape == faces.shape:
                    current_height = float(fused_vertices[:, 2].max() - fused_vertices[:, 2].min())
                    target_height = _fusion_target_height_m(params, fused_vertices)
                    if current_height > 1e-8 and target_height > 1e-8:
                        fused_vertices *= float(target_height) / current_height
                    fused_vertices[:, 2] -= float(fused_vertices[:, 2].min())
                    vertices = fused_vertices
                    faces = fused_faces
        except Exception:
            pass

    torso_faces = faces[~arm_mask[faces].any(axis=1)]
    if len(torso_faces) == 0:
        return None
    return trimesh.Trimesh(
        vertices=vertices,
        faces=torso_faces,
        process=False,
    )


def refine_torso_measurements(measurements: dict, body: MhrBody, torso_mesh: trimesh.Trimesh | None) -> dict:
    mesh = body.mesh
    height = float(np.asarray(mesh.vertices)[:, 2].max())
    if height <= 0:
        return measurements

    upperarm = joint_anchored_upperarm_measurement(mesh, body.joints, height)
    if upperarm is not None and (float(measurements.get("upperarm_cm", 0.0) or 0.0) <= 0.0):
        measurements["upperarm_cm"], measurements["_upperarm_z"], measurements["_upperarm_pct"] = upperarm

    if torso_mesh is None:
        return measurements

    if "bust_cm" in measurements or "_bust_z" in measurements:
        bust_zs = np.arange(height * 0.68, height * 0.76, 0.002)
        if len(bust_zs) > 0:
            slicer = MeshSlicer(torso_mesh)
            bust_circs = np.array([
                slicer.circumference_at_z(z, combine_fragments=True)
                for z in bust_zs
            ])
            valid = bust_circs > 0.30
            if valid.any():
                idx = int(np.argmax(np.where(valid, bust_circs, -1.0)))
                z = float(bust_zs[idx])
                measurements["bust_cm"] = float(bust_circs[idx] * 100.0)
                measurements["_bust_z"] = z
                measurements["_bust_pct"] = z / height * 100.0

    if "hip_cm" in measurements or "_hip_z" in measurements:
        hip_region = REGIONS["hip"]
        hip_zs = np.arange(
            height * hip_region["low_pct"],
            height * hip_region["high_pct"],
            0.002,
        )
        if len(hip_zs) > 0:
            hip_circs = np.array([
                torso_circumference_at_z(
                    torso_mesh,
                    z,
                    max_x_extent=0.80,
                    combine_fragments=True,
                )
                for z in hip_zs
            ])
            valid = hip_circs > 0.30
            if valid.any():
                idx = int(np.argmax(np.where(valid, hip_circs, -1.0)))
                z = float(hip_zs[idx])
                measurements["hip_cm"] = float(hip_circs[idx] * 100.0)
                measurements["_hip_z"] = z
                measurements["_hip_pct"] = z / height * 100.0

    waist_z = float(measurements.get("_waist_z", height * 0.61) or 0.0)
    if waist_z > 0 and ("waist_cm" in measurements or "_waist_z" in measurements):
        waist_circ = torso_circumference_at_z(
            torso_mesh,
            waist_z,
            max_x_extent=0.60,
            combine_fragments=True,
        )
        if waist_circ > 0.30:
            measurements["waist_cm"] = float(waist_circ * 100.0)
            measurements["_waist_z"] = waist_z
            measurements["_waist_pct"] = waist_z / height * 100.0

    hip_z = float(measurements.get("_hip_z", 0.0) or 0.0)
    if hip_z > 0 and waist_z > hip_z and ("stomach_cm" in measurements or "_stomach_z" in measurements):
        # Avoid letting the hip contour double as the stomach contour; search the
        # abdomen span above the hip and below the waist.
        lo = hip_z + (waist_z - hip_z) * 0.20
        hi = waist_z
        stomach_zs = np.arange(lo, hi, 0.002)
        if len(stomach_zs) > 0:
            verts = np.asarray(torso_mesh.vertices)
            best_z = None
            best_front_y = float("inf")
            for z in stomach_zs:
                band = verts[np.abs(verts[:, 2] - z) < 0.002]
                if len(band) < 3:
                    continue
                front_y = float(band[:, 1].min())
                if front_y < best_front_y:
                    best_front_y = front_y
                    best_z = float(z)
            if best_z is not None:
                span = waist_z - hip_z
                if span > 0 and (best_z < hip_z + span * 0.15 or best_z > waist_z - span * 0.15):
                    best_z = float((hip_z + waist_z) * 0.5)
                stomach_circ = torso_circumference_at_z(
                    torso_mesh,
                    best_z,
                    max_x_extent=0.60,
                    combine_fragments=True,
                )
                if stomach_circ > 0.30:
                    measurements["stomach_cm"] = float(stomach_circ * 100.0)
                    measurements["_stomach_z"] = best_z
                    measurements["_stomach_pct"] = best_z / height * 100.0

    measurements["contours"] = extract_measurement_contours(mesh, measurements, torso_mesh=torso_mesh)
    measurements["_torso_mesh_source"] = "fusion_vertices_clad_torso_only"
    measurements["_torso_mesh"] = torso_mesh
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="SAM 3D Body MHR params JSON.")
    parser.add_argument("--out-json", required=True, help="Measurement JSON output path.")
    parser.add_argument("--render", default="", help="Optional 4-view render PNG output path.")
    parser.add_argument("--preset", default="all", help="CLAD measurement preset.")
    parser.add_argument(
        "--device",
        default=None,
        help="Measurement device. Default lets CLAD auto-select CUDA when available.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated measurement keys. Overrides preset when provided.",
    )
    args = parser.parse_args()

    params_path = Path(args.params)
    out_json_path = Path(args.out_json)
    render_path = Path(args.render) if args.render else None
    only = [part.strip() for part in args.only.split(",") if part.strip()] or None

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)

    body = load_body_for_measurement(params_path)
    torso_mesh = build_torso_mesh_for_measurement(body, params_path)
    measurements = measure(
        body,
        preset=None if only else args.preset,
        only=only,
        render_path=None,
        device=args.device,
    )
    measurements = refine_torso_measurements(measurements, body, torso_mesh)
    if render_path is not None:
        render_4view(
            body.mesh,
            measurements,
            str(render_path),
            title=params_path.stem,
            model_label="MHR",
            torso_mesh=torso_mesh,
        )

    with out_json_path.open("w") as f:
        json.dump(to_jsonable(measurements), f, indent=2, sort_keys=True)

    print(f"Saved measurements: {out_json_path}")
    if render_path is not None:
        print(f"Saved render      : {render_path}")


if __name__ == "__main__":
    main()
