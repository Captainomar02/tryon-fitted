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
from clad_body.measure._circumferences import find_measurement
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
        value = params.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None:
            try:
                return float(value) / 100.0
            except (TypeError, ValueError):
                pass
    return float(fused_vertices[:, 2].max() - fused_vertices[:, 2].min())


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


def load_body_for_measurement(params_path: Path) -> MhrBody:
    """Measure final fused torso geometry while preserving upstream CLAD's A-pose arms."""
    body = load_mhr_from_params(str(params_path))

    with params_path.open() as f:
        params = json.load(f)

    if not _truthy_param(params.get("fusion_prefer_vertices_for_clad", False)):
        return body

    vertices = params.get("fusion_vertices_clad")
    faces = params.get("fusion_faces_clad")
    if vertices is None or faces is None:
        return body

    fused_vertices = np.asarray(vertices, dtype=np.float32)
    fused_faces = np.asarray(faces, dtype=np.int32)
    rest_faces = np.asarray(body.mesh.faces, dtype=np.int32)
    if fused_vertices.shape != np.asarray(body.mesh.vertices).shape or fused_faces.shape != rest_faces.shape:
        return body

    target_height_m = _fusion_target_height_m(params, fused_vertices)
    body = _scaled_body_to_height(body, target_height_m)

    measurement_vertices = np.asarray(body.mesh.vertices, dtype=np.float32).copy()
    transfer_mask = _fused_transfer_mask(body.mesh, body.joints)
    measurement_vertices[transfer_mask] = fused_vertices[transfer_mask]
    measurement_vertices[:, 2] -= float(measurement_vertices[:, 2].min())

    mesh = trimesh.Trimesh(vertices=measurement_vertices, faces=rest_faces, process=False)
    return MhrBody(
        mesh=mesh,
        source=f"fusion_torso_on_clad_apose:{params_path.name}",
        obj_path=body.obj_path,
        sam3d_params=body.sam3d_params,
        joints=body.joints,
    )

def build_torso_mesh_for_measurement(body: MhrBody) -> trimesh.Trimesh | None:
    if not body.joints:
        return None
    transfer_mask = _fused_transfer_mask(body.mesh, body.joints)
    arm_mask = ~transfer_mask
    if not arm_mask.any():
        return None
    faces = np.asarray(body.mesh.faces, dtype=np.int32)
    torso_faces = faces[~arm_mask[faces].any(axis=1)]
    if len(torso_faces) == 0:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(body.mesh.vertices),
        faces=torso_faces,
        process=False,
    )


def refine_torso_measurements(measurements: dict, body: MhrBody, torso_mesh: trimesh.Trimesh | None) -> dict:
    if torso_mesh is None:
        return measurements

    mesh = body.mesh
    height = float(np.asarray(mesh.vertices)[:, 2].max())
    if height <= 0:
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
    torso_mesh = build_torso_mesh_for_measurement(body)
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
