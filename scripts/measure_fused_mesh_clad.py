#!/usr/bin/env python3
"""Measure a fused SDF mesh with CLAD Body.

This script is intentionally standalone: it reads the fused mesh output from a
SAM/front-side fusion JSON or from an OBJ/PLY/STL mesh, wraps that mesh as a
CLAD MhrBody, and runs CLAD measurements/rendering on the fused mesh directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAD_ROOT = REPO_ROOT / "clad-body"
if str(CLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(CLAD_ROOT))

from clad_body.load.mhr import MhrBody, load_mhr_from_params
from clad_body.measure import measure
from clad_body.measure._render import render_4view
from clad_body.measure._slicer import MeshSlicer


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
        out = {}
        for key, item in value.items():
            converted = jsonable(item)
            if converted is not None:
                out[str(key)] = converted
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            converted = jsonable(item)
            if converted is not None:
                out.append(converted)
        return out
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Expected vertices shaped [N,3], got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Expected faces shaped [M,3], got {f.shape}")
    if len(v) == 0 or len(f) == 0:
        raise ValueError("Fused mesh is empty")
    if f.min() < 0 or f.max() >= len(v):
        if f.min() == 1 and f.max() == len(v):
            f = f - 1
        else:
            raise ValueError(f"Face indices {int(f.min())}..{int(f.max())} do not fit {len(v)} vertices")
    return v, f


def read_mesh_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(parts)
    return mesh_arrays(mesh.vertices, mesh.faces)


def read_fused_source(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        params = json.loads(path.read_text(encoding="utf-8"))
        if "fusion_vertices_clad" in params and "fusion_faces_clad" in params:
            v, f = mesh_arrays(params["fusion_vertices_clad"], params["fusion_faces_clad"])
            return v, f, params
        for key in ("fusion_clad_obj_path", "fusion_mesh_path", "fusion_sdf_mesh_path"):
            raw = scalar(params.get(key))
            if raw:
                mesh_path = Path(str(raw)).expanduser()
                if not mesh_path.is_absolute():
                    mesh_path = path.parent / mesh_path
                if mesh_path.exists():
                    v, f = read_mesh_file(mesh_path)
                    return v, f, params
        raise ValueError(f"JSON has no fused mesh arrays or valid fused mesh path: {path}")
    v, f = read_mesh_file(path)
    return v, f, {}


def target_height_m(params: dict[str, Any], explicit_height_cm: float | None, vertices: np.ndarray) -> float:
    if explicit_height_cm and explicit_height_cm > 0:
        return explicit_height_cm / 100.0
    for key in ("fusion_target_height_cm", "fusion_target_height"):
        value = scalar(params.get(key))
        if value is not None:
            try:
                return float(value) / 100.0
            except (TypeError, ValueError):
                pass
    return float(vertices[:, 2].max() - vertices[:, 2].min())


def canonicalize(vertices: np.ndarray, height_m: float) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float32).copy()
    current_height = float(v[:, 2].max() - v[:, 2].min())
    if current_height > 1e-8 and height_m > 1e-8:
        v *= float(height_m) / current_height
    v[:, 2] -= float(v[:, 2].min())
    center_xy = (v[:, :2].min(axis=0) + v[:, :2].max(axis=0)) * 0.5
    v[:, 0] -= float(center_xy[0])
    v[:, 1] -= float(center_xy[1])
    return v.astype(np.float32)


def scaled_reference_geometry(params_path: Path | None, height_m: float) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None]:
    if params_path is None or params_path.suffix.lower() != ".json":
        return None, None
    try:
        ref = load_mhr_from_params(str(params_path))
    except Exception as exc:
        print(f"[measure-fused] reference MHR geometry unavailable: {exc}", file=sys.stderr)
        return None, None

    ref_vertices = np.asarray(ref.mesh.vertices, dtype=np.float32)
    ref_height = float(ref_vertices[:, 2].max() - ref_vertices[:, 2].min())
    scale = float(height_m) / ref_height if ref_height > 1e-8 and height_m > 1e-8 else 1.0
    scaled_vertices = (ref_vertices * scale).astype(np.float32)
    joints = None
    if ref.joints:
        joints = {name: np.asarray(point, dtype=np.float32) * scale for name, point in ref.joints.items()}
    return joints, scaled_vertices



_ARM_SKINNING_CACHE: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None


def _subprocess_env_with_torch_lib() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = Path(spec.origin).resolve().parent / "lib"
            current = env.get("LD_LIBRARY_PATH", "")
            if str(torch_lib) not in current.split(":"):
                env["LD_LIBRARY_PATH"] = f"{torch_lib}:{current}" if current else str(torch_lib)
    except Exception:
        pass
    return env


def mhr_arm_skinning_masks(n_vertices: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    # Same-topology complete arm/hand masks from MHR skinning weights.
    global _ARM_SKINNING_CACHE
    if _ARM_SKINNING_CACHE is not None:
        masks, scores = _ARM_SKINNING_CACHE
        if len(masks["l"]) == n_vertices:
            return masks, scores
        return None

    script = """
import sys
import pymomentum.geometry  # noqa: F401
import numpy as np
from mhr.mhr import MHR

model = MHR.from_files(device="cpu", wants_pose_correctives=False)
names = list(model.character_torch.skeleton.joint_names)
lbs = model.character_torch.linear_blend_skinning
vi = lbs.vert_indices_flattened.cpu().numpy().astype(np.int64)
ji = lbs.skin_indices_flattened.cpu().numpy().astype(np.int64)
wt = lbs.skin_weights_flattened.cpu().numpy().astype(np.float32)
n_vertices = int(lbs.num_vertices)
TOKENS = ("uparm", "lowarm", "wrist", "pinky", "ring", "middle", "index", "thumb")

def side_score(prefix):
    bones = [i for i, name in enumerate(names) if name.lower().startswith(prefix) and any(tok in name.lower() for tok in TOKENS)]
    keep = np.isin(ji, np.asarray(bones, dtype=np.int64))
    score = np.bincount(vi[keep], weights=wt[keep], minlength=n_vertices).astype(np.float32)
    return score

l_score = side_score("l_")
r_score = side_score("r_")
np.savez(
    sys.argv[1],
    l_score=l_score,
    r_score=r_score,
    l_mask=(l_score >= 0.05),
    r_mask=(r_score >= 0.05),
)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as nf:
        out_path = nf.name
    try:
        result = subprocess.run(
            [sys.executable, script_path, out_path],
            cwd=str(REPO_ROOT),
            env=_subprocess_env_with_torch_lib(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr[-800:] if result.stderr else "unknown error"
            print(f"[measure-fused] MHR skinning masks unavailable: {err}", file=sys.stderr)
            return None
        data = np.load(out_path)
        masks = {"l": data["l_mask"].astype(bool), "r": data["r_mask"].astype(bool)}
        scores = {"l": data["l_score"].astype(np.float32), "r": data["r_score"].astype(np.float32)}
        _ARM_SKINNING_CACHE = masks, scores
        if len(masks["l"]) != n_vertices:
            return None
        return masks, scores
    finally:
        for tmp in (script_path, out_path):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _infer_current_arm_angle(vertices: np.ndarray, mask: np.ndarray, shoulder: np.ndarray, sign: float) -> float:
    arm = vertices[mask]
    if len(arm) < 20:
        return 0.0
    z_cut = np.percentile(arm[:, 2], 30.0)
    lower = arm[arm[:, 2] <= z_cut]
    if len(lower) < 10:
        lower = arm
    center = np.median(lower, axis=0)
    outward = sign * float(center[0] - shoulder[0])
    downward = max(float(shoulder[2] - center[2]), 0.05)
    return float(np.arctan2(outward, downward))


def pose_fused_mesh_soft_apose(
    vertices: np.ndarray,
    reference_vertices: np.ndarray | None,
    joints: dict[str, np.ndarray] | None,
    angle_deg: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Abduct complete fused-mesh arm/hand components into a mild A-pose.
    meta: dict[str, Any] = {
        "enabled": False,
        "method": "mhr_skinning_complete_arm_soft_abduction",
        "requested_angle_deg": float(angle_deg),
        "left_vertices": 0,
        "right_vertices": 0,
        "left_delta_deg": 0.0,
        "right_delta_deg": 0.0,
    }
    if joints is None or reference_vertices is None or len(reference_vertices) != len(vertices):
        meta["reason"] = "missing_same_topology_reference_geometry"
        return vertices, meta

    skinning = mhr_arm_skinning_masks(len(vertices))
    if skinning is None:
        meta["reason"] = "missing_mhr_skinning_masks"
        return vertices, meta
    masks, scores = skinning

    posed = np.asarray(vertices, dtype=np.float32).copy()
    target = np.deg2rad(float(angle_deg))
    max_delta = np.deg2rad(45.0)
    any_posed = False

    for side, sign, label in (("l", 1.0, "left"), ("r", -1.0, "right")):
        shoulder = joints.get(f"{side}_shoulder")
        if shoulder is None:
            continue
        mask = masks[side]
        score = scores[side]
        count = int(np.count_nonzero(mask))
        meta[f"{label}_vertices"] = count
        if count < 200:
            continue

        current = _infer_current_arm_angle(vertices, mask, shoulder, sign)
        delta = float(np.clip(target - current, 0.0, max_delta))
        meta[f"{label}_delta_deg"] = float(np.rad2deg(delta))
        if delta <= np.deg2rad(1.0):
            continue

        pts = vertices[mask]
        rel_u = sign * (pts[:, 0] - float(shoulder[0]))
        rel_z = pts[:, 2] - float(shoulder[2])
        c = float(np.cos(delta))
        s = float(np.sin(delta))
        rot_u = c * rel_u - s * rel_z
        rot_z = s * rel_u + c * rel_z

        rotated = pts.copy()
        rotated[:, 0] = float(shoulder[0]) + sign * rot_u
        rotated[:, 2] = float(shoulder[2]) + rot_z

        down_from_shoulder = float(shoulder[2]) - pts[:, 2]
        shoulder_weight = _smoothstep(0.015, 0.13, down_from_shoulder).astype(np.float32)
        skin_weight = _smoothstep(0.05, 0.45, score[mask]).astype(np.float32)
        weights = np.maximum(
            shoulder_weight * skin_weight,
            (score[mask] >= 0.75).astype(np.float32) * shoulder_weight,
        )
        posed[mask] = pts + (rotated - pts) * weights[:, None]
        any_posed = True

    meta["enabled"] = bool(any_posed)
    if not any_posed:
        meta["reason"] = "no_arm_vertices_rotated"
    return posed.astype(np.float32), meta




def arm_excluded_torso_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh | None, dict[str, Any]]:
    skinning = mhr_arm_skinning_masks(len(mesh.vertices))
    meta: dict[str, Any] = {"enabled": False, "method": "mhr_skinning_arm_face_removal"}
    if skinning is None:
        meta["reason"] = "missing_mhr_skinning_masks"
        return None, meta
    masks, _ = skinning
    arm_mask = masks["l"] | masks["r"]
    faces = np.asarray(mesh.faces, dtype=np.int32)
    keep_faces = ~arm_mask[faces].any(axis=1)
    kept = int(np.count_nonzero(keep_faces))
    meta.update({
        "enabled": kept > 0,
        "removed_vertices": int(np.count_nonzero(arm_mask)),
        "kept_faces": kept,
        "removed_faces": int(len(faces) - kept),
    })
    if kept <= 0:
        meta["reason"] = "all_faces_removed"
        return None, meta
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces[keep_faces], process=False), meta


def _best_torso_circumference(
    slicer: MeshSlicer,
    zs: np.ndarray,
    *,
    max_x_extent: float = 0.85,
) -> tuple[float, float]:
    best_circ = 0.0
    best_z = 0.0
    for z in zs:
        circ = float(slicer.circumference_at_z(float(z), max_x_extent=max_x_extent, combine_fragments=True))
        if circ > best_circ:
            best_circ = circ
            best_z = float(z)
    return best_circ, best_z


def apply_arm_excluded_torso_measurements(
    measurements: dict[str, Any],
    body: MhrBody,
    requested_keys: set[str] | None = None,
) -> trimesh.Trimesh | None:
    torso_mesh, meta = arm_excluded_torso_mesh(body.mesh)
    measurements["_torso_measurement_source"] = "fused_mesh_mhr_skinning_arm_excluded"
    measurements["_torso_arm_exclusion"] = meta
    if torso_mesh is None:
        return None

    height = float(np.asarray(body.mesh.vertices)[:, 2].max() - np.asarray(body.mesh.vertices)[:, 2].min())
    if height <= 0.0:
        return torso_mesh
    slicer = MeshSlicer(torso_mesh)

    def wants(name: str) -> bool:
        return requested_keys is None or f"{name}_cm" in requested_keys

    def set_measurement(name: str, circ_m: float, z: float) -> None:
        if not wants(name) or circ_m <= 0.30 or z <= 0.0:
            return
        measurements[f"{name}_cm"] = float(circ_m * 100.0)
        measurements[f"_{name}_z"] = float(z)
        measurements[f"_{name}_pct"] = float(z / height * 100.0)

    if wants("bust"):
        circ, z = _best_torso_circumference(
            slicer,
            np.arange(height * 0.68, height * 0.76, 0.002),
            max_x_extent=0.85,
        )
        set_measurement("bust", circ, z)

    waist_z = height * 0.61
    if wants("waist"):
        waist_circ = float(slicer.circumference_at_z(waist_z, max_x_extent=0.75, combine_fragments=True))
        set_measurement("waist", waist_circ, waist_z)

    if wants("hip"):
        circ, z = _best_torso_circumference(
            slicer,
            np.arange(height * 0.46, height * 0.54, 0.002),
            max_x_extent=0.95,
        )
        set_measurement("hip", circ, z)

    hip_z = float(measurements.get("_hip_z", height * 0.50) or height * 0.50)
    hi = float(measurements.get("_waist_z", waist_z) or waist_z)
    lo = min(hip_z + max(hi - hip_z, 0.0) * 0.20, hi)
    if wants("stomach") and hi > lo:
        circ, z = _best_torso_circumference(
            slicer,
            np.arange(lo, hi, 0.002),
            max_x_extent=0.85,
        )
        set_measurement("stomach", circ, z)

    return torso_mesh
def fused_body(
    source: Path,
    height_cm: float | None,
    *,
    pose_arms: bool = False,
    arm_pose_deg: float = 30.0,
) -> tuple[MhrBody, dict[str, Any]]:
    raw_vertices, faces, params = read_fused_source(source)
    height_m = target_height_m(params, height_cm, raw_vertices)
    vertices = canonicalize(raw_vertices, height_m)
    joints, reference_vertices = scaled_reference_geometry(source if source.suffix.lower() == ".json" else None, height_m)

    pose_meta: dict[str, Any] = {"enabled": False, "method": "disabled"}
    if pose_arms:
        vertices, pose_meta = pose_fused_mesh_soft_apose(vertices, reference_vertices, joints, arm_pose_deg)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if params:
        params["_fused_mesh_pose_edit"] = pose_meta
    source_prefix = "fresh_fused_mesh_soft_apose" if pose_meta.get("enabled") else "fresh_fused_mesh"
    body = MhrBody(
        mesh=mesh,
        source=f"{source_prefix}:{source.name}",
        obj_path=str(source),
        sam3d_params=params or None,
        joints=joints,
    )
    return body, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a fused SDF mesh directly with CLAD Body.")
    parser.add_argument("--mesh", "--params", dest="mesh", required=True, help="Fused params JSON or mesh file.")
    parser.add_argument("--out-json", required=True, help="Measurement JSON output path.")
    parser.add_argument("--render", default="", help="Optional 4-view render PNG output path.")
    parser.add_argument("--height-cm", type=float, default=0.0, help="Optional target height override in centimeters.")
    parser.add_argument("--preset", default="all", help="CLAD measurement preset.")
    parser.add_argument("--only", default="", help="Comma-separated measurement keys. Overrides preset.")
    parser.add_argument("--device", default=None, help="Accepted for runner compatibility; MHR CLAD currently ignores it.")
    parser.add_argument("--pose-arms", action="store_true", help="Visibly soft-pose arms before measuring. Off by default to preserve shoulders.")
    parser.add_argument("--no-pose-arms", action="store_true", help="Compatibility flag; arm posing is already off by default.")
    parser.add_argument("--arm-pose-deg", type=float, default=30.0, help="Target arm abduction angle from vertical when --pose-arms is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.mesh).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    render_path = Path(args.render).expanduser().resolve() if args.render else None
    only = [item.strip() for item in args.only.split(",") if item.strip()] or None

    body, params = fused_body(
        source,
        args.height_cm if args.height_cm > 0 else None,
        pose_arms=bool(args.pose_arms and not args.no_pose_arms),
        arm_pose_deg=args.arm_pose_deg,
    )
    measurements = measure(body, preset=None if only else args.preset, only=only, render_path=None)
    torso_mesh = apply_arm_excluded_torso_measurements(measurements, body, set(only) if only else None)
    measurements["_measurement_mesh_source"] = body.source
    measurements["_measurement_mesh_vertices"] = int(len(body.mesh.vertices))
    measurements["_measurement_mesh_faces"] = int(len(body.mesh.faces))
    if params:
        measurements["_fusion_rule"] = scalar(params.get("fusion_rule"), "")
        measurements["_fused_mesh_pose_edit"] = params.get("_fused_mesh_pose_edit", {"enabled": False})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(measurements), indent=2, sort_keys=True), encoding="utf-8")

    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_4view(body.mesh, measurements, str(render_path), title=source.stem, model_label="Fused SDF", torso_mesh=torso_mesh)

    print(f"Saved measurements: {out_json}")
    if render_path is not None:
        print(f"Saved render      : {render_path}")
    print(f"Measured mesh     : {body.source} ({len(body.mesh.vertices)} verts, {len(body.mesh.faces)} faces)")


if __name__ == "__main__":
    main()
