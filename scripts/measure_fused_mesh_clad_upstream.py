#!/usr/bin/env python3
"""Measure a fused mesh using a clean upstream CLAD Body checkout.

The mesh remains the fused SDF surface. Linear joint landmarks come from the
front MHR parameters retained in the fusion JSON and are transformed into the
fused mesh's canonical measurement frame.
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
UPSTREAM_CLAD_ROOT = REPO_ROOT / "vendor" / "clad-body-upstream"
if not UPSTREAM_CLAD_ROOT.is_dir():
    raise RuntimeError(f"Upstream CLAD checkout is missing: {UPSTREAM_CLAD_ROOT}")
sys.path.insert(0, str(UPSTREAM_CLAD_ROOT))

from clad_body.load.mhr import MhrBody, load_mhr_from_params  # noqa: E402
from clad_body.measure import measure  # noqa: E402
from clad_body.measure._render import render_4view  # noqa: E402
import clad_body.measure._lengths as upstream_lengths  # noqa: E402


_MHR_SKINNING_CACHE: dict[str, Any] | None = None
_MHR_JOINT_NAMES_CACHE: list[str] | None = None
_ARM_SKINNING_CACHE: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
_ARM_SKINNING_GROUP_CACHE: dict[str, np.ndarray] | None = None
BUST_FULL_TORSO_WIDTH_FACTOR = float(os.environ.get("FUSION_BUST_FULL_TORSO_WIDTH_FACTOR", "1.15"))
ARM_MASK_CACHE_PATH = REPO_ROOT / "checkpoints" / "mhr-assets" / "assets" / "arm_skinning_masks.npz"


def _mhr_subprocess_env() -> dict[str, str]:
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


def _mhr_joint_names() -> list[str] | None:
    global _MHR_JOINT_NAMES_CACHE
    if _MHR_JOINT_NAMES_CACHE is not None:
        return _MHR_JOINT_NAMES_CACHE
    script = """
import json
import pymomentum.geometry  # noqa: F401
from mhr.mhr import MHR
model = MHR.from_files(device='cpu', wants_pose_correctives=False)
print(json.dumps(list(model.character_torch.skeleton.joint_names)), flush=True)
"""
    try:
        result = subprocess.run([sys.executable, "-c", script], cwd=str(REPO_ROOT), env=_mhr_subprocess_env(), capture_output=True, text=True, timeout=120)
        if result.stdout:
            names = json.loads(result.stdout.strip().splitlines()[-1])
            _MHR_JOINT_NAMES_CACHE = [str(name) for name in names]
            return _MHR_JOINT_NAMES_CACHE
    except Exception:
        pass
    return None


def _mhr_skinning_data() -> dict[str, Any] | None:
    """Load the exact MHR vertex-to-joint weights for the 127-joint order."""
    global _MHR_SKINNING_CACHE
    if _MHR_SKINNING_CACHE is not None:
        return _MHR_SKINNING_CACHE
    script = """
import os
import sys
import numpy as np
import pymomentum.geometry  # noqa: F401
from mhr.mhr import MHR
model = MHR.from_files(device='cpu', wants_pose_correctives=False)
lbs = model.character_torch.linear_blend_skinning
names = np.asarray(model.character_torch.skeleton.joint_names)
vi = lbs.vert_indices_flattened.cpu().numpy().copy()
ji = lbs.skin_indices_flattened.cpu().numpy().copy()
wt = lbs.skin_weights_flattened.cpu().numpy().copy()
n = np.asarray([lbs.num_vertices])
np.savez(sys.argv[1], names=names, vi=vi, ji=ji, wt=wt, n=n)
# MHR/Momentum can fault during interpreter teardown after a successful save.
# np.savez has already closed the archive, so skip native destructors.
os._exit(0)
"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as nf:
        data_path = nf.name
    try:
        result = subprocess.run([sys.executable, script_path, data_path], cwd=str(REPO_ROOT), env=_mhr_subprocess_env(), capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[measure-fused] MHR skinning unavailable (exit {result.returncode}): {result.stderr[-600:] or result.stdout[-600:]}", file=sys.stderr)
            return None
        data = np.load(data_path, allow_pickle=False)
        _MHR_SKINNING_CACHE = {
            "names": [str(name) for name in data["names"].tolist()],
            "vi": data["vi"].astype(np.int64), "ji": data["ji"].astype(np.int64),
            "wt": data["wt"].astype(np.float64), "vertex_count": int(data["n"][0]),
        }
        return _MHR_SKINNING_CACHE
    except Exception as exc:
        print(f"[measure-fused] MHR skinning unavailable: {exc}", file=sys.stderr)
        return None
    finally:
        for path in (script_path, data_path):
            try: os.unlink(path)
            except OSError: pass


def mhr_arm_skinning_masks(n_vertices: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Return complete arm/hand masks from MHR skinning weights."""
    global _ARM_SKINNING_CACHE, _ARM_SKINNING_GROUP_CACHE
    if _ARM_SKINNING_CACHE is not None:
        masks, scores = _ARM_SKINNING_CACHE
        return (masks, scores) if len(masks["l"]) == n_vertices else None
    if ARM_MASK_CACHE_PATH.is_file():
        try:
            data = np.load(ARM_MASK_CACHE_PATH, allow_pickle=False)
            masks = {"l": data["l_mask"].astype(bool), "r": data["r_mask"].astype(bool)}
            scores = {"l": data["l_score"].astype(np.float32), "r": data["r_score"].astype(np.float32)}
            if len(masks["l"]) == n_vertices:
                _ARM_SKINNING_GROUP_CACHE = {
                    key: data[key].astype(np.float32)
                    for key in ("l_upper", "l_lower", "l_wrist", "r_upper", "r_lower", "r_wrist")
                }
                _ARM_SKINNING_CACHE = (masks, scores)
                return masks, scores
        except (KeyError, OSError, ValueError):
            pass
    data = _mhr_skinning_data()
    if data is None or data["vertex_count"] != n_vertices:
        return None
    names, vi, ji, wt = data["names"], data["vi"], data["ji"], data["wt"]

    def score(prefix: str, tokens: tuple[str, ...]) -> np.ndarray:
        bones = [i for i, name in enumerate(names) if name.lower().startswith(prefix) and any(token in name.lower() for token in tokens)]
        keep = np.isin(ji, np.asarray(bones, dtype=np.int64))
        return np.bincount(vi[keep], weights=wt[keep], minlength=n_vertices).astype(np.float32)

    arm_tokens = ("uparm", "lowarm", "wrist", "pinky", "ring", "middle", "index", "thumb")
    wrist_tokens = ("wrist", "pinky", "ring", "middle", "index", "thumb")
    _ARM_SKINNING_GROUP_CACHE = {
        "l_upper": score("l_", ("uparm",)), "l_lower": score("l_", ("lowarm",)), "l_wrist": score("l_", wrist_tokens),
        "r_upper": score("r_", ("uparm",)), "r_lower": score("r_", ("lowarm",)), "r_wrist": score("r_", wrist_tokens),
    }
    scores = {"l": score("l_", arm_tokens), "r": score("r_", arm_tokens)}
    masks = {side: values >= 0.05 for side, values in scores.items()}
    _ARM_SKINNING_CACHE = (masks, scores)
    return masks, scores


def mhr_arm_skinning_groups(n_vertices: int) -> dict[str, np.ndarray] | None:
    return _ARM_SKINNING_GROUP_CACHE if mhr_arm_skinning_masks(n_vertices) is not None else None


def arm_excluded_torso_mesh(mesh: trimesh.Trimesh, joints: dict[str, np.ndarray] | None = None) -> tuple[trimesh.Trimesh | None, dict[str, Any]]:
    """Remove arm faces from transferred shoulder/elbow/wrist landmarks.

    This stays in the fused mesh's coordinate system and avoids the unstable
    native Momentum skinning extractor that previously disabled chest loops.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    meta: dict[str, Any] = {"enabled": False, "method": "joint_chain_arm_face_removal"}
    arm_mask = np.zeros(len(vertices), dtype=bool)
    height = float(np.ptp(vertices[:, 2]))
    # A forearm can be much thinner than the upper arm adjacent to the bust.
    # The floor is height-relative (not body-specific) and prevents that
    # forearm estimate from leaking upper-arm vertices into a chest slice.
    min_radius, max_radius = 0.045 * height, 0.090 * height

    def point_to_segment_distance(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = end - start
        length_sq = float(direction @ direction)
        if length_sq <= 1e-10:
            return np.full(len(vertices), np.inf), np.zeros(len(vertices))
        fraction = np.clip(((vertices - start) @ direction) / length_sq, 0.0, 1.0)
        closest = start + fraction[:, None] * direction
        return np.linalg.norm(vertices - closest, axis=1), fraction

    side_radii: dict[str, float] = {}
    if joints:
        for side in ("l", "r"):
            chain = [joints.get(f"{side}_shoulder"), joints.get(f"{side}_elbow"), joints.get(f"{side}_wrist")]
            if any(point is None for point in chain):
                continue
            chain = [np.asarray(point, dtype=np.float64) for point in chain]
            # Measure this person's forearm radius away from the torso, then
            # use a modest margin so the entire arm surface is excluded.
            forearm_distance, forearm_fraction = point_to_segment_distance(chain[1], chain[2])
            midpoint_x = float((chain[1][0] + chain[2][0]) * 0.5)
            outward = 1.0 if side == "l" else -1.0
            samples = forearm_distance[
                (forearm_fraction >= 0.15) & (forearm_fraction <= 0.85)
                & (outward * (vertices[:, 0] - midpoint_x) >= -0.015 * height)
                & (forearm_distance >= 0.010 * height) & (forearm_distance <= 0.160 * height)
            ]
            if len(samples) >= 30:
                radius = float(np.clip(np.percentile(samples, 75) * 1.15, min_radius, max_radius))
                source = "mesh_adaptive_forearm_radius"
            else:
                radius = float(np.clip(0.060 * height, min_radius, max_radius))
                source = "height_scaled_radius_fallback"
            side_radii[side] = radius
            for start, end in zip(chain, chain[1:]):
                distance, _ = point_to_segment_distance(start, end)
                arm_mask |= distance <= radius
            meta[f"{side}_arm_radius_m"] = radius
            meta[f"{side}_arm_radius_source"] = source
    if not arm_mask.any():
        meta.update({"reason": "missing_arm_joint_chains_full_mesh_fallback", "fallback": True})
        return mesh.copy(), meta
    faces = np.asarray(mesh.faces, dtype=np.int32)
    keep_faces = ~arm_mask[faces].any(axis=1)
    kept = int(np.count_nonzero(keep_faces))
    meta.update({"enabled": kept > 0, "removed_vertices": int(np.count_nonzero(arm_mask)), "kept_faces": kept, "removed_faces": int(len(faces) - kept)})
    if kept <= 0:
        meta["reason"] = "all_faces_removed"
        return None, meta
    return trimesh.Trimesh(vertices=vertices, faces=faces[keep_faces], process=False), meta


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


def transfer_joints_to_fused_mesh(
    joints: dict[str, np.ndarray],
    reference_vertices: np.ndarray | None,
    fused_vertices: np.ndarray,
    height_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Carry front MHR landmarks through the saved same-topology fusion field.

    Joints remain skeletal (they are not snapped to skin).  A compact Gaussian
    neighbourhood over corresponding vertices transfers the local fused depth
    edit while avoiding a global, unrelated surface displacement.
    """
    meta: dict[str, Any] = {"enabled": False, "reason": "missing_reference", "joints": {}}
    if reference_vertices is None:
        return joints, meta
    ref = np.asarray(reference_vertices, dtype=np.float64)
    fused = np.asarray(fused_vertices, dtype=np.float64)
    if ref.shape != fused.shape or ref.ndim != 2 or ref.shape[1] != 3:
        meta["reason"] = "reference_topology_mismatch"
        return joints, meta
    delta = fused - ref
    radius = max(0.04, 0.09 * float(height_m))
    max_shift = max(0.08, 0.15 * float(height_m))
    transferred: dict[str, np.ndarray] = {}
    invalid: list[str] = []
    for name, point in joints.items():
        p = np.asarray(point, dtype=np.float64)
        distances = np.linalg.norm(ref - p[None, :], axis=1)
        indices = np.argpartition(distances, min(63, len(distances) - 1))[: min(64, len(distances))]
        local_dist = distances[indices]
        weights = np.exp(-0.5 * (local_dist / radius) ** 2)
        if float(weights.sum()) < 1e-8:
            shift = np.zeros(3, dtype=np.float64)
        else:
            shift = np.sum(delta[indices] * weights[:, None], axis=0) / float(weights.sum())
        magnitude = float(np.linalg.norm(shift))
        accepted = magnitude <= max_shift
        if not accepted:
            invalid.append(name)
            shift[:] = 0.0
        transferred[name] = (p + shift).astype(np.float32)
        meta["joints"][name] = {
            "displacement_m": float(magnitude),
            "nearest_reference_distance_m": float(local_dist.min()),
            "accepted": accepted,
        }
    meta.update({
        "enabled": True,
        "reason": "ok" if not invalid else "implausible_joint_displacement",
        "radius_m": radius,
        "max_accepted_displacement_m": max_shift,
        "invalid_joints": invalid,
    })
    return transferred, meta


def transfer_posed_joints_by_skinning(
    posed_joints: np.ndarray,
    reference_vertices: np.ndarray | None,
    fused_vertices: np.ndarray,
    height_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transfer all posed SAM3D joints through same-index fused deformation."""
    meta: dict[str, Any] = {"enabled": False, "method": "mhr_lbs_same_topology_deformation", "joints": {}, "invalid_joints": []}
    skinning = _mhr_skinning_data()
    joints = np.asarray(posed_joints, dtype=np.float64)
    if skinning is None or reference_vertices is None:
        meta["reason"] = "missing_skinning_or_reference"
        return {}, meta
    ref, fused = np.asarray(reference_vertices, dtype=np.float64), np.asarray(fused_vertices, dtype=np.float64)
    names = skinning["names"]
    if ref.shape != fused.shape or len(ref) != skinning["vertex_count"] or joints.shape != (len(names), 3):
        meta["reason"] = "posed_joint_or_topology_mismatch"
        return {}, meta
    delta = fused - ref
    vi, ji, wt = skinning["vi"], skinning["ji"], skinning["wt"]
    result: dict[str, np.ndarray] = {}
    max_shift = 0.12 * float(height_m)
    for index, name in enumerate(names):
        rows = ji == index
        vertices, weights = vi[rows], wt[rows]
        weight_sum = float(weights.sum())
        support = int(np.count_nonzero(weights > 0.01))
        valid = support >= 4 and weight_sum > 1e-6
        shift = np.sum(delta[vertices] * weights[:, None], axis=0) / weight_sum if valid else np.zeros(3)
        magnitude = float(np.linalg.norm(shift))
        if magnitude > max_shift:
            valid = False
            shift[:] = 0.0
        result[name] = (joints[index] + shift).astype(np.float32)
        meta["joints"][name] = {"index": index, "support_vertices": support, "support_weight": weight_sum, "displacement_m": magnitude, "accepted": bool(valid)}
        if not valid:
            meta["invalid_joints"].append(name)
    meta.update({"enabled": True, "reason": "ok" if not meta["invalid_joints"] else "invalid_joint_support", "joint_count": len(names), "max_accepted_displacement_m": max_shift})
    return result, meta


def transfer_posed_joints_by_saved_bindings(
    posed_joints: np.ndarray,
    bind_indices: np.ndarray,
    bind_weights: np.ndarray,
    reference_vertices: np.ndarray | None,
    fused_vertices: np.ndarray,
    height_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the fusion deformation through bindings made in the same posed mesh."""
    meta: dict[str, Any] = {"enabled": False, "method": "sam3d_posed_same_topology_vertex_bindings", "joints": {}, "invalid_joints": []}
    points = np.asarray(posed_joints, dtype=np.float64)
    indices, weights = np.asarray(bind_indices, dtype=np.int64), np.asarray(bind_weights, dtype=np.float64)
    if reference_vertices is None:
        meta["reason"] = "missing_reference"
        return {}, meta
    ref, fused = np.asarray(reference_vertices, dtype=np.float64), np.asarray(fused_vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or indices.shape != weights.shape or indices.shape[0] != len(points) or ref.shape != fused.shape:
        meta["reason"] = "binding_shape_mismatch"
        return {}, meta
    if np.any(indices < 0) or np.any(indices >= len(ref)):
        meta["reason"] = "binding_index_out_of_range"
        return {}, meta
    delta = fused - ref
    names = _mhr_joint_names() or [f"joint_{i}" for i in range(len(points))]
    result: dict[str, np.ndarray] = {}
    max_shift = 0.12 * float(height_m)
    for i, name in enumerate(names):
        w = np.maximum(weights[i], 0.0)
        total = float(w.sum())
        shift = np.sum(delta[indices[i]] * w[:, None], axis=0) / total if total > 1e-8 else np.zeros(3)
        magnitude = float(np.linalg.norm(shift))
        valid = total > 0.99 and magnitude <= max_shift
        if not valid:
            shift[:] = 0.0
            meta["invalid_joints"].append(name)
        result[name] = (points[i] + shift).astype(np.float32)
        meta["joints"][name] = {"index": i, "support_vertices": int(len(indices[i])), "support_weight": total, "displacement_m": magnitude, "accepted": bool(valid)}
    meta.update({"enabled": True, "reason": "ok" if not meta["invalid_joints"] else "invalid_joint_binding", "joint_count": len(points), "max_accepted_displacement_m": max_shift})
    return result, meta


def shoulder_neck_surface_anchor(vertices: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Locate the posterior neck base between two validated acromions."""
    points = np.asarray(vertices, dtype=np.float64)
    height = float(np.ptp(points[:, 2]))
    if height <= 0:
        return None, {"reason": "invalid_mesh_height"}
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    midpoint = (left + right) * 0.5
    shoulder_top = max(float(left[2]), float(right[2]))
    candidates = points[
        (np.abs(points[:, 0] - midpoint[0]) <= 0.035 * height)
        & (points[:, 2] >= shoulder_top)
        & (points[:, 2] <= shoulder_top + 0.020 * height)
    ]
    if len(candidates) < 8:
        return None, {"reason": "neck_base_surface_region_missing"}
    # Restrict to the posterior portion of the neck band, then take the
    # highest point in it.  This excludes the low upper-back surface and the
    # head/face while retaining the actual neck-base transition.
    posterior = candidates[candidates[:, 1] >= np.quantile(candidates[:, 1], 0.80)]
    point = posterior[int(np.argmax(posterior[:, 2]))].copy()
    point[0] = midpoint[0]
    return point, {
        "reason": "posterior_neck_base_between_acromions",
        "shoulder_top_z_m": shoulder_top,
        "search_count": int(len(candidates)),
    }


def posterior_surface_projection(vertices: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    """Project a landmark to rear skin at its lateral/height location."""
    point = np.asarray(point, dtype=np.float64)
    for x_tol, z_tol in ((0.018, 0.018), (0.035, 0.030), (0.060, 0.055)):
        nearby = vertices[
            (np.abs(vertices[:, 0] - point[0]) <= x_tol)
            & (np.abs(vertices[:, 2] - point[2]) <= z_tol)
        ]
        if len(nearby):
            # In CLAD canonical space, more-negative Y is the posterior skin.
            return np.array([point[0], float(nearby[:, 1].min()), point[2]], dtype=np.float64)
    return None


def surface_landmarks_from_skeleton(vertices: np.ndarray, skeletal: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Find anatomically constrained C7 and acromion points on fused skin.

    C7 is a posterior *mid-neck* landmark.  It must not drift onto the
    shoulder cap merely because that is the nearest visible surface.  The
    acromions, conversely, are the lateral shoulder caps themselves and must
    not be pushed backwards onto the scapular surface.
    """
    from clad_body.measure.mhr import find_acromion
    out = dict(skeletal)
    meta: dict[str, Any] = {"method": "fused_topology_surface_landmarks", "landmarks": {}, "invalid": []}
    points = np.asarray(vertices, dtype=np.float64)
    height = float(np.ptp(points[:, 2]))
    c7 = skeletal.get("c7")
    acromions: dict[str, np.ndarray] = {}
    for prefix, side in (("l", "left"), ("r", "right")):
        key = f"{prefix}_shoulder"
        seed = skeletal.get(key)
        acromion = find_acromion(vertices, seed, side=side) if seed is not None else None
        if acromion is None or seed is None or height <= 0:
            meta["invalid"].append(key)
            continue
        acromion = np.asarray(acromion, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        sign = 1.0 if side == "left" else -1.0
        lateral = sign * (acromion[0] - seed[0])
        plausible = lateral >= -0.005 and abs(acromion[2] - seed[2]) <= 0.06 * height
        if not plausible:
            meta["invalid"].append(key)
            continue
        # find_acromion already returns the actual outer shoulder skin point.
        # Do not project it to the back surface: that was the source of the
        # visibly rearward shoulder endpoints in the render.
        out[key] = acromion.astype(np.float32)
        acromions[key] = acromion
        meta["landmarks"][key] = {
            "type": "lateral_acromion_surface",
            "accepted": True,
            "point_m": acromion.tolist(),
            "seed_distance_m": float(np.linalg.norm(acromion - seed)),
            "lateral_offset_m": float(lateral),
        }
    c7_surface = None
    c7_meta: dict[str, Any] = {}
    if {"l_shoulder", "r_shoulder"} <= set(acromions):
        c7_surface, c7_meta = shoulder_neck_surface_anchor(vertices, acromions["l_shoulder"], acromions["r_shoulder"])
    if c7_surface is None:
        meta["invalid"].append("c7")
    else:
        out["c7"] = c7_surface.astype(np.float32)
        meta["landmarks"]["c7"] = {
            "type": "posterior_neck_base_surface",
            "accepted": True,
            "point_m": c7_surface.tolist(),
            "skeletal_seed_distance_m": float(np.linalg.norm(c7_surface - np.asarray(c7))) if c7 is not None else None,
            **c7_meta,
        }
    meta["enabled"] = not meta["invalid"]
    return out, meta


def measure_hps_to_crotch_front_surface(
    joints: dict[str, np.ndarray], mesh: trimesh.Trimesh, crotch_z: float, step: float = 0.005,
) -> tuple[float, np.ndarray | None, dict[str, Any]]:
    """Use CLAD's original ISO side-neck and convex front-surface trace.

    The original implementation is deliberate: it starts at the neck-base
    side-neck point, skips unstable shoulder-cap slices, and takes a lower
    convex hull over the chest and stomach profile.  The prior replacement
    re-invented this logic from shoulder-joint locations and consequently
    started on the shoulder instead of the neck.
    """
    from clad_body.measure._slicer import MeshSlicer

    if joints.get("c7") is None or crotch_z <= 0:
        return 0.0, None, {"confidence": "low", "reason": "missing_neck_or_crotch"}
    shoulder = joints.get("l_shoulder")
    if shoulder is None:
        return 0.0, None, {"confidence": "low", "reason": "missing_left_shoulder"}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    height = float(np.ptp(vertices[:, 2]))
    if height <= 0:
        return 0.0, None, {"confidence": "low", "reason": "invalid_mesh_height"}
    slicer = MeshSlicer(mesh)
    shoulder = np.asarray(shoulder, dtype=np.float64)
    # Follow the lateral silhouette upward from the acromion.  Use the inner
    # part of the shoulder-to-neck contraction so the tape starts beside the
    # neck and captures the main front torso, rather than riding near the
    # shoulder cap like a sleeve seam.
    zs = np.arange(float(shoulder[2]) + 0.002 * height, float(shoulder[2]) + 0.060 * height, 0.003)
    samples: list[tuple[float, float, float]] = []
    for z in zs:
        contours = slicer.contours_at_z(float(z))
        if not contours:
            continue
        points = np.vstack([np.asarray(contour[0], dtype=np.float64) for contour in contours])
        side_points = points[points[:, 0] > 0]
        if len(side_points) < 2:
            continue
        edge_x = float(side_points[:, 0].max())
        edge = side_points[side_points[:, 0] >= edge_x - 0.010 * height]
        samples.append((float(z), edge_x, float(edge[:, 1].min())))
    if len(samples) < 5:
        return 0.0, None, {"confidence": "low", "reason": "neck_transition_slices_missing"}
    sample_array = np.asarray(samples, dtype=np.float64)
    edge_start, edge_min = float(sample_array[0, 1]), float(sample_array[:, 1].min())
    neck_contraction = 0.70
    target_edge = edge_start - neck_contraction * (edge_start - edge_min)
    transition_index = int(np.flatnonzero(sample_array[:, 1] <= target_edge)[0]) if np.any(sample_array[:, 1] <= target_edge) else int(np.argmin(abs(sample_array[:, 1] - target_edge)))
    side_neck = np.asarray([sample_array[transition_index, 1], sample_array[transition_index, 2], sample_array[transition_index, 0]], dtype=np.float64)

    # Delegate the front chest/stomach curvature to the upstream routine.  It
    # skips the unstable shoulder cap and applies its established lower convex
    # hull, but receives the anatomically corrected side-neck start above.
    original_finder = upstream_lengths.find_side_neck_point
    upstream_lengths.find_side_neck_point = lambda _slicer, _c7_z: side_neck.copy()
    try:
        length_cm, trace = upstream_lengths.measure_shirt_length(
            joints, mesh, crotch_z, measurements=None, step=step, end_offset=0.0,
        )
    finally:
        upstream_lengths.find_side_neck_point = original_finder
    if trace is None or len(trace) < 3 or length_cm <= 0:
        return 0.0, None, {"confidence": "low", "reason": "original_clad_side_neck_trace_failed"}
    return float(length_cm), np.asarray(trace, dtype=np.float32), {
        "confidence": "high",
        "reason": "shoulder_to_neck_transition_then_original_clad_front_convex_hull",
        "rendered_side": "left",
        "start_point_m": np.asarray(trace[0], dtype=np.float64).tolist(),
        "shoulder_to_neck_contraction": neck_contraction,
        "shoulder_edge_x_m": edge_start,
        "neck_edge_x_m": edge_min,
        "trace_count": 1,
    }


def _canonical_clad_joints(all_joints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from clad_body.measure.mhr import MHR_JOINT_MAP
    result: dict[str, np.ndarray] = {}
    for canonical, candidates in MHR_JOINT_MAP.items():
        for name in candidates:
            if name in all_joints:
                result[canonical] = all_joints[name]
                break
    return result


def suppress_measurements_for_invalid_joints(measurements: dict[str, Any], joint_meta: dict[str, Any], surface_meta: dict[str, Any]) -> list[str]:
    """Keep torso outputs while removing values whose required landmarks failed."""
    invalid = set(joint_meta.get("invalid_joints", [])) | set(surface_meta.get("invalid", []))
    suppressed: list[str] = []
    if {"c_neck", "l_uparm", "r_uparm", "c7", "l_shoulder", "r_shoulder"} & invalid:
        for key in ("shoulder_width_cm", "shirt_length_cm", "regular_tshirt_length_cm", "hps_to_crotch_cm"):
            if key in measurements:
                measurements.pop(key, None)
                suppressed.append(key)
    if {"l_lowarm", "r_lowarm", "l_wrist", "r_wrist"} & invalid:
        if "sleeve_length_cm" in measurements:
            measurements.pop("sleeve_length_cm", None)
            suppressed.append("sleeve_length_cm")
    return suppressed

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
    reference_raw = params.get("fusion_front_reference_vertices_clad")
    posed_raw = params.get("fusion_joint_coords_clad")
    bind_indices_raw = params.get("fusion_joint_bind_indices")
    bind_weights_raw = params.get("fusion_joint_bind_weights")
    reference_vertices = None
    if reference_raw is not None:
        try:
            reference_vertices, reference_transform = canonicalize_with_transform(np.asarray(reference_raw, dtype=np.float32), height_m)
        except Exception:
            reference_vertices = None
            reference_transform = None
    else:
        reference_transform = None
    if posed_raw is None or bind_indices_raw is None or bind_weights_raw is None or reference_transform is None:
        all_joints, joint_transfer = {}, {"enabled": False, "reason": "missing_fusion_joint_coords"}
    else:
        posed = np.asarray(posed_raw, dtype=np.float32).copy()
        posed *= float(reference_transform["scale"])
        posed[:, 2] -= float(reference_transform["z_offset_m"])
        posed[:, :2] -= np.asarray(reference_transform["xy_center_offset_m"], dtype=np.float32)
        all_joints, joint_transfer = transfer_posed_joints_by_saved_bindings(
            posed, np.asarray(bind_indices_raw), np.asarray(bind_weights_raw), reference_vertices, vertices, height_m
        )
    joints = _canonical_clad_joints(all_joints)
    joints, surface_landmarks = surface_landmarks_from_skeleton(vertices, joints) if joints else (joints, {"enabled": False, "invalid": ["all"]})
    params["_fused_joint_transfer"] = joint_transfer
    params["_fused_joint_landmarks"] = surface_landmarks
    params["_fused_joint_coords"] = all_joints
    body = MhrBody(
        mesh=trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        source=f"fused_sdf_with_front_mhr_joints:{source.name}",
        obj_path=str(source),
        sam3d_params=params,
        joints=joints,
    )
    return body, fused_transform, {"source": "saved_sam3d_posed_joint_coords"}





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
    """Measure/render sleeves from transferred skin landmarks without Momentum."""
    if "sleeve_length_cm" not in measurements or not body.joints:
        return
    from clad_body.measure.mhr import find_acromion

    vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    traces = []
    for prefix, side_name, side_sign in (("l", "left", 1.0), ("r", "right", -1.0)):
        shoulder = body.joints.get(f"{prefix}_shoulder")
        elbow = body.joints.get(f"{prefix}_elbow")
        wrist = body.joints.get(f"{prefix}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            continue
        acromion = find_acromion(vertices, np.asarray(shoulder), side=side_name)
        trace = np.asarray([acromion, elbow, wrist], dtype=np.float64)
        length_cm = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum() * 100.0)
        traces.append((prefix, length_cm, trace))

    if not traces:
        measurements["_sleeve_length_source"] = "upstream_joint_chain_fallback"
        return
    mean_cm = float(np.mean([item[1] for item in traces]))
    shown = min(traces, key=lambda item: abs(item[1] - mean_cm))
    measurements["sleeve_length_cm"] = mean_cm
    measurements["_sleeve_length_source"] = "transferred_joint_outer_arm_chain"
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


def apply_surface_shoulder_width(measurements: dict[str, Any], body: MhrBody) -> None:
    """Measure the upper shoulder seam through a visible neck-centre anchor.

    The anatomical C7 joint can sit below both shoulder caps.  It remains in
    the saved skeleton, but is not suitable as the control point for this
    garment shoulder line.  Instead, find the posterior neck/shoulder
    transition on the fused surface and require it to be at shoulder height
    or higher.
    """
    if not body.joints or not {"l_shoulder", "r_shoulder"} <= set(body.joints):
        return
    vertices = np.asarray(body.mesh.vertices, dtype=np.float64)
    left_seed = np.asarray(body.joints["l_shoulder"], dtype=np.float64)
    right_seed = np.asarray(body.joints["r_shoulder"], dtype=np.float64)
    left = posterior_surface_projection(vertices, left_seed)
    right = posterior_surface_projection(vertices, right_seed)
    left = left_seed if left is None else left
    right = right_seed if right is None else right
    height = float(np.ptp(np.asarray(body.mesh.vertices)[:, 2]))
    if height <= 0:
        return
    neck_anchor, _ = shoulder_neck_surface_anchor(vertices, left, right)
    if neck_anchor is None:
        # Do not silently reintroduce a sagging centre when neck geometry is
        # missing: use a conservative raised midpoint and record the fallback.
        neck_anchor = (left + right) * 0.5
        neck_anchor[2] = max(float(left[2]), float(right[2])) + 0.008 * height

    # Two shallow quadratic segments preserve a genuine neck-centre curve
    # while making it impossible for the shoulder path to drape downward.
    left_control = (left + neck_anchor) * 0.5
    right_control = (right + neck_anchor) * 0.5
    left_control[2] = max(left_control[2], left[2], neck_anchor[2])
    right_control[2] = max(right_control[2], right[2], neck_anchor[2])
    t = np.linspace(0.0, 1.0, 16)[:, None]
    right_half = ((1.0 - t) ** 2) * right + 2.0 * (1.0 - t) * t * right_control + (t ** 2) * neck_anchor
    left_half = ((1.0 - t) ** 2) * neck_anchor + 2.0 * (1.0 - t) * t * left_control + (t ** 2) * left
    arc = np.vstack([right_half, left_half[1:]])
    width_cm = float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum() * 100.0)
    measurements["shoulder_width_cm"] = float(width_cm)
    measurements["_shoulder_arc_pts"] = np.asarray(arc, dtype=np.float32)
    measurements["_shoulder_width_source"] = "posterior_surface_acromion_to_upper_neck_seam"
    measurements["_shoulder_neck_anchor_m"] = neck_anchor.astype(np.float32)
    polylines = measurements.get("_linear_polylines")
    if isinstance(polylines, dict):
        polylines["shoulder_width"] = np.asarray(arc, dtype=np.float32)



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
    from clad_body.measure._slicer import MeshSlicer

    quality: dict[str, Any] = {}
    torso_mesh, torso_meta = arm_excluded_torso_mesh(body.mesh, body.joints)
    height = float(np.ptp(np.asarray(body.mesh.vertices)[:, 2]))
    if torso_mesh is None or height <= 0:
        for name in ("bust", "waist", "stomach", "hip", "inseam", "side_neck_to_waist"):
            quality[name] = _quality("low", "arm_excluded_torso_unavailable")
        measurements["_measurement_quality"] = quality
        return None

    slicer = MeshSlicer(torso_mesh)
    full_slicer = MeshSlicer(body.mesh)
    # The shoulder roots plus the removal radii define a person-specific
    # torso envelope.  Points outside it belong to the axilla/upper-arm side
    # of a cut, even when their remaining mesh fragment is still centred.
    left_root = body.joints.get("l_shoulder") if body.joints else None
    right_root = body.joints.get("r_shoulder") if body.joints else None
    left_limit = None
    right_limit = None
    if left_root is not None:
        left_limit = float(np.asarray(left_root)[0]) - float(torso_meta.get("l_arm_radius_m", 0.0))
    if right_root is not None:
        right_limit = float(np.asarray(right_root)[0]) + float(torso_meta.get("r_arm_radius_m", 0.0))

    def arm_safe_torso_points(z: float) -> list[np.ndarray]:
        """Return only torso cross-section points inside both shoulder roots."""
        parts: list[np.ndarray] = []
        for points, x_extent, x_center in slicer.contours_at_z(float(z)):
            if x_extent < 0.12 or abs(x_center) > 0.08:
                continue
            keep = np.ones(len(points), dtype=bool)
            if left_limit is not None:
                keep &= points[:, 0] <= left_limit
            if right_limit is not None:
                keep &= points[:, 0] >= right_limit
            clipped = points[keep]
            if len(clipped) >= 3:
                parts.append(clipped)
        return parts

    def centered_full_torso_circumference(z: float, extent: float, torso_extent_limit: float) -> tuple[float, int, float]:
        """Use the intact, centred torso loop when arms are separate slices.

        Removing arm faces can open the upper torso mesh at the axilla.  At a
        bust height where the full mesh has a single large centred component
        and separate small arm components, the full component is the more
        complete torso circumference.
        """
        contours = full_slicer.contours_at_z(float(z))
        centered = [
            (points, x_extent) for points, x_extent, x_center in contours
            if x_extent < torso_extent_limit and x_extent >= 0.12 and abs(x_center) <= 0.14 * height
        ]
        if len(centered) != 1:
            return 0.0, len(centered), 0.0
        points, x_extent = centered[0]
        try:
            from scipy.spatial import ConvexHull
            return float(ConvexHull(points).area), 1, float(x_extent)
        except Exception:
            closed = np.vstack([points, points[:1]])
            return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()), 1, float(x_extent)

    def centered_arm_excluded_torso_circumference(z: float) -> tuple[float, int]:
        """Measure only centred torso points inside the shoulder-root envelope."""
        parts = arm_safe_torso_points(z)
        if not parts:
            return 0.0, 0
        points = np.vstack(parts)
        try:
            from scipy.spatial import ConvexHull
            return float(ConvexHull(points).area), len(parts)
        except Exception:
            ordered = points[np.argsort(np.arctan2(points[:, 1] - points[:, 1].mean(), points[:, 0] - points[:, 0].mean()))]
            return float(np.linalg.norm(np.diff(np.vstack([ordered, ordered[:1]]), axis=0), axis=1).sum()), len(parts)

    def select_max(name: str, low: float, high: float, extent: float, *, prefer_full_centered: bool = False, arm_excluded_only: bool = False) -> tuple[float, float, dict[str, Any], str]:
        zs = np.arange(low, high, 0.002)
        if arm_excluded_only:
            # Garment bust circumference must never traverse arm geometry.
            # The arm-chain cut may leave axilla fragments, so combine only
            # centred torso fragments and never inspect the intact full mesh.
            values = [centered_arm_excluded_torso_circumference(float(z)) for z in zs]
            circs = np.asarray([value[0] for value in values])
            components = np.asarray([value[1] for value in values])
            component_widths = np.zeros_like(circs)
            used_full = np.zeros_like(circs, dtype=bool)
            torso_width_baseline = 0.0
            torso_extent_limit = float(extent)
            baseline_source = "arm_excluded_torso_only"
        elif prefer_full_centered:
            # First pass: get the protected arm-excluded torso width through
            # the lower part of this person's bust range.  This is the
            # baseline; the intact full-mesh loop is allowed to be only a
            # configurable multiple of it before it is considered arm-merged.
            arm_excluded_widths = []
            for z in zs:
                fragments = [points for points, x_extent, _ in slicer.contours_at_z(float(z)) if x_extent < extent]
                if fragments:
                    combined = np.vstack(fragments)
                    arm_excluded_widths.append(float(combined[:, 0].max() - combined[:, 0].min()))
                else:
                    arm_excluded_widths.append(np.nan)
            arm_excluded_widths = np.asarray(arm_excluded_widths, dtype=np.float64)
            raw_components = []
            for z in zs:
                candidates = [
                    (x_extent, x_center) for _, x_extent, x_center in full_slicer.contours_at_z(float(z))
                    if x_extent >= 0.12 and abs(x_center) <= 0.14 * height
                ]
                raw_components.append(candidates[0][0] if len(candidates) == 1 else np.nan)
            raw_components = np.asarray(raw_components, dtype=np.float64)
            lower_count = max(4, len(arm_excluded_widths) // 3)
            lower_band = arm_excluded_widths[:lower_count]
            baseline_samples = lower_band[np.isfinite(lower_band)]
            baseline_source = "arm_excluded_lower_bust_torso_width"
            if len(baseline_samples) < 3:
                baseline_samples = raw_components[np.isfinite(raw_components)]
                baseline_source = "full_mesh_fallback_no_arm_excluded_baseline"
            if len(baseline_samples) >= 3:
                torso_width_baseline = float(np.percentile(baseline_samples, 50))
                torso_extent_limit = min(float(extent), torso_width_baseline * BUST_FULL_TORSO_WIDTH_FACTOR)
            else:
                torso_width_baseline = 0.0
                torso_extent_limit = float(extent)
                baseline_source = "unavailable"
            values = [centered_full_torso_circumference(float(z), extent, torso_extent_limit) for z in zs]
            circs = np.asarray([value[0] for value in values])
            components = np.asarray([value[1] for value in values])
            component_widths = np.asarray([value[2] for value in values])
            fallback_values = [centered_arm_excluded_torso_circumference(float(z)) for z in zs]
            fallback_circs = np.asarray([value[0] for value in fallback_values])
            used_full = circs > 0.30
            circs = np.where(used_full, circs, fallback_circs)
            components = np.where(used_full, components, np.asarray([value[1] for value in fallback_values]))
        else:
            circs = np.asarray([
                slicer.circumference_at_z(float(z), max_x_extent=extent, combine_fragments=True)
                for z in zs
            ])
            components = np.asarray([len(slicer.contours_at_z(float(z))) for z in zs])
            component_widths = np.zeros_like(circs)
        valid = circs > 0.30
        if not valid.any():
            return 0.0, 0.0, _quality("low", "no_valid_torso_contour"), "unavailable", {}
        index = int(np.argmax(np.where(valid, circs, -1.0)))
        z = float(zs[index])
        fragments = int(components[index])
        edge = index <= 2 or index >= len(zs) - 3
        confidence = "high" if fragments == 1 and not edge else "medium"
        reasons = []
        if fragments != 1:
            reasons.append("fragmented_slice")
        if edge:
            reasons.append("selected_search_boundary")
        source = "arm_excluded_torso_mesh"
        if prefer_full_centered and bool(used_full[index]):
            reasons.append("intact_centered_full_mesh_torso_component")
            source = "intact_centered_full_mesh_torso_component"
        elif prefer_full_centered:
            reasons.append("arm_excluded_fallback_no_unambiguous_full_torso_component")
        details = {
            "full_torso_width_baseline_m": float(torso_width_baseline) if prefer_full_centered else None,
            "full_torso_width_limit_m": float(torso_extent_limit) if prefer_full_centered else None,
            "full_torso_width_factor": float(BUST_FULL_TORSO_WIDTH_FACTOR) if prefer_full_centered else None,
            "full_torso_width_baseline_source": baseline_source if prefer_full_centered else None,
            "selected_full_torso_width_m": float(component_widths[index]) if prefer_full_centered and bool(used_full[index]) else None,
        }
        return float(circs[index]), z, _quality(confidence, *reasons), source, details

    def centered_hip_circumference(low: float, high: float) -> tuple[float, float, dict[str, Any]]:
        """Find the widest centred pelvis loop, never the two leg loops."""
        candidates: list[tuple[float, float]] = []
        for z in np.arange(low, high, 0.002):
            parts = [
                points for points, x_extent, x_center in slicer.contours_at_z(float(z))
                if x_extent >= 0.12 and abs(x_center) <= 0.08
            ]
            if not parts:
                continue
            points = np.vstack(parts)
            try:
                from scipy.spatial import ConvexHull
                circumference = float(ConvexHull(points).area)
            except Exception:
                ordered = points[np.argsort(np.arctan2(points[:, 1] - points[:, 1].mean(), points[:, 0] - points[:, 0].mean()))]
                circumference = float(np.linalg.norm(np.diff(np.vstack([ordered, ordered[:1]]), axis=0), axis=1).sum())
            if circumference > 0.30:
                candidates.append((circumference, float(z)))
        if not candidates:
            return 0.0, 0.0, _quality("low", "no_centered_pelvis_contour")
        circumference, z = max(candidates, key=lambda item: item[0])
        edge = z <= low + 0.004 or z >= high - 0.004
        return circumference, z, _quality("medium" if edge else "high", "selected_search_boundary" if edge else "centred_pelvis_component")

    # Select the maximum arm/shoulder-free torso circumference between this
    # person's waist landmark and bilateral shoulder line.  This replaces a
    # fixed-percent chest band with anatomical endpoints.
    upstream_waist_z = float(measurements.get("_waist_z", 0.61 * height) or 0.61 * height)
    shoulder_zs = [
        float(np.asarray(body.joints[key], dtype=np.float64)[2])
        for key in ("l_shoulder", "r_shoulder")
        if body.joints and key in body.joints
    ]
    shoulder_line_z = float(np.mean(shoulder_zs)) if shoulder_zs else 0.80 * height
    bust_low = min(upstream_waist_z, shoulder_line_z - 0.010)
    bust_high = max(shoulder_line_z, bust_low + 0.010)
    bust_m, bust_z, bust_q, bust_source, bust_details = select_max(
        "bust", bust_low, bust_high, 0.85, arm_excluded_only=True,
    )
    bust_source = "waist_to_bilateral_shoulder_line_max_on_arm_shoulder_excluded_torso"
    bust_render_pts = None
    render_parts = arm_safe_torso_points(bust_z) if bust_z > 0 else []
    if render_parts:
        try:
            from scipy.spatial import ConvexHull
            hull_pts = np.vstack(render_parts)[ConvexHull(np.vstack(render_parts)).vertices]
            bust_render_pts = np.column_stack([hull_pts, np.full(len(hull_pts), bust_z)]).astype(np.float32)
        except Exception:
            pass
    bust_details = {
        **bust_details,
        "waist_z_m": upstream_waist_z,
        "shoulder_line_z_m": shoulder_line_z,
        "search_low_pct": bust_low / height * 100.0,
        "search_high_pct": bust_high / height * 100.0,
        "arm_shoulder_excluded": True,
        "left_torso_limit_m": left_limit,
        "right_torso_limit_m": right_limit,
    }
    hip_m, hip_z, hip_q = centered_hip_circumference(0.40 * height, 0.60 * height)
    if bust_m:
        measurements.update({
            "bust_cm": bust_m * 100.0,
            "_bust_z": bust_z,
            "_bust_pct": bust_z / height * 100.0,
            "_bust_measurement_source": bust_source,
            "_bust_full_torso_component": bust_details,
            "_bust_search_high_pct": bust_details["search_high_pct"],
            "_bust_search_high_source": "bilateral_shoulder_line",
        })
        if bust_render_pts is not None:
            measurements["_bust_render_pts"] = bust_render_pts
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
    # Regular-fit T-shirt target: 90% of a high-point-shoulder path which
    # crosses the anterior chest before running down the front torso.  It
    # deliberately has no waist or hip dependency.
    hps_to_crotch_cm, hps_to_crotch_pts = (0.0, None)
    shirt_start_meta: dict[str, Any] = {"confidence": "low", "reason": "crotch_unavailable"}
    crotch_z = float(measurements.get("_inseam_z", 0.0))
    if crotch_z > 0:
        hps_to_crotch_cm, hps_to_crotch_pts, shirt_start_meta = measure_hps_to_crotch_front_surface(
            body.joints or {}, body.mesh, crotch_z,
        )
    regular_tshirt_cm, regular_tshirt_pts = truncate_polyline_at_fraction(hps_to_crotch_pts, 0.90) if hps_to_crotch_pts is not None else (0.0, None)
    if regular_tshirt_pts is not None and regular_tshirt_cm > 0:
        measurements["hps_to_crotch_cm"] = float(hps_to_crotch_cm)
        measurements["regular_tshirt_length_cm"] = float(regular_tshirt_cm)
        measurements["shirt_length_cm"] = float(regular_tshirt_cm)
        measurements["_hps_to_crotch_pts"] = hps_to_crotch_pts
        measurements["_regular_tshirt_length_pts"] = regular_tshirt_pts
        measurements["_shirt_length_pts"] = regular_tshirt_pts
        measurements["_shirt_length_source"] = "regular_fit_90pct_hps_front_surface_trace"
        measurements["_shirt_length_definition"] = "0.90 * hps_to_crotch_surface_length"
        measurements["_shirt_start"] = shirt_start_meta
        polylines = measurements.get("_linear_polylines")
        if isinstance(polylines, dict):
            polylines["shirt_length"] = regular_tshirt_pts
        start_conf = str(measurements.get("_shirt_start", {}).get("confidence", "low"))
        side_q = _quality(start_conf, "regular_fit_90pct_hps_front_surface_trace")
    else:
        side_q = _quality("low", "hps_to_crotch_trace_failed")
        measurements["_shirt_start"] = shirt_start_meta
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
    measurements = measure(body, preset=None if only else args.preset, only=only, render_path=None)
    # Keep the upstream CLAD definition: acromion → posterior C7 → acromion.
    # In particular, do not substitute the garment seam anchor for C7.
    apply_surface_sleeve_length(measurements, body)
    apply_distinct_stomach_measurement(measurements, body)
    apply_surface_crotch_length(measurements, body)
    torso_mesh = apply_production_core_measurements(measurements, body)
    joint_meta = body.sam3d_params.get("_fused_joint_transfer", {})
    surface_meta = body.sam3d_params.get("_fused_joint_landmarks", {})
    suppressed = suppress_measurements_for_invalid_joints(measurements, joint_meta, surface_meta)
    measurements.setdefault("_shirt_length_source", "regular_fit_90pct_hps_front_surface_trace")
    measurements.update({
        "_measurement_engine": "datar-psa/clad-body@a2140a7",
        "_measurement_mesh_source": "fused_sdf_mesh",
        "_joint_landmark_source": "front_sam3d_posed_joints_with_mhr_skinning_same_topology_deformation",
        "_joint_transform_source": front_transform.get("source", "saved_sam3d_posed_joint_coords"),
        "_fused_mesh_transform": fused_transform,
        "_front_mhr_joint_transform": front_transform,
        "_fused_joint_transfer": joint_meta,
        "_fused_joint_landmarks": surface_meta,
        "_joint_validation": joint_meta,
        "_suppressed_measurements": suppressed,
        "_fusion_quality_status": scalar(body.sam3d_params.get("fusion_quality_status"), "unknown"),
        "_fusion_quality_errors": body.sam3d_params.get("fusion_quality_errors", []),
        "_measurement_mesh_vertices": len(body.mesh.vertices),
        "_measurement_mesh_faces": len(body.mesh.faces),
        "_fusion_rule": scalar(body.sam3d_params.get("fusion_rule"), ""),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(measurements), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        # Render the contour from the same arm-excluded torso mesh used for
        # the value, so the front panel cannot show a bust loop on an arm.
        render_4view(body.mesh, measurements, str(render_path), title=source.stem, model_label="Fused SDF / upstream CLAD", torso_mesh=torso_mesh)
    print(f"Saved measurements: {out_json}")
    if render_path is not None:
        print(f"Saved render      : {render_path}")
    print(f"Measured mesh     : {body.source} ({len(body.mesh.vertices)} verts, {len(body.mesh.faces)} faces)")


if __name__ == "__main__":
    main()
