"""Load MHR body meshes from SAM3D params JSON via pymomentum.

All outputs are Z-up, metres, XY-centred, feet at Z = 0, +Y=front.
Matches Anny convention — both loaders produce the same canonical orientation.

Uses :func:`load_mhr_from_params` — generates from SAM3D params JSON via
pymomentum subprocess.  Deterministic coordinate conversion (no heuristic
orientation detection).
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


@dataclass
class MhrBody:
    """MHR body mesh in canonical rest-pose convention.

    Coordinate system: Z-up, metres, XY-centred, feet at Z = 0, +Y=front.
    Matches Anny convention — both loaders produce the same orientation.
    """

    mesh: trimesh.Trimesh        # canonical-positioned mesh
    source: str                  # e.g. "params:sam3d_mhr_restpose_params.json"
    obj_path: Optional[str] = None
    sam3d_params: Optional[dict] = None  # shape_params, scale_params, raw_height_m
    joints: Optional[dict] = None  # canonical joint name → (3,) position (Z-up, m)

    @property
    def height_m(self) -> float:
        verts = np.asarray(self.mesh.vertices)
        return float(verts[:, 2].max() - verts[:, 2].min())

    @property
    def n_vertices(self) -> int:
        return len(self.mesh.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.mesh.faces)

    def __repr__(self) -> str:
        params_str = " +params" if self.sam3d_params else ""
        return (
            f"MhrBody({self.n_vertices} verts, {self.n_faces} faces, "
            f"height={self.height_m:.3f}m, source='{self.source}'{params_str})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mhr_yup_cm_to_canonical(verts_yup_cm: np.ndarray) -> np.ndarray:
    """Convert pymomentum output (Y-up, cm) to canonical (Z-up, m, +Y=front).

    Deterministic transformation — no orientation detection needed.
    MHR native: Y=height, -Z=front.  Target: Z=height, +Y=front.
    """
    v = np.zeros_like(verts_yup_cm, dtype=np.float32)
    v[:, 0] = verts_yup_cm[:, 0] / 100.0        # X stays X, cm → m
    v[:, 1] = -verts_yup_cm[:, 2] / 100.0       # MHR -Z=front → +Y=front
    v[:, 2] = verts_yup_cm[:, 1] / 100.0        # MHR Y=height → Z=height
    v[:, 2] -= v[:, 2].min()                     # feet at Z = 0
    return v


def _resolve_params_json(path: str) -> str:
    """Resolve path to sam3d_mhr_restpose_params.json."""
    if path.endswith(".json") and os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "sam3d_mhr_restpose_params.json")
        if os.path.exists(candidate):
            return candidate
    # Try sibling of OBJ
    if path.endswith(".obj"):
        candidate = path.replace(".obj", "_params.json")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No params JSON found for: {path}")


def _extract_target_height_m(params: dict) -> Optional[float]:
    """Read optional target height as meters from params JSON.

    Unit conventions:
    - All target-height fields are interpreted as centimeters.
    - Internal return value remains meters for mesh scaling.
    """
    for key in ("fusion_target_height_cm", "fusion_target_height", "target_height"):
        if key in params:
            try:
                value_cm = float(params[key])
            except (TypeError, ValueError):
                continue
            if value_cm > 0:
                return value_cm / 100.0
    return None


def _normalise_canonical_vertices(verts: np.ndarray, target_height_m: Optional[float]) -> np.ndarray:
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3).copy()
    if not np.isfinite(verts).all():
        raise ValueError("Fusion vertex override contains non-finite values")

    verts[:, 2] -= verts[:, 2].min()
    center_xy = (verts[:, :2].max(axis=0) + verts[:, :2].min(axis=0)) / 2
    verts[:, 0] -= center_xy[0]
    verts[:, 1] -= center_xy[1]

    if target_height_m is not None:
        cur_h = float(verts[:, 2].max() - verts[:, 2].min())
        if cur_h > 1e-8:
            verts *= float(target_height_m) / cur_h
    return verts


def _extract_fusion_vertices_override(
    params: dict,
    faces: np.ndarray,
    target_height_m: Optional[float],
) -> Optional[np.ndarray]:
    """Return CLAD-canonical fusion vertices when a fusion JSON opts in."""
    if not bool(params.get("fusion_prefer_vertices_for_clad", False)):
        return None

    raw_vertices = params.get("fusion_vertices_clad")
    if raw_vertices is None:
        return None

    verts = _normalise_canonical_vertices(np.asarray(raw_vertices), target_height_m)
    if faces.size and int(np.asarray(faces).max()) >= len(verts):
        raise ValueError(
            "Fusion vertex override does not match MHR topology: "
            f"{len(verts)} vertices for face max index {int(np.asarray(faces).max())}"
        )
    return verts


def _extract_fusion_faces_override(params: dict) -> Optional[np.ndarray]:
    """Return face topology stored with a fusion vertex override, if present."""
    raw_faces = params.get("fusion_faces_clad")
    if raw_faces is None:
        raw_faces = params.get("fusion_faces")
    if raw_faces is None:
        return None

    faces = np.asarray(raw_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Fusion face override must have shape [N,3], got {faces.shape}")
    if faces.size and int(faces.min()) < 0:
        raise ValueError("Fusion face override contains negative indices")
    return faces.astype(np.int32, copy=False)


def _load_fusion_vertex_body_if_available(params_json: str, params: dict) -> Optional[MhrBody]:
    """Build a body directly from fusion vertices/faces without pymomentum.

    Fusion JSONs already contain CLAD-canonical vertices. When they also carry
    topology, measuring the mesh does not need the MHR/pymomentum subprocess.
    """
    if not bool(params.get("fusion_prefer_vertices_for_clad", False)):
        return None

    faces = _extract_fusion_faces_override(params)
    if faces is None:
        return None

    target_height_m = _extract_target_height_m(params)
    verts = _extract_fusion_vertices_override(params, faces, target_height_m)
    if verts is None:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return MhrBody(
        mesh=mesh,
        source=f"params:{os.path.basename(params_json)}+fusion_vertices",
        sam3d_params=params,
        joints=None,
    )


def _profile_band_weight(height_pct: np.ndarray, center: float, half_width: float) -> np.ndarray:
    distance = np.abs(height_pct - float(center))
    weight = np.clip(1.0 - (distance / float(half_width)), 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def _float_param(params: dict, key: str, default: float = 1.0) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _apply_profile_depth_to_mhr_restpose(verts: np.ndarray, params: dict) -> np.ndarray:
    """Apply fusion-derived bust/hip profile depth to a valid MHR rest-pose mesh."""
    if not bool(params.get("fusion_apply_profile_depth_to_mhr", False)):
        return verts

    bust_scale = _float_param(params, "fusion_profile_depth_bust_scale", 1.0)
    hip_scale = _float_param(params, "fusion_profile_depth_hip_scale", 1.0)
    if max(bust_scale, hip_scale) <= 1.0001:
        return verts

    corrected = np.asarray(verts, dtype=np.float32).copy()
    z_min = float(corrected[:, 2].min())
    height = float(corrected[:, 2].max() - z_min)
    if height <= 1e-8:
        return corrected

    pct = (corrected[:, 2] - z_min) / height
    total_weight = np.zeros(len(corrected), dtype=np.float32)
    total_weight += _profile_band_weight(pct, 0.725, 0.075) * (bust_scale - 1.0)
    total_weight += _profile_band_weight(pct, 0.50, 0.09) * (hip_scale - 1.0)

    if float(total_weight.max()) <= 0.0:
        return corrected

    unique_bins = np.floor(pct * 200).astype(np.int32)
    y_center = np.zeros(len(corrected), dtype=np.float32)
    for bin_id in np.unique(unique_bins):
        bin_mask = unique_bins == bin_id
        y_center[bin_mask] = np.median(corrected[bin_mask, 1])
    corrected[:, 1] = y_center + (corrected[:, 1] - y_center) * (1.0 + total_weight)
    return corrected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mhr_from_params_dict(
    params: dict,
    elbow_bend: float = -0.5,
) -> MhrBody:
    """Generate MHR body from a SAM3D params dict (no file on disk required).

    Writes a temporary JSON file and delegates to :func:`load_mhr_from_params`.

    Args:
        params: SAM3D params dict with at least ``shape_params`` and ``scale_params``.
        elbow_bend: MHR elbow_bend parameter (default: -0.5).

    Returns:
        :class:`MhrBody` in canonical rest-pose (Z-up, m, +Y=front, XY-centred).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as jf:
        json.dump(params, jf)
        tmp_path = jf.name
    try:
        body = load_mhr_from_params(tmp_path, elbow_bend=elbow_bend)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return body


def load_mhr_from_params(
    path: str,
    elbow_bend: float = -0.5,
) -> MhrBody:
    """Generate MHR body from SAM3D params JSON via pymomentum.

    Deterministic coordinate conversion, no heuristic orientation detection.
    Runs pymomentum in a subprocess (import-order isolation:
    ``pymomentum.geometry`` must come before ``torch``).

    Args:
        path: Path to params JSON, SAM3D results directory, or restpose OBJ
              (auto-finds companion ``_params.json``).
        elbow_bend: MHR elbow_bend parameter (default: -0.5).

    Returns:
        :class:`MhrBody` in canonical rest-pose (Z-up, m, +Y=front, XY-centred).
    """
    params_json = _resolve_params_json(path)
    with open(params_json) as f:
        sam3d_params = json.load(f)

    fusion_body = _load_fusion_vertex_body_if_available(params_json, sam3d_params)
    if fusion_body is not None:
        return fusion_body

    # Ensure LD_LIBRARY_PATH includes torch/lib for pymomentum-cpu
    import importlib.util
    _torch_spec = importlib.util.find_spec("torch")
    if _torch_spec and _torch_spec.origin:
        _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _torch_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{_ld}" if _ld else _torch_lib

    repo_root = Path(__file__).resolve().parents[3]
    default_mhr_assets_dir = repo_root / "checkpoints" / "mhr-assets" / "assets"
    mhr_assets_dir = Path(os.environ.get("MHR_ASSETS_DIR", str(default_mhr_assets_dir))).expanduser().resolve()

    # Subprocess script — pymomentum.geometry MUST import before torch
    script = f"""\
import sys, os
from pathlib import Path
import pymomentum.geometry  # noqa: F401 — MUST come before torch
import pymomentum.skel_state as pym_skel_state
import json, torch, numpy as np
from mhr.mhr import MHR

with open({params_json!r}) as f:
    params = json.load(f)

shape_t = torch.tensor(params["shape_params"], dtype=torch.float32).unsqueeze(0)

asset_dir = Path({str(mhr_assets_dir)!r})
model = MHR.from_files(folder=asset_dir, device="cpu", wants_pose_correctives=False)

# Rest pose: zero translation/rotation/pose, keep scale from model output.
# Layout: [trans(3)|rot(3)|pose(130)|scale(68)] = 204
mp = torch.zeros(1, 204, dtype=torch.float32)
mp[0, 36] = {elbow_bend}   # r_elbow_bend
mp[0, 46] = {elbow_bend}   # l_elbow_bend

# Scale params: either from full mhr_model_params[136:] or standalone scale_params
if "mhr_model_params" in params:
    mhr_params = torch.tensor(params["mhr_model_params"], dtype=torch.float32)
    mp[0, 136:] = mhr_params[136:]
elif "scale_params" in params:
    scale_t = torch.tensor(params["scale_params"], dtype=torch.float32)
    mp[0, 136:136+len(scale_t)] = scale_t

with torch.no_grad():
    verts, skel_state = model(shape_t, mp, None, apply_correctives=True)

v = verts[0].cpu().numpy().astype(np.float32)
faces = np.array(model.character.mesh.faces, dtype=np.int32)

# Extract joint positions from skeleton state
joint_positions, _, _ = pym_skel_state.split(skel_state)
joint_pos = joint_positions[0].cpu().numpy().astype(np.float32)
joint_names = model.character_torch.skeleton.joint_names

np.savez(sys.argv[1], vertices=v, faces=faces,
         joint_positions=joint_pos,
         joint_names=np.array(joint_names))
"""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as nf:
        npz_path = nf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(script)
        script_path = sf.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, npz_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr[-800:] if result.stderr else "unknown error"
            raise RuntimeError(f"pymomentum subprocess failed:\n{err}")

        data = np.load(npz_path, allow_pickle=True)
        verts_yup_cm = data["vertices"]
        faces = data["faces"]
        joint_pos_raw = data.get("joint_positions")
        joint_names_raw = data.get("joint_names")
    finally:
        for p in (npz_path, script_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Deterministic conversion: Y-up cm → Z-up m, +Y=front
    verts = _mhr_yup_cm_to_canonical(verts_yup_cm)

    # XY-centre
    center_xy = (verts[:, :2].max(axis=0) + verts[:, :2].min(axis=0)) / 2
    verts[:, 0] -= center_xy[0]
    verts[:, 1] -= center_xy[1]

    # Optional post-fit global scaling:
    # if the params JSON carries a desired target height, enforce it directly
    # on the reconstructed canonical mesh.
    target_height_m = _extract_target_height_m(sam3d_params)
    if target_height_m is not None:
        cur_h = float(verts[:, 2].max() - verts[:, 2].min())
        if cur_h > 1e-8:
            h_scale = target_height_m / cur_h
            verts *= h_scale

    verts = _apply_profile_depth_to_mhr_restpose(verts, sam3d_params)

    fusion_override_verts = _extract_fusion_vertices_override(
        sam3d_params,
        faces,
        target_height_m,
    )
    mesh_verts = fusion_override_verts if fusion_override_verts is not None else verts
    mesh = trimesh.Trimesh(vertices=mesh_verts, faces=faces, process=False)

    # Extract canonical joint positions (same coordinate transform as vertices)
    joints = None
    if joint_pos_raw is not None and joint_names_raw is not None:
        from clad_body.measure.mhr import MHR_JOINT_MAP
        from clad_body.measure._lengths import extract_joints_from_names
        # Joint positions are in MHR native Y-up cm → convert to Z-up m
        joint_pos_canonical = _mhr_yup_cm_to_canonical(joint_pos_raw)
        # Apply same XY centering as vertices
        joint_pos_canonical[:, 0] -= center_xy[0]
        joint_pos_canonical[:, 1] -= center_xy[1]
        if target_height_m is not None:
            cur_h = float(joint_pos_canonical[:, 2].max() - joint_pos_canonical[:, 2].min())
            if cur_h > 1e-8:
                j_scale = target_height_m / cur_h
                joint_pos_canonical *= j_scale
        joint_names = list(joint_names_raw)
        joints = extract_joints_from_names(
            joint_names, joint_pos_canonical, MHR_JOINT_MAP)

    source_suffix = "+fusion_vertices" if fusion_override_verts is not None else ""
    return MhrBody(
        mesh=mesh,
        source=f"params:{os.path.basename(params_json)}{source_suffix}",
        sam3d_params=sam3d_params,
        joints=joints,
    )
