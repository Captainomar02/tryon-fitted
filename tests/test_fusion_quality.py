import numpy as np
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_quality import (
    alignment_report,
    silhouette_report,
    similarity_align,
    torso_mask_from_landmarks,
)


UPSTREAM_MODULE = ROOT / "scripts" / "measure_fused_mesh_clad_upstream.py"
upstream_spec = importlib.util.spec_from_file_location("measure_fused_mesh_clad_upstream_joint_test", UPSTREAM_MODULE)
upstream = importlib.util.module_from_spec(upstream_spec)
assert upstream_spec.loader is not None
upstream_spec.loader.exec_module(upstream)


def _points(seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(160, 3)).astype(np.float64)


def test_similarity_alignment_recovers_scale_rotation_and_translation():
    source = _points()
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    target = 1.13 * (source @ rotation.T) + np.array([0.4, -0.2, 0.3])
    aligned, transform = similarity_align(source, target, np.ones(len(source), dtype=bool))
    assert np.allclose(aligned, target, atol=1e-5)
    assert abs(transform["scale"] - 1.13) < 1e-5


def test_alignment_report_rejects_large_pose_residual():
    front = _points()
    side = front.copy()
    side[:80, 2] += 0.8
    report = alignment_report(front, side, np.ones(len(front), dtype=bool), 1.7, {"scale": 1.0})
    assert "front_side_torso_pose_mismatch" in report["errors"]


def test_torso_mask_excludes_far_limb_vertices():
    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [3, 0, 0]], dtype=float)
    keypoints = np.zeros((11, 3), dtype=float)
    keypoints[5], keypoints[6] = [0.5, 1, 0], [-0.5, 1, 0]
    keypoints[9], keypoints[10] = [0.4, -1, 0], [-0.4, -1, 0]
    mask = torso_mask_from_landmarks(vertices, keypoints)
    assert mask.tolist() == [True, True, False]


def test_silhouette_report_rejects_shifted_mask():
    expected = np.zeros((100, 100), dtype=bool)
    expected[25:75, 25:75] = True
    shifted = np.zeros_like(expected)
    shifted[25:75, 70:100] = True
    report = silhouette_report(shifted, expected)
    assert "silhouette_iou_too_low" in report["errors"]


def test_joint_transfer_follows_same_topology_fusion_displacement():
    reference = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float32)
    fused = reference.copy()
    fused[:, 1] += 0.03
    joints, meta = upstream.transfer_joints_to_fused_mesh({"pelvis": np.array([0.02, 0.01, 0.0])}, reference, fused, 1.7)
    assert meta["enabled"] is True
    assert np.isclose(joints["pelvis"][1], 0.04, atol=2e-3)


def test_saved_joint_binding_uses_only_its_bound_vertex_deformation(monkeypatch):
    monkeypatch.setattr(upstream, "_mhr_joint_names", lambda: ["root", "l_uparm"])
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    fused = reference.copy()
    fused[1, 2] = 0.04
    joints, meta = upstream.transfer_posed_joints_by_saved_bindings(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 2], [1, 2]], dtype=np.int32),
        np.array([[1, 0], [1, 0]], dtype=np.float32),
        reference, fused, 1.7,
    )
    assert meta["enabled"] is True
    assert np.allclose(joints["root"], [0, 0, 0])
    assert np.allclose(joints["l_uparm"], [1, 0, 0.04])
