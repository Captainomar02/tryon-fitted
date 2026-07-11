import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure_fused_mesh_clad_upstream.py"
spec = importlib.util.spec_from_file_location("measure_fused_mesh_clad_upstream", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _profile(iliac_z: float, rib_z: float) -> tuple[np.ndarray, np.ndarray]:
    zs = np.arange(0.47, 0.76, 0.002)
    extent = 0.22 + 0.035 * np.exp(-((zs - iliac_z) / 0.012) ** 2)
    extent += 0.040 * np.exp(-((zs - rib_z) / 0.012) ** 2)
    return zs, extent


def test_iso_waist_is_bilateral_rib_iliac_midpoint():
    zs, left = _profile(0.56, 0.68)
    _, right = _profile(0.56, 0.68)
    result, reasons = module.infer_iso_waist_from_side_profiles(zs, left, right, 1.0)
    assert reasons == []
    assert result is not None
    assert abs(result["waist_z"] - 0.62) < 0.006


def test_iso_waist_rejects_bilateral_disagreement():
    zs, left = _profile(0.56, 0.68)
    _, right = _profile(0.58, 0.72)
    result, reasons = module.infer_iso_waist_from_side_profiles(zs, left, right, 1.0)
    assert result is None
    assert reasons == ["bilateral_disagreement"]


def test_iso_waist_rejects_missing_or_fragmented_profile():
    zs = np.arange(0.47, 0.76, 0.002)
    missing = np.full_like(zs, np.nan)
    result, reasons = module.infer_iso_waist_from_side_profiles(zs, missing, missing, 1.0)
    assert result is None
    assert "insufficient_side_profile" in " ".join(reasons)


def test_regular_tshirt_target_is_ninety_percent_of_full_hip_trace():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.8, 0.6]])
    length_cm, truncated = module.truncate_polyline_at_fraction(points, 0.90)
    assert length_cm == 126.0
    assert np.allclose(truncated[-1], [0.0, 0.66, 0.6])
