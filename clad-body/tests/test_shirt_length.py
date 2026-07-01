"""Regression tests for the standard shirt-length target.

These tests focus on the fitted-top path in ``clad_body.measure``:
the same stored shirt polyline must drive both the cm measurement and the
dark-blue render overlay.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clad_body.load.mhr import load_mhr_from_params
from clad_body.measure import measure
from clad_body.measure._lengths import (
    extract_linear_measurement_polylines,
)

TESTDATA_DIR = ROOT / "clad_body" / "measure" / "testdata" / "mhr"
SUBJECTS = [
    "female_average",
    "female_slim",
    "female_curvy",
    "female_plus_size",
    "male_plus_size",
]
TOLERANCE_CM = 0.1


def _load_body(subject: str):
    return load_mhr_from_params(str(TESTDATA_DIR / subject / "mhr_params.json"))


@pytest.mark.parametrize("subject", SUBJECTS)
def test_shirt_length_matches_updated_regression(subject):
    body = _load_body(subject)
    with open(TESTDATA_DIR / subject / "expected_measurements.json") as fh:
        expected = json.load(fh)

    measured = measure(body, only=["shirt_length_cm"])

    assert abs(measured["shirt_length_cm"] - expected["shirt_length_cm"]) <= TOLERANCE_CM


@pytest.mark.parametrize("subject", SUBJECTS)
def test_shirt_length_ends_at_hip_line(subject):
    body = _load_body(subject)
    measured = measure(body, only=["shirt_length_cm"])

    crotch_z = measured["_inseam_z"]
    target_z = measured["_hip_z"]
    hem_z = float(measured["_shirt_length_pts"][-1, 2])

    assert crotch_z < hem_z
    assert abs(hem_z - target_z) <= 0.015


def test_shirt_length_render_polyline_matches_measurement():
    body = _load_body("female_average")
    measured = measure(body, only=["shirt_length_cm"])

    polylines = extract_linear_measurement_polylines(
        measured["mesh"],
        measured,
        body.joints or {},
    )

    np.testing.assert_allclose(
        polylines["shirt_length"],
        measured["_shirt_length_pts"],
    )
