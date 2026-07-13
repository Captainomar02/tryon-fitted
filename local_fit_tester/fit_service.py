#!/usr/bin/env python3
"""HTTP adapter from the TryOn worker to the SAM-3D + CLAD measurement pipeline.

This service implements the worker's existing POST /measure contract.  It is
strictly for fitted mode; it does not receive or alter preview-mode requests.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = REPO_ROOT / "local_fit_tester" / "extract_body_measurements.py"
JOB_ROOT = Path(os.environ.get("FIT_SERVICE_JOB_ROOT", REPO_ROOT / "local_fit_tester" / "service_jobs"))
HOST = os.environ.get("FIT_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FIT_SERVICE_PORT", "8098"))
TOKEN = os.environ.get("FIT_EXTERNAL_BODY_SERVICE_TOKEN", "")
MAX_REQUEST_BYTES = int(os.environ.get("FIT_SERVICE_REQUEST_MAX_BYTES", "25000000"))


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _copy_number(target: dict[str, Any], source: dict[str, Any], source_key: str, target_key: str) -> None:
    value = _number(source.get(source_key))
    if value is not None:
        target[target_key] = value


def canonical_measurements(clad: dict[str, Any]) -> dict[str, Any]:
    """Map current CLAD JSON to the schema consumed by TryOn's fit_report.py."""
    canonical: dict[str, Any] = {
        "measurement_unit": "cm",
        "measurement_source": "sam3d_clad_fused_mesh",
    }
    for source_key, target_key in (
        ("height_cm", "body_height"),
        ("bust_cm", "chest_circumference"),
        ("bust_cm", "bust_circumference"),
        ("upperarm_cm", "bicep_circumference"),
        ("neck_cm", "neck_circumference"),
        ("stomach_cm", "stomach_circumference"),
        ("waist_cm", "stomach_circumference"),
        ("thigh_cm", "thigh_circumference"),
        ("shoulder_width_cm", "shoulder_to_shoulder_length"),
        ("sleeve_length_cm", "arm_length"),
        ("inseam_cm", "leg_length"),
        ("shirt_length_cm", "neck_to_hip_length"),
        ("back_neck_to_waist_cm", "neck_to_hip_length"),
        ("hip_cm", "hip_circumference"),
    ):
        # Prefer a more-specific value when the target is already populated.
        if target_key not in canonical:
            _copy_number(canonical, clad, source_key, target_key)

    canonical["clad_measurements_cm"] = {
        key: value for key, value in clad.items() if not str(key).startswith("_") and isinstance(value, (int, float))
    }
    return canonical


def _safe_job_id(value: Any) -> str:
    proposed = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return proposed[:100] or f"fit_{int(time.time())}"


def _decode_image(value: Any, label: str) -> bytes:
    try:
        data = base64.b64decode(str(value or ""), validate=True)
    except Exception as exc:
        raise ValueError(f"invalid_{label}_image_base64") from exc
    if not data:
        raise ValueError(f"empty_{label}_image")
    return data


def run_measurement(*, job_id: str, height_cm: float, front: bytes, side: bytes) -> dict[str, Any]:
    job_dir = JOB_ROOT / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "front.jpg").write_bytes(front)
    (input_dir / "side.jpg").write_bytes(side)

    cmd = [
        sys.executable,
        str(EXTRACTOR),
        "--front", str(input_dir / "front.jpg"),
        "--side", str(input_dir / "side.jpg"),
        "--height-cm", str(height_cm),
        "--work-dir", str(JOB_ROOT),
        "--run-name", job_id,
        "--output-format", "paths",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"measurement_pipeline_failed:{detail}")

    measurement_path = job_dir / "body_measurements.json"
    if not measurement_path.is_file():
        raise RuntimeError("measurement_pipeline_missing_output")
    raw = json.loads(measurement_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("measurement_pipeline_invalid_output")
    return canonical_measurements(raw)


class Handler(BaseHTTPRequestHandler):
    server_version = "TryOnFittedService/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        self._send(200, {"ok": True, "service": "sam3d_clad_fitted", "jobRoot": str(JOB_ROOT)})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/measure":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        if TOKEN and self.headers.get("x-fit-service-token", "") != TOKEN:
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send(400, {"ok": False, "error": "invalid_payload_size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload_must_be_object")
            job_id = _safe_job_id(payload.get("jobId"))
            height_cm = _number(payload.get("heightCm"))
            if height_cm is None or not 50.0 <= height_cm <= 300.0:
                raise ValueError("invalid_height_cm")
            measurements = run_measurement(
                job_id=job_id,
                height_cm=height_cm,
                front=_decode_image(payload.get("frontImageBase64"), "front"),
                side=_decode_image(payload.get("sideImageBase64"), "side"),
            )
            self._send(200, {"ok": True, "jobId": job_id, "pipeline": "sam3d_clad_fitted", "bodyMeasurements": measurements})
        except Exception as exc:
            self._send(500, {"ok": False, "error": type(exc).__name__, "reason": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve SAM-3D + CLAD measurements to the TryOn fitted worker.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[fit-service] listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
