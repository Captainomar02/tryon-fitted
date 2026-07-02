#!/usr/bin/env python3
"""Run front/side fusion, then extract CLAD body measurements and render."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
DEFAULT_CONDA_EXE = HOME / "miniconda3" / "bin" / "conda"


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    _err(f"[body-measure] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command_failed:{proc.returncode}")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _image_destination(input_dir: Path, stem: str, src: Path) -> Path:
    suffix = src.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    return input_dir / f"{stem}{suffix}"


def _resolve_conda_exe(conda_arg: str) -> Path | None:
    """Return a conda executable, or None when the current Python should be used."""
    if conda_arg.strip():
        conda_path = Path(conda_arg).expanduser().resolve()
        if conda_path.exists():
            return conda_path
        raise FileNotFoundError(f"Conda executable not found: {conda_path}")

    env_conda = os.environ.get("CONDA_EXE", "").strip()
    if env_conda:
        conda_path = Path(env_conda).expanduser().resolve()
        if conda_path.exists():
            return conda_path

    path_conda = shutil.which("conda")
    if path_conda:
        return Path(path_conda).resolve()

    if DEFAULT_CONDA_EXE.exists():
        return DEFAULT_CONDA_EXE.resolve()

    _err("[body-measure] conda not found; running both steps with the current Python.")
    return None


def _python_command(conda_exe: Path | None, env_name: str, script: Path) -> list[str]:
    if conda_exe is None:
        return [sys.executable, str(script)]
    return [str(conda_exe), "run", "-n", env_name, "python", str(script)]


def _with_pythonpath(*paths: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    path_parts = [str(path) for path in paths if path.exists()]
    if existing:
        path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM-3D front/side fusion from a front image, side image, and "
            "height, then extract CLAD body measurements and a render."
        )
    )
    parser.add_argument("--front", required=True, help="Path to the front image.")
    parser.add_argument("--side", required=True, help="Path to the side image.")
    parser.add_argument("--height-cm", required=True, type=float, help="Person height in centimeters.")
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "local_fit_tester" / "runs"),
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run folder name. Defaults to measure_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--conda-exe",
        default="",
        help="Optional path to conda. Defaults to CONDA_EXE, PATH, then current Python if conda is unavailable.",
    )
    parser.add_argument(
        "--sam3d-conda-env",
        default="sam_3d_body",
        help="Conda env used to run SAM-3D front/side fusion when conda is available.",
    )
    parser.add_argument(
        "--clad-conda-env",
        default="sam_3d_body_unified",
        help="Conda env used to run CLAD measurement extraction when conda is available.",
    )
    parser.add_argument(
        "--measure-preset",
        default="all",
        help="CLAD measurement preset passed to scripts/measure_mhr_params.py.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated measurement keys. Overrides --measure-preset when provided.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional CLAD measurement device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "paths"],
        default="json",
        help="Print a JSON summary or only output paths.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    front_path = Path(args.front).expanduser().resolve()
    side_path = Path(args.side).expanduser().resolve()
    work_root = Path(args.work_dir).expanduser().resolve()
    conda_exe = _resolve_conda_exe(args.conda_exe)

    for path in (front_path, side_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    run_name = args.run_name.strip() or time.strftime("measure_%Y%m%d_%H%M%S")
    run_dir = work_root / run_name
    input_dir = run_dir / "input"
    fusion_output_dir = run_dir / "fusion_output"
    body_measurements_path = run_dir / "body_measurements.json"
    render_path = run_dir / "clad_body_render.png"

    input_dir.mkdir(parents=True, exist_ok=True)
    fusion_output_dir.mkdir(parents=True, exist_ok=True)

    _copy_image(front_path, _image_destination(input_dir, "front", front_path))
    _copy_image(side_path, _image_destination(input_dir, "side", side_path))

    fusion_cmd = _python_command(
        conda_exe,
        args.sam3d_conda_env,
        REPO_ROOT / "run_front_side_fusion.py",
    ) + [
        "--target-height",
        str(float(args.height_cm)),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(fusion_output_dir),
    ]
    _run(fusion_cmd, cwd=REPO_ROOT)

    params_path = fusion_output_dir / "front_fused_all_body_params_scaled.json"
    if not params_path.exists():
        raise FileNotFoundError(f"SAM-3D output JSON missing: {params_path}")

    measure_cmd = _python_command(
        conda_exe,
        args.clad_conda_env,
        REPO_ROOT / "scripts" / "measure_mhr_params.py",
    ) + [
        "--params",
        str(params_path),
        "--out-json",
        str(body_measurements_path),
        "--render",
        str(render_path),
        "--preset",
        args.measure_preset,
    ]
    if args.only.strip():
        measure_cmd.extend(["--only", args.only.strip()])
    if args.device.strip():
        measure_cmd.extend(["--device", args.device.strip()])
    measure_env = _with_pythonpath(REPO_ROOT / "clad-body")
    _run(measure_cmd, cwd=REPO_ROOT, env=measure_env)

    if not body_measurements_path.exists():
        raise FileNotFoundError(f"CLAD body measurements JSON missing: {body_measurements_path}")
    if not render_path.exists():
        raise FileNotFoundError(f"CLAD render missing: {render_path}")

    if args.output_format == "paths":
        print(f"params_json={params_path}")
        print(f"measurements_json={body_measurements_path}")
        print(f"render_png={render_path}")
        print(f"run_dir={run_dir}")
    else:
        payload = {
            "run_dir": str(run_dir),
            "params_json": str(params_path),
            "measurements_json": str(body_measurements_path),
            "render_png": str(render_path),
            "body_measurements_cm": _load_json(body_measurements_path),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    _err(f"[body-measure] run artifacts saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
