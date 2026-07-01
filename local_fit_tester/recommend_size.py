#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
DEFAULT_TRYON3D_ROOT = HOME / "tryon-3d"
DEFAULT_CLAD_BODY_ROOT = HOME / "clad-body"
DEFAULT_CONDA_EXE = HOME / "miniconda3" / "bin" / "conda"


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    _err(f"[local-fit] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command_failed:{proc.returncode}")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _import_fit_report_modules(tryon3d_root: Path):
    body_dir = tryon3d_root / "body"
    body_dir_str = str(body_dir)
    if body_dir_str not in sys.path:
        sys.path.insert(0, body_dir_str)

    from fit_report import build_fit_report_with_variants, load_all_variants_json  # type: ignore

    return build_fit_report_with_variants, load_all_variants_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local fit-mode tester: front image + side image + height => recommended size."
    )
    parser.add_argument("--front", required=True, help="Path to front image.")
    parser.add_argument("--side", required=True, help="Path to side image.")
    parser.add_argument("--height-cm", required=True, type=float, help="Person height in centimeters.")
    parser.add_argument(
        "--variants-json",
        required=True,
        help="Path to all-variants garment measurements JSON.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "local_fit_tester" / "runs"),
        help="Directory where temporary test runs will be written.",
    )
    parser.add_argument(
        "--tryon3d-root",
        default=str(DEFAULT_TRYON3D_ROOT),
        help="Path to the tryon-3d repo.",
    )
    parser.add_argument(
        "--clad-body-root",
        default=str(DEFAULT_CLAD_BODY_ROOT),
        help="Path to the clad-body repo.",
    )
    parser.add_argument(
        "--conda-exe",
        default=str(DEFAULT_CONDA_EXE),
        help="Path to the conda executable.",
    )
    parser.add_argument(
        "--sam3d-conda-env",
        default="sam_3d_body",
        help="Conda env used to run SAM-3D.",
    )
    parser.add_argument(
        "--clad-conda-env",
        default="sam_3d_body_unified",
        help="Conda env used to run Clad measurement extraction.",
    )
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        help="Keep generated run artifacts instead of deleting them.",
    )
    parser.add_argument(
        "--output-format",
        choices=["size", "json"],
        default="json",
        help="Output only the recommended size or a JSON object with size and body measurements.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Generate and keep the Clad render image.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    front_path = Path(args.front).expanduser().resolve()
    side_path = Path(args.side).expanduser().resolve()
    variants_path = Path(args.variants_json).expanduser().resolve()
    work_root = Path(args.work_dir).expanduser().resolve()
    tryon3d_root = Path(args.tryon3d_root).expanduser().resolve()
    clad_body_root = Path(args.clad_body_root).expanduser().resolve()
    conda_exe = Path(args.conda_exe).expanduser().resolve()

    for path in (front_path, side_path, variants_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    if not conda_exe.exists():
        raise FileNotFoundError(f"Conda executable not found: {conda_exe}")

    run_id = time.strftime("fit_%Y%m%d_%H%M%S")
    run_dir = work_root / run_id
    input_dir = run_dir / "input"
    sam_output_dir = run_dir / "sam_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    sam_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _copy_image(front_path, input_dir / "front.jpg")
        _copy_image(side_path, input_dir / "side.jpg")

        sam_cmd = [
            str(conda_exe),
            "run",
            "-n",
            args.sam3d_conda_env,
            "python",
            str(REPO_ROOT / "run_front_side_fusion.py"),
            "--target-height",
            str(float(args.height_cm)),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(sam_output_dir),
        ]
        _run(sam_cmd, cwd=REPO_ROOT)

        params_path = sam_output_dir / "front_fused_all_body_params_scaled.json"
        if not params_path.exists():
            raise FileNotFoundError(f"SAM-3D output JSON missing: {params_path}")

        body_measurements_path = run_dir / "body_measurements.json"
        render_path = run_dir / "clad_render.png"
        clad_cmd = [
            str(conda_exe),
            "run",
            "-n",
            args.clad_conda_env,
            "python",
            str(tryon3d_root / "body" / "clad_body_extractor.py"),
            "--params",
            str(params_path),
            "--out_json",
            str(body_measurements_path),
            "--out_render",
            str(render_path) if args.render else "",
            "--model",
            "auto",
            "--clad_body_root",
            str(clad_body_root),
        ]
        _run(clad_cmd)

        if not body_measurements_path.exists():
            raise FileNotFoundError(f"Clad body measurements JSON missing: {body_measurements_path}")

        body_measurements = _load_json(body_measurements_path)
        build_fit_report_with_variants, load_all_variants_json = _import_fit_report_modules(tryon3d_root)
        variants = load_all_variants_json(str(variants_path))
        report = build_fit_report_with_variants(
            float(args.height_cm),
            body_measurements,
            body_measurements,
            variants,
        )

        recommendation = report.get("shopify_recommendation") or {}
        recommended_size = recommendation.get("recommended_size_label")
        if not isinstance(recommended_size, str) or not recommended_size.strip():
            raise RuntimeError("no_recommended_size")

        report_path = run_dir / "fit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        if args.output_format == "size":
            print(recommended_size.strip())
        else:
            payload = {
                "recommended_size": recommended_size.strip(),
                "body_measurements_cm": body_measurements,
                "render_path": str(render_path) if args.render and render_path.exists() else None,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        _err(f"[local-fit] report saved to {report_path}")
        return 0
    finally:
        if not args.keep_run_dir and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
