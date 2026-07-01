#!/usr/bin/env python
"""Measure a SAM 3D Body MHR parameter JSON with CLAD Body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clad_body.load import load_mhr_from_params
from clad_body.measure import measure


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


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

    body = load_mhr_from_params(str(params_path))
    measurements = measure(
        body,
        preset=None if only else args.preset,
        only=only,
        render_path=str(render_path) if render_path else None,
        device=args.device,
    )

    with out_json_path.open("w") as f:
        json.dump(to_jsonable(measurements), f, indent=2, sort_keys=True)

    print(f"Saved measurements: {out_json_path}")
    if render_path is not None:
        print(f"Saved render      : {render_path}")


if __name__ == "__main__":
    main()
