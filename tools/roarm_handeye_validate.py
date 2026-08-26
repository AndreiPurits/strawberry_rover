#!/usr/bin/env python3
"""Validate a solved T_link5_camera on independently measured base-frame points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402
from pipelines.roarm_calibration.validation import (  # noqa: E402
    KnownPointObservation,
    validate_known_points,
)


def transform(value):
    if value.get("units") != "mm":
        raise ValueError("expected millimetre transform")
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("known_points", type=Path)
    parser.add_argument("--max-error-mm", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    point_payload = json.loads(args.known_points.read_text(encoding="utf-8"))
    value = calibration["report"]["T_link5_camera"]
    link5_camera = transform(value)
    observations = [
        KnownPointObservation(
            observation_id=item["id"],
            transform_base_link5=transform(item["T_base_link5"]),
            point_camera_mm=np.asarray(item["point_camera_mm"], dtype=np.float64),
            point_base_expected_mm=np.asarray(item["point_base_expected_mm"], dtype=np.float64),
        )
        for item in point_payload["observations"]
    ]
    report = validate_known_points(
        link5_camera, observations, max_error_mm=args.max_error_mm
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "KIN_GATE": report["KIN_GATE"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
