#!/usr/bin/env python3
"""Offline pixel/depth -> camera/base -> standoff IK/FK audit on saved detections."""
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
from pipelines.roarm_m3_kinematics import JointVector  # noqa: E402
from pipelines.roarm_m3_kinematics.cartesian import (  # noqa: E402
    pixel_depth_to_camera_mm,
    plan_cartesian_pose,
    point_camera_to_base_mm,
    standoff_target_pose,
)


def transform(value: dict) -> RigidTransform:
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("approach_run", type=Path)
    parser.add_argument("perception_run", type=Path)
    parser.add_argument("--standoff-mm", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    source = json.loads(args.observations.read_text(encoding="utf-8"))
    approach = json.loads(args.approach_run.read_text(encoding="utf-8"))
    perception = json.loads(args.perception_run.read_text(encoding="utf-8"))
    link5_camera = transform(calibration["report"]["T_link5_camera"])
    camera = source["metadata"]["intrinsics"]
    matrix = np.asarray(camera["K"], dtype=np.float64)
    intrinsics = (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])
    joints = approach["joints_end"]
    q = JointVector(*(float(joints[name]) for name in ("base", "shoulder", "elbow", "wrist", "roll")))

    berry = perception["framing"]["berry"]
    last = perception["last_perception"]
    points = {
        "berry_center": {
            "pixel": [float(berry["px"]), float(berry["py"])],
            "depth_m": float(berry["depth_m"]),
        },
        "calyx": {
            "pixel": [float(last["calyx"]["x"]), float(last["calyx"]["y"])],
            "depth_m": float(last["depth_m"]),
        },
        "grasp_point": {
            "pixel": list(map(float, perception["grasp_point"]["xy"])),
            "depth_m": float(perception["grasp_point"]["depth_m"]),
        },
    }
    rows = []
    for name, value in points.items():
        p_camera = pixel_depth_to_camera_mm(
            value["pixel"][0], value["pixel"][1], value["depth_m"], intrinsics
        )
        p_base = point_camera_to_base_mm(p_camera, q, link5_camera)
        target = standoff_target_pose(
            p_camera,
            p_base,
            q,
            link5_camera,
            standoff_mm=args.standoff_mm,
        )
        plan = plan_cartesian_pose(target, q, stage="coarse")
        rows.append(
            {
                "name": name,
                "pixel": value["pixel"],
                "depth_m": value["depth_m"],
                "p_camera_mm": p_camera.tolist(),
                "p_base_mm": p_base.tolist(),
                "target_standoff_pose": target.as_dict(),
                "plan": plan.as_dict(),
            }
        )
    passed = all(row["plan"]["status"] == "PASS" for row in rows)
    report = {
        "status": "PASS" if passed else "FAIL",
        "motion": "NONE_OFFLINE_ONLY",
        "transform": "T_link5_camera_v1",
        "standoff_mm": args.standoff_mm,
        "q_source": q.as_dict(),
        "points": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
