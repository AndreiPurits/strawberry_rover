#!/usr/bin/env python3
"""Measure RoArm repositioning repeatability against a fixed checkerboard."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "ops/axm-monitor/agent"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pipelines.roarm_calibration.chessboard import CameraIntrinsics, detect_chessboard  # noqa: E402
from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402
from pipelines.roarm_calibration.urdf_fk import base_to_link5  # noqa: E402
from pipelines.roarm_m3_kinematics.conventions import (  # noqa: E402
    JOINT_NAMES,
    JointVector,
    joint_limit_margins,
    within_limits,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402
from tools.roarm_rgbd_board_validate import stats  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"


def feedback() -> dict:
    response = execute_rpc("feedback", {})
    value = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("T:105 feedback unavailable")
    return {key: float(value[key]) for key in ("b", "s", "e", "t", "r", "x", "y", "z")}


def q_from_feedback(value: dict) -> JointVector:
    return JointVector(value["b"], value["s"], value["e"], value["t"], value["r"])


def move(q: JointVector, settle_s: float) -> None:
    if not within_limits(q, "hard") or min(joint_limit_margins(q, "hard").values()) < 0.10:
        raise RuntimeError("unsafe joint target: {}".format(q.as_dict()))
    params = q.as_dict()
    params.update({"hand": 1.08, "spd": 0.0, "acc": 6.0})
    response = execute_rpc("joints_move", params)
    if not response.get("ok"):
        raise RuntimeError("joint move failed: {}".format(response))
    time.sleep(settle_s)


def wait_intrinsics(provider: Ros2RgbDepthProvider, timeout_s: float) -> CameraIntrinsics:
    deadline = time.monotonic() + timeout_s
    msg = provider._color_info_msg
    while msg is None and time.monotonic() < deadline:
        time.sleep(0.02)
        msg = provider._color_info_msg
    if msg is None:
        raise RuntimeError("CameraInfo unavailable")
    return CameraIntrinsics(
        int(msg.width), int(msg.height), np.asarray(msg.k).reshape(3, 3), np.asarray(msg.d),
        str(msg.header.frame_id) or "stereo_camera_color_optical_frame", True,
    )


def transform(value: dict) -> RigidTransform:
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"], dtype=np.float64))


def point_stats(points: np.ndarray) -> dict:
    center = np.mean(points, axis=0)
    errors = np.linalg.norm(points - center, axis=1)
    return {
        "mean_xyz_mm": center.tolist(),
        "xyz_std_mm": np.std(points, axis=0).tolist(),
        "xyz_span_mm": (np.max(points, axis=0) - np.min(points, axis=0)).tolist(),
        "norm_scatter_mm": stats(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--settle-s", type=float, default=3.2)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor = JointVector(0.0, -0.72, 2.95, 0.0, 0.0)
    away = (
        JointVector(0.25, -0.64, 2.82, 0.12, 0.12),
        JointVector(-0.25, -0.80, 3.02, -0.12, -0.12),
    )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    link5_camera = transform(calibration["report"]["T_link5_camera"])

    provider = Ros2RgbDepthProvider(
        rgb_topic=RGB_TOPIC, depth_topic=DEPTH_TOPIC, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=INFO_TOPIC)
    observations = []
    try:
        intrinsics = wait_intrinsics(provider, args.timeout_s)
        move(anchor, args.settle_s)
        for index in range(args.cycles):
            move(away[index % 2], args.settle_s)
            move(anchor, args.settle_s)
            fresh = feedback()
            provider._rgb_buf.clear()
            provider._depth_buf.clear()
            frame = provider.read(timeout_s=args.timeout_s)
            detection = detect_chessboard(
                frame.rgb_bgr,
                intrinsics,
                cols=9,
                rows=6,
                square_size_mm=18.0,
            )
            if detection is None:
                raise RuntimeError("board lost after return {}".format(index + 1))
            q_actual = q_from_feedback(fresh)
            base_board = base_to_link5(q_actual.as_array()).then(link5_camera).then(
                detection.transform_camera_board
            )
            observations.append(
                {
                    "cycle": index + 1,
                    "away_direction": "positive" if index % 2 == 0 else "negative",
                    "feedback": fresh,
                    "q_error_from_command_rad": (q_actual.as_array() - anchor.as_array()).tolist(),
                    "T_base_board": base_board.matrix.tolist(),
                    "reprojection_rmse_px": detection.reprojection_rmse_px,
                }
            )
            print(json.dumps({"cycle": index + 1, "feedback": fresh}), flush=True)
    finally:
        provider.close()

    base_board_xyz = np.asarray([np.asarray(row["T_base_board"])[:3, 3] for row in observations])
    feedback_xyz = np.asarray(
        [[row["feedback"][key] for key in ("x", "y", "z")] for row in observations]
    )
    q_actual = np.asarray(
        [[row["feedback"][key] for key in ("b", "s", "e", "t", "r")] for row in observations]
    )
    report = {
        "status": "MEASURED_REPOSITIONING",
        "cycles": args.cycles,
        "anchor_command_rad": anchor.as_dict(),
        "away_commands_rad": [value.as_dict() for value in away],
        "board_derived_cartesian_repeatability": point_stats(base_board_xyz),
        "firmware_T105_cartesian_repeatability": point_stats(feedback_xyz),
        "actual_joint_return": {
            "mean_rad": dict(zip(JOINT_NAMES, np.mean(q_actual, axis=0).tolist())),
            "std_rad": dict(zip(JOINT_NAMES, np.std(q_actual, axis=0).tolist())),
            "span_rad": dict(zip(JOINT_NAMES, (np.max(q_actual, axis=0) - np.min(q_actual, axis=0)).tolist())),
        },
        "reprojection_rmse_px": stats([row["reprojection_rmse_px"] for row in observations]),
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
