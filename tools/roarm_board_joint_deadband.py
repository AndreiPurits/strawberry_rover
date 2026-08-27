#!/usr/bin/env python3
"""Measure per-joint command deadband against a fixed checkerboard."""
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
from pipelines.roarm_m3_kinematics.conventions import (  # noqa: E402
    JOINT_NAMES,
    JointVector,
    joint_limit_margins,
    within_limits,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"


def feedback() -> dict:
    response = execute_rpc("feedback", {})
    value = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("T:105 feedback unavailable")
    return {key: float(value[key]) for key in ("b", "s", "e", "t", "r", "x", "y", "z")}


def move(q: JointVector, settle_s: float) -> None:
    if not within_limits(q, "hard") or min(joint_limit_margins(q, "hard").values()) < 0.10:
        raise RuntimeError("unsafe joint target: {}".format(q.as_dict()))
    params = q.as_dict()
    params.update({"hand": 1.08, "spd": 0.0, "acc": 6.0})
    response = execute_rpc("joints_move", params)
    if not response.get("ok"):
        raise RuntimeError("joint move failed: {}".format(response))
    time.sleep(settle_s)


def move_single(joint_index: int, target_rad: float, settle_s: float) -> None:
    response = execute_rpc(
        "joint_move",
        {"joint": joint_index + 1, "rad": float(target_rad), "spd": 0.0, "acc": 6.0},
    )
    if not response.get("ok"):
        raise RuntimeError("single joint move failed: {}".format(response))
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


def observe(provider: Ros2RgbDepthProvider, intrinsics: CameraIntrinsics, timeout_s: float):
    fresh = feedback()
    # Ros2RgbDepthProvider keeps a synchronization window. Drop pre-motion
    # messages so a very well synchronized old pair cannot win over a fresh one.
    provider._rgb_buf.clear()
    provider._depth_buf.clear()
    frame = provider.read(timeout_s=timeout_s)
    detection = detect_chessboard(
        frame.rgb_bgr, intrinsics, cols=9, rows=6, square_size_mm=18.0
    )
    if detection is None:
        raise RuntimeError("board lost during deadband probe")
    return fresh, detection


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sensor_floor", type=Path)
    parser.add_argument("--settle-s", type=float, default=1.8)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchor = JointVector(0.0, -0.72, 2.95, 0.0, 0.0)
    command_steps = (0.003, 0.006, 0.012, 0.025, 0.050)
    preload = 0.12
    floor = json.loads(args.sensor_floor.read_text(encoding="utf-8"))
    translation_threshold_mm = max(
        0.12, 3.0 * float(floor["pnp_corner_camera"]["norm_scatter_mm"]["p95"])
    )
    rotation_threshold_deg = max(
        0.05, 3.0 * float(floor["pnp_rotation_scatter_deg"]["p95"])
    )

    provider = Ros2RgbDepthProvider(
        rgb_topic=RGB_TOPIC, depth_topic=DEPTH_TOPIC, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=INFO_TOPIC)
    rows = []
    try:
        intrinsics = wait_intrinsics(provider, args.timeout_s)
        move(anchor, args.settle_s)
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            for direction in (1.0, -1.0):
                anchor_value = float(anchor.as_array()[joint_index])
                move_single(joint_index, anchor_value - direction * preload, args.settle_s)
                move_single(joint_index, anchor_value, args.settle_s)
                baseline_feedback, baseline = observe(provider, intrinsics, args.timeout_s)
                for delta in command_steps:
                    move_single(joint_index, anchor_value + direction * delta, args.settle_s)
                    actual, detection = observe(provider, intrinsics, args.timeout_s)
                    key = ("b", "s", "e", "t", "r")[joint_index]
                    actual_delta = float(actual[key] - baseline_feedback[key])
                    all_feedback_delta = {
                        name: float(actual[feedback_key] - baseline_feedback[feedback_key])
                        for name, feedback_key in zip(JOINT_NAMES, ("b", "s", "e", "t", "r"))
                    }
                    translation_mm = float(
                        np.linalg.norm(
                            detection.transform_camera_board.matrix[:3, 3]
                            - baseline.transform_camera_board.matrix[:3, 3]
                        )
                    )
                    rotation_deg = rotation_delta_deg(
                        baseline.transform_camera_board.matrix[:3, :3],
                        detection.transform_camera_board.matrix[:3, :3],
                    )
                    measurable = bool(
                        translation_mm >= translation_threshold_mm
                        or rotation_deg >= rotation_threshold_deg
                    )
                    row = {
                        "joint": joint_name,
                        "direction": "positive" if direction > 0 else "negative",
                        "command_delta_rad": direction * delta,
                        "actual_feedback_delta_rad": actual_delta,
                        "all_feedback_delta_rad": all_feedback_delta,
                        "board_translation_delta_mm": translation_mm,
                        "board_rotation_delta_deg": rotation_deg,
                        "measurable_motion": measurable,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        move(anchor, args.settle_s)
    finally:
        provider.close()

    deadband = {}
    for joint_name in JOINT_NAMES:
        directions = {}
        for direction in ("positive", "negative"):
            candidates = [
                abs(row["command_delta_rad"])
                for row in rows
                if row["joint"] == joint_name
                and row["direction"] == direction
                and row["measurable_motion"]
            ]
            directions[direction] = min(candidates) if candidates else None
        finite = [value for value in directions.values() if value is not None]
        deadband[joint_name] = {
            "min_measurable_command_rad": directions,
            "conservative_bidirectional_deadband_rad": max(finite) if len(finite) == 2 else None,
        }
    report = {
        "status": "MEASURED_DEADBAND",
        "anchor_command_rad": anchor.as_dict(),
        "command_steps_rad": list(command_steps),
        "physical_detection_thresholds": {
            "board_translation_mm": translation_threshold_mm,
            "board_rotation_deg": rotation_threshold_deg,
        },
        "deadband": deadband,
        "probes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "probes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
