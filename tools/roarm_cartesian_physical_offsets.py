#!/usr/bin/env python3
"""Standalone gated Cartesian IK offsets against a fixed checkerboard; no grasp APIs."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "ops/axm-monitor/agent"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pipelines.roarm_calibration.chessboard import CameraIntrinsics, detect_chessboard  # noqa: E402
from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402
from pipelines.roarm_calibration.urdf_fk import base_to_link5  # noqa: E402
from pipelines.roarm_m3_kinematics import (  # noqa: E402
    JointVector,
    Pose5,
    fk,
    jacobian_condition,
    pose_error,
    within_limits,
)
from pipelines.roarm_m3_kinematics.cartesian import (  # noqa: E402
    EMPIRICAL_GATES_MM,
    MAX_JACOBIAN_CONDITION,
    plan_cartesian_pose,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"


def transform(value: dict) -> RigidTransform:
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def feedback() -> dict:
    response = execute_rpc("feedback", {})
    value = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("fresh T:105 unavailable")
    return {key: float(value[key]) for key in ("b", "s", "e", "t", "r", "x", "y", "z")}


def q_from_feedback(value: dict) -> JointVector:
    return JointVector(*(value[key] for key in ("b", "s", "e", "t", "r")))


def move(q: JointVector, settle_s: float) -> None:
    if not within_limits(q, "hard"):
        raise RuntimeError("hard limit gate")
    params = q.as_dict()
    params.update({"hand": 1.08, "spd": 0.0, "acc": 6.0})
    response = execute_rpc("joints_move", params)
    if not response.get("ok"):
        raise RuntimeError("T:102 failed: {}".format(response))
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


def observe_board(provider, intrinsics, timeout_s):
    fresh = feedback()
    provider._rgb_buf.clear()
    provider._depth_buf.clear()
    frame = provider.read(timeout_s=timeout_s)
    detection = detect_chessboard(
        frame.rgb_bgr, intrinsics, cols=9, rows=6, square_size_mm=18.0
    )
    if detection is None:
        raise RuntimeError("fresh camera verification failed: board not 54/54")
    return fresh, detection


def physical_delta_base(before_q, before_board, after_board, link5_camera):
    base_board = base_to_link5(before_q.as_array()).then(link5_camera).then(before_board)
    board_camera_before = np.linalg.inv(before_board.matrix)
    board_camera_after = np.linalg.inv(after_board.matrix)
    delta_board = board_camera_after[:3, 3] - board_camera_before[:3, 3]
    delta_base = base_board.matrix[:3, :3] @ delta_board
    relative_rotation_board = board_camera_before[:3, :3].T @ board_camera_after[:3, :3]
    rotation_vector_board, _ = cv2.Rodrigues(relative_rotation_board)
    rotation_vector_base = base_board.matrix[:3, :3] @ rotation_vector_board.reshape(3)
    return delta_base, rotation_vector_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--settle-s", type=float, default=3.2)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("physical motion requires explicit --execute")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    link5_camera = transform(calibration["report"]["T_link5_camera"])
    work_q = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
    if jacobian_condition(work_q) > MAX_JACOBIAN_CONDITION:
        raise RuntimeError("working pose Jacobian gate")
    offsets = (
        ("+X", np.asarray([8.0, 0.0, 0.0, 0.0, 0.0])),
        ("-X", np.asarray([-8.0, 0.0, 0.0, 0.0, 0.0])),
        ("+Y", np.asarray([0.0, 8.0, 0.0, 0.0, 0.0])),
        ("-Y", np.asarray([0.0, -8.0, 0.0, 0.0, 0.0])),
        ("+Z", np.asarray([0.0, 0.0, 8.0, 0.0, 0.0])),
        ("-Z", np.asarray([0.0, 0.0, -8.0, 0.0, 0.0])),
        ("+pitch", np.asarray([0.0, 0.0, 0.0, 0.04, 0.0])),
        ("+roll", np.asarray([0.0, 0.0, 0.0, 0.0, 0.03])),
    )

    provider = Ros2RgbDepthProvider(
        rgb_topic=RGB_TOPIC, depth_topic=DEPTH_TOPIC, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=INFO_TOPIC)
    rows = []
    try:
        intrinsics = wait_intrinsics(provider, args.timeout_s)
        print(json.dumps({"dry_run_work_pose": work_q.as_dict()}), flush=True)
        move(work_q, args.settle_s)
        for name, requested in offsets:
            before_feedback, before_detection = observe_board(provider, intrinsics, args.timeout_s)
            before_q = q_from_feedback(before_feedback)
            before_pose = fk(before_q)
            target = Pose5(*(before_pose.as_array() + requested))
            plan = plan_cartesian_pose(
                target,
                before_q,
                stage="visual",
                requested_delta=requested,
                max_joint_step_rad=0.20,
            )
            print(json.dumps({"offset": name, "dry_run": plan.as_dict()}), flush=True)
            if not plan.ok:
                raise RuntimeError("{} dry-run failed: {}".format(name, plan.reason))
            assert plan.executable_q is not None
            move(plan.executable_q, args.settle_s)
            after_feedback, after_detection = observe_board(provider, intrinsics, args.timeout_s)
            after_q = q_from_feedback(after_feedback)
            after_pose = fk(after_q)
            actual_delta_fk = after_pose.as_array() - before_pose.as_array()
            actual_delta_fk[3:] = -pose_error(after_pose, before_pose)[3:]
            command_error = pose_error(after_pose, target)
            physical_translation, physical_rotation = physical_delta_base(
                before_q,
                before_detection.transform_camera_board,
                after_detection.transform_camera_board,
                link5_camera,
            )
            row = {
                "offset": name,
                "commanded_delta": requested.tolist(),
                "plan": plan.as_dict(),
                "fresh_T105_before": before_feedback,
                "fresh_T105_after": after_feedback,
                "actual_delta_fk": actual_delta_fk.tolist(),
                "commanded_minus_actual_fk": command_error.tolist(),
                "camera_measured_translation_delta_base_mm": physical_translation.tolist(),
                "camera_measured_translation_norm_mm": float(np.linalg.norm(physical_translation)),
                "camera_measured_rotation_vector_base_rad": physical_rotation.tolist(),
                "camera_measured_rotation_deg": math.degrees(float(np.linalg.norm(physical_rotation))),
                "board_reprojection_before_px": before_detection.reprojection_rmse_px,
                "board_reprojection_after_px": after_detection.reprojection_rmse_px,
            }
            row["post_move_translation_error_mm"] = float(
                np.linalg.norm(np.asarray(row["commanded_minus_actual_fk"][:3]))
            )
            row["post_move_KIN_GATE"] = (
                "PASS"
                if row["post_move_translation_error_mm"] <= EMPIRICAL_GATES_MM["visual"]
                else "FAIL"
            )
            rows.append(row)
            print(json.dumps({"offset_result": row}), flush=True)
            move(work_q, args.settle_s)
        final_feedback, final_detection = observe_board(provider, intrinsics, args.timeout_s)
    finally:
        provider.close()

    translation_errors = [
        float(np.linalg.norm(np.asarray(row["commanded_minus_actual_fk"][:3]))) for row in rows
    ]
    passed = all(row["post_move_KIN_GATE"] == "PASS" for row in rows)
    report = {
        "status": "PASS" if passed else "FAIL",
        "motion": "CANONICAL_IK_ONLY_NO_GRASP",
        "transform": "T_link5_camera_v1",
        "work_q": work_q.as_dict(),
        "results": rows,
        "translation_error_mm": {
            "mean": float(np.mean(translation_errors)),
            "p95": float(np.percentile(translation_errors, 95)),
            "max": float(np.max(translation_errors)),
        },
        "final_fresh_T105": final_feedback,
        "final_board_reprojection_px": final_detection.reprojection_rmse_px,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
