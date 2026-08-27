#!/usr/bin/env python3
"""Phase D.1 fixed-target Cartesian closed loop; checkerboard only, no grasp APIs."""
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
from pipelines.roarm_m3_kinematics import JointVector, Pose5, fk, pose_error, within_limits  # noqa: E402
from pipelines.roarm_m3_kinematics.cartesian import (  # noqa: E402
    EMPIRICAL_GATES_MM,
    JOINT_DEADBAND_RAD,
    MAX_JACOBIAN_CONDITION,
    plan_cartesian_pose,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"
JOINT_FEEDBACK_KEYS = ("b", "s", "e", "t", "r")
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist", "roll")


def transform(value: dict) -> RigidTransform:
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def feedback() -> dict:
    response = execute_rpc("feedback", {})
    value = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("fresh T:105 unavailable")
    return {key: float(value[key]) for key in (*JOINT_FEEDBACK_KEYS, "x", "y", "z")}


def q_from_feedback(value: dict) -> JointVector:
    return JointVector(*(value[key] for key in JOINT_FEEDBACK_KEYS))


def move_all(q: JointVector, settle_s: float) -> None:
    if not within_limits(q, "hard"):
        raise RuntimeError("hard limit gate")
    params = q.as_dict()
    params.update({"hand": 1.08, "spd": 0.0, "acc": 6.0})
    result = execute_rpc("joints_move", params)
    if not result.get("ok"):
        raise RuntimeError("T:102 failed")
    time.sleep(settle_s)


def move_single(joint_index: int, target_rad: float, settle_s: float) -> None:
    result = execute_rpc(
        "joint_move",
        {"joint": joint_index + 1, "rad": float(target_rad), "spd": 0.0, "acc": 6.0},
    )
    if not result.get("ok"):
        raise RuntimeError("T:101 failed")
    time.sleep(settle_s)


def wait_intrinsics(provider, timeout_s):
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


def observe(provider, intrinsics, timeout_s):
    fresh = feedback()
    provider._rgb_buf.clear()
    provider._depth_buf.clear()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = provider.read(timeout_s=max(0.2, deadline - time.monotonic()))
        board = detect_chessboard(frame.rgb_bgr, intrinsics, cols=9, rows=6, square_size_mm=18.0)
        if board is not None:
            return fresh, board
    raise RuntimeError("fresh camera verification failed")


def rotation_vector(rotation):
    value, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return value.reshape(3)


def camera_motion_base(q0, board0, board1, link5_camera):
    base_board = base_to_link5(q0.as_array()).then(link5_camera).then(board0)
    board_camera0 = np.linalg.inv(board0.matrix)
    board_camera1 = np.linalg.inv(board1.matrix)
    delta_board = board_camera1[:3, 3] - board_camera0[:3, 3]
    delta_base = base_board.matrix[:3, :3] @ delta_board
    relative_board = board_camera0[:3, :3].T @ board_camera1[:3, :3]
    rotvec_base = base_board.matrix[:3, :3] @ rotation_vector(relative_board)
    return delta_base, rotvec_base


def task_score(error):
    value = np.asarray(error, dtype=np.float64)
    return float(np.linalg.norm(value[:3]) + 100.0 * np.linalg.norm(value[3:]))


def joint_log(q_start, q_target, q_feedback):
    start, target, actual = q_start.as_array(), q_target.as_array(), q_feedback.as_array()
    return {
        name: {
            "q_start": float(start[i]),
            "q_target": float(target[i]),
            "q_feedback": float(actual[i]),
            "command_delta": float(target[i] - start[i]),
            "actual_delta": float(actual[i] - start[i]),
            "target_minus_feedback": float(target[i] - actual[i]),
            "deadband": float(JOINT_DEADBAND_RAD[name]),
        }
        for i, name in enumerate(JOINT_NAMES)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--case", choices=("x", "y", "z", "pitch", "roll"), required=True)
    parser.add_argument("--delta", type=float, required=True, help="mm for XYZ, rad for pitch/roll")
    parser.add_argument("--strategy", choices=("direct", "elbow_below", "elbow_above"), default="direct")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--preload-rad", type=float, default=0.03)
    parser.add_argument("--settle-s", type=float, default=3.2)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("explicit --execute required")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    link5_camera = transform(calibration["report"]["T_link5_camera"])
    work_q = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
    provider = Ros2RgbDepthProvider(rgb_topic=RGB_TOPIC, depth_topic=DEPTH_TOPIC, sync_slop_s=0.15)
    provider.open(camera_info_topic=INFO_TOPIC)
    repetitions = []
    try:
        intrinsics = wait_intrinsics(provider, args.timeout_s)
        for repeat in range(args.repeats):
            move_all(work_q, args.settle_s)
            baseline_feedback, baseline_board = observe(provider, intrinsics, args.timeout_s)
            baseline_q = q_from_feedback(baseline_feedback)
            baseline_pose = fk(baseline_q)
            requested = np.zeros(5, dtype=np.float64)
            requested[("x", "y", "z", "pitch", "roll").index(args.case)] = args.delta
            fixed_target = Pose5(*(baseline_pose.as_array() + requested))
            initial_plan = plan_cartesian_pose(
                fixed_target,
                baseline_q,
                stage="visual",
                requested_delta=requested,
                max_joint_step_rad=0.20,
            )
            if not initial_plan.ok or initial_plan.raw_q is None:
                raise RuntimeError("initial dry-run failed: {}".format(initial_plan.reason))
            preload = None
            current_feedback, current_board = baseline_feedback, baseline_board
            if args.strategy != "direct":
                sign = -1.0 if args.strategy == "elbow_below" else 1.0
                preload_target = float(initial_plan.raw_q.elbow + sign * args.preload_rad)
                before_preload_q = q_from_feedback(current_feedback)
                if args.case == "pitch":
                    # Keep shoulder+elbow+wrist pitch constant while placing both
                    # elbow and wrist on the same side of their final targets.
                    preload_q = JointVector(
                        initial_plan.raw_q.base,
                        initial_plan.raw_q.shoulder - 2.0 * sign * args.preload_rad,
                        preload_target,
                        initial_plan.raw_q.wrist + sign * args.preload_rad,
                        initial_plan.raw_q.roll,
                    )
                else:
                    preload_q = JointVector(
                        initial_plan.raw_q.base,
                        initial_plan.raw_q.shoulder,
                        preload_target,
                        initial_plan.raw_q.wrist - sign * args.preload_rad,
                        initial_plan.raw_q.roll,
                    )
                move_all(preload_q, args.settle_s)
                current_feedback, current_board = observe(provider, intrinsics, args.timeout_s)
                after_preload_q = q_from_feedback(current_feedback)
                preload = {
                    "strategy": args.strategy,
                    "target_elbow": preload_target,
                    "target_q": preload_q.as_dict(),
                    "fresh_T105": current_feedback,
                    "joints": joint_log(before_preload_q, preload_q, after_preload_q),
                }

            iterations = []
            previous_score = task_score(pose_error(fk(q_from_feedback(current_feedback)), fixed_target))
            stop_reason = "MAX_ITERATIONS"
            for iteration in range(1, args.max_iterations + 1):
                q_start = q_from_feedback(current_feedback)
                residual_before = pose_error(fk(q_start), fixed_target)
                translation_before = float(np.linalg.norm(residual_before[:3]))
                orientation_before = float(np.linalg.norm(residual_before[3:]))
                orientation_limit = 0.015 if args.case == "pitch" else 0.03
                if iterations and translation_before <= EMPIRICAL_GATES_MM["visual"] and orientation_before <= orientation_limit:
                    stop_reason = "TARGET_GATE_REACHED"
                    break
                plan = plan_cartesian_pose(
                    fixed_target,
                    q_start,
                    stage="visual",
                    max_joint_step_rad=0.20,
                )
                if not plan.ok or plan.executable_q is None:
                    stop_reason = "PLAN_{}".format(plan.reason)
                    break
                q_target = plan.executable_q
                print(json.dumps({"repeat": repeat + 1, "iteration": iteration, "dry_run": plan.as_dict()}), flush=True)
                move_all(q_target, args.settle_s)
                next_feedback, next_board = observe(provider, intrinsics, args.timeout_s)
                q_actual = q_from_feedback(next_feedback)
                residual_after = pose_error(fk(q_actual), fixed_target)
                score_after = task_score(residual_after)
                joints = joint_log(q_start, q_target, q_actual)
                stalled = [
                    name
                    for name, row in joints.items()
                    if abs(row["command_delta"]) >= row["deadband"]
                    and abs(row["actual_delta"]) < row["deadband"]
                ]
                camera_delta, camera_rotation = camera_motion_base(
                    baseline_q, baseline_board.transform_camera_board, next_board.transform_camera_board, link5_camera
                )
                row = {
                    "iteration": iteration,
                    "q_start": q_start.as_dict(),
                    "q_target": q_target.as_dict(),
                    "q_feedback": q_actual.as_dict(),
                    "joint_errors": joints,
                    "stalled_joints": stalled,
                    "residual_before": residual_before.tolist(),
                    "residual_after": residual_after.tolist(),
                    "translation_error_after_mm": float(np.linalg.norm(residual_after[:3])),
                    "orientation_error_after_rad": float(np.linalg.norm(residual_after[3:])),
                    "camera_delta_base_mm": camera_delta.tolist(),
                    "camera_rotation_vector_base_rad": camera_rotation.tolist(),
                    "camera_rotation_deg": math.degrees(float(np.linalg.norm(camera_rotation))),
                    "fresh_T105": next_feedback,
                    "board_reprojection_px": next_board.reprojection_rmse_px,
                }
                iterations.append(row)
                print(json.dumps({"closed_loop": row}), flush=True)
                current_feedback, current_board = next_feedback, next_board
                if stalled:
                    stop_reason = "JOINT_STALL_{}".format("_".join(stalled))
                    break
                if score_after > previous_score + 0.25:
                    stop_reason = "ERROR_GREW"
                    break
                previous_score = score_after

            final_q = q_from_feedback(current_feedback)
            final_residual = pose_error(fk(final_q), fixed_target)
            final_translation = float(np.linalg.norm(final_residual[:3]))
            final_orientation = float(np.linalg.norm(final_residual[3:]))
            orientation_limit = 0.015 if args.case == "pitch" else 0.03
            passed = bool(
                final_translation <= EMPIRICAL_GATES_MM["visual"]
                and final_orientation <= orientation_limit
                and stop_reason != "ERROR_GREW"
                and not stop_reason.startswith("JOINT_STALL_")
            )
            repetitions.append(
                {
                    "repeat": repeat + 1,
                    "case": args.case,
                    "delta": args.delta,
                    "strategy": args.strategy,
                    "fixed_target": fixed_target.as_dict(),
                    "baseline_q": baseline_q.as_dict(),
                    "preload": preload,
                    "iterations": iterations,
                    "stop_reason": stop_reason,
                    "final_q": final_q.as_dict(),
                    "final_residual": final_residual.tolist(),
                    "final_translation_error_mm": final_translation,
                    "final_orientation_error_rad": final_orientation,
                    "PASS": passed,
                }
            )
            move_all(work_q, args.settle_s)
    finally:
        try:
            move_all(work_q, args.settle_s)
        except Exception as exc:
            print(json.dumps({"restore_work_pose_error": str(exc)}), flush=True)
        provider.close()

    report = {
        "status": "PASS" if all(row["PASS"] for row in repetitions) else "FAIL",
        "case": args.case,
        "delta": args.delta,
        "strategy": args.strategy,
        "repeats": args.repeats,
        "transform": "T_link5_camera_v1",
        "repetitions": repetitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
