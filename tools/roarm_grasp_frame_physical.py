#!/usr/bin/env python3
"""Board-only physical validation of the fixed RoArm-M3 grasp TCP."""
from __future__ import annotations

import argparse
import json
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
from pipelines.roarm_calibration.urdf_fk import base_to_grasp, base_to_link5, link5_to_grasp  # noqa: E402
from pipelines.roarm_m3_kinematics import JointVector, Pose5, grasp_fk, grasp_ik, grasp_jacobian, pose_error, within_limits  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


def feedback() -> dict:
    value = (execute_rpc("feedback", {}) or {}).get("feedback")
    if not isinstance(value, dict):
        raise RuntimeError("fresh T:105 unavailable")
    return value


def q_from_feedback(value: dict) -> JointVector:
    return JointVector(*(float(value[key]) for key in ("b", "s", "e", "t", "r")))


def move(q: JointVector, settle_s: float) -> None:
    if not within_limits(q, "hard"):
        raise RuntimeError("hard limit gate")
    params = q.as_dict()
    params.update({"hand": 1.08, "spd": 0.0, "acc": 6.0})
    if not (execute_rpc("joints_move", params) or {}).get("ok"):
        raise RuntimeError("T:102 failed")
    time.sleep(settle_s)


def observe(provider, intrinsics, link5_camera, timeout_s):
    fresh = feedback()
    provider._rgb_buf.clear()
    provider._depth_buf.clear()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = provider.read(timeout_s=max(0.2, deadline - time.monotonic()))
        board = detect_chessboard(frame.rgb_bgr, intrinsics, cols=9, rows=6, square_size_mm=18.0)
        if board is not None:
            q = q_from_feedback(fresh)
            base_board = base_to_link5(q.as_array()).then(link5_camera).then(board.transform_camera_board)
            return fresh, q, board, base_board
    raise RuntimeError("fresh 54/54 board verification failed")


def point_in_board(base_board, point_base):
    return base_board.inverse().transform_point(point_base, frame="base_link")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--delta-mm", nargs=3, type=float, default=(-8.0, 0.0, 0.0))
    parser.add_argument("--settle-s", type=float, default=3.2)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    value = calibration["report"]["T_link5_camera"]
    link5_camera = RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))
    provider = Ros2RgbDepthProvider(
        rgb_topic="/stereo_camera/color/image_rect_raw",
        depth_topic="/stereo_camera/aligned_depth_to_color/image_raw",
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic="/stereo_camera/color/camera_info")
    work_q = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
    moved = False
    try:
        deadline = time.monotonic() + args.timeout_s
        msg = provider._color_info_msg
        while msg is None and time.monotonic() < deadline:
            time.sleep(0.02)
            msg = provider._color_info_msg
        if msg is None:
            raise RuntimeError("CameraInfo unavailable")
        intrinsics = CameraIntrinsics(int(msg.width), int(msg.height), np.asarray(msg.k).reshape(3, 3), np.asarray(msg.d), str(msg.header.frame_id), True)

        if args.execute:
            move(work_q, args.settle_s)
        fb0, q0, board0, base_board0 = observe(provider, intrinsics, link5_camera, args.timeout_s)
        start = grasp_fk(q0)
        delta = np.asarray(args.delta_mm, dtype=np.float64)
        target = Pose5(*(start.as_array() + np.r_[delta, 0.0, 0.0]))
        solution = grasp_ik(target, q0)
        if not solution.ok or solution.q is None:
            raise RuntimeError("grasp IK failed: {}".format(solution.reason))
        joint_delta = solution.q.as_array() - q0.as_array()
        condition = float(np.linalg.cond(np.diag([1 / 200.0] * 3 + [1.0, 1.0]) @ grasp_jacobian(solution.q)))
        dry_run = {
            "q_start": q0.as_dict(),
            "target_grasp_pose": target.as_dict(),
            "q_target": solution.q.as_dict(),
            "joint_delta_rad": joint_delta.tolist(),
            "max_joint_delta_rad": float(np.max(np.abs(joint_delta))),
            "jacobian_condition": condition,
            "hard_limits_ok": within_limits(solution.q, "hard"),
            "fk_round_trip_error": pose_error(grasp_fk(solution.q), target).tolist(),
        }
        if dry_run["max_joint_delta_rad"] > 0.20 or condition > 80.0:
            raise RuntimeError("dry-run gate failed")
        report = {"status": "DRY_RUN_PASS", "T_link5_grasp": link5_to_grasp().to_dict(), "dry_run": dry_run}
        if args.execute:
            move(solution.q, args.settle_s)
            moved = True
            fb1, q1, board1, base_board1 = observe(provider, intrinsics, link5_camera, args.timeout_s)
            actual = grasp_fk(q1)
            target_board = point_in_board(base_board0, target.as_array()[:3])
            actual_board = point_in_board(base_board1, actual.as_array()[:3])
            report.update({
                "status": "PASS" if np.linalg.norm(actual_board - target_board) <= 8.0 else "FAIL",
                "q_feedback": q1.as_dict(),
                "actual_grasp_pose": actual.as_dict(),
                "commanded_vs_fk_actual": pose_error(actual, target).tolist(),
                "commanded_vs_fk_actual_mm": float(np.linalg.norm(pose_error(actual, target)[:3])),
                "known_point_target_board_mm": target_board.tolist(),
                "known_point_actual_board_mm": actual_board.tolist(),
                "known_point_error_board_mm": (target_board - actual_board).tolist(),
                "known_point_error_norm_mm": float(np.linalg.norm(target_board - actual_board)),
                "board_pose_shift_mm": float(np.linalg.norm(base_board1.translation_mm - base_board0.translation_mm)),
                "fresh_T105_before": fb0,
                "fresh_T105_after": fb1,
                "board_reprojection_before_px": board0.reprojection_rmse_px,
                "board_reprojection_after_px": board1.reprojection_rmse_px,
            })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report))
    finally:
        if args.execute and moved:
            try:
                move(work_q, args.settle_s)
                feedback()
            except Exception as exc:
                print(json.dumps({"restore_work_pose_error": str(exc)}), flush=True)
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
