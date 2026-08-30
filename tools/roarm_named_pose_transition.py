#!/usr/bin/env python3
"""Fail-closed segmented joint-space transition between named RoArm poses."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
for item in (REPO, REPO / "ops/axm-monitor/agent"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402
from pipelines.roarm_m3_kinematics import JointVector  # noqa: E402
from pipelines.roarm_m3_kinematics.conventions import within_limits  # noqa: E402
from pipelines.roarm_m3_kinematics.stereo_safety import stereo_motion_gate  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc, reset_client  # noqa: E402

JOINT_KEYS = ("b", "s", "e", "t", "r")
POSE_KEYS = ("base", "shoulder", "elbow", "wrist", "roll")


def read_feedback() -> tuple[JointVector, float, dict]:
    value = execute_rpc("feedback", {}).get("feedback")
    if not isinstance(value, dict) or int(value.get("T", -1)) != 1051:
        raise RuntimeError("FRESH_T105_UNAVAILABLE")
    if any(int(value.get(f"torswitch{name}", -1)) != 1 for name in "BSETRG"):
        raise RuntimeError("TORQUE_FEEDBACK_NOT_ALL_ON")
    q = JointVector(*(float(value[key]) for key in JOINT_KEYS))
    return q, float(value["g"]), value


def physical_q(q: JointVector, roll_zero: float) -> JointVector:
    values = q.as_array().copy()
    values[4] -= float(roll_zero)
    return JointVector(*map(float, values))


def fresh_rgbd(provider: Ros2RgbDepthProvider, timeout_s: float):
    provider._rgb_buf.clear()
    provider._depth_buf.clear()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = provider.read(timeout_s=max(0.5, deadline - time.monotonic()))
        if frame.depth_m is not None and provider.get_intrinsics() is not None:
            return frame, provider.get_intrinsics()
        time.sleep(0.1)
    raise RuntimeError("FRESH_RGBD_UNAVAILABLE")


def load_transform(path: Path) -> RigidTransform:
    document = json.loads(path.read_text(encoding="utf-8"))
    value = document["report"]["T_link5_camera"]
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--handeye", type=Path, required=True)
    ap.add_argument("--roll-zero-offset-rad", type=float, default=1.89)
    ap.add_argument("--max-segment-rad", type=float, default=0.18)
    ap.add_argument("--minimum-clearance-mm", type=float, default=55.0)
    ap.add_argument("--settle-s", type=float, default=3.2)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--feedback-tol-rad", type=float, default=0.055)
    ap.add_argument("--elbow-preload-below-rad", type=float, default=0.0)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    pose = config.get(args.pose)
    if not isinstance(pose, dict):
        raise RuntimeError(f"UNKNOWN_POSE:{args.pose}")
    target = JointVector(*(float(pose[name]) for name in POSE_KEYS))
    if not within_limits(physical_q(target, args.roll_zero_offset_rad), "hard"):
        raise RuntimeError("TARGET_PHYSICAL_LIMIT")
    link5_camera = load_transform(args.handeye)

    reset_client()
    start, hand, raw = read_feedback()
    if not within_limits(physical_q(start, args.roll_zero_offset_rad), "hard"):
        raise RuntimeError("START_PHYSICAL_LIMIT")
    delta = target.as_array() - start.as_array()
    segments = max(1, int(math.ceil(float(np.max(np.abs(delta))) / args.max_segment_rad)))
    waypoints = [
        JointVector(*map(float, start.as_array() + (i / segments) * delta))
        for i in range(1, segments + 1)
    ]
    report = {
        "pose": args.pose,
        "q_start": start.as_dict(),
        "q_target": target.as_dict(),
        "segments": segments,
        "execute": bool(args.execute),
        "steps": [],
        "status": "DRY_RUN" if not args.execute else "RUNNING",
    }

    provider = Ros2RgbDepthProvider(
        rgb_topic="/stereo_camera/color/image_rect_raw",
        depth_topic="/stereo_camera/depth/image_rect_raw",
        sync_slop_s=0.3,
        sync_enabled=False,
    )
    provider.open(camera_info_topic="/stereo_camera/color/camera_info")
    try:
        current = start
        for index, waypoint in enumerate(waypoints, 1):
            frame, intrinsics = fresh_rgbd(provider, args.timeout_s)
            if args.execute:
                fresh, hand, _ = read_feedback()
                if float(np.max(np.abs(fresh.as_array() - current.as_array()))) > args.feedback_tol_rad:
                    raise RuntimeError("UNEXPECTED_JOINT_CHANGE_BEFORE_SEGMENT")
            else:
                fresh = current

            if args.elbow_preload_below_rad > 0.0 and waypoint.elbow < fresh.elbow:
                preload_values = waypoint.as_array().copy()
                preload_values[2] -= float(args.elbow_preload_below_rad)
                preload = JointVector(*map(float, preload_values))
                preload_step = float(np.max(np.abs(preload.as_array() - fresh.as_array())))
                if preload_step > 0.20:
                    raise RuntimeError("PRELOAD_JOINT_STEP_GATE")
                if not within_limits(physical_q(preload, args.roll_zero_offset_rad), "hard"):
                    raise RuntimeError("PRELOAD_PHYSICAL_LIMIT")
                preload_gate = stereo_motion_gate(
                    frame.depth_m,
                    intrinsics,
                    fresh,
                    preload,
                    link5_camera,
                    minimum_clearance_mm=args.minimum_clearance_mm,
                    roll_zero_offset_rad=args.roll_zero_offset_rad,
                )
                preload_item = {
                    "index": index,
                    "kind": "elbow_preload_below",
                    "q_start": fresh.as_dict(),
                    "q_target": preload.as_dict(),
                    "max_step_rad": preload_step,
                    "rgbd_stamp_s": float(frame.stamp_s),
                    "stereo_gate": preload_gate.as_dict(),
                }
                report["steps"].append(preload_item)
                print(json.dumps(preload_item), flush=True)
                if not preload_gate.ok:
                    raise RuntimeError(f"STEREO_STOP:{preload_gate.reason}")
                if args.execute:
                    params = preload.as_dict()
                    params.update({"hand": hand, "spd": 0.0, "acc": 6.0})
                    result = execute_rpc("joints_move", params)
                    if not result.get("ok"):
                        raise RuntimeError("PRELOAD_T102_FAILED")
                    time.sleep(args.settle_s)
                    actual_preload, hand, _ = read_feedback()
                    preload_item["q_feedback"] = actual_preload.as_dict()
                    if actual_preload.elbow >= fresh.elbow - 0.012:
                        raise RuntimeError("PRELOAD_NO_PROGRESS")
                    fresh = actual_preload
                    frame, intrinsics = fresh_rgbd(provider, args.timeout_s)
                else:
                    fresh = preload
            gate = stereo_motion_gate(
                frame.depth_m,
                intrinsics,
                fresh,
                waypoint,
                link5_camera,
                minimum_clearance_mm=args.minimum_clearance_mm,
                roll_zero_offset_rad=args.roll_zero_offset_rad,
            )
            item = {
                "index": index,
                "kind": "segment_target",
                "q_start": fresh.as_dict(),
                "q_target": waypoint.as_dict(),
                "max_step_rad": float(np.max(np.abs(waypoint.as_array() - fresh.as_array()))),
                "rgbd_stamp_s": float(frame.stamp_s),
                "stereo_gate": gate.as_dict(),
            }
            report["steps"].append(item)
            print(json.dumps(item), flush=True)
            if not gate.ok:
                raise RuntimeError(f"STEREO_STOP:{gate.reason}")
            if not args.execute:
                current = waypoint
                continue
            params = waypoint.as_dict()
            params.update({"hand": hand, "spd": 0.0, "acc": 6.0})
            result = execute_rpc("joints_move", params)
            if not result.get("ok"):
                raise RuntimeError("T102_FAILED")
            time.sleep(args.settle_s)
            actual, _, _ = read_feedback()
            item["q_feedback"] = actual.as_dict()
            item["target_error_rad"] = {
                name: float(waypoint.as_array()[i] - actual.as_array()[i])
                for i, name in enumerate(POSE_KEYS)
            }
            if float(np.max(np.abs(waypoint.as_array() - actual.as_array()))) > args.feedback_tol_rad:
                raise RuntimeError("SEGMENT_FEEDBACK_ERROR")
            fresh_rgbd(provider, args.timeout_s)
            current = actual
        report["status"] = "PASS" if args.execute else "DRY_RUN_PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = str(exc)
    finally:
        provider.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] in ("PASS", "DRY_RUN_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
