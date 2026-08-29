#!/usr/bin/env python3
"""Read-only check of replacement-arm roll zero against the fixed Phase-B board."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "ops/axm-monitor/agent"):
    sys.path.insert(0, str(item))

from pipelines.roarm_calibration.chessboard import CameraIntrinsics, detect_chessboard
from pipelines.roarm_calibration.se3 import RigidTransform
from pipelines.roarm_calibration.urdf_fk import base_to_link5
from pipelines.roarm_m3_kinematics import JointVector
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider
from roarm_proxy import execute_rpc


def rotation_error_deg(a, b):
    rv, _ = cv2.Rodrigues(np.asarray(a[:3, :3]).T @ np.asarray(b[:3, :3]))
    return math.degrees(float(np.linalg.norm(rv)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration", type=Path)
    args = ap.parse_args()
    data = json.loads(args.calibration.read_text())
    report = data["report"]
    selected = report["selected_method"]
    method = report["methods"][selected]
    reference = np.asarray(method["validation"]["estimated_T_base_board_mean"]["matrix"])
    c = report["T_link5_camera"]
    link5_camera = RigidTransform(c["parent"], c["child"], np.asarray(c["matrix"]))

    response = execute_rpc("feedback", {})
    fb = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(fb, dict) or int(fb.get("T", -1)) != 1051:
        raise RuntimeError("fresh T:105 unavailable")
    q = JointVector(*(float(fb[k]) for k in "bsetr"))

    provider = Ros2RgbDepthProvider(
        rgb_topic="/stereo_camera/color/image_rect_raw",
        depth_topic="/stereo_camera/aligned_depth_to_color/image_raw",
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic="/stereo_camera/color/camera_info")
    try:
        deadline = time.monotonic() + 10.0
        while provider._color_info_msg is None and time.monotonic() < deadline:
            time.sleep(0.02)
        msg = provider._color_info_msg
        if msg is None:
            raise RuntimeError("CameraInfo unavailable")
        intr = CameraIntrinsics(
            int(msg.width), int(msg.height), np.asarray(msg.k).reshape(3, 3),
            np.asarray(msg.d), str(msg.header.frame_id), True,
        )
        board = None
        while board is None and time.monotonic() < deadline:
            frame = provider.read(timeout_s=max(0.2, deadline - time.monotonic()))
            board = detect_chessboard(frame.rgb_bgr, intr, cols=9, rows=6, square_size_mm=18.0)
        if board is None:
            raise RuntimeError("board not 54/54")
    finally:
        provider.close()

    rows = []
    for offset in np.linspace(-math.pi, math.pi, 2881):
        adjusted = JointVector(q.base, q.shoulder, q.elbow, q.wrist, q.roll - offset)
        estimate = base_to_link5(adjusted.as_array()).then(link5_camera).then(
            board.transform_camera_board
        ).matrix
        t = float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3]))
        r = rotation_error_deg(estimate, reference)
        rows.append((t + 5.0 * r, t, r, float(offset)))
    rows.sort()

    def evaluate(offset):
        adjusted = JointVector(q.base, q.shoulder, q.elbow, q.wrist, q.roll - offset)
        estimate = base_to_link5(adjusted.as_array()).then(link5_camera).then(
            board.transform_camera_board
        ).matrix
        return {
            "offset_rad": offset,
            "translation_error_mm": float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3])),
            "rotation_error_deg": rotation_error_deg(estimate, reference),
        }

    print(json.dumps({
        "read_only": True,
        "q_feedback": q.as_dict(),
        "selected_method": selected,
        "board_reprojection_px": board.reprojection_rmse_px,
        "raw_offset_zero": evaluate(0.0),
        "dom_roll_as_offset": evaluate(1.89),
        "best": [
            {"score": s, "translation_error_mm": t, "rotation_error_deg": r, "offset_rad": o}
            for s, t, r, o in rows[:5]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
