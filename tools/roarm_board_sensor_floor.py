#!/usr/bin/env python3
"""Measure static checkerboard PnP and aligned RGB-D sensor noise."""
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
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402
from tools.roarm_rgbd_board_validate import robust_depth, stats  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"


def feedback() -> dict:
    response = execute_rpc("feedback", {})
    value = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("T:105 feedback unavailable")
    return {key: float(value[key]) for key in ("b", "s", "e", "t", "r", "x", "y", "z")}


def wait_intrinsics(provider: Ros2RgbDepthProvider, timeout_s: float) -> CameraIntrinsics:
    deadline = time.monotonic() + timeout_s
    msg = provider._color_info_msg
    while msg is None and time.monotonic() < deadline:
        time.sleep(0.02)
        msg = provider._color_info_msg
    if msg is None:
        raise RuntimeError("CameraInfo unavailable")
    return CameraIntrinsics(
        int(msg.width),
        int(msg.height),
        np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
        np.asarray(msg.d, dtype=np.float64),
        str(msg.header.frame_id) or "stereo_camera_color_optical_frame",
        True,
    )


def rotation_angle_deg(rotation: np.ndarray, mean_rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(mean_rotation.T @ rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


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
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=18.0)
    parser.add_argument("--corner-col", type=int, default=4)
    parser.add_argument("--corner-row", type=int, default=3)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--interval-s", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = feedback()
    provider = Ros2RgbDepthProvider(
        rgb_topic=RGB_TOPIC,
        depth_topic=DEPTH_TOPIC,
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic=INFO_TOPIC)
    pnp_points = []
    pnp_rotations = []
    depth_points = []
    reprojection = []
    depth_vs_pnp = []
    failed = 0
    try:
        intrinsics = wait_intrinsics(provider, args.timeout_s)
        fx, fy = intrinsics.camera_matrix[0, 0], intrinsics.camera_matrix[1, 1]
        cx, cy = intrinsics.camera_matrix[0, 2], intrinsics.camera_matrix[1, 2]
        for _ in range(args.frames):
            frame = provider.read(timeout_s=args.timeout_s)
            detection = detect_chessboard(
                frame.rgb_bgr,
                intrinsics,
                cols=args.cols,
                rows=args.rows,
                square_size_mm=args.square_mm,
            )
            if detection is None or frame.depth_m is None:
                failed += 1
                continue
            corners = detection.corners_px.reshape(args.rows, args.cols, 2)
            u, v = map(float, corners[args.corner_row, args.corner_col])
            depth_m = robust_depth(frame.depth_m, u, v, args.depth_radius)
            depth_point = np.asarray(
                [(u - cx) * depth_m * 1000.0 / fx, (v - cy) * depth_m * 1000.0 / fy, depth_m * 1000.0]
            )
            board_point = np.asarray(
                [args.corner_col * args.square_mm, args.corner_row * args.square_mm, 0.0]
            )
            pnp_corner = detection.transform_camera_board.transform_point(
                board_point, frame="calibration_board"
            )
            pnp_points.append(pnp_corner)
            pnp_rotations.append(detection.transform_camera_board.matrix[:3, :3])
            depth_points.append(depth_point)
            reprojection.append(detection.reprojection_rmse_px)
            depth_vs_pnp.append(float(np.linalg.norm(depth_point - pnp_corner)))
            time.sleep(args.interval_s)
    finally:
        provider.close()
    after = feedback()

    if len(pnp_points) < max(10, int(args.frames * 0.9)):
        raise RuntimeError("insufficient valid 54-corner frames: {} / {}".format(len(pnp_points), args.frames))
    pnp_points_array = np.asarray(pnp_points)
    depth_points_array = np.asarray(depth_points)
    rotation_sum = np.sum(np.asarray(pnp_rotations), axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    mean_rotation = u @ vt
    rotation_errors = [rotation_angle_deg(value, mean_rotation) for value in pnp_rotations]
    joint_drift = {
        key: after[key] - before[key] for key in ("b", "s", "e", "t", "r", "x", "y", "z")
    }
    report = {
        "status": "MEASURED_STATIC_NO_MOTION",
        "requested_frames": args.frames,
        "valid_54_corner_frames": len(pnp_points),
        "failed_frames": failed,
        "feedback_before": before,
        "feedback_after": after,
        "feedback_drift": joint_drift,
        "pnp_corner_camera": point_stats(pnp_points_array),
        "pnp_rotation_scatter_deg": stats(rotation_errors),
        "reprojection_rmse_px": stats(reprojection),
        "rgbd_corner_camera": point_stats(depth_points_array),
        "rgbd_vs_pnp_mm": stats(depth_vs_pnp),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
