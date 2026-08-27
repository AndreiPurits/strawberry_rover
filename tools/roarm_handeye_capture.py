#!/usr/bin/env python3
"""Capture one static chessboard observation; contains no robot motion API."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "ops/axm-monitor/agent"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pipelines.roarm_calibration.chessboard import (  # noqa: E402
    CameraIntrinsics,
    detect_chessboard,
    draw_detection,
)
from pipelines.roarm_calibration.handeye import HandEyeObservation  # noqa: E402
from pipelines.roarm_calibration.io import (  # noqa: E402
    SCHEMA,
    load_dataset,
    observation_to_dict,
)
from pipelines.roarm_calibration.urdf_fk import base_to_link5  # noqa: E402
from pipelines.roarm_m3_kinematics.conventions import JointVector  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


RGB_TOPIC = "/stereo_camera/color/image_rect_raw"
DEPTH_TOPIC = "/stereo_camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/stereo_camera/color/camera_info"


def feedback_joints() -> JointVector:
    response = execute_rpc("feedback", {})
    feedback = response.get("feedback") if isinstance(response, dict) else None
    if not isinstance(feedback, dict):
        raise RuntimeError("T:105 feedback unavailable")
    return JointVector(
        base=float(feedback["b"]),
        shoulder=float(feedback["s"]),
        elbow=float(feedback["e"]),
        wrist=float(feedback.get("t", feedback.get("w"))),
        roll=float(feedback["r"]),
    )


def camera_intrinsics(
    provider: Ros2RgbDepthProvider,
    frame_shape,
    timeout_s: float,
) -> CameraIntrinsics:
    deadline = time.monotonic() + timeout_s
    msg = provider._color_info_msg  # cached immutable ROS CameraInfo
    while msg is None and time.monotonic() < deadline:
        time.sleep(0.02)
        msg = provider._color_info_msg
    if msg is None:
        raise RuntimeError("CameraInfo unavailable")
    frame = str(msg.header.frame_id) or "stereo_camera_color_optical_frame"
    if frame != "stereo_camera_color_optical_frame":
        raise RuntimeError("unexpected camera optical frame: {}".format(frame))
    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        camera_matrix=np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(msg.d, dtype=np.float64),
        frame=frame,
        rectified=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--cols", type=int, required=True, help="inner chessboard corners")
    parser.add_argument("--rows", type=int, required=True, help="inner chessboard corners")
    parser.add_argument("--square-mm", type=float, required=True, help="measured square edge")
    parser.add_argument("--timeout-s", type=float, default=6.0)
    parser.add_argument("--max-joint-drift-rad", type=float, default=0.003)
    args = parser.parse_args()

    provider = Ros2RgbDepthProvider(
        rgb_topic=RGB_TOPIC,
        depth_topic=DEPTH_TOPIC,
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic=INFO_TOPIC)
    try:
        q_before = feedback_joints()
        frame = provider.read(timeout_s=args.timeout_s)
        q_after = feedback_joints()
        joint_drift = float(np.max(np.abs(q_after.as_array() - q_before.as_array())))
        if joint_drift > args.max_joint_drift_rad:
            raise RuntimeError("arm was not static; max joint drift {:.6f} rad".format(joint_drift))
        q = JointVector(*map(float, 0.5 * (q_before.as_array() + q_after.as_array())))
        intrinsics = camera_intrinsics(provider, frame.rgb_bgr.shape, args.timeout_s)
        detection = detect_chessboard(
            frame.rgb_bgr,
            intrinsics,
            cols=args.cols,
            rows=args.rows,
            square_size_mm=args.square_mm,
        )
        if detection is None:
            raise RuntimeError("CHESSBOARD_NOT_DETECTED")
    finally:
        provider.close()

    observation = HandEyeObservation(
        observation_id=args.id,
        split=args.split,
        transform_base_link5=base_to_link5(q.as_array()),
        transform_camera_board=detection.transform_camera_board,
        reprojection_rmse_px=detection.reprojection_rmse_px,
        q_rad=tuple(map(float, q.as_array())),
    )
    existing = load_dataset(args.dataset) if args.dataset.exists() else []
    if any(item.observation_id == args.id for item in existing):
        raise RuntimeError("duplicate observation id: {}".format(args.id))

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    capture_dir = args.dataset.parent / (args.dataset.stem + "_raw")
    capture_dir.mkdir(parents=True, exist_ok=True)
    image_path = capture_dir / (args.id + ".png")
    overlay_path = capture_dir / (args.id + "_overlay.png")
    depth_path = capture_dir / (args.id + "_depth_m.npy")
    if not cv2.imwrite(str(image_path), frame.rgb_bgr):
        raise RuntimeError("failed to save raw image")
    cv2.imwrite(
        str(overlay_path),
        draw_detection(frame.rgb_bgr, detection, cols=args.cols, rows=args.rows),
    )
    if frame.depth_m is None:
        raise RuntimeError("aligned depth unavailable")
    np.save(str(depth_path), np.asarray(frame.depth_m, dtype=np.float32))

    payload = {
        "schema": SCHEMA,
        "metadata": {
            "camera_mount": "eye-in-hand on link5",
            "camera_frame": intrinsics.frame,
            "image_topic": RGB_TOPIC,
            "depth_topic": DEPTH_TOPIC,
            "board": {"cols": args.cols, "rows": args.rows, "square_size_mm": args.square_mm},
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "K": intrinsics.camera_matrix.tolist(),
                "D": intrinsics.distortion.tolist(),
                "rectified": intrinsics.rectified,
            },
        },
        "observations": [observation_to_dict(item) for item in existing + [observation]],
    }
    temporary = args.dataset.with_suffix(args.dataset.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.dataset)
    print(
        json.dumps(
            {
                "status": "CAPTURED_NO_MOTION",
                "id": args.id,
                "split": args.split,
                "reprojection_rmse_px": detection.reprojection_rmse_px,
                "q": q.as_dict(),
                "raw_image": str(image_path),
                "overlay": str(overlay_path),
                "aligned_depth_m": str(depth_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
