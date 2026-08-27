#!/usr/bin/env python3
"""Capture a live board frame and project CAD-derived link5 grasp landmarks."""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "ops/axm-monitor/agent"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pipelines.roarm_calibration.chessboard import CameraIntrinsics, detect_chessboard  # noqa: E402
from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402
from pipelines.roarm_calibration.urdf_fk import base_to_grasp, base_to_link5  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from roarm_proxy import execute_rpc  # noqa: E402


def stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    vertices = np.empty((count, 3, 3), dtype=np.float64)
    offset = 84
    for index in range(count):
        vertices[index] = np.frombuffer(raw, dtype="<f4", count=9, offset=offset + 12).reshape(3, 3)
        offset += 50
    return vertices.reshape(-1, 3)


def rotation_rpy(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.asarray([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                       [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                       [-sy, cy * sx, cy * cx]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    args = parser.parse_args()

    value = json.loads(args.calibration.read_text(encoding="utf-8"))["report"]["T_link5_camera"]
    link5_camera = RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))
    camera_link5 = link5_camera.inverse()

    provider = Ros2RgbDepthProvider(
        rgb_topic="/stereo_camera/color/image_rect_raw",
        depth_topic="/stereo_camera/aligned_depth_to_color/image_raw",
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic="/stereo_camera/color/camera_info")
    try:
        frame = provider.read(timeout_s=12.0)
        msg = provider._color_info_msg
        if msg is None:
            raise RuntimeError("CameraInfo unavailable")
        intrinsics = CameraIntrinsics(
            int(msg.width), int(msg.height), np.asarray(msg.k).reshape(3, 3),
            np.asarray(msg.d), str(msg.header.frame_id), True,
        )
        board = detect_chessboard(frame.rgb_bgr, intrinsics, cols=9, rows=6, square_size_mm=18.0)
        image = frame.rgb_bgr.copy()
        if board is not None:
            cv2.drawChessboardCorners(image, (9, 6), board.corners_px.astype(np.float32), True)

        landmarks_link5 = {
            "official_tcp": [0.0, 0.0, 115.428],
            "grasp_center": [-8.004633, 0.0, 115.428],
            "fixed_inner": [-10.025849, 0.0, 115.428],
            "moving_inner": [-5.983416, 0.0, 115.428],
            "approach_back": [-8.004633, 0.0, 100.0],
        }
        colors = {
            "official_tcp": (0, 255, 255), "grasp_center": (0, 0, 255),
            "fixed_inner": (255, 0, 255), "moving_inner": (255, 0, 255),
            "approach_back": (0, 128, 255),
        }
        projected = {}
        for name, point in landmarks_link5.items():
            p = camera_link5.transform_point(point, frame="link5")
            uv = intrinsics.camera_matrix @ p
            uv = uv[:2] / uv[2]
            projected[name] = {"camera_mm": p.tolist(), "pixel": uv.tolist()}
            center = tuple(int(round(v)) for v in uv)
            cv2.circle(image, center, 6 if name == "grasp_center" else 4, colors[name], -1)
            cv2.putText(image, name, (center[0] + 7, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
        a = tuple(int(round(v)) for v in projected["fixed_inner"]["pixel"])
        b = tuple(int(round(v)) for v in projected["moving_inner"]["pixel"])
        cv2.line(image, a, b, (255, 0, 255), 2)
        a = tuple(int(round(v)) for v in projected["approach_back"]["pixel"])
        b = tuple(int(round(v)) for v in projected["grasp_center"]["pixel"])
        cv2.arrowedLine(image, a, b, (0, 128, 255), 2)

        feedback = (execute_rpc("feedback", {}) or {}).get("feedback", {})
        gripper_g = float(feedback.get("g", 0.0))
        q = [float(feedback[name]) for name in ("b", "s", "e", "t", "r")]
        base_board = None if board is None else base_to_link5(q).then(link5_camera).then(board.transform_camera_board)
        base_grasp = base_to_grasp(q)
        cad_alignment = None
        if args.mesh_dir is not None:
            link5_vertices = stl_vertices(args.mesh_dir / "link5.stl")
            gripper_vertices = stl_vertices(args.mesh_dir / "gripper_link.stl")
            origin = np.asarray([0.0, 18.821, 52.035])
            gripper_rotation = rotation_rpy(-math.pi / 2.0, -math.pi / 2.0, 0.0) @ rotation_rpy(0.0, 0.0, gripper_g)
            gripper_vertices = gripper_vertices @ gripper_rotation.T + origin
            for label, vertices, color in (("fixed", link5_vertices, (0, 255, 0)), ("moving", gripper_vertices, (255, 255, 0))):
                vertices = vertices[vertices[:, 2] > 85.0]
                if len(vertices) == 0:
                    continue
                points_camera = np.asarray([
                    camera_link5.transform_point(point, frame="link5") for point in vertices[::10]
                ])
                valid = points_camera[:, 2] > 1.0
                uv = (intrinsics.camera_matrix @ points_camera[valid].T).T
                uv = np.rint(uv[:, :2] / uv[:, 2:3]).astype(np.int32)
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < intrinsics.width) & (uv[:, 1] >= 0) & (uv[:, 1] < intrinsics.height)]
                if label == "fixed" and len(uv):
                    dark = cv2.cvtColor(frame.rgb_bgr, cv2.COLOR_BGR2GRAY) < 85
                    distance = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3)
                    values = distance[uv[:, 1], uv[:, 0]]
                    cad_alignment = {
                        "dark_pixel_distance_mean_px": float(np.mean(values)),
                        "dark_pixel_distance_p95_px": float(np.percentile(values, 95)),
                        "dark_pixel_distance_max_px": float(np.max(values)),
                    }
                for x, y in uv:
                    image[y, x] = color

        args.output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_dir / "rgb.png"), frame.rgb_bgr)
        cv2.imwrite(str(args.output_dir / "overlay.png"), image)
        report = {
            "board_found": board is not None,
            "board_reprojection_px": None if board is None else board.reprojection_rmse_px,
            "landmarks_link5_mm": landmarks_link5,
            "projected": projected,
            "gripper_g": gripper_g,
            "q_rad": q,
            "T_base_grasp": base_grasp.to_dict(),
            "T_base_board": None if base_board is None else base_board.to_dict(),
            "cad_alignment": cad_alignment,
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
