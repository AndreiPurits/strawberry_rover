#!/usr/bin/env python3
"""Measure one physical checkerboard corner through RGB-D -> camera -> base."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.roarm_calibration.chessboard import CameraIntrinsics, detect_chessboard  # noqa: E402
from pipelines.roarm_calibration.io import load_dataset  # noqa: E402
from pipelines.roarm_calibration.se3 import RigidTransform  # noqa: E402


def transform(value):
    return RigidTransform(value["parent"], value["child"], np.asarray(value["matrix"]))


def robust_depth(depth_m: np.ndarray, u: float, v: float, radius: int) -> float:
    x, y = int(round(u)), int(round(v))
    y0, y1 = max(0, y - radius), min(depth_m.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth_m.shape[1], x + radius + 1)
    values = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 3:
        raise RuntimeError("insufficient valid depth around ({:.1f}, {:.1f})".format(u, v))
    return float(np.median(values))


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--corner-col", type=int, default=4)
    parser.add_argument("--corner-row", type=int, default=3)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    link5_camera = transform(calibration["report"]["T_link5_camera"])
    source = json.loads(args.dataset.read_text(encoding="utf-8"))
    metadata = source["metadata"]
    board = metadata["board"]
    camera = metadata["intrinsics"]
    cols, rows = int(board["cols"]), int(board["rows"])
    if not (0 <= args.corner_col < cols and 0 <= args.corner_row < rows):
        raise ValueError("corner index outside checkerboard")
    intrinsics = CameraIntrinsics(
        int(camera["width"]),
        int(camera["height"]),
        np.asarray(camera["K"]),
        np.asarray(camera["D"]),
        str(metadata["camera_frame"]),
        bool(camera["rectified"]),
    )
    raw_dir = args.dataset.parent / (args.dataset.stem + "_raw")
    observations = load_dataset(args.dataset)
    rows_out = []
    for observation in observations:
        image = cv2.imread(str(raw_dir / (observation.observation_id + ".png")))
        depth = np.load(str(raw_dir / (observation.observation_id + "_depth_m.npy")))
        detection = detect_chessboard(
            image,
            intrinsics,
            cols=cols,
            rows=rows,
            square_size_mm=float(board["square_size_mm"]),
        )
        if detection is None:
            raise RuntimeError("board lost in {}".format(observation.observation_id))
        corners = detection.corners_px.reshape(rows, cols, 2)
        u, v = map(float, corners[args.corner_row, args.corner_col])
        depth_m = robust_depth(depth, u, v, args.depth_radius)
        fx, fy = intrinsics.camera_matrix[0, 0], intrinsics.camera_matrix[1, 1]
        cx, cy = intrinsics.camera_matrix[0, 2], intrinsics.camera_matrix[1, 2]
        point_camera_mm = np.asarray(
            [(u - cx) * depth_m * 1000.0 / fx, (v - cy) * depth_m * 1000.0 / fy, depth_m * 1000.0]
        )
        point_base_mm = observation.transform_base_link5.then(link5_camera).transform_point(
            point_camera_mm, frame=link5_camera.child
        )
        point_board_mm = np.asarray(
            [args.corner_col * float(board["square_size_mm"]), args.corner_row * float(board["square_size_mm"]), 0.0]
        )
        pnp_camera_mm = detection.transform_camera_board.transform_point(
            point_board_mm, frame="calibration_board"
        )
        rows_out.append(
            {
                "id": observation.observation_id,
                "pixel": [u, v],
                "depth_mm": depth_m * 1000.0,
                "point_camera_mm": point_camera_mm.tolist(),
                "point_base_mm": point_base_mm.tolist(),
                "depth_vs_pnp_camera_error_mm": float(np.linalg.norm(point_camera_mm - pnp_camera_mm)),
            }
        )
    points = np.asarray([row["point_base_mm"] for row in rows_out])
    center = np.mean(points, axis=0)
    errors = np.linalg.norm(points - center, axis=1)
    report = {
        "status": "MEASURED_THRESHOLD_NOT_SET",
        "KIN_GATE": "FAIL",
        "corner": {"col": args.corner_col, "row": args.corner_row},
        "count": len(rows_out),
        "mean_point_base_mm": center.tolist(),
        "xyz_std_mm": np.std(points, axis=0).tolist(),
        "xyz_span_mm": (np.max(points, axis=0) - np.min(points, axis=0)).tolist(),
        "scatter_norm_mm": stats(errors),
        "depth_vs_pnp_camera_error_mm": stats(
            [row["depth_vs_pnp_camera_error_mm"] for row in rows_out]
        ),
        "observations": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scatter_norm_mm": report["scatter_norm_mm"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
