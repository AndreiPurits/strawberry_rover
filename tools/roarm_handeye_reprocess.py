#!/usr/bin/env python3
"""Recompute PnP from retained raw images without changing the source dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.roarm_calibration.chessboard import (  # noqa: E402
    CameraIntrinsics,
    detect_chessboard,
    draw_detection,
)
from pipelines.roarm_calibration.handeye import HandEyeObservation  # noqa: E402
from pipelines.roarm_calibration.io import load_dataset, save_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import json

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    metadata = dict(payload["metadata"])
    camera = metadata["intrinsics"]
    board = metadata["board"]
    intrinsics = CameraIntrinsics(
        width=int(camera["width"]),
        height=int(camera["height"]),
        camera_matrix=np.asarray(camera["K"], dtype=np.float64),
        distortion=np.asarray(camera["D"], dtype=np.float64),
        frame=str(metadata["camera_frame"]),
        rectified=bool(camera["rectified"]),
    )
    observations = load_dataset(args.source)
    raw_dir = args.source.parent / (args.source.stem + "_raw")
    overlay_dir = args.output.parent / (args.output.stem + "_overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    reprocessed = []
    for observation in observations:
        image_path = raw_dir / (observation.observation_id + ".png")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("missing raw image: {}".format(image_path))
        detection = detect_chessboard(
            image,
            intrinsics,
            cols=int(board["cols"]),
            rows=int(board["rows"]),
            square_size_mm=float(board["square_size_mm"]),
        )
        if detection is None:
            raise RuntimeError("CHESSBOARD_NOT_DETECTED: {}".format(observation.observation_id))
        reprocessed.append(
            HandEyeObservation(
                observation_id=observation.observation_id,
                transform_base_link5=observation.transform_base_link5,
                transform_camera_board=detection.transform_camera_board,
                split=observation.split,
                reprojection_rmse_px=detection.reprojection_rmse_px,
                q_rad=observation.q_rad,
            )
        )
        cv2.imwrite(
            str(overlay_dir / (observation.observation_id + "_overlay.png")),
            draw_detection(
                image,
                detection,
                cols=int(board["cols"]),
                rows=int(board["rows"]),
            ),
        )
    metadata["corner_order_convention"] = "lower_right_image_origin"
    metadata["reprocessed_from"] = str(args.source)
    save_dataset(args.output, reprocessed, metadata=metadata)
    print("reprocessed {} observations -> {}".format(len(reprocessed), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
