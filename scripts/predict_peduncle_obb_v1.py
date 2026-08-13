#!/usr/bin/env python3
"""Run peduncle OBB model on val (or any folder) and save visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from yolo_jetson_compat import apply_torchvision_nms_patch
except ImportError:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch  # type: ignore


def main() -> int:
    apply_torchvision_nms_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=str(REPO_ROOT / "runs" / "peduncle_obb" / "yolov8n_peduncle_v1" / "weights" / "best.pt"),
    )
    ap.add_argument(
        "--source",
        default=str(REPO_ROOT / "data" / "peduncle_obb_approved_v1" / "images" / "val"),
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs" / "peduncle_obb" / "yolov8n_peduncle_v1_predict_val"),
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"missing weights: {w}")

    model = YOLO(str(w))
    model.predict(
        source=str(args.source),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        device=str(args.device),
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=True,
        save=True,
    )
    print(f"saved predictions → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
