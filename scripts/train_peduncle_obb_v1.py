#!/usr/bin/env python3
"""Train small YOLO-OBB peduncle model on approved close-up crops."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from yolo_jetson_compat import apply_torchvision_nms_patch
except ImportError:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch  # type: ignore


def main() -> int:
    apply_torchvision_nms_patch()
    ap = argparse.ArgumentParser(description="Train yolov8n-obb on peduncle OBB approved set.")
    ap.add_argument(
        "--data",
        default=str(REPO_ROOT / "data" / "peduncle_obb_approved_v1" / "data.yaml"),
    )
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "peduncle_obb"))
    ap.add_argument("--name", default="yolov8n_peduncle_v1")
    ap.add_argument("--amp", action="store_true", help="Enable AMP (default off on Jetson).")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"missing data.yaml: {data}")

    t0 = time.time()
    model = YOLO(args.model)
    train_kwargs: Dict[str, Any] = dict(
        data=str(data),
        epochs=int(args.epochs),
        patience=int(args.patience),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        seed=int(args.seed),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
        pretrained=True,
        amp=bool(args.amp),
        exist_ok=True,
        verbose=True,
        # small dataset: more geometry noise helps OBB
        degrees=30.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.8,
        close_mosaic=15,
    )
    results = model.train(**train_kwargs)
    dt = time.time() - t0
    print(f"Training finished in {dt:.1f}s")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")
    if best.is_file():
        print("Running quick val…")
        m = YOLO(str(best))
        m.val(data=str(data), imgsz=int(args.imgsz), device=str(args.device), split="val")
    return 0 if results is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
