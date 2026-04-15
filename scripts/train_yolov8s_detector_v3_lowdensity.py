#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Train yolov8s on dataset_v3 (low-density) with early stopping.")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v3" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32, help="Safe default for Orin. Increase only if you know it fits.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "detect_benchmark_v3"))
    ap.add_argument("--name", default="yolov8s_v3_lowdensity")
    ap.add_argument("--amp", action="store_true", help="Enable AMP (default off).")
    args = ap.parse_args()

    t0 = time.time()
    model = YOLO("yolov8s.pt")
    train_kwargs: Dict[str, Any] = dict(
        data=str(args.data),
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
        exist_ok=False,
        verbose=True,
    )
    _ = model.train(**train_kwargs)
    dt = time.time() - t0
    print(f"Training finished in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

