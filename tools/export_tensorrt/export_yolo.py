#!/usr/bin/env python3
"""Export YOLO .pt to ONNX and TensorRT .engine (run on target Jetson)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLO weights to ONNX + TensorRT engine.")
    ap.add_argument("--weights", required=True, help="Path to best.pt")
    ap.add_argument("--task", choices=("detect", "segment"), default="detect")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 export")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--engine-out", default="", help="Canonical .engine path (copied after export).")
    ap.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export if engine-out already exists.")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Jetson torchvision ABI fix for ultralytics NMS
    try:
        from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

        apply_torchvision_nms_patch()
    except Exception:
        pass

    from ultralytics import YOLO

    engine_out = Path(args.engine_out) if str(args.engine_out).strip() else None
    if engine_out and engine_out.exists() and bool(args.skip_onnx):
        print(f"SKIP exists: {engine_out}", flush=True)
        return 0

    model = YOLO(str(weights))
    print(f"export task={args.task} imgsz={args.imgsz} half={args.half} -> {outdir}", flush=True)

    # ONNX (onnxsim may be unavailable on Jetson — fall back to unsimplified)
    if not bool(args.skip_onnx):
        try:
            onnx_path = model.export(
                format="onnx",
                imgsz=int(args.imgsz),
                half=bool(args.half),
                simplify=True,
                device=str(args.device),
            )
        except Exception as exc:
            print(f"onnx simplify failed ({exc}); retrying simplify=False", flush=True)
            onnx_path = model.export(
                format="onnx",
                imgsz=int(args.imgsz),
                half=bool(args.half),
                simplify=False,
                device=str(args.device),
            )
        print(f"onnx: {onnx_path}", flush=True)

    # TensorRT engine (ultralytics export)
    engine_path = model.export(
        format="engine",
        imgsz=int(args.imgsz),
        half=bool(args.half),
        device=str(args.device),
        workspace=4,
    )
    engine_path = Path(str(engine_path))
    print(f"engine: {engine_path}", flush=True)

    if engine_out:
        engine_out.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(engine_path, engine_out)
        print(f"canonical: {engine_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
