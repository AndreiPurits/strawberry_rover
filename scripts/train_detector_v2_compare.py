#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_baseline_metrics(progress_json: Path, *, model_name: str) -> Optional[Dict[str, Any]]:
    if not progress_json.is_file():
        return None
    data = json.loads(progress_json.read_text(encoding="utf-8"))
    for r in data.get("runs", []):
        if r.get("model_name") == model_name and r.get("status") == "ok":
            return r
    return None


def _metrics_from_val(res: Any) -> Dict[str, float]:
    """
    Ultralytics Results has nested metrics; keep robust extraction.
    """
    out: Dict[str, float] = {}
    try:
        out["mAP50"] = float(res.box.map50)
        out["mAP50_95"] = float(res.box.map)
        out["precision"] = float(res.box.mp)
        out["recall"] = float(res.box.mr)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train best detector on dataset v2 and compare metrics vs baseline.")
    ap.add_argument("--model", default="yolov8s.pt", help="Ultralytics model name or path (best-quality baseline was yolov8s).")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v2" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP (mixed precision).",
    )
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "detect_benchmark_v2"))
    ap.add_argument("--name", default="yolov8s_v2")
    ap.add_argument("--device", default="", help="Empty=auto. Example: 0")
    ap.add_argument("--baseline-progress", default=str(REPO_ROOT / "reports" / "detect_benchmark" / "benchmark_progress.json"))
    ap.add_argument("--baseline-model-name", default="yolov8s")
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "reports" / "detect_benchmark_v2"))
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline_metrics(Path(args.baseline_progress), model_name=str(args.baseline_model_name))

    t0 = time.time()
    model = YOLO(args.model)
    train_kwargs: Dict[str, Any] = dict(
        data=args.data,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        seed=int(args.seed),
        project=str(args.project),
        name=str(args.name),
        exist_ok=False,
        pretrained=True,
        amp=bool(args.amp),
        verbose=True,
    )
    if str(args.device).strip():
        train_kwargs["device"] = str(args.device).strip()

    train_res = model.train(**train_kwargs)
    train_time = time.time() - t0

    # Validate best.pt (Ultralytics will load best weights automatically when calling val() on model trained in-place)
    val_res = model.val(data=args.data, imgsz=int(args.imgsz), batch=int(args.batch), workers=int(args.workers))
    m = _metrics_from_val(val_res)

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": args.data,
        "train": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "seed": args.seed,
            "project": args.project,
            "name": args.name,
            "train_time_s": round(train_time, 3),
        },
        "val_metrics": m,
        "baseline": baseline or {},
    }

    out_json = report_dir / "detector_v2_compare.json"
    out_md = report_dir / "detector_v2_compare.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x: Any) -> str:
        try:
            return f"{float(x):.6f}"
        except Exception:
            return "-"

    md = []
    md.append("# Detector v2 compare")
    md.append("")
    md.append(f"- Generated (UTC): `{out['ts']}`")
    md.append(f"- Dataset: `{args.data}`")
    md.append(f"- Model: `{args.model}`")
    md.append("")
    md.append("## Metrics")
    md.append("")
    md.append("| run | mAP50 | mAP50-95 | P | R |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        "| baseline (v1) | "
        + f"{f((baseline or {}).get('mAP50'))} | {f((baseline or {}).get('mAP50_95'))} | {f((baseline or {}).get('precision'))} | {f((baseline or {}).get('recall'))} |"
    )
    md.append(f"| v2 | {f(m.get('mAP50'))} | {f(m.get('mAP50_95'))} | {f(m.get('precision'))} | {f(m.get('recall'))} |")
    md.append("")
    md.append("## Training")
    md.append("")
    md.append(f"- Train time (s): `{out['train']['train_time_s']}`")
    md.append(f"- Run dir: `{Path(args.project) / args.name}`")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

