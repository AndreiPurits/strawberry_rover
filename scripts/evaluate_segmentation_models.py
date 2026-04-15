#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvalRow:
    model_name: str
    weights_path: str
    weights_size_mb: float
    epochs_completed: int
    run_status: str
    box_precision: float
    box_recall: float
    box_map50: float
    box_map50_95: float
    mask_precision: float
    mask_recall: float
    mask_map50: float
    mask_map50_95: float
    inference_time_ms_per_image: float
    fps: float


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _weights_size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def _count_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    # results.csv includes header; epoch rows are non-empty lines after first.
    lines = [ln for ln in results_csv.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    return len(lines)


def _run_status(epochs_completed: int, expected: int = 150) -> str:
    if epochs_completed >= expected:
        return "completed"
    if epochs_completed > 0:
        return "partial"
    return "unknown"


def _val_metrics_and_speed(model: YOLO, *, data_yaml: str, device: str, imgsz: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    r = model.val(data=data_yaml, split="test", imgsz=imgsz, device=device, verbose=False)

    metrics: Dict[str, float] = {}
    metrics["box_precision"] = float(getattr(r.box, "mp", 0.0))
    metrics["box_recall"] = float(getattr(r.box, "mr", 0.0))
    metrics["box_map50"] = float(getattr(r.box, "map50", 0.0))
    metrics["box_map50_95"] = float(getattr(r.box, "map", 0.0))

    metrics["mask_precision"] = float(getattr(r.seg, "mp", 0.0))
    metrics["mask_recall"] = float(getattr(r.seg, "mr", 0.0))
    metrics["mask_map50"] = float(getattr(r.seg, "map50", 0.0))
    metrics["mask_map50_95"] = float(getattr(r.seg, "map", 0.0))

    speed: Dict[str, float] = {}
    # r.speed typically contains preprocess/inference/loss/postprocess in ms
    sp = getattr(r, "speed", None)
    if isinstance(sp, dict):
        for k, v in sp.items():
            try:
                speed[str(k)] = float(v)
            except Exception:
                pass
    return metrics, speed


def _inference_speed(model: YOLO, images: List[Path], *, device: str, imgsz: int) -> Tuple[float, float]:
    if not images:
        return 0.0, 0.0
    # Warmup
    _ = model.predict(source=str(images[0]), imgsz=imgsz, device=device, verbose=False)
    t0 = time.time()
    for p in images:
        _ = model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    dt = time.time() - t0
    ms = (dt / len(images)) * 1000.0
    fps = (len(images) / dt) if dt > 0 else 0.0
    return ms, fps


def _save_previews(model: YOLO, images: List[Path], out_dir: Path, *, device: str, imgsz: int) -> None:
    _safe_mkdir(out_dir)
    for p in images:
        res = model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)[0]
        arr = res.plot()  # BGR
        im = Image.fromarray(arr[..., ::-1])  # to RGB
        im.save(out_dir / p.name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate YOLOv8-seg best.pt weights on the same test split.")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_segmentation_dataset" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--preview_k", type=int, default=50)
    ap.add_argument("--speed_n", type=int, default=250)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in torch; refusing to run evaluation.")

    runs_root = REPO_ROOT / "runs" / "segment_benchmark"
    models = [
        ("yolov8n-seg", runs_root / "yolov8n_seg_benchmark" / "weights" / "best.pt", runs_root / "yolov8n_seg_benchmark" / "results.csv"),
        ("yolov8s-seg", runs_root / "yolov8s_seg_benchmark" / "weights" / "best.pt", runs_root / "yolov8s_seg_benchmark" / "results.csv"),
        ("yolov8m-seg", runs_root / "yolov8m_seg_benchmark" / "weights" / "best.pt", runs_root / "yolov8m_seg_benchmark" / "results.csv"),
    ]

    test_dir = REPO_ROOT / "data" / "yolo_segmentation_dataset" / "images" / "test"
    test_images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    preview_images = test_images[: int(args.preview_k)]
    speed_images = test_images[: int(args.speed_n)]

    preview_root = REPO_ROOT / "data" / "segmentation_preview_compare"
    reports_dir = REPO_ROOT / "reports" / "segment_benchmark"
    _safe_mkdir(preview_root)
    _safe_mkdir(reports_dir)

    rows: List[EvalRow] = []

    for name, wpath, results_csv in models:
        if not wpath.is_file():
            rows.append(
                EvalRow(
                    model_name=name,
                    weights_path=str(wpath),
                    weights_size_mb=0.0,
                    epochs_completed=_count_epochs(results_csv),
                    run_status="missing_weights",
                    box_precision=0.0,
                    box_recall=0.0,
                    box_map50=0.0,
                    box_map50_95=0.0,
                    mask_precision=0.0,
                    mask_recall=0.0,
                    mask_map50=0.0,
                    mask_map50_95=0.0,
                    inference_time_ms_per_image=0.0,
                    fps=0.0,
                )
            )
            continue

        epochs = _count_epochs(results_csv)
        status = _run_status(epochs, expected=150)

        model = YOLO(str(wpath))
        metrics, speed = _val_metrics_and_speed(model, data_yaml=str(args.data), device=str(args.device), imgsz=int(args.imgsz))
        ms, fps = _inference_speed(model, speed_images, device=str(args.device), imgsz=int(args.imgsz))
        _save_previews(model, preview_images, preview_root / name.replace("-", "_"), device=str(args.device), imgsz=int(args.imgsz))

        # Prefer pure inference timing we measured; also keep a note if val speed differs.
        _ = speed  # kept for potential future extension

        rows.append(
            EvalRow(
                model_name=name,
                weights_path=str(wpath),
                weights_size_mb=_weights_size_mb(wpath),
                epochs_completed=epochs,
                run_status=status,
                box_precision=float(metrics["box_precision"]),
                box_recall=float(metrics["box_recall"]),
                box_map50=float(metrics["box_map50"]),
                box_map50_95=float(metrics["box_map50_95"]),
                mask_precision=float(metrics["mask_precision"]),
                mask_recall=float(metrics["mask_recall"]),
                mask_map50=float(metrics["mask_map50"]),
                mask_map50_95=float(metrics["mask_map50_95"]),
                inference_time_ms_per_image=float(ms),
                fps=float(fps),
            )
        )

    # Write CSV
    csv_path = reports_dir / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Write JSON
    json_path = reports_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Basic recommendation logic
    best_quality = max(rows, key=lambda r: (r.mask_map50_95, r.mask_recall))
    best_speed = max(rows, key=lambda r: (r.fps, -r.inference_time_ms_per_image))

    # speed/quality tradeoff: maximize mask_map50_95 per log(size) and latency penalty
    def score_tradeoff(r: EvalRow) -> float:
        size_pen = 1.0 + (r.weights_size_mb / 50.0)
        lat_pen = 1.0 + (r.inference_time_ms_per_image / 30.0)
        return (r.mask_map50_95 * r.mask_recall) / (size_pen * lat_pen)

    best_tradeoff = max(rows, key=score_tradeoff)

    md = []
    md.append("## Segmentation benchmark (fair test split)")
    md.append("")
    md.append(f"- test split: `{test_dir}`")
    md.append(f"- data.yaml: `{args.data}`")
    md.append(f"- imgsz: `{args.imgsz}`  device: `{args.device}`")
    md.append("")
    md.append("| model | status | epochs | mask mAP50-95 | mask R | fps | ms/img | size (MB) | weights |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r.model_name} | {r.run_status} | {r.epochs_completed} | {r.mask_map50_95:.4f} | {r.mask_recall:.4f} | {r.fps:.2f} | {r.inference_time_ms_per_image:.2f} | {r.weights_size_mb:.1f} | `{r.weights_path}` |"
        )
    md.append("")
    md.append("### Answers")
    md.append(f"1. Best mask quality (mAP50-95, recall): **{best_quality.model_name}**")
    md.append(f"2. Best speed (FPS): **{best_speed.model_name}**")
    md.append(f"3. Best speed/quality tradeoff: **{best_tradeoff.model_name}**")
    md.append("")
    md.append("### Recommendations")
    md.append(f"- Production segmentation default: **{best_tradeoff.model_name}** (balanced).")
    md.append(f"- Stereo/grasp pipeline candidate: **{best_tradeoff.model_name}** (prefers recall + latency).")
    md.append("")
    md.append("### Notes on incomplete runs")
    for r in rows:
        if r.run_status != "completed":
            md.append(f"- {r.model_name}: run_status=`{r.run_status}` (epochs_completed={r.epochs_completed}), evaluated by `best.pt`.")
    md.append("")
    md.append("### Do we need to continue training s/m?")
    md.append("- If a partial model is already best or close (< ~0.5% mAP50-95 gap), further training may not help and can overfit; consider stopping early.")
    md.append("- If quality gap is significant and latency is acceptable, continue training with early stopping enabled.")

    md_path = reports_dir / "benchmark_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    print(f"Previews: {preview_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

