#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelResult:
    model_name: str
    batch_size: int
    epochs_completed: int
    train_time_total_s: float
    avg_epoch_time_s: float
    box_map50: float
    box_map50_95: float
    mask_map50: float
    mask_map50_95: float
    precision: float
    recall: float
    inference_time_ms_per_image: float
    fps: float
    best_weights_path: str
    weights_size_mb: float
    status: str


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now() -> float:
    return time.time()


def _weights_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0


def _pick_batch(model_pt: str) -> int:
    # Conservative defaults for Orin; will auto-reduce on OOM.
    if "yolov8n" in model_pt:
        return 32
    if "yolov8s" in model_pt:
        return 16
    return 8


def _train_one(
    model_pt: str,
    *,
    data_yaml: str,
    project: str,
    name: str,
    epochs: int,
    patience: int,
    imgsz: int,
    device: str,
    workers: int,
    seed: int,
) -> Tuple[Optional[Path], int, float, float, str]:
    run_dir = Path(project) / name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    # If already finished, skip retraining.
    if best.is_file() and results_csv.is_file():
        try:
            epochs_completed = sum(1 for _ in results_csv.read_text(encoding="utf-8").splitlines()[1:] if _.strip())
        except Exception:
            epochs_completed = 0
        if epochs_completed >= epochs:
            return best, _pick_batch(model_pt), 0.0, 0.0, "ok"

    batch = _pick_batch(model_pt)
    start = _now()
    last_err = ""
    for _ in range(6):
        try:
            # Resume if we have a checkpoint.
            if last.is_file():
                model = YOLO(str(last))
                _ = model.train(resume=True, device=device, verbose=True)
            else:
                model = YOLO(model_pt)
                _ = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    patience=patience,
                    imgsz=imgsz,
                    device=device,
                    workers=workers,
                    seed=seed,
                    batch=batch,
                    pretrained=True,
                    project=project,
                    name=name,
                    exist_ok=False,
                    verbose=True,
                )

            dt = _now() - start
            # Try to estimate epochs completed from results.csv if present.
            epochs_completed = 0
            if results_csv.is_file():
                epochs_completed = sum(1 for _ in results_csv.read_text(encoding="utf-8").splitlines()[1:] if _.strip())
            avg_epoch = (dt / epochs_completed) if epochs_completed > 0 else 0.0
            return best if best.is_file() else None, batch, dt, avg_epoch, "ok"
        except FileNotFoundError as e:
            # Occasionally hit transient filesystem issues; retry a few times.
            last_err = str(e)
            time.sleep(2.0)
            continue
        except RuntimeError as e:
            last_err = str(e)
            if "CUDA out of memory" in last_err or "out of memory" in last_err:
                batch = max(1, batch // 2)
                continue
            return None, batch, _now() - start, 0.0, f"failed: {last_err[:120]}"
    return None, batch, _now() - start, 0.0, f"failed: OOM (last_err={last_err[:120]})"


def _val_metrics(best_pt: Path, data_yaml: str, device: str) -> Dict[str, float]:
    model = YOLO(str(best_pt))
    r = model.val(data=data_yaml, split="test", imgsz=640, device=device, verbose=False)
    # Ultralytics metrics are a bit version-dependent; use safest attributes.
    out: Dict[str, float] = {}
    try:
        out["box_map50"] = float(r.box.map50)
        out["box_map50_95"] = float(r.box.map)
        out["mask_map50"] = float(r.seg.map50)
        out["mask_map50_95"] = float(r.seg.map)
        out["precision"] = float(r.seg.mp)  # mean precision (mask)
        out["recall"] = float(r.seg.mr)  # mean recall (mask)
    except Exception:
        # fallback: attempt dict conversion
        dd = getattr(r, "results_dict", None)
        if callable(dd):
            d = dd()
        else:
            d = {}
        # common keys (best-effort)
        for k in ("metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"):
            if k in d:
                pass
    return out


def _inference_speed(best_pt: Path, test_dir: Path, device: str, n: int = 250) -> Tuple[float, float]:
    model = YOLO(str(best_pt))
    imgs = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    imgs = imgs[:n]
    if not imgs:
        return 0.0, 0.0
    # Warmup
    _ = model.predict(source=str(imgs[0]), imgsz=640, device=device, verbose=False)
    t0 = _now()
    for p in imgs:
        _ = model.predict(source=str(p), imgsz=640, device=device, verbose=False)
    dt = _now() - t0
    ms = (dt / len(imgs)) * 1000.0
    fps = (len(imgs) / dt) if dt > 0 else 0.0
    return ms, fps


def _save_previews(best_pt: Path, test_dir: Path, out_dir: Path, device: str, k: int = 50) -> None:
    _safe_mkdir(out_dir)
    model = YOLO(str(best_pt))
    imgs = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:k]
    for p in imgs:
        res = model.predict(source=str(p), imgsz=640, device=device, verbose=False)[0]
        # `plot()` returns a BGR numpy array
        arr = res.plot()
        im = Image.fromarray(arr[..., ::-1])  # BGR->RGB
        im.save(out_dir / p.name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and benchmark YOLOv8 segmentation models on Orin.")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_segmentation_dataset" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default=str(REPO_ROOT / "runs" / "segment_benchmark"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in torch; refusing to run benchmark.")

    project = str(args.project)
    _safe_mkdir(Path(project))
    reports_dir = REPO_ROOT / "reports" / "segment_benchmark"
    _safe_mkdir(reports_dir)

    preview_root = REPO_ROOT / "data" / "segmentation_preview"
    _safe_mkdir(preview_root)

    models = [
        ("yolov8n_seg_benchmark", "yolov8n-seg.pt"),
        ("yolov8s_seg_benchmark", "yolov8s-seg.pt"),
        ("yolov8m_seg_benchmark", "yolov8m-seg.pt"),
    ]

    results: List[ModelResult] = []
    test_images = REPO_ROOT / "data" / "yolo_segmentation_dataset" / "images" / "test"

    for run_name, model_pt in models:
        best, batch, train_dt, avg_epoch, status = _train_one(
            model_pt,
            data_yaml=str(args.data),
            project=project,
            name=run_name,
            epochs=int(args.epochs),
            patience=int(args.patience),
            imgsz=int(args.imgsz),
            device=str(args.device),
            workers=int(args.workers),
            seed=int(args.seed),
        )

        if status != "ok" or best is None:
            results.append(
                ModelResult(
                    model_name=model_pt,
                    batch_size=batch,
                    epochs_completed=0,
                    train_time_total_s=train_dt,
                    avg_epoch_time_s=avg_epoch,
                    box_map50=0.0,
                    box_map50_95=0.0,
                    mask_map50=0.0,
                    mask_map50_95=0.0,
                    precision=0.0,
                    recall=0.0,
                    inference_time_ms_per_image=0.0,
                    fps=0.0,
                    best_weights_path="",
                    weights_size_mb=0.0,
                    status=status,
                )
            )
            continue

        metrics = _val_metrics(best, str(args.data), str(args.device))
        ms, fps = _inference_speed(best, test_images, str(args.device), n=250)
        _save_previews(best, test_images, preview_root / model_pt.replace(".pt", ""), str(args.device), k=50)

        epochs_completed = 0
        res_csv = Path(project) / run_name / "results.csv"
        if res_csv.is_file():
            epochs_completed = sum(1 for line in res_csv.read_text(encoding="utf-8").splitlines()[1:] if line.strip())

        results.append(
            ModelResult(
                model_name=model_pt.replace(".pt", ""),
                batch_size=batch,
                epochs_completed=epochs_completed,
                train_time_total_s=train_dt,
                avg_epoch_time_s=avg_epoch,
                box_map50=float(metrics.get("box_map50", 0.0)),
                box_map50_95=float(metrics.get("box_map50_95", 0.0)),
                mask_map50=float(metrics.get("mask_map50", 0.0)),
                mask_map50_95=float(metrics.get("mask_map50_95", 0.0)),
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                inference_time_ms_per_image=ms,
                fps=fps,
                best_weights_path=str(best),
                weights_size_mb=_weights_size_mb(best),
                status="ok",
            )
        )

    # Write CSV
    csv_path = reports_dir / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_name",
                "batch_size",
                "epochs_completed",
                "train_time_total_s",
                "avg_epoch_time_s",
                "box_map50",
                "box_map50_95",
                "mask_map50",
                "mask_map50_95",
                "precision",
                "recall",
                "inference_time_ms_per_image",
                "fps",
                "best_weights_path",
                "weights_size_mb",
                "status",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.model_name,
                    r.batch_size,
                    r.epochs_completed,
                    f"{r.train_time_total_s:.3f}",
                    f"{r.avg_epoch_time_s:.3f}",
                    f"{r.box_map50:.6f}",
                    f"{r.box_map50_95:.6f}",
                    f"{r.mask_map50:.6f}",
                    f"{r.mask_map50_95:.6f}",
                    f"{r.precision:.6f}",
                    f"{r.recall:.6f}",
                    f"{r.inference_time_ms_per_image:.3f}",
                    f"{r.fps:.3f}",
                    r.best_weights_path,
                    f"{r.weights_size_mb:.3f}",
                    r.status,
                ]
            )

    # Write MD summary (simple)
    md_path = reports_dir / "benchmark_summary.md"
    lines = ["## Segmentation benchmark summary", ""]
    lines.append(f"- dataset: `{args.data}`")
    lines.append(f"- project runs: `{project}`")
    lines.append("")
    lines.append("| model | status | mask mAP50-95 | mask recall | fps | best.pt |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.status} | {r.mask_map50_95:.4f} | {r.recall:.4f} | {r.fps:.2f} | `{r.best_weights_path}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reports: {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

