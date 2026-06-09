#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    pass


@dataclass(frozen=True)
class StageTimesMs:
    detector_ms: float
    roi_prep_ms: float
    classifier_ms: float
    segmentation_ms: float
    postprocess_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class ImageCounts:
    detections: int
    classified_crops: int
    segmentation_masks: int


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _weights_size_mb(p: Path) -> float:
    try:
        return float(p.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _iter_images_sorted(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: str(p).lower())
    return imgs


def _clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(0, min(w - 1, x2))
    y2c = max(0, min(h - 1, y2))
    if x2c <= x1c:
        x2c = min(w - 1, x1c + 1)
    if y2c <= y1c:
        y2c = min(h - 1, y1c + 1)
    return x1c, y1c, x2c, y2c


def _pad_xyxy(x1: int, y1: int, x2: int, y2: int, pad_frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    return _clamp_xyxy(x1 - px, y1 - py, x2 + px, y2 + py, w=w, h=h)


def _device_for_ultralytics(device: str) -> str | int:
    d = str(device).strip().lower()
    if d in ("cuda", "cuda:0", "0", "gpu", "gpu:0"):
        return 0
    if d.startswith("cuda:") and d[5:].isdigit():
        return int(d[5:])
    if d.isdigit():
        return int(d)
    return device


def _load_model_group(group_name: str) -> Dict[str, object]:
    name = str(group_name).strip()
    group_dir = REPO_ROOT / "models" / "model_groups" / name
    group_json = group_dir / "group.json"
    if not group_json.exists():
        available = sorted(
            p.name for p in (REPO_ROOT / "models" / "model_groups").iterdir() if p.is_dir()
        )
        raise SystemExit(f"Unknown model group: {name}. Available: {', '.join(available)}")
    data = json.loads(group_json.read_text(encoding="utf-8"))
    weights = data.get("weights") or {}
    for role in ("detector", "classifier", "segmenter"):
        if role not in weights:
            raise SystemExit(f"Model group {name} missing weights.{role} in {group_json}")
    return data


def _apply_model_group_to_args(args: argparse.Namespace) -> None:
    group_name = str(getattr(args, "model_group", "") or "").strip()
    if not group_name:
        return
    data = _load_model_group(group_name)
    weights = data["weights"]
    def _resolve_weight(p: str) -> str:
        wp = Path(str(p))
        if wp.is_absolute():
            return str(wp)
        return str((REPO_ROOT / wp).resolve())

    args.detector = _resolve_weight(str(weights["detector"]))
    args.classifier = _resolve_weight(str(weights["classifier"]))
    args.segmenter = _resolve_weight(str(weights["segmenter"]))
    preset_name = str(getattr(args, "preset", "custom")).strip().lower()
    preset = data.get("runtime_preset_recommended") or data.get("runtime_preset")
    # Apply group runtime defaults only when caller did not request a custom preset.
    if preset and preset_name not in ("custom", "all", ""):
        args.imgsz_det = int(preset.get("det_imgsz", args.imgsz_det))
        args.imgsz_seg = int(preset.get("seg_imgsz", args.imgsz_seg))
        args.conf_det = float(preset.get("det_conf", args.conf_det))
        args.max_det = int(preset.get("max_det", args.max_det))
        args.seg_every = int(preset.get("seg_every", args.seg_every))


def _apply_preset_to_args(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "preset", "") or "").strip().lower()
    if not preset or preset == "custom":
        return

    presets: Dict[str, Dict[str, object]] = {
        # Baseline matches current pipeline defaults.
        "baseline": {
            "imgsz_det": 640,
            "imgsz_seg": 384,
            "conf_det": 0.35,
            "max_det": int(args.max_det),
            "seg_every": 2,
        },
        "fast": {
            "imgsz_det": 512,
            "imgsz_seg": 320,
            "conf_det": 0.40,
            "max_det": 10,
            "seg_every": 2,
        },
        "very_fast": {
            "imgsz_det": 480,
            "imgsz_seg": 320,
            "conf_det": 0.45,
            "max_det": 5,
            "seg_every": 3,
        },
        "ultra_low": {
            "imgsz_det": 416,
            "imgsz_seg": 256,
            "conf_det": 0.45,
            "max_det": 5,
            "seg_every": 2,
        },
    }
    if preset not in presets:
        raise SystemExit(f"Unknown preset: {preset}. Expected one of: {', '.join(sorted(presets.keys()))} or 'all'.")
    for k, v in presets[preset].items():
        setattr(args, k, v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark full vision pipeline FPS on holdout/unseen frames.")
    ap.add_argument(
        "--holdout",
        default=str(REPO_ROOT / "data" / "new_photos"),
        help="Path to holdout/unseen images root directory (recursive).",
    )
    ap.add_argument(
        "--detector",
        default=str(REPO_ROOT / "runs" / "detect_benchmark_v3" / "yolov8s_v3_lowdensity" / "weights" / "best.pt"),
    )
    ap.add_argument(
        "--classifier",
        default=str(REPO_ROOT / "runs" / "classification_benchmark_v2" / "efficientnet_b0" / "best.pt"),
    )
    ap.add_argument(
        "--segmenter",
        default=str(REPO_ROOT / "runs" / "segment_benchmark" / "yolov8n_seg_benchmark" / "weights" / "best.pt"),
    )
    ap.add_argument(
        "--model-group",
        default="",
        help="Load weights from models/model_groups/<name>/group.json (e.g. 01_fast_initial, 02_lightened_current).",
    )
    ap.add_argument("--device", default="cuda:0", help="Use cuda:0 if available, else CPU.")
    ap.add_argument("--imgsz-det", type=int, default=640)
    ap.add_argument("--imgsz-seg", type=int, default=384)
    ap.add_argument("--conf-det", type=float, default=0.35)
    ap.add_argument("--iou-det", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--max-rois", type=int, default=8)
    ap.add_argument("--crop-pad-frac", type=float, default=0.15)
    ap.add_argument(
        "--cls-every",
        type=int,
        default=8,
        help="Run classification once per N frames (matches pipeline default = 8). Use 1 for every frame.",
    )
    ap.add_argument(
        "--seg-every",
        type=int,
        default=2,
        help="Run segmentation once per N frames (matches run_strawberry_ensemble.py default).",
    )
    ap.add_argument("--seg-max-rois", type=int, default=8)
    ap.add_argument("--seg-min-det-conf", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20, help="Warmup frames before measurement.")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, benchmark only first N images in stable sorted order.",
    )
    ap.add_argument("--preview", action="store_true", help="Optionally save preview for first 20 frames (not counted).")
    ap.add_argument("--preview-n", type=int, default=20)
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "runs" / "holdout_full_pipeline_fps"),
        help="Output directory for summary + per-image timing logs.",
    )
    ap.add_argument(
        "--preset",
        default="custom",
        help="Benchmark preset: baseline|fast|very_fast|all (custom = use provided flags).",
    )
    args = ap.parse_args()
    # Preserve explicit CLI weight overrides when --model-group is also set (e.g. TRT engines).
    det_cli = Path(args.detector)
    cls_cli = Path(args.classifier)
    seg_cli = Path(args.segmenter)
    det_default = Path(
        str(REPO_ROOT / "runs" / "detect_benchmark_v3" / "yolov8s_v3_lowdensity" / "weights" / "best.pt")
    )
    cls_default = Path(str(REPO_ROOT / "runs" / "classification_benchmark_v2" / "efficientnet_b0" / "best.pt"))
    seg_default = Path(
        str(REPO_ROOT / "runs" / "segment_benchmark" / "yolov8n_seg_benchmark" / "weights" / "best.pt")
    )
    det_override = det_cli != det_default
    cls_override = cls_cli != cls_default
    seg_override = seg_cli != seg_default
    _apply_model_group_to_args(args)
    if det_override:
        args.detector = str(det_cli)
    if cls_override:
        args.classifier = str(cls_cli)
    if seg_override:
        args.segmenter = str(seg_cli)

    holdout_root = Path(args.holdout)
    if not holdout_root.exists():
        raise SystemExit(f"Holdout dir not found: {str(holdout_root)}")

    det_w = Path(args.detector)
    cls_w = Path(args.classifier)
    seg_w = Path(args.segmenter)
    for p in (det_w, cls_w, seg_w):
        if not p.exists():
            raise SystemExit(f"Weights not found: {str(p)}")

    outdir = Path(args.outdir)
    _safe_mkdir(outdir)
    preview_dir = outdir / "preview"
    if bool(args.preview):
        _safe_mkdir(preview_dir)

    # Import pipeline components to match current implementation.
    from pipelines.strawberry_ensemble import RipenessClassifier  # noqa: WPS433
    from ultralytics import YOLO  # noqa: WPS433

    cuda_ok = bool(torch.cuda.is_available())
    device_req = str(args.device)
    if "cuda" in device_req and not cuda_ok:
        device_req = "cpu"
    ul_dev = _device_for_ultralytics(device_req)

    # Load models (.engine needs explicit task for segmenter).
    detector = YOLO(str(det_w), task="detect")
    segmenter = YOLO(str(seg_w), task="segment")
    classifier = RipenessClassifier(str(cls_w), device=str(device_req if device_req != "cpu" else "cpu"))

    # Gather holdout images in stable order.
    images = _iter_images_sorted(holdout_root)
    if not images:
        raise SystemExit(f"No images found under holdout dir: {str(holdout_root)}")
    if int(args.limit) > 0:
        images = images[: int(args.limit)]

    def bench_one_preset(
        preset_name: str,
        *,
        per_image_csv_path: Path,
        summary_dir: Path,
    ) -> Dict[str, object]:
        # Clone args (shallow) then apply preset overrides.
        a = argparse.Namespace(**vars(args))
        a.preset = preset_name
        if preset_name != "custom":
            _apply_preset_to_args(a)

        print(
            "benchmark_config "
            f"preset={preset_name} "
            f"holdout={str(holdout_root)} "
            f"device_req={str(args.device)} cuda_ok={int(cuda_ok)} effective_device={str(device_req)} ul_device={str(ul_dev)} "
            f"imgsz_det={int(a.imgsz_det)} conf_det={float(a.conf_det)} iou_det={float(a.iou_det)} "
            f"imgsz_seg={int(a.imgsz_seg)} cls_every={int(a.cls_every)} seg_every={int(a.seg_every)} "
            f"max_det={int(a.max_det)} max_rois={int(a.max_rois)} seg_max_rois={int(a.seg_max_rois)} seg_min_det_conf={float(a.seg_min_det_conf)}",
            flush=True,
        )

        warmup_n = max(0, int(a.warmup))
        preview_n = max(0, int(a.preview_n))

        def run_one(
            image_path: Path,
            frame_i: int,
            *,
            do_preview: bool,
        ) -> Tuple[StageTimesMs, ImageCounts]:
            frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read image: {str(image_path)}")
            h, w = frame_bgr.shape[:2]

            t0 = _now_ms()

            # 1) Detector inference.
            t_det0 = _now_ms()
            det_res = detector.predict(
                source=frame_bgr,
                imgsz=int(a.imgsz_det),
                conf=float(a.conf_det),
                iou=float(a.iou_det),
                device=ul_dev,
                max_det=int(a.max_det),
                verbose=False,
            )[0]
            t_det1 = _now_ms()

            xyxy = None
            confs = None
            if det_res.boxes is not None and det_res.boxes.xyxy is not None and det_res.boxes.conf is not None:
                xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
                confs = det_res.boxes.conf.detach().cpu().numpy()
            dets: List[Tuple[Tuple[int, int, int, int], float]] = []
            if xyxy is not None and confs is not None:
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    dets.append(((int(x1), int(y1), int(x2), int(y2)), float(c)))
            dets.sort(key=lambda d: float(d[1]), reverse=True)
            dets = dets[: max(0, int(a.max_rois))]

            # 2) ROI preparation (pad + crop).
            t_roi0 = _now_ms()
            crops: List[Optional[np.ndarray]] = []
            pads: List[Tuple[int, int, int, int]] = []
            for (x1, y1, x2, y2), _c in dets:
                x1p, y1p, x2p, y2p = _pad_xyxy(x1, y1, x2, y2, float(a.crop_pad_frac), w=w, h=h)
                crop = frame_bgr[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    crops.append(None)
                    pads.append((x1p, y1p, x2p, y2p))
                else:
                    crops.append(crop)
                    pads.append((x1p, y1p, x2p, y2p))
            t_roi1 = _now_ms()

            # 3) Classifier inference.
            cls_every = max(1, int(a.cls_every))
            do_cls_this_frame = (frame_i % cls_every) == 0
            t_cls0 = _now_ms()
            classified_crops = 0
            if do_cls_this_frame:
                valid = [c for c in crops if c is not None]
                if valid:
                    _ = classifier.infer_crops_bgr(valid)
                    classified_crops = int(len(valid))
            t_cls1 = _now_ms()

            # 4) Segmentation inference.
            seg_every = max(1, int(a.seg_every))
            do_seg_this_frame = (frame_i % seg_every) == 0
            seg_masks: List[Optional[np.ndarray]] = []
            seg_ok = 0
            t_seg0 = _now_ms()
            if do_seg_this_frame:
                for i, crop in enumerate(crops):
                    if crop is None:
                        seg_masks.append(None)
                        continue
                    if i >= int(a.seg_max_rois):
                        seg_masks.append(None)
                        continue
                    det_conf = float(dets[i][1]) if i < len(dets) else 0.0
                    if det_conf < float(a.seg_min_det_conf):
                        seg_masks.append(None)
                        continue
                    seg_res = segmenter.predict(
                        source=crop,
                        imgsz=int(a.imgsz_seg),
                        conf=0.25,
                        iou=0.7,
                        device=ul_dev,
                        verbose=False,
                    )[0]
                    m = None
                    if (
                        seg_res.masks is not None
                        and seg_res.masks.data is not None
                        and seg_res.boxes is not None
                        and seg_res.boxes.conf is not None
                    ):
                        masks = seg_res.masks.data.detach().cpu().numpy()
                        sconfs = seg_res.boxes.conf.detach().cpu().numpy()
                        if masks.size and sconfs.size:
                            best_i = int(np.argmax(sconfs))
                            m = (masks[best_i] > 0.5).astype(np.uint8) * 255
                    seg_masks.append(m)
                    if m is not None:
                        seg_ok += 1
            t_seg1 = _now_ms()

            # 5) Postprocess (visualization-free).
            t_post0 = _now_ms()
            if do_seg_this_frame and seg_masks:
                for i, m in enumerate(seg_masks):
                    if m is None:
                        continue
                    crop = crops[i]
                    if crop is None:
                        continue
                    x1p, y1p, x2p, y2p = pads[i]
                    mh, mw = m.shape[:2]
                    if (mw, mh) != (crop.shape[1], crop.shape[0]):
                        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_full = np.zeros((h, w), dtype=np.uint8)
                    mask_full[y1p:y2p, x1p:x2p] = m
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    _ = contours
            t_post1 = _now_ms()

            t1 = _now_ms()

            if do_preview:
                # Preview is intentionally excluded from timing.
                vis = frame_bgr.copy()
                for i, ((x1, y1, x2, y2), c) in enumerate(dets):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{c:.2f}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if do_seg_this_frame and i < len(seg_masks) and seg_masks[i] is not None:
                        x1p, y1p, x2p, y2p = pads[i]
                        mm = seg_masks[i]
                        if mm is not None:
                            if mm.shape[:2] != (y2p - y1p, x2p - x1p):
                                mm = cv2.resize(mm, (x2p - x1p, y2p - y1p), interpolation=cv2.INTER_NEAREST)
                            color = np.zeros((y2p - y1p, x2p - x1p, 3), dtype=np.uint8)
                            color[:, :, 1] = 200
                            msel = (mm > 0)[:, :, None]
                            roi = vis[y1p:y2p, x1p:x2p]
                            vis[y1p:y2p, x1p:x2p] = np.where(
                                msel, (roi * 0.65 + color * 0.35).astype(np.uint8), roi
                            )
                out_path = preview_dir / f"{preset_name}_{frame_i:06d}_{image_path.name}"
                cv2.imwrite(str(out_path), vis)

            times = StageTimesMs(
                detector_ms=float(t_det1 - t_det0),
                roi_prep_ms=float(t_roi1 - t_roi0),
                classifier_ms=float(t_cls1 - t_cls0),
                segmentation_ms=float(t_seg1 - t_seg0),
                postprocess_ms=float(t_post1 - t_post0),
                end_to_end_ms=float(t1 - t0),
            )
            counts = ImageCounts(
                detections=int(len(dets)),
                classified_crops=int(classified_crops),
                segmentation_masks=int(seg_ok),
            )
            return times, counts

        # Warmup
        for i, p in enumerate(images[:warmup_n]):
            _ = run_one(p, i + 1, do_preview=False)

        # Measure
        rows: List[Dict[str, object]] = []
        sum_det = sum_roi = sum_cls = sum_seg = sum_post = sum_e2e = 0.0
        sum_dets = sum_seg_masks = 0
        t_all0 = time.perf_counter()
        for j, p in enumerate(images):
            frame_i = warmup_n + j + 1
            do_preview = bool(a.preview) and (j < preview_n)
            times, counts = run_one(p, frame_i, do_preview=do_preview)

            sum_det += times.detector_ms
            sum_roi += times.roi_prep_ms
            sum_cls += times.classifier_ms
            sum_seg += times.segmentation_ms
            sum_post += times.postprocess_ms
            sum_e2e += times.end_to_end_ms
            sum_dets += counts.detections
            sum_seg_masks += counts.segmentation_masks

            rows.append(
                {
                    "index": j,
                    "path": str(p),
                    "detector_ms": times.detector_ms,
                    "roi_prep_ms": times.roi_prep_ms,
                    "classifier_ms": times.classifier_ms,
                    "segmentation_ms": times.segmentation_ms,
                    "postprocess_ms": times.postprocess_ms,
                    "end_to_end_ms": times.end_to_end_ms,
                    "detections": counts.detections,
                    "segmentation_masks": counts.segmentation_masks,
                }
            )
        t_all1 = time.perf_counter()
        wall_ms = float((t_all1 - t_all0) * 1000.0)

        n = max(1, len(images))
        total_ms_per_img = sum_e2e / n
        fps = 1000.0 / max(1e-9, total_ms_per_img)

        with per_image_csv_path.open("w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
            wcsv.writeheader()
            for r in rows:
                wcsv.writerow(r)

        summary = {
            "preset": preset_name,
            "images_count": int(len(images)),
            "fps": float(fps),
            "ms_per_img": float(total_ms_per_img),
            "detector_ms_per_img": float(sum_det / n),
            "classifier_ms_per_img": float(sum_cls / n),
            "segmentation_ms_per_img": float(sum_seg / n),
            "avg_detections_per_img": float(sum_dets / n),
            "avg_masks_per_img": float(sum_seg_masks / n),
            "det_imgsz": int(a.imgsz_det),
            "seg_imgsz": int(a.imgsz_seg),
            "det_conf": float(a.conf_det),
            "max_det": int(a.max_det),
            "seg_every": int(a.seg_every),
            "timing_notes": {
                "warmup_frames": int(warmup_n),
                "measurement_wall_ms": float(wall_ms),
            },
        }
        # Also keep a full record for audit/debug.
        full_summary = {
            **summary,
            "device": str(device_req),
            "ultralytics_device": str(ul_dev),
            "model_paths": {"detector": str(det_w), "classifier": str(cls_w), "segmenter": str(seg_w)},
            "weights_sizes_mb": {
                "detector": _weights_size_mb(det_w),
                "classifier": _weights_size_mb(cls_w),
                "segmenter": _weights_size_mb(seg_w),
            },
        }

        # Save per-preset json for convenience.
        with (summary_dir / f"summary_{preset_name}.json").open("w") as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)

        print(
            "preset_summary "
            f"preset={preset_name} images={int(len(images))} fps={float(fps):.2f} ms/img={float(total_ms_per_img):.2f} "
            f"det={float(sum_det/n):.2f} cls={float(sum_cls/n):.2f} seg={float(sum_seg/n):.2f} "
            f"avg_det/img={float(sum_dets/n):.2f} avg_masks/img={float(sum_seg_masks/n):.2f}",
            flush=True,
        )
        return full_summary

    # Preset handling.
    preset = str(args.preset).strip().lower()
    if preset != "all":
        if preset != "custom":
            _apply_preset_to_args(args)
        # Backwards-compatible single-run behavior: keep previous output filenames.
        # (This path still writes per_image.csv, summary.csv, summary.json in outdir.)
        warmup_n = max(0, int(args.warmup))
        preview_n = max(0, int(args.preview_n))

        def run_one(
            image_path: Path,
            frame_i: int,
            *,
            do_preview: bool,
        ) -> Tuple[StageTimesMs, ImageCounts]:
            frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read image: {str(image_path)}")
            h, w = frame_bgr.shape[:2]

            t0 = _now_ms()

            # 1) Detector inference.
            t_det0 = _now_ms()
            det_res = detector.predict(
                source=frame_bgr,
                imgsz=int(args.imgsz_det),
                conf=float(args.conf_det),
                iou=float(args.iou_det),
                device=ul_dev,
                max_det=int(args.max_det),
                verbose=False,
            )[0]
            t_det1 = _now_ms()

            xyxy = None
            confs = None
            if det_res.boxes is not None and det_res.boxes.xyxy is not None and det_res.boxes.conf is not None:
                xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
                confs = det_res.boxes.conf.detach().cpu().numpy()
            dets: List[Tuple[Tuple[int, int, int, int], float]] = []
            if xyxy is not None and confs is not None:
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    dets.append(((int(x1), int(y1), int(x2), int(y2)), float(c)))
            dets.sort(key=lambda d: float(d[1]), reverse=True)
            dets = dets[: max(0, int(args.max_rois))]

            # 2) ROI preparation (pad + crop).
            t_roi0 = _now_ms()
            crops: List[Optional[np.ndarray]] = []
            pads: List[Tuple[int, int, int, int]] = []
            for (x1, y1, x2, y2), _c in dets:
                x1p, y1p, x2p, y2p = _pad_xyxy(x1, y1, x2, y2, float(args.crop_pad_frac), w=w, h=h)
                crop = frame_bgr[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    crops.append(None)
                    pads.append((x1p, y1p, x2p, y2p))
                else:
                    crops.append(crop)
                    pads.append((x1p, y1p, x2p, y2p))
            t_roi1 = _now_ms()

            # 3) Classifier inference.
            cls_every = max(1, int(args.cls_every))
            do_cls_this_frame = (frame_i % cls_every) == 0
            t_cls0 = _now_ms()
            classified_crops = 0
            if do_cls_this_frame:
                valid = [c for c in crops if c is not None]
                if valid:
                    _ = classifier.infer_crops_bgr(valid)
                    classified_crops = int(len(valid))
            t_cls1 = _now_ms()

            # 4) Segmentation inference.
            seg_every = max(1, int(args.seg_every))
            do_seg_this_frame = (frame_i % seg_every) == 0
            seg_masks: List[Optional[np.ndarray]] = []
            seg_ok = 0
            t_seg0 = _now_ms()
            if do_seg_this_frame:
                for i, crop in enumerate(crops):
                    if crop is None:
                        seg_masks.append(None)
                        continue
                    if i >= int(args.seg_max_rois):
                        seg_masks.append(None)
                        continue
                    det_conf = float(dets[i][1]) if i < len(dets) else 0.0
                    if det_conf < float(args.seg_min_det_conf):
                        seg_masks.append(None)
                        continue
                    seg_res = segmenter.predict(
                        source=crop,
                        imgsz=int(args.imgsz_seg),
                        conf=0.25,
                        iou=0.7,
                        device=ul_dev,
                        verbose=False,
                    )[0]
                    m = None
                    if (
                        seg_res.masks is not None
                        and seg_res.masks.data is not None
                        and seg_res.boxes is not None
                        and seg_res.boxes.conf is not None
                    ):
                        masks = seg_res.masks.data.detach().cpu().numpy()
                        sconfs = seg_res.boxes.conf.detach().cpu().numpy()
                        if masks.size and sconfs.size:
                            best_i = int(np.argmax(sconfs))
                            m = (masks[best_i] > 0.5).astype(np.uint8) * 255
                    seg_masks.append(m)
                    if m is not None:
                        seg_ok += 1
            t_seg1 = _now_ms()

            # 5) Postprocess (visualization-free).
            t_post0 = _now_ms()
            if do_seg_this_frame and seg_masks:
                for i, m in enumerate(seg_masks):
                    if m is None:
                        continue
                    crop = crops[i]
                    if crop is None:
                        continue
                    x1p, y1p, x2p, y2p = pads[i]
                    mh, mw = m.shape[:2]
                    if (mw, mh) != (crop.shape[1], crop.shape[0]):
                        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_full = np.zeros((h, w), dtype=np.uint8)
                    mask_full[y1p:y2p, x1p:x2p] = m
                    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    _ = contours
            t_post1 = _now_ms()

            t1 = _now_ms()

            if do_preview:
                vis = frame_bgr.copy()
                for i, ((x1, y1, x2, y2), c) in enumerate(dets):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{c:.2f}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if do_seg_this_frame and i < len(seg_masks) and seg_masks[i] is not None:
                        x1p, y1p, x2p, y2p = pads[i]
                        mm = seg_masks[i]
                        if mm is not None:
                            if mm.shape[:2] != (y2p - y1p, x2p - x1p):
                                mm = cv2.resize(mm, (x2p - x1p, y2p - y1p), interpolation=cv2.INTER_NEAREST)
                            color = np.zeros((y2p - y1p, x2p - x1p, 3), dtype=np.uint8)
                            color[:, :, 1] = 200
                            msel = (mm > 0)[:, :, None]
                            roi = vis[y1p:y2p, x1p:x2p]
                            vis[y1p:y2p, x1p:x2p] = np.where(
                                msel, (roi * 0.65 + color * 0.35).astype(np.uint8), roi
                            )
                out_path = preview_dir / f"{frame_i:06d}_{image_path.name}"
                cv2.imwrite(str(out_path), vis)

            times = StageTimesMs(
                detector_ms=float(t_det1 - t_det0),
                roi_prep_ms=float(t_roi1 - t_roi0),
                classifier_ms=float(t_cls1 - t_cls0),
                segmentation_ms=float(t_seg1 - t_seg0),
                postprocess_ms=float(t_post1 - t_post0),
                end_to_end_ms=float(t1 - t0),
            )
            counts = ImageCounts(
                detections=int(len(dets)),
                classified_crops=int(classified_crops),
                segmentation_masks=int(seg_ok),
            )
            return times, counts

        # Warmup
        for i, p in enumerate(images[:warmup_n]):
            _ = run_one(p, i + 1, do_preview=False)

        # Measure
        rows: List[Dict[str, object]] = []
        sum_det = sum_roi = sum_cls = sum_seg = sum_post = sum_e2e = 0.0
        sum_dets = sum_cls_crops = sum_seg_masks = 0
        t_all0 = time.perf_counter()
        for j, p in enumerate(images):
            frame_i = warmup_n + j + 1
            do_preview = bool(args.preview) and (j < preview_n)
            times, counts = run_one(p, frame_i, do_preview=do_preview)

            sum_det += times.detector_ms
            sum_roi += times.roi_prep_ms
            sum_cls += times.classifier_ms
            sum_seg += times.segmentation_ms
            sum_post += times.postprocess_ms
            sum_e2e += times.end_to_end_ms
            sum_dets += counts.detections
            sum_cls_crops += counts.classified_crops
            sum_seg_masks += counts.segmentation_masks

            rows.append(
                {
                    "index": j,
                    "path": str(p),
                    "detector_ms": times.detector_ms,
                    "roi_prep_ms": times.roi_prep_ms,
                    "classifier_ms": times.classifier_ms,
                    "segmentation_ms": times.segmentation_ms,
                    "postprocess_ms": times.postprocess_ms,
                    "end_to_end_ms": times.end_to_end_ms,
                    "detections": counts.detections,
                    "classified_crops": counts.classified_crops,
                    "segmentation_masks": counts.segmentation_masks,
                }
            )
        t_all1 = time.perf_counter()
        wall_ms = float((t_all1 - t_all0) * 1000.0)

        n = max(1, len(images))
        total_ms_per_img = sum_e2e / n
        fps = 1000.0 / max(1e-9, total_ms_per_img)

        summary = {
            "images_count": int(len(images)),
            "total_end_to_end_fps": float(fps),
            "total_ms_per_img": float(total_ms_per_img),
            "detector_ms_per_img": float(sum_det / n),
            "roi_prep_ms_per_img": float(sum_roi / n),
            "classifier_ms_per_img": float(sum_cls / n),
            "segmentation_ms_per_img": float(sum_seg / n),
            "postprocess_ms_per_img": float(sum_post / n),
            "avg_detections_per_img": float(sum_dets / n),
            "avg_classified_crops_per_img": float(sum_cls_crops / n),
            "avg_segmentation_masks_per_img": float(sum_seg_masks / n),
            "device": str(device_req),
            "ultralytics_device": str(ul_dev),
            "model_paths": {
                "detector": str(det_w),
                "classifier": str(cls_w),
                "segmenter": str(seg_w),
            },
            "weights_sizes_mb": {
                "detector": _weights_size_mb(det_w),
                "classifier": _weights_size_mb(cls_w),
                "segmenter": _weights_size_mb(seg_w),
            },
            "pipeline_params": {
                "imgsz_det": int(args.imgsz_det),
                "conf_det": float(args.conf_det),
                "iou_det": float(args.iou_det),
                "max_det": int(args.max_det),
                "max_rois": int(args.max_rois),
                "crop_pad_frac": float(args.crop_pad_frac),
                "cls_every": int(max(1, int(args.cls_every))),
                "seg_every": int(args.seg_every),
                "seg_max_rois": int(args.seg_max_rois),
                "seg_min_det_conf": float(args.seg_min_det_conf),
                "imgsz_seg": int(args.imgsz_seg),
                "seg_conf": 0.25,
                "seg_iou": 0.7,
            },
            "timing_notes": {
                "warmup_frames": int(warmup_n),
                "measurement_wall_ms": float(wall_ms),
                "note": "Per-image end_to_end_ms is measured around full detector->ROI->classifier->segmenter->postprocess (no saves). Preview is excluded.",
            },
        }

        # Write outputs
        per_image_csv = outdir / "per_image.csv"
        summary_json = outdir / "summary.json"
        summary_csv = outdir / "summary.csv"

        with per_image_csv.open("w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
            wcsv.writeheader()
            for r in rows:
                wcsv.writerow(r)

        with summary_json.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with summary_csv.open("w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["key", "value"])
            for k in [
                "images_count",
                "total_end_to_end_fps",
                "total_ms_per_img",
                "detector_ms_per_img",
                "roi_prep_ms_per_img",
                "classifier_ms_per_img",
                "segmentation_ms_per_img",
                "postprocess_ms_per_img",
                "avg_detections_per_img",
                "avg_classified_crops_per_img",
                "avg_segmentation_masks_per_img",
                "device",
            ]:
                wcsv.writerow([k, summary[k]])

        print(f"Wrote {str(per_image_csv)}", flush=True)
        print(f"Wrote {str(summary_csv)}", flush=True)
        print(f"Wrote {str(summary_json)}", flush=True)
        print(
            "summary "
            f"images={int(len(images))} "
            f"fps={float(fps):.2f} ms/img={float(total_ms_per_img):.2f} "
            f"det={float(sum_det/n):.2f} roi={float(sum_roi/n):.2f} cls={float(sum_cls/n):.2f} "
            f"seg={float(sum_seg/n):.2f} post={float(sum_post/n):.2f}",
            flush=True,
        )
        return 0

    # All presets mode: write consolidated outputs in outdir.
    preset_outdir = outdir
    _safe_mkdir(preset_outdir)
    preset_names = ["baseline", "fast", "very_fast"]
    per_preset: List[Dict[str, object]] = []
    for name in preset_names:
        per_image_csv_path = preset_outdir / f"per_image_{name}.csv"
        full = bench_one_preset(name, per_image_csv_path=per_image_csv_path, summary_dir=preset_outdir)
        per_preset.append(full)

    summary_csv = preset_outdir / "summary.csv"
    summary_json = preset_outdir / "summary.json"

    # Write consolidated summary.
    with summary_json.open("w") as f:
        json.dump(per_preset, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", newline="") as f:
        cols = [
            "preset",
            "fps",
            "ms_per_img",
            "detector_ms_per_img",
            "classifier_ms_per_img",
            "segmentation_ms_per_img",
            "avg_detections_per_img",
            "avg_masks_per_img",
            "det_imgsz",
            "seg_imgsz",
            "det_conf",
            "max_det",
            "seg_every",
        ]
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        for s in per_preset:
            row = {k: s.get(k) for k in cols}
            wcsv.writerow(row)

    print(f"Wrote {str(summary_csv)}", flush=True)
    print(f"Wrote {str(summary_json)}", flush=True)
    return 0

    warmup_n = max(0, int(args.warmup))
    preview_n = max(0, int(args.preview_n))

    def run_one(
        image_path: Path,
        frame_i: int,
        *,
        do_preview: bool,
    ) -> Tuple[StageTimesMs, ImageCounts]:
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read image: {str(image_path)}")
        h, w = frame_bgr.shape[:2]

        t0 = _now_ms()

        # 1) Detector inference.
        t_det0 = _now_ms()
        det_res = detector.predict(
            source=frame_bgr,
            imgsz=int(args.imgsz_det),
            conf=float(args.conf_det),
            iou=float(args.iou_det),
            device=ul_dev,
            max_det=int(args.max_det),
            verbose=False,
        )[0]
        t_det1 = _now_ms()

        xyxy = None
        confs = None
        if det_res.boxes is not None and det_res.boxes.xyxy is not None and det_res.boxes.conf is not None:
            xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
            confs = det_res.boxes.conf.detach().cpu().numpy()
        dets: List[Tuple[Tuple[int, int, int, int], float]] = []
        if xyxy is not None and confs is not None:
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                dets.append(((int(x1), int(y1), int(x2), int(y2)), float(c)))
        dets.sort(key=lambda d: float(d[1]), reverse=True)
        dets = dets[: max(0, int(args.max_rois))]

        # 2) ROI preparation (pad + crop).
        t_roi0 = _now_ms()
        crops: List[Optional[np.ndarray]] = []
        pads: List[Tuple[int, int, int, int]] = []
        for (x1, y1, x2, y2), _c in dets:
            x1p, y1p, x2p, y2p = _pad_xyxy(x1, y1, x2, y2, float(args.crop_pad_frac), w=w, h=h)
            crop = frame_bgr[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                crops.append(None)
                pads.append((x1p, y1p, x2p, y2p))
            else:
                crops.append(crop)
                pads.append((x1p, y1p, x2p, y2p))
        t_roi1 = _now_ms()

        # 3) Classifier inference (matches pipeline default: every 8th frame).
        cls_every = max(1, int(args.cls_every))
        do_cls_this_frame = (frame_i % cls_every) == 0
        t_cls0 = _now_ms()
        classified_crops = 0
        cls_out: List[Tuple[str, float]] = []
        if do_cls_this_frame:
            valid = [c for c in crops if c is not None]
            if valid:
                cls_out = classifier.infer_crops_bgr(valid)
                classified_crops = int(len(valid))
        t_cls1 = _now_ms()

        # 4) Segmentation inference (matches pipeline gates).
        seg_every = max(1, int(args.seg_every))
        do_seg_this_frame = (frame_i % seg_every) == 0
        seg_masks: List[Optional[np.ndarray]] = []
        seg_ok = 0
        t_seg0 = _now_ms()
        if do_seg_this_frame:
            for i, crop in enumerate(crops):
                if crop is None:
                    seg_masks.append(None)
                    continue
                if i >= int(args.seg_max_rois):
                    seg_masks.append(None)
                    continue
                det_conf = float(dets[i][1]) if i < len(dets) else 0.0
                if det_conf < float(args.seg_min_det_conf):
                    seg_masks.append(None)
                    continue
                seg_res = segmenter.predict(
                    source=crop,
                    imgsz=int(args.imgsz_seg),
                    conf=0.25,
                    iou=0.7,
                    device=ul_dev,
                    verbose=False,
                )[0]
                m = None
                if (
                    seg_res.masks is not None
                    and seg_res.masks.data is not None
                    and seg_res.boxes is not None
                    and seg_res.boxes.conf is not None
                ):
                    masks = seg_res.masks.data.detach().cpu().numpy()  # (N,H,W)
                    sconfs = seg_res.boxes.conf.detach().cpu().numpy()  # (N,)
                    if masks.size and sconfs.size:
                        best_i = int(np.argmax(sconfs))
                        m = (masks[best_i] > 0.5).astype(np.uint8) * 255
                seg_masks.append(m)
                if m is not None:
                    seg_ok += 1
        t_seg1 = _now_ms()

        # 5) Postprocess (visualization-free): place masks back to fullres + find contours (like pipeline).
        t_post0 = _now_ms()
        full_masks = 0
        if do_seg_this_frame and seg_masks:
            for i, m in enumerate(seg_masks):
                if m is None:
                    continue
                crop = crops[i]
                if crop is None:
                    continue
                x1p, y1p, x2p, y2p = pads[i]
                mh, mw = m.shape[:2]
                if (mw, mh) != (crop.shape[1], crop.shape[0]):
                    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_full = np.zeros((h, w), dtype=np.uint8)
                mask_full[y1p:y2p, x1p:x2p] = m
                contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _ = contours  # keep same work as pipeline; no visualization.
                full_masks += 1
        t_post1 = _now_ms()

        t1 = _now_ms()

        if do_preview:
            # Preview is intentionally excluded from timing; minimal overlay + save.
            vis = frame_bgr.copy()
            for i, ((x1, y1, x2, y2), c) in enumerate(dets):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{c:.2f}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if do_seg_this_frame and i < len(seg_masks) and seg_masks[i] is not None:
                    x1p, y1p, x2p, y2p = pads[i]
                    mm = seg_masks[i]
                    if mm is not None:
                        if mm.shape[:2] != (y2p - y1p, x2p - x1p):
                            mm = cv2.resize(mm, (x2p - x1p, y2p - y1p), interpolation=cv2.INTER_NEAREST)
                        color = np.zeros((y2p - y1p, x2p - x1p, 3), dtype=np.uint8)
                        color[:, :, 1] = 200
                        msel = (mm > 0)[:, :, None]
                        roi = vis[y1p:y2p, x1p:x2p]
                        vis[y1p:y2p, x1p:x2p] = np.where(msel, (roi * 0.65 + color * 0.35).astype(np.uint8), roi)
            out_path = preview_dir / f"{frame_i:06d}_{image_path.name}"
            cv2.imwrite(str(out_path), vis)

        times = StageTimesMs(
            detector_ms=float(t_det1 - t_det0),
            roi_prep_ms=float(t_roi1 - t_roi0),
            classifier_ms=float(t_cls1 - t_cls0),
            segmentation_ms=float(t_seg1 - t_seg0),
            postprocess_ms=float(t_post1 - t_post0),
            end_to_end_ms=float(t1 - t0),
        )
        counts = ImageCounts(
            detections=int(len(dets)),
            classified_crops=int(classified_crops),
            segmentation_masks=int(seg_ok),
        )
        return times, counts

    # Warmup
    for i, p in enumerate(images[:warmup_n]):
        _ = run_one(p, i + 1, do_preview=False)

    # Measure
    rows: List[Dict[str, object]] = []
    sum_det = sum_roi = sum_cls = sum_seg = sum_post = sum_e2e = 0.0
    sum_dets = sum_cls_crops = sum_seg_masks = 0
    t_all0 = time.perf_counter()
    for j, p in enumerate(images):
        frame_i = warmup_n + j + 1
        do_preview = bool(args.preview) and (j < preview_n)
        times, counts = run_one(p, frame_i, do_preview=do_preview)

        sum_det += times.detector_ms
        sum_roi += times.roi_prep_ms
        sum_cls += times.classifier_ms
        sum_seg += times.segmentation_ms
        sum_post += times.postprocess_ms
        sum_e2e += times.end_to_end_ms
        sum_dets += counts.detections
        sum_cls_crops += counts.classified_crops
        sum_seg_masks += counts.segmentation_masks

        rows.append(
            {
                "index": j,
                "path": str(p),
                "detector_ms": times.detector_ms,
                "roi_prep_ms": times.roi_prep_ms,
                "classifier_ms": times.classifier_ms,
                "segmentation_ms": times.segmentation_ms,
                "postprocess_ms": times.postprocess_ms,
                "end_to_end_ms": times.end_to_end_ms,
                "detections": counts.detections,
                "classified_crops": counts.classified_crops,
                "segmentation_masks": counts.segmentation_masks,
            }
        )
    t_all1 = time.perf_counter()
    wall_ms = float((t_all1 - t_all0) * 1000.0)

    n = max(1, len(images))
    total_ms_per_img = sum_e2e / n
    fps = 1000.0 / max(1e-9, total_ms_per_img)

    summary = {
        "images_count": int(len(images)),
        "total_end_to_end_fps": float(fps),
        "total_ms_per_img": float(total_ms_per_img),
        "detector_ms_per_img": float(sum_det / n),
        "roi_prep_ms_per_img": float(sum_roi / n),
        "classifier_ms_per_img": float(sum_cls / n),
        "segmentation_ms_per_img": float(sum_seg / n),
        "postprocess_ms_per_img": float(sum_post / n),
        "avg_detections_per_img": float(sum_dets / n),
        "avg_classified_crops_per_img": float(sum_cls_crops / n),
        "avg_segmentation_masks_per_img": float(sum_seg_masks / n),
        "device": str(device_req),
        "ultralytics_device": str(ul_dev),
        "model_paths": {
            "detector": str(det_w),
            "classifier": str(cls_w),
            "segmenter": str(seg_w),
        },
        "weights_sizes_mb": {
            "detector": _weights_size_mb(det_w),
            "classifier": _weights_size_mb(cls_w),
            "segmenter": _weights_size_mb(seg_w),
        },
        "pipeline_params": {
            "imgsz_det": int(args.imgsz_det),
            "conf_det": float(args.conf_det),
            "iou_det": float(args.iou_det),
            "max_det": int(args.max_det),
            "max_rois": int(args.max_rois),
            "crop_pad_frac": float(args.crop_pad_frac),
            "cls_every": int(max(1, int(args.cls_every))),
            "seg_every": int(args.seg_every),
            "seg_max_rois": int(args.seg_max_rois),
            "seg_min_det_conf": float(args.seg_min_det_conf),
            "imgsz_seg": int(args.imgsz_seg),
            "seg_conf": 0.25,
            "seg_iou": 0.7,
        },
        "timing_notes": {
            "warmup_frames": int(warmup_n),
            "measurement_wall_ms": float(wall_ms),
            "note": "Per-image end_to_end_ms is measured around full detector->ROI->classifier->segmenter->postprocess (no saves). Preview is excluded.",
        },
    }

    # Write outputs
    per_image_csv = outdir / "per_image.csv"
    summary_json = outdir / "summary.json"
    summary_csv = outdir / "summary.csv"

    with per_image_csv.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["key", "value"])
        for k in [
            "images_count",
            "total_end_to_end_fps",
            "total_ms_per_img",
            "detector_ms_per_img",
            "roi_prep_ms_per_img",
            "classifier_ms_per_img",
            "segmentation_ms_per_img",
            "postprocess_ms_per_img",
            "avg_detections_per_img",
            "avg_classified_crops_per_img",
            "avg_segmentation_masks_per_img",
            "device",
        ]:
            wcsv.writerow([k, summary[k]])

    print(f"Wrote {str(per_image_csv)}", flush=True)
    print(f"Wrote {str(summary_csv)}", flush=True)
    print(f"Wrote {str(summary_json)}", flush=True)
    print(
        "summary "
        f"images={int(len(images))} "
        f"fps={float(fps):.2f} ms/img={float(total_ms_per_img):.2f} "
        f"det={float(sum_det/n):.2f} roi={float(sum_roi/n):.2f} cls={float(sum_cls/n):.2f} "
        f"seg={float(sum_seg/n):.2f} post={float(sum_post/n):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

