#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _iter_images_sorted(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: str(p).lower())
    return imgs


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _device_for_ultralytics(device: str) -> str | int:
    d = str(device).strip().lower()
    if d in ("cuda", "cuda:0", "0", "gpu", "gpu:0"):
        return 0
    if d.startswith("cuda:") and d[5:].isdigit():
        return int(d[5:])
    if d.isdigit():
        return int(d)
    return device


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


def _mask_to_polygon_xy_norm(mask_u8: np.ndarray, *, x_off: int, y_off: int, w_img: int, h_img: int) -> Optional[List[float]]:
    if mask_u8 is None or mask_u8.size == 0:
        return None
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if not bool(np.any(m)):
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if c.shape[0] < 3:
        return None
    # Simplify to reduce label size.
    peri = float(cv2.arcLength(c, True))
    eps = max(1.0, 0.01 * peri)
    c2 = cv2.approxPolyDP(c, epsilon=eps, closed=True)
    pts = c2.reshape(-1, 2).astype(np.float32)
    out: List[float] = []
    for x, y in pts:
        xf = (float(x) + float(x_off)) / float(max(1, w_img))
        yf = (float(y) + float(y_off)) / float(max(1, h_img))
        xf = min(1.0, max(0.0, xf))
        yf = min(1.0, max(0.0, yf))
        out.extend([xf, yf])
    if len(out) < 6:
        return None
    return out


@dataclass(frozen=True)
class Det:
    xyxy: Tuple[int, int, int, int]
    conf: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-label FPS dataset from reserve images using existing models.")
    ap.add_argument(
        "--src",
        default=str(REPO_ROOT / "data" / "segmentation_project_dataset" / "reserve"),
        help="Source reserve directory (recursive).",
    )
    ap.add_argument(
        "--dst",
        default=str(REPO_ROOT / "data" / "ФПС ДАТАСЕТ"),
        help="Destination dataset root containing images/ labels/ images_with_labels/.",
    )
    ap.add_argument("--limit", type=int, default=600, help="How many images to label (0 = all).")
    ap.add_argument("--device", default="cuda:0")
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
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--det-conf", type=float, default=0.35)
    ap.add_argument("--det-iou", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--max-rois", type=int, default=8)
    ap.add_argument("--crop-pad-frac", type=float, default=0.15)
    ap.add_argument("--seg-imgsz", type=int, default=384)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--seg-iou", type=float, default=0.7)
    ap.add_argument("--skip-existing", action="store_true", help="Skip images that already have labels in dst/labels.")
    ap.add_argument(
        "--only-positive",
        action="store_true",
        help="Keep only images where detector found at least one strawberry (move negatives to rejected dir).",
    )
    ap.add_argument(
        "--rejected-dir",
        default="rejected_no_strawberry",
        help="Subdir under dst to store negatives (0 detections).",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"Source not found: {str(src)}")
    if not dst.exists():
        raise SystemExit(f"Destination not found: {str(dst)}")

    dst_images = dst / "images"
    dst_labels = dst / "labels"
    dst_vis = dst / "images_with_labels"
    for p in (dst_images, dst_labels, dst_vis):
        _safe_mkdir(p)
    rejected_root = dst / str(args.rejected_dir)
    rejected_images = rejected_root / "images"
    rejected_labels = rejected_root / "labels"
    rejected_vis = rejected_root / "images_with_labels"
    for p in (rejected_images, rejected_labels, rejected_vis):
        _safe_mkdir(p)

    imgs = _iter_images_sorted(src)
    if not imgs:
        raise SystemExit(f"No images under: {str(src)}")
    if int(args.limit) > 0:
        imgs = imgs[: int(args.limit)]

    cuda_ok = bool(torch.cuda.is_available())
    device_req = str(args.device)
    if "cuda" in device_req and not cuda_ok:
        device_req = "cpu"
    ul_dev = _device_for_ultralytics(device_req)

    from pipelines.strawberry_ensemble import RipenessClassifier  # noqa: WPS433
    from ultralytics import YOLO  # noqa: WPS433

    det_model = YOLO(str(Path(args.detector)))
    seg_model = YOLO(str(Path(args.segmenter)))
    cls_model = RipenessClassifier(str(Path(args.classifier)), device=str(device_req if device_req != "cpu" else "cpu"))

    class_to_id: Dict[str, int] = {str(c): i for i, c in enumerate(cls_model.classes)}

    print(
        "label_config "
        f"src={str(src)} dst={str(dst)} n={len(imgs)} "
        f"device={device_req} ul_device={str(ul_dev)} "
        f"det_imgsz={int(args.det_imgsz)} det_conf={float(args.det_conf)} det_iou={float(args.det_iou)} "
        f"seg_imgsz={int(args.seg_imgsz)} seg_conf={float(args.seg_conf)} seg_iou={float(args.seg_iou)} "
        f"classes={','.join(cls_model.classes)} "
        f"only_positive={int(bool(args.only_positive))} rejected_dir={str(rejected_root)}",
        flush=True,
    )

    kept = 0
    rejected = 0
    for idx, p in enumerate(imgs):
        rel = p.relative_to(src)
        # Flatten path to a safe filename while keeping uniqueness.
        stem = "__".join(rel.with_suffix("").parts)
        out_img = dst_images / f"{stem}{p.suffix.lower()}"
        out_lbl = dst_labels / f"{stem}.txt"
        out_vis = dst_vis / f"{stem}.jpg"
        rej_img = rejected_images / f"{stem}{p.suffix.lower()}"
        rej_lbl = rejected_labels / f"{stem}.txt"
        rej_vis = rejected_vis / f"{stem}.jpg"

        if bool(args.skip_existing) and out_lbl.exists() and out_img.exists():
            continue

        # Read frame from source (don't copy until we know it's positive).
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        det_res = det_model.predict(
            source=frame,
            imgsz=int(args.det_imgsz),
            conf=float(args.det_conf),
            iou=float(args.det_iou),
            device=ul_dev,
            max_det=int(args.max_det),
            verbose=False,
        )[0]

        dets: List[Det] = []
        if det_res.boxes is not None and det_res.boxes.xyxy is not None and det_res.boxes.conf is not None:
            xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
            confs = det_res.boxes.conf.detach().cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                dets.append(Det(xyxy=(int(x1), int(y1), int(x2), int(y2)), conf=float(c)))
        dets.sort(key=lambda d: float(d.conf), reverse=True)
        dets = dets[: max(0, int(args.max_rois))]

        if bool(args.only_positive) and (len(dets) == 0):
            # If already present in main dataset (from a previous run), move it out to rejected.
            if out_img.exists():
                shutil.move(str(out_img), str(rej_img))
            if out_lbl.exists():
                shutil.move(str(out_lbl), str(rej_lbl))
            if out_vis.exists():
                shutil.move(str(out_vis), str(rej_vis))
            rejected += 1
            if (idx + 1) % 50 == 0:
                print(f"processed {idx+1}/{len(imgs)} kept={kept} rejected={rejected}", flush=True)
            continue

        # Positive (or keeping negatives allowed): write into main dataset.
        if not out_img.exists():
            shutil.copy2(str(p), str(out_img))

        # Build crops for classification + segmentation.
        crops: List[Optional[np.ndarray]] = []
        pads: List[Tuple[int, int, int, int]] = []
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            x1p, y1p, x2p, y2p = _pad_xyxy(x1, y1, x2, y2, float(args.crop_pad_frac), w=w, h=h)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                crops.append(None)
                pads.append((x1p, y1p, x2p, y2p))
            else:
                crops.append(crop)
                pads.append((x1p, y1p, x2p, y2p))

        valid_idx = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_idx if crops[i] is not None]  # type: ignore[list-item]
        cls_out = cls_model.infer_crops_bgr(valid_crops) if valid_crops else []

        # Assign classes per detection index.
        cls_by_i: Dict[int, Tuple[str, float]] = {}
        for j, i in enumerate(valid_idx):
            cls_by_i[i] = (str(cls_out[j][0]), float(cls_out[j][1]))

        label_lines: List[str] = []
        vis = frame.copy()

        for i, d in enumerate(dets):
            crop = crops[i]
            if crop is None:
                continue
            x1, y1, x2, y2 = d.xyxy
            x1p, y1p, x2p, y2p = pads[i]
            cls_name, cls_conf = cls_by_i.get(i, ("green", 0.0))
            cls_id = int(class_to_id.get(cls_name, 0))

            seg_res = seg_model.predict(
                source=crop,
                imgsz=int(args.seg_imgsz),
                conf=float(args.seg_conf),
                iou=float(args.seg_iou),
                device=ul_dev,
                verbose=False,
            )[0]

            mask_crop = None
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
                    mask_crop = (masks[best_i] > 0.5).astype(np.uint8) * 255

            # Polygon in full-image coords.
            poly = None
            if mask_crop is not None:
                if mask_crop.shape[:2] != (crop.shape[0], crop.shape[1]):
                    mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                poly = _mask_to_polygon_xy_norm(mask_crop, x_off=int(x1p), y_off=int(y1p), w_img=int(w), h_img=int(h))

            # YOLOv8-seg label: class cx cy w h poly...
            bw = float(max(1, x2 - x1))
            bh = float(max(1, y2 - y1))
            cx = (float(x1) + float(x2)) * 0.5 / float(max(1, w))
            cy = (float(y1) + float(y2)) * 0.5 / float(max(1, h))
            ww = bw / float(max(1, w))
            hh = bh / float(max(1, h))
            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            ww = min(1.0, max(0.0, ww))
            hh = min(1.0, max(0.0, hh))

            if poly is None:
                # If segmentation failed, still write bbox-only (as a degenerate polygon box).
                # Four corners (normalized) so label stays YOLO-seg compatible.
                x1n = float(x1) / float(max(1, w))
                y1n = float(y1) / float(max(1, h))
                x2n = float(x2) / float(max(1, w))
                y2n = float(y2) / float(max(1, h))
                poly = [x1n, y1n, x2n, y1n, x2n, y2n, x1n, y2n]

            line = " ".join([str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{ww:.6f}", f"{hh:.6f}"] + [f"{v:.6f}" for v in poly])
            label_lines.append(line)

            # Visualization
            color = (20, 220, 20) if cls_name == "ripe" else (0, 215, 255) if cls_name == "turning" else (0, 180, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            txt = f"{cls_name}:{cls_conf:.2f} det:{d.conf:.2f}"
            cv2.putText(vis, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if mask_crop is not None:
                msel = (mask_crop > 0)[:, :, None]
                roi = vis[y1p:y2p, x1p:x2p]
                if roi.size and roi.shape[:2] == mask_crop.shape[:2]:
                    overlay = np.zeros_like(roi, dtype=np.uint8)
                    overlay[:, :, 1] = 200
                    vis[y1p:y2p, x1p:x2p] = np.where(msel, (roi * 0.65 + overlay * 0.35).astype(np.uint8), roi)

        out_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        cv2.imwrite(str(out_vis), vis)
        kept += 1

        if (idx + 1) % 50 == 0:
            print(f"processed {idx+1}/{len(imgs)} kept={kept} rejected={rejected}", flush=True)

    print(f"done kept={kept} rejected={rejected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

