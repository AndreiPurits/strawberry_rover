#!/usr/bin/env python3
"""
Build a peduncle labeling queue from FPS dataset frames.

Filters (configurable):
- distance 12-15 cm via pinhole model,
- visible peduncle above berry bbox,
- optional ripe-only / red-ratio / sharpness / close-up checks,
- optional one-best-berry-per source frame.

Output layouts:
  queue:   crops/ + labels_berry/ + labels_peduncle/
  dataset: images/ + labels/ + labels_peduncle/ + bbox_vis/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_BERRY = 0
CLASS_PEDUNCLE = 1


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _iter_label_image_pairs(src_root: Path) -> List[Tuple[Path, Path]]:
    """Return (label_file, images_dir) for flat, YOLO split, or co-located split layouts."""
    pairs: List[Tuple[Path, Path]] = []
    images_root = src_root / "images"
    labels_root = src_root / "labels"
    if images_root.is_dir() and labels_root.is_dir():
        split_dirs = [p for p in labels_root.iterdir() if p.is_dir()]
        if split_dirs:
            for split in sorted(split_dirs):
                img_dir = images_root / split.name
                if not img_dir.is_dir():
                    continue
                for lf in sorted(split.glob("*.txt")):
                    pairs.append((lf, img_dir))
        else:
            for lf in sorted(labels_root.glob("*.txt")):
                pairs.append((lf, images_root))

    for split_name in ("training", "validation", "train", "val", "test"):
        split_dir = src_root / split_name
        if not split_dir.is_dir():
            continue
        for lf in sorted(split_dir.glob("*.txt")):
            if _find_image(split_dir, lf.stem) is not None:
                pairs.append((lf, split_dir))

    return pairs


def _parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float, List[Tuple[float, float]]]]:
    toks = line.strip().split()
    if len(toks) < 5:
        return None
    try:
        cid = int(float(toks[0]))
        a, b, c, d = map(float, toks[1:5])
    except Exception:
        return None

    poly: List[Tuple[float, float]] = []
    if len(toks) > 5:
        coords = list(map(float, toks[5:]))
        if len(coords) % 2 == 0:
            for i in range(0, len(coords), 2):
                poly.append((coords[i], coords[i + 1]))

    # Standard YOLO bbox header (seg labels from FPS / detection).
    if 0.0 < c <= 1.0 and 0.0 < d <= 1.0 and (a - c / 2.0) >= -0.05 and (a + c / 2.0) <= 1.05:
        return cid, a, b, c, d, poly

    # Polygon-only label: derive bbox from polygon points.
    all_coords = list(map(float, toks[1:]))
    if len(all_coords) < 6 or len(all_coords) % 2 != 0:
        return None
    xs = all_coords[0::2]
    ys = all_coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    for i in range(0, len(all_coords), 2):
        poly.append((all_coords[i], all_coords[i + 1]))
    return cid, xc, yc, w, h, poly


def _parse_yolo_seg_line(line: str) -> Optional[Tuple[int, float, float, float, float, List[Tuple[float, float]]]]:
    return _parse_yolo_line(line)


def _yolo_norm_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    bw = w * img_w
    bh = h * img_h
    x1 = int(math.floor(xc * img_w - bw / 2.0))
    y1 = int(math.floor(yc * img_h - bh / 2.0))
    x2 = int(math.ceil(xc * img_w + bw / 2.0))
    y2 = int(math.ceil(yc * img_h + bh / 2.0))
    return (
        _clamp(x1, 0, img_w - 1),
        _clamp(y1, 0, img_h - 1),
        _clamp(max(x2, x1 + 1), 1, img_w),
        _clamp(max(y2, y1 + 1), 1, img_h),
    )


def _xyxy_to_yolo_norm(x1: int, y1: int, x2: int, y2: int, crop_w: int, crop_h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = (x1 + x2) / 2.0 / crop_w
    yc = (y1 + y2) / 2.0 / crop_h
    return xc, yc, bw / crop_w, bh / crop_h


def _estimate_distance_cm(berry_diameter_px: float, *, focal_px: float, berry_diameter_mm: float) -> float:
    if berry_diameter_px <= 1.0:
        return float("inf")
    return (focal_px * berry_diameter_mm) / (berry_diameter_px * 10.0)


def _asymmetric_crop_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    pad_top_frac: float,
    pad_bottom_frac: float,
    pad_side_frac: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad_side_frac))
    pt = int(round(bh * pad_top_frac))
    pb = int(round(bh * pad_bottom_frac))
    cx1 = _clamp(x1 - px, 0, img_w - 1)
    cy1 = _clamp(y1 - pt, 0, img_h - 1)
    cx2 = _clamp(x2 + px, 1, img_w)
    cy2 = _clamp(y2 + pb, 1, img_h)
    if cx2 <= cx1:
        cx2 = min(img_w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(img_h, cy1 + 1)
    return cx1, cy1, cx2, cy2


@dataclass
class PeduncleHeuristic:
    green_ratio: float
    vertical_span_frac: float
    top_margin_frac: float
    score: float
    passed: bool
    reason: str


def _laplacian_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _red_ratio_in_berry(frame_bgr: np.ndarray, berry_xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = berry_xyxy
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 55, 45), (12, 255, 255))
    m2 = cv2.inRange(hsv, (160, 55, 45), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _score_peduncle_visibility(
    frame_bgr: np.ndarray,
    berry_xyxy: Tuple[int, int, int, int],
    *,
    top_search_frac: float,
    min_green_ratio: float,
    min_vertical_span_frac: float,
    min_top_margin_frac: float,
) -> PeduncleHeuristic:
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = berry_xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    zone_h = int(round(bh * top_search_frac))
    if zone_h < 4:
        return PeduncleHeuristic(0.0, 0.0, 0.0, 0.0, False, "berry_too_small")

    zx1 = _clamp(int(x1 + 0.15 * bw), 0, w_img - 1)
    zx2 = _clamp(int(x2 - 0.15 * bw), zx1 + 1, w_img)
    zy2 = _clamp(y1 + int(0.05 * bh), 1, h_img)
    zy1 = _clamp(y1 - zone_h, 0, zy2 - 1)

    zone = frame_bgr[zy1:zy2, zx1:zx2]
    if zone.size == 0:
        return PeduncleHeuristic(0.0, 0.0, 0.0, 0.0, False, "empty_zone")

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask, cv2.inRange(gray, 25, 255))

    green_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    top_margin_frac = float(zy1) / float(max(1, h_img))

    vertical_span_frac = 0.0
    if green_ratio > 0.01:
        ys, _xs = np.where(mask > 0)
        if ys.size > 0:
            vertical_span_frac = float(ys.max() - ys.min() + 1) / float(max(1, zone.shape[0]))

    score = 0.55 * min(1.0, green_ratio / max(min_green_ratio, 1e-6))
    score += 0.35 * min(1.0, vertical_span_frac / max(min_vertical_span_frac, 1e-6))
    score += 0.10 * min(1.0, top_margin_frac / max(min_top_margin_frac, 1e-6))

    if top_margin_frac < min_top_margin_frac:
        return PeduncleHeuristic(green_ratio, vertical_span_frac, top_margin_frac, score, False, "peduncle_at_frame_edge")
    if green_ratio < min_green_ratio:
        return PeduncleHeuristic(green_ratio, vertical_span_frac, top_margin_frac, score, False, "low_green_ratio")
    if vertical_span_frac < min_vertical_span_frac:
        return PeduncleHeuristic(green_ratio, vertical_span_frac, top_margin_frac, score, False, "short_stem")

    return PeduncleHeuristic(green_ratio, vertical_span_frac, top_margin_frac, score, True, "ok")


def _peduncle_matches_berry(
    ped_xyxy: Tuple[int, int, int, int],
    berry_xyxy: Tuple[int, int, int, int],
) -> bool:
    px1, py1, px2, py2 = ped_xyxy
    bx1, by1, bx2, by2 = berry_xyxy
    bh = max(1, by2 - by1)
    if py2 > by1 + int(0.12 * bh):
        return False
    overlap = min(px2, bx2) - max(px1, bx1)
    min_w = min(max(1, px2 - px1), max(1, bx2 - bx1))
    return overlap >= int(0.20 * min_w)


def _find_annotated_peduncle(
    peduncles: List[Tuple[int, int, int, int]],
    berry_xyxy: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    best: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0
    bx1, by1, bx2, by2 = berry_xyxy
    bcx = (bx1 + bx2) / 2.0
    for ped in peduncles:
        if not _peduncle_matches_berry(ped, berry_xyxy):
            continue
        px1, py1, px2, py2 = ped
        pcx = (px1 + px2) / 2.0
        score = float(by1 - py2) - 0.5 * abs(pcx - bcx)
        if score > best_score:
            best_score = score
            best = ped
    return best


@dataclass
class QueueItem:
    crop_id: str
    source_image: str
    source_label: str
    obj_idx: int
    ripeness_class: int
    distance_cm: float
    berry_xyxy_full: Tuple[int, int, int, int]
    crop_xyxy_full: Tuple[int, int, int, int]
    peduncle_score: float
    peduncle_reason: str


@dataclass
class Candidate:
    crop_id: str
    crop: np.ndarray
    berry_line: str
    item: QueueItem
    ped: PeduncleHeuristic
    sharpness: float
    red_ratio: float
    berry_frac: float
    manifest_row: Dict[str, object]
    vis: Optional[np.ndarray]
    peduncle_line: Optional[str] = None


@dataclass
class QueueStats:
    images_seen: int = 0
    berries_seen: int = 0
    kept: int = 0
    rejected_distance: int = 0
    rejected_peduncle: int = 0
    rejected_ripeness: int = 0
    rejected_quality: int = 0
    rejected_read_error: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)

    def bump_reject(self, reason: str) -> None:
        self.reject_reasons[reason] = int(self.reject_reasons.get(reason, 0)) + 1


def build_from_precropped(args: argparse.Namespace) -> QueueStats:
    """Import already-cropped queue items (crops/ + labels_berry/) with filtering."""
    src_root = Path(args.src)
    src_tag = str(args.src_tag).strip() or "pq"
    out_root = Path(args.out)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    ped_dir = out_root / "labels_peduncle"
    meta_dir = out_root / "meta"
    for p in (images_dir, labels_dir, ped_dir, meta_dir):
        _safe_mkdir(p)

    ripeness_filter = _parse_class_filter(str(args.ripeness_classes))
    crops_dir = src_root / "crops"
    berry_dir = src_root / "labels_berry"
    meta_src = src_root / "meta"

    stats = QueueStats()
    for img_path in sorted(crops_dir.glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        stats.images_seen += 1
        berry_lbl = berry_dir / f"{stem}.txt"
        if not berry_lbl.exists():
            stats.bump_reject("missing_label")
            continue

        meta_path = meta_src / f"{stem}.json"
        ripeness_class = -1
        if meta_path.exists():
            try:
                ripeness_class = int(json.loads(meta_path.read_text(encoding="utf-8")).get("ripeness_class", -1))
            except Exception:
                pass
        if ripeness_filter is not None and ripeness_class not in ripeness_filter:
            stats.rejected_ripeness += 1
            stats.bump_reject("ripeness_filtered")
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            stats.bump_reject("broken_image")
            continue
        h, w = bgr.shape[:2]

        parsed = _parse_berry_label_from_file(berry_lbl)
        if parsed is None:
            stats.bump_reject("bad_label")
            continue
        x1, y1, x2, y2 = _yolo_norm_to_xyxy(*parsed, w, h)

        if _noise_std(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) > float(getattr(args, "max_noise_std", 13.0)):
            stats.rejected_quality += 1
            stats.bump_reject("too_noisy")
            continue
        if _count_red_blobs_import(bgr) > 1:
            stats.rejected_quality += 1
            stats.bump_reject("multiple_berries")
            continue

        sharpness = _laplacian_sharpness(bgr)
        if sharpness < float(args.min_sharpness):
            stats.rejected_quality += 1
            stats.bump_reject("too_blurry")
            continue
        red_ratio = _red_ratio_in_berry(bgr, (x1, y1, x2, y2))
        if float(args.min_red_ratio) > 0 and red_ratio < float(args.min_red_ratio):
            stats.rejected_quality += 1
            stats.bump_reject("not_red_enough")
            continue

        from scripts.cull_no_peduncle_images import assess_peduncle_visible

        ped = assess_peduncle_visible(
            bgr, (x1, y1, x2, y2),
            min_green_ratio=float(args.min_green_ratio),
            min_stem_span_frac=0.28,
            min_thin_aspect=1.8,
        )
        if not ped.passed:
            stats.rejected_peduncle += 1
            stats.bump_reject(ped.reason)
            continue

        crop_id = f"{src_tag}__{stem}"
        out_img = images_dir / f"{crop_id}.jpg"
        if bool(args.skip_existing) and out_img.exists():
            continue

        cv2.imwrite(str(out_img), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        shutil.copy2(str(berry_lbl), str(labels_dir / f"{crop_id}.txt"))
        ped_lbl = ped_dir / f"{crop_id}.txt"
        if not ped_lbl.exists() or bool(args.overwrite):
            ped_lbl.write_text("", encoding="utf-8")
        stats.kept += 1

    return stats


def _load_coco_annotations(coco_json: Path) -> Tuple[Dict[int, dict], Dict[int, List[Tuple[int, List[float]]]]]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images_by_id: Dict[int, dict] = {}
    for im in data.get("images", []):
        try:
            images_by_id[int(im["id"])] = im
        except Exception:
            continue
    anns_by_image: Dict[int, List[Tuple[int, List[float]]]] = {}
    for ann in data.get("annotations", []):
        try:
            iid = int(ann["image_id"])
            cid = int(ann["category_id"])
            bbox = ann.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x, y, w, h = map(float, bbox)
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        anns_by_image.setdefault(iid, []).append((cid, [x, y, w, h]))
    return images_by_id, anns_by_image


def _coco_bbox_to_xyxy(b: List[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = b
    return int(math.floor(x)), int(math.floor(y)), int(math.ceil(x + w)), int(math.ceil(y + h))


def _iter_coco_splits(src_root: Path) -> List[Tuple[Path, Path]]:
    splits: List[Tuple[Path, Path]] = []
    for coco_json in sorted(src_root.rglob("_annotations.coco.json")):
        img_dir = coco_json.parent
        if img_dir.is_dir():
            splits.append((coco_json, img_dir))
    if not splits:
        for name in ("train", "valid", "val", "test"):
            ann_dir = src_root / "annotations"
            for pattern in (f"instances_{name}.json", f"{name}.json"):
                coco_json = ann_dir / pattern if ann_dir.is_dir() else src_root / pattern
                if coco_json.exists():
                    img_dir = src_root / "images" / name
                    if not img_dir.is_dir():
                        img_dir = src_root / name
                    if img_dir.is_dir():
                        splits.append((coco_json, img_dir))
    return splits


def build_from_coco(args: argparse.Namespace) -> QueueStats:
    """Build crops from COCO detection annotations (Roboflow / segmentation_project)."""
    src_root = Path(args.src)
    src_tag = str(args.src_tag).strip() or "coco"
    out_root = Path(args.out)
    out_layout = str(args.layout).strip().lower()
    if out_layout == "dataset":
        images_dir = out_root / "images"
        labels_dir = out_root / "labels"
        preview_dir = out_root / "bbox_vis"
    else:
        images_dir = out_root / "crops"
        labels_dir = out_root / "labels_berry"
        preview_dir = out_root / "preview"
    ped_dir = out_root / "labels_peduncle"
    meta_dir = out_root / "meta"
    for p in (images_dir, labels_dir, ped_dir, meta_dir, preview_dir):
        _safe_mkdir(p)

    ripeness_filter = _parse_class_filter(str(args.ripeness_classes))
    stats = QueueStats()
    preview_budget = int(args.preview_count)

    for coco_json, img_dir in _iter_coco_splits(src_root):
        images_by_id, anns_by_image = _load_coco_annotations(coco_json)
        for iid, im in images_by_id.items():
            stats.images_seen += 1
            file_name = str(im.get("file_name", ""))
            img_path = img_dir / file_name
            if not img_path.exists():
                stats.bump_reject("missing_image")
                continue
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame is None:
                stats.bump_reject("broken_image")
                continue
            h_img, w_img = frame.shape[:2]
            focal_px = float(args.focal_px) if float(args.focal_px) > 0 else float(w_img) * float(args.focal_scale)

            source_candidates: List[Candidate] = []
            obj_idx = 0
            for cid, bbox in anns_by_image.get(iid, []):
                stats.berries_seen += 1
                if ripeness_filter is not None and cid not in ripeness_filter:
                    stats.rejected_ripeness += 1
                    stats.bump_reject("ripeness_filtered")
                    obj_idx += 1
                    continue

                x, y, bw, bh = bbox
                x1 = _clamp(int(math.floor(x)), 0, w_img - 1)
                y1 = _clamp(int(math.floor(y)), 0, h_img - 1)
                x2 = _clamp(int(math.ceil(x + bw)), 1, w_img)
                y2 = _clamp(int(math.ceil(y + bh)), 1, h_img)
                berry_diam_px = max(float(x2 - x1), float(y2 - y1))
                dist_cm = _estimate_distance_cm(berry_diam_px, focal_px=focal_px, berry_diameter_mm=float(args.berry_diameter_mm))
                if not (float(args.distance_min_cm) <= dist_cm <= float(args.distance_max_cm)):
                    stats.rejected_distance += 1
                    stats.bump_reject("distance_out_of_range")
                    obj_idx += 1
                    continue

                ped = _score_peduncle_visibility(
                    frame, (x1, y1, x2, y2),
                    top_search_frac=float(args.top_search_frac),
                    min_green_ratio=float(args.min_green_ratio),
                    min_vertical_span_frac=float(args.min_vertical_span_frac),
                    min_top_margin_frac=float(args.min_top_margin_frac),
                )
                if not ped.passed:
                    stats.rejected_peduncle += 1
                    stats.bump_reject(ped.reason)
                    obj_idx += 1
                    continue

                cx1, cy1, cx2, cy2 = _asymmetric_crop_xyxy(
                    x1, y1, x2, y2,
                    pad_top_frac=float(args.pad_top_frac),
                    pad_bottom_frac=float(args.pad_bottom_frac),
                    pad_side_frac=float(args.pad_side_frac),
                    img_w=w_img, img_h=h_img,
                )
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    stats.bump_reject("empty_crop")
                    obj_idx += 1
                    continue
                crop_h, crop_w = crop.shape[:2]
                if crop_w < int(args.min_crop_px) or crop_h < int(args.min_crop_px):
                    stats.rejected_peduncle += 1
                    stats.bump_reject("crop_too_small")
                    obj_idx += 1
                    continue
                if berry_diam_px < float(args.min_berry_px):
                    stats.rejected_quality += 1
                    stats.bump_reject("berry_too_small")
                    obj_idx += 1
                    continue
                sharpness = _laplacian_sharpness(crop)
                if sharpness < float(args.min_sharpness):
                    stats.rejected_quality += 1
                    stats.bump_reject("too_blurry")
                    obj_idx += 1
                    continue
                red_ratio = _red_ratio_in_berry(frame, (x1, y1, x2, y2))
                if float(args.min_red_ratio) > 0 and red_ratio < float(args.min_red_ratio):
                    stats.rejected_quality += 1
                    stats.bump_reject("not_red_enough")
                    obj_idx += 1
                    continue

                bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
                bxc, byc, bw_n, bh_n = _xyxy_to_yolo_norm(bx1, by1, bx2, by2, crop_w, crop_h)
                berry_frac = float((bx2 - bx1) * (by2 - by1)) / float(max(1, crop_w * crop_h))
                if berry_frac < float(args.min_berry_frac):
                    stats.rejected_quality += 1
                    stats.bump_reject("berry_not_closeup")
                    obj_idx += 1
                    continue

                stem = Path(file_name).stem
                crop_id = f"{src_tag}__{stem}__obj{obj_idx:03d}"
                berry_line = f"{CLASS_BERRY} {bxc:.6f} {byc:.6f} {bw_n:.6f} {bh_n:.6f}"
                item = QueueItem(
                    crop_id=crop_id, source_image=str(img_path), source_label=str(coco_json),
                    obj_idx=obj_idx, ripeness_class=cid, distance_cm=round(dist_cm, 2),
                    berry_xyxy_full=(x1, y1, x2, y2), crop_xyxy_full=(cx1, cy1, cx2, cy2),
                    peduncle_score=round(ped.score, 4), peduncle_reason=ped.reason,
                )
                vis = crop.copy()
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
                source_candidates.append(Candidate(
                    crop_id=crop_id, crop=crop, berry_line=berry_line, item=item, ped=ped,
                    sharpness=sharpness, red_ratio=red_ratio, berry_frac=berry_frac,
                    manifest_row={"crop_id": crop_id, "distance_cm": round(dist_cm, 2)},
                    vis=vis,
                ))
                obj_idx += 1

            if bool(args.one_per_source) and len(source_candidates) > 1:
                source_candidates.sort(key=lambda c: (c.ped.score, c.berry_frac, c.sharpness, c.red_ratio), reverse=True)
                source_candidates = source_candidates[:1]

            for cand in source_candidates:
                if _write_candidate(
                    cand, images_dir=images_dir, labels_dir=labels_dir, ped_dir=ped_dir,
                    meta_dir=meta_dir, overwrite=bool(args.overwrite),
                    jpeg_quality=int(args.jpeg_quality), skip_existing=bool(args.skip_existing),
                ):
                    stats.kept += 1

            if stats.images_seen % 500 == 0:
                print(f"progress images={stats.images_seen} berries={stats.berries_seen} kept={stats.kept}", flush=True)

        if args.limit > 0 and stats.images_seen >= int(args.limit):
            break

    return stats


def _parse_berry_label_from_file(path: Path) -> Optional[Tuple[float, float, float, float]]:
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) == 5:
            try:
                return tuple(map(float, toks[1:5]))
            except Exception:
                continue
    return None


def _noise_std(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))


def _count_red_blobs_import(bgr: np.ndarray) -> int:
    h, w = bgr.shape[:2]
    min_area = float(h * w) * 0.04
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 55, 45), (12, 255, 255))
    m2 = cv2.inRange(hsv, (160, 55, 45), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if cv2.contourArea(c) >= min_area)


def _parse_class_filter(text: str) -> Optional[set[int]]:
    raw = str(text).strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out or None


def _write_candidate(
    cand: Candidate,
    *,
    images_dir: Path,
    labels_dir: Path,
    ped_dir: Path,
    meta_dir: Path,
    overwrite: bool,
    jpeg_quality: int,
    skip_existing: bool,
) -> bool:
    crop_path = images_dir / f"{cand.crop_id}.jpg"
    if skip_existing and crop_path.exists():
        return False
    berry_lbl_path = labels_dir / f"{cand.crop_id}.txt"
    ped_lbl_path = ped_dir / f"{cand.crop_id}.txt"
    meta_path = meta_dir / f"{cand.crop_id}.json"

    cv2.imwrite(str(crop_path), cand.crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    berry_lbl_path.write_text(cand.berry_line + "\n", encoding="utf-8")
    if overwrite or not ped_lbl_path.exists():
        ped_lbl_path.write_text(
            (cand.peduncle_line + "\n") if cand.peduncle_line else "",
            encoding="utf-8",
        )

    meta_path.write_text(
        json.dumps(
            {
                **asdict(cand.item),
                "sharpness": round(cand.sharpness, 2),
                "red_ratio": round(cand.red_ratio, 4),
                "berry_frac": round(cand.berry_frac, 4),
                "peduncle_heuristic": asdict(cand.ped),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def build_queue(args: argparse.Namespace) -> QueueStats:
    src_root = Path(args.src)
    src_tag = str(args.src_tag).strip() or "src"
    out_root = Path(args.out)
    out_layout = str(args.layout).strip().lower()

    if out_layout == "dataset":
        images_dir = out_root / "images"
        labels_dir = out_root / "labels"
        preview_dir = out_root / "bbox_vis"
    else:
        images_dir = out_root / "crops"
        labels_dir = out_root / "labels_berry"
        preview_dir = out_root / "preview"

    ped_dir = out_root / "labels_peduncle"
    meta_dir = out_root / "meta"
    reports_dir = out_root / "reports"
    for p in (images_dir, labels_dir, ped_dir, meta_dir, preview_dir, reports_dir):
        _safe_mkdir(p)

    ripeness_filter = _parse_class_filter(str(args.ripeness_classes))
    peduncle_class = int(getattr(args, "peduncle_class", -1))
    require_annotated_peduncle = bool(getattr(args, "require_annotated_peduncle", False))
    label_pairs = _iter_label_image_pairs(src_root)
    if args.limit > 0:
        label_pairs = label_pairs[: int(args.limit)]

    stats = QueueStats()
    manifest_rows: List[Dict[str, object]] = []
    preview_budget = int(args.preview_count)

    for lf, images_dir_src in label_pairs:
        stats.images_seen += 1
        img_path = _find_image(images_dir_src, lf.stem)
        if img_path is None:
            stats.rejected_read_error += 1
            stats.bump_reject("missing_image")
            continue

        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            stats.rejected_read_error += 1
            stats.bump_reject("broken_image")
            continue

        h_img, w_img = frame.shape[:2]
        focal_px = float(args.focal_px) if float(args.focal_px) > 0 else float(w_img) * float(args.focal_scale)

        try:
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            stats.rejected_read_error += 1
            stats.bump_reject("label_read_error")
            continue

        source_candidates: List[Candidate] = []
        obj_idx = 0

        parsed_objects: List[Tuple[int, float, float, float, float]] = []
        for line in lines:
            parsed = _parse_yolo_seg_line(line)
            if parsed is not None:
                cid, xc, yc, w, h, _poly = parsed
                parsed_objects.append((cid, xc, yc, w, h))

        peduncle_xyxys: List[Tuple[int, int, int, int]] = []
        if peduncle_class >= 0:
            for cid, xc, yc, w, h in parsed_objects:
                if cid == peduncle_class:
                    peduncle_xyxys.append(_yolo_norm_to_xyxy(xc, yc, w, h, w_img, h_img))

        for cid, xc, yc, w, h in parsed_objects:
            if peduncle_class >= 0 and cid == peduncle_class:
                continue
            stats.berries_seen += 1

            if ripeness_filter is not None and cid not in ripeness_filter:
                stats.rejected_ripeness += 1
                stats.bump_reject("ripeness_filtered")
                obj_idx += 1
                continue

            x1, y1, x2, y2 = _yolo_norm_to_xyxy(xc, yc, w, h, w_img, h_img)
            berry_diam_px = max(float(x2 - x1), float(y2 - y1))
            dist_cm = _estimate_distance_cm(
                berry_diam_px,
                focal_px=focal_px,
                berry_diameter_mm=float(args.berry_diameter_mm),
            )
            if not (float(args.distance_min_cm) <= dist_cm <= float(args.distance_max_cm)):
                stats.rejected_distance += 1
                stats.bump_reject("distance_out_of_range")
                obj_idx += 1
                continue

            annotated_ped = _find_annotated_peduncle(peduncle_xyxys, (x1, y1, x2, y2)) if peduncle_xyxys else None
            if require_annotated_peduncle and annotated_ped is None:
                stats.rejected_peduncle += 1
                stats.bump_reject("no_annotated_peduncle")
                obj_idx += 1
                continue

            if annotated_ped is not None:
                ped = PeduncleHeuristic(1.0, 1.0, 0.0, 1.0, True, "annotated")
            else:
                ped = _score_peduncle_visibility(
                    frame,
                    (x1, y1, x2, y2),
                    top_search_frac=float(args.top_search_frac),
                    min_green_ratio=float(args.min_green_ratio),
                    min_vertical_span_frac=float(args.min_vertical_span_frac),
                    min_top_margin_frac=float(args.min_top_margin_frac),
                )
            if not ped.passed:
                stats.rejected_peduncle += 1
                stats.bump_reject(ped.reason)
                obj_idx += 1
                continue

            cx1, cy1, cx2, cy2 = _asymmetric_crop_xyxy(
                x1, y1, x2, y2,
                pad_top_frac=float(args.pad_top_frac),
                pad_bottom_frac=float(args.pad_bottom_frac),
                pad_side_frac=float(args.pad_side_frac),
                img_w=w_img,
                img_h=h_img,
            )
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                stats.rejected_peduncle += 1
                stats.bump_reject("empty_crop")
                obj_idx += 1
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_w < int(args.min_crop_px) or crop_h < int(args.min_crop_px):
                stats.rejected_peduncle += 1
                stats.bump_reject("crop_too_small")
                obj_idx += 1
                continue

            if berry_diam_px < float(args.min_berry_px):
                stats.rejected_quality += 1
                stats.bump_reject("berry_too_small")
                obj_idx += 1
                continue

            sharpness = _laplacian_sharpness(crop)
            if sharpness < float(args.min_sharpness):
                stats.rejected_quality += 1
                stats.bump_reject("too_blurry")
                obj_idx += 1
                continue

            red_ratio = _red_ratio_in_berry(frame, (x1, y1, x2, y2))
            if float(args.min_red_ratio) > 0.0 and red_ratio < float(args.min_red_ratio):
                stats.rejected_quality += 1
                stats.bump_reject("not_red_enough")
                obj_idx += 1
                continue

            bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
            bxc, byc, bw_n, bh_n = _xyxy_to_yolo_norm(bx1, by1, bx2, by2, crop_w, crop_h)
            berry_frac = float((bx2 - bx1) * (by2 - by1)) / float(max(1, crop_w * crop_h))
            if berry_frac < float(args.min_berry_frac):
                stats.rejected_quality += 1
                stats.bump_reject("berry_not_closeup")
                obj_idx += 1
                continue

            peduncle_line: Optional[str] = None
            if annotated_ped is not None:
                px1, py1, px2, py2 = annotated_ped
                cpx1, cpy1, cpx2, cpy2 = px1 - cx1, py1 - cy1, px2 - cx1, py2 - cy1
                pxc, pyc, pw_n, ph_n = _xyxy_to_yolo_norm(cpx1, cpy1, cpx2, cpy2, crop_w, crop_h)
                peduncle_line = f"{CLASS_PEDUNCLE} {pxc:.6f} {pyc:.6f} {pw_n:.6f} {ph_n:.6f}"

            crop_id = f"{src_tag}__{lf.stem}__obj{obj_idx:03d}"
            berry_line = f"{CLASS_BERRY} {bxc:.6f} {byc:.6f} {bw_n:.6f} {bh_n:.6f}"
            item = QueueItem(
                crop_id=crop_id,
                source_image=str(img_path),
                source_label=str(lf),
                obj_idx=obj_idx,
                ripeness_class=cid,
                distance_cm=round(dist_cm, 2),
                berry_xyxy_full=(x1, y1, x2, y2),
                crop_xyxy_full=(cx1, cy1, cx2, cy2),
                peduncle_score=round(ped.score, 4),
                peduncle_reason=ped.reason,
            )

            vis = crop.copy()
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
            if annotated_ped is not None:
                cpx1, cpy1, cpx2, cpy2 = annotated_ped[0] - cx1, annotated_ped[1] - cy1, annotated_ped[2] - cx1, annotated_ped[3] - cy1
                cv2.rectangle(vis, (cpx1, cpy1), (cpx2, cpy2), (0, 120, 255), 2)
            else:
                zone_h = int(round((y2 - y1) * float(args.top_search_frac)))
                zy1 = max(0, (y1 - cy1) - zone_h)
                zy2 = max(0, y1 - cy1)
                cv2.rectangle(vis, (0, zy1), (crop_w - 1, zy2), (0, 180, 255), 1)
            cv2.putText(
                vis,
                f"{dist_cm:.1f}cm red={red_ratio:.2f} sharp={sharpness:.0f}",
                (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            rel_img = images_dir.relative_to(out_root) / f"{crop_id}.jpg"
            rel_lbl = labels_dir.relative_to(out_root) / f"{crop_id}.txt"
            source_candidates.append(
                Candidate(
                    crop_id=crop_id,
                    crop=crop,
                    berry_line=berry_line,
                    item=item,
                    ped=ped,
                    sharpness=sharpness,
                    red_ratio=red_ratio,
                    berry_frac=berry_frac,
                    manifest_row={
                        "crop_id": crop_id,
                        "source_image": img_path.name,
                        "obj_idx": obj_idx,
                        "distance_cm": round(dist_cm, 2),
                        "ripeness_class": cid,
                        "peduncle_score": round(ped.score, 4),
                        "red_ratio": round(red_ratio, 4),
                        "sharpness": round(sharpness, 2),
                        "berry_frac": round(berry_frac, 4),
                        "image_path": str(rel_img),
                        "label_path": str(rel_lbl),
                    },
                    vis=vis,
                    peduncle_line=peduncle_line,
                )
            )
            obj_idx += 1

        if bool(args.one_per_source) and len(source_candidates) > 1:
            source_candidates.sort(
                key=lambda c: (c.ped.score, c.berry_frac, c.sharpness, c.red_ratio),
                reverse=True,
            )
            source_candidates = source_candidates[:1]

        for cand in source_candidates:
            if _write_candidate(
                cand,
                images_dir=images_dir,
                labels_dir=labels_dir,
                ped_dir=ped_dir,
                meta_dir=meta_dir,
                overwrite=bool(args.overwrite),
                jpeg_quality=int(args.jpeg_quality),
                skip_existing=bool(args.skip_existing),
            ):
                if preview_budget > 0 and cand.vis is not None:
                    cv2.imwrite(str(preview_dir / f"{cand.crop_id}.jpg"), cand.vis)
                    preview_budget -= 1
                manifest_rows.append(cand.manifest_row)
                stats.kept += 1

        if stats.images_seen % 500 == 0:
            print(
                f"progress images={stats.images_seen} berries={stats.berries_seen} kept={stats.kept}",
                flush=True,
            )

    summary = {
        "src": str(src_root),
        "src_tag": src_tag,
        "out": str(out_root),
        "layout": out_layout,
        "config": {
            "distance_cm": [float(args.distance_min_cm), float(args.distance_max_cm)],
            "berry_diameter_mm": float(args.berry_diameter_mm),
            "focal_px": float(args.focal_px) if float(args.focal_px) > 0 else None,
            "focal_scale": float(args.focal_scale),
            "pad_top_frac": float(args.pad_top_frac),
            "pad_bottom_frac": float(args.pad_bottom_frac),
            "pad_side_frac": float(args.pad_side_frac),
            "min_green_ratio": float(args.min_green_ratio),
            "min_vertical_span_frac": float(args.min_vertical_span_frac),
            "min_top_margin_frac": float(args.min_top_margin_frac),
            "ripeness_classes": sorted(ripeness_filter) if ripeness_filter else None,
            "one_per_source": bool(args.one_per_source),
            "min_berry_px": float(args.min_berry_px),
            "min_berry_frac": float(args.min_berry_frac),
            "min_sharpness": float(args.min_sharpness),
            "min_red_ratio": float(args.min_red_ratio),
            "jpeg_quality": int(args.jpeg_quality),
        },
        "stats": {**asdict(stats), "keep_rate": round(stats.kept / max(1, stats.berries_seen), 4)},
        "labeling_instructions": {
            "berry": "Pre-filled in labels/ as class 0 (one bbox per image).",
            "peduncle": "Manually annotate labels_peduncle/ as class 1.",
            "format": "YOLO: class_id x_center y_center width height (normalized to crop).",
            "one_berry_per_image": bool(args.one_per_source),
        },
    }
    (reports_dir / "queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (reports_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(json.dumps(summary["stats"], ensure_ascii=False, indent=2), flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build peduncle labeling queue from FPS dataset.")
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "ФПС ДАТАСЕТ"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "peduncle_labeling_queue"))
    ap.add_argument("--layout", choices=("queue", "dataset", "precropped", "coco"), default="queue")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--src-tag", default="src", help="Prefix for output crop ids (avoid collisions across sources).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ripeness-classes", default="", help="e.g. '2' for ripe red only.")
    ap.add_argument("--one-per-source", action="store_true", help="1 best berry per source frame.")

    ap.add_argument("--distance-min-cm", type=float, default=11.0)
    ap.add_argument("--distance-max-cm", type=float, default=16.0)
    ap.add_argument("--berry-diameter-mm", type=float, default=30.0)
    ap.add_argument("--focal-px", type=float, default=0.0)
    ap.add_argument("--focal-scale", type=float, default=0.95)

    ap.add_argument("--pad-top-frac", type=float, default=0.85)
    ap.add_argument("--pad-bottom-frac", type=float, default=0.15)
    ap.add_argument("--pad-side-frac", type=float, default=0.20)
    ap.add_argument("--min-crop-px", type=int, default=120)
    ap.add_argument("--min-berry-px", type=float, default=120.0)
    ap.add_argument("--min-berry-frac", type=float, default=0.22)
    ap.add_argument("--min-sharpness", type=float, default=80.0)
    ap.add_argument("--min-red-ratio", type=float, default=0.18)
    ap.add_argument("--jpeg-quality", type=int, default=95)

    ap.add_argument("--top-search-frac", type=float, default=0.90)
    ap.add_argument("--min-green-ratio", type=float, default=0.06)
    ap.add_argument("--min-vertical-span-frac", type=float, default=0.18)
    ap.add_argument("--min-top-margin-frac", type=float, default=0.02)
    ap.add_argument("--skip-existing", action="store_true", help="Skip crops that already exist in output.")
    ap.add_argument("--peduncle-class", type=int, default=-1, help="YOLO class id for annotated peduncle (Luke dataset: 2).")
    ap.add_argument("--require-annotated-peduncle", action="store_true")
    ap.add_argument("--preview-count", type=int, default=200)
    ap.add_argument("--max-noise-std", type=float, default=13.0)
    args = ap.parse_args()

    layout = str(getattr(args, "layout", "queue"))
    if layout == "precropped":
        stats = build_from_precropped(args)
    elif layout == "coco":
        stats = build_from_coco(args)
    else:
        stats = build_queue(args)
    print(json.dumps({**stats.__dict__, "keep_rate": round(stats.kept / max(1, stats.berries_seen), 4)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
