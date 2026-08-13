#!/usr/bin/env python3
"""
Visually strict cull: remove crop images where peduncle (stem) is NOT clearly visible.

Uses berry bbox from labels/ to inspect the zone above the berry crown.
Moves rejected samples to rejected_no_peduncle/ mirroring dataset layout.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _noise_std(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))


def _count_red_blobs(bgr: np.ndarray, min_area_frac: float = 0.04) -> int:
    h, w = bgr.shape[:2]
    min_area = float(h * w) * min_area_frac
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 55, 45), (12, 255, 255))
    m2 = cv2.inRange(hsv, (160, 55, 45), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if cv2.contourArea(c) >= min_area)


def _parse_berry_label(text: str) -> Optional[Tuple[float, float, float, float]]:
    for raw in text.splitlines():
        toks = raw.strip().split()
        if len(toks) != 5:
            continue
        try:
            return tuple(map(float, toks[1:5]))  # xc yc w h
        except Exception:
            continue
    return None


def _yolo_to_xyxy(xc: float, yc: float, w: float, h: float, iw: int, ih: int) -> Tuple[int, int, int, int]:
    bw, bh = w * iw, h * ih
    x1 = int(math.floor(xc * iw - bw / 2))
    y1 = int(math.floor(yc * ih - bh / 2))
    x2 = int(math.ceil(xc * iw + bw / 2))
    y2 = int(math.ceil(yc * ih + bh / 2))
    return max(0, x1), max(0, y1), min(iw, max(x2, x1 + 1)), min(ih, max(y2, y1 + 1))


@dataclass
class PeduncleVisualVerdict:
    passed: bool
    reason: str
    green_ratio: float
    stem_span_frac: float
    center_path: bool
    thin_stem: bool
    top_open: bool


def assess_peduncle_visible(
    bgr: np.ndarray,
    berry_xyxy: Tuple[int, int, int, int],
    *,
    min_green_ratio: float,
    min_stem_span_frac: float,
    min_thin_aspect: float,
) -> PeduncleVisualVerdict:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = berry_xyxy
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    berry_cx = (x1 + x2) // 2

    zone_y2 = min(h, y1 + int(0.06 * bh))
    zone_y1 = 0
    if zone_y2 - zone_y1 < 10:
        return PeduncleVisualVerdict(False, "no_room_above_berry", 0, 0, False, False, False)

    # Full berry-width band above crown — stem is thin in center; leaf cover fills evenly.
    zx1 = max(0, int(x1 + 0.05 * bw))
    zx2 = min(w, int(x2 - 0.05 * bw))
    zone = bgr[zone_y1:zone_y2, zx1:zx2]
    if zone.size == 0:
        return PeduncleVisualVerdict(False, "empty_zone", 0, 0, False, False, False)

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    green = cv2.inRange(hsv, (28, 42, 42), (95, 255, 255))
    green = cv2.bitwise_and(green, cv2.inRange(gray, 30, 255))

    upper = gray[: max(1, int(0.30 * zone.shape[0])), :]
    dark_upper = float(np.count_nonzero(upper < 35)) / float(upper.size)
    top_open = dark_upper < 0.65

    green_ratio = float(np.count_nonzero(green)) / float(green.size)
    if green_ratio < min_green_ratio:
        return PeduncleVisualVerdict(False, "no_green_stem", green_ratio, 0, False, False, top_open)

    zh, zw = zone.shape[:2]
    upper_green = green[: max(1, int(0.70 * zh)), :]
    if upper_green.size:
        third = max(1, zw // 3)
        left_g = float(np.count_nonzero(upper_green[:, :third]))
        center_g = float(np.count_nonzero(upper_green[:, third : 2 * third]))
        right_g = float(np.count_nonzero(upper_green[:, 2 * third :]))
        side_avg = (left_g + right_g) / 2.0
        if center_g > 20 and side_avg / max(1.0, center_g) > 0.82:
            return PeduncleVisualVerdict(False, "leaf_mass_not_stem", green_ratio, 0, False, False, top_open)

    # Thin vertical stem in center third.
    third = max(1, zw // 3)
    center_green = green[:, third : 2 * third]
    k_stem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(9, int(0.35 * zh))))
    stem = cv2.morphologyEx(center_green, cv2.MORPH_OPEN, k_stem)
    stem = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    row_proj = np.count_nonzero(stem, axis=1)
    if row_proj.max() == 0:
        return PeduncleVisualVerdict(False, "no_stem_pixels", green_ratio, 0, False, False, top_open)

    active = row_proj > 0
    touch_band = max(2, int(0.22 * zh))
    if not bool(active[-touch_band:].any()):
        # Allow attachment via dilated calyx bridge.
        bridge = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        bridge_proj = np.count_nonzero(bridge, axis=1)
        if not bool(bridge_proj[-touch_band:].any()):
            return PeduncleVisualVerdict(False, "stem_not_attached", green_ratio, 0, False, False, top_open)
        active = bridge_proj > 0
        row_proj = bridge_proj
        stem = bridge

    ys = np.where(active)[0]
    stem_span_frac = float(ys.max() - ys.min() + 1) / float(max(1, zh))
    if stem_span_frac < min_stem_span_frac:
        return PeduncleVisualVerdict(False, "stem_too_short", green_ratio, stem_span_frac, False, False, top_open)

    # Width along skeleton — stem stays narrow; leaves do not.
    widths: List[int] = []
    for row in range(ys.min(), ys.max() + 1):
        xs = np.where(stem[row] > 0)[0]
        if xs.size:
            widths.append(int(xs.max() - xs.min() + 1))
    if not widths:
        return PeduncleVisualVerdict(False, "no_stem_component", green_ratio, stem_span_frac, False, False, top_open)

    med_w = float(np.median(widths))
    max_w = max(widths)
    max_allowed_w = max(8.0, 0.50 * bw)
    if max_w > max_allowed_w:
        return PeduncleVisualVerdict(False, "only_leaves_no_stem", green_ratio, stem_span_frac, True, False, top_open)

    thin_stem = (stem_span_frac * zh) / max(1.0, med_w) >= min_thin_aspect
    if not thin_stem:
        return PeduncleVisualVerdict(False, "stem_not_thin", green_ratio, stem_span_frac, True, False, top_open)

    if not top_open and stem_span_frac < (min_stem_span_frac + 0.10):
        return PeduncleVisualVerdict(False, "stem_cut_at_top", green_ratio, stem_span_frac, True, thin_stem, top_open)

    return PeduncleVisualVerdict(True, "ok", green_ratio, stem_span_frac, True, thin_stem, top_open)


def _move_pair(stem: str, src_root: Path, dst_root: Path, subdirs: List[str]) -> None:
    for sub in subdirs:
        src_dir = src_root / sub
        dst_dir = dst_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("",) if sub == "meta" else IMG_EXTS:
            if sub == "meta":
                p = src_dir / f"{stem}.json"
                if p.exists():
                    shutil.move(str(p), str(dst_dir / p.name))
                continue
            for e in (ext,) if ext else IMG_EXTS:
                p = src_dir / f"{stem}{e}"
                if p.exists():
                    shutil.move(str(p), str(dst_dir / p.name))
        p = src_dir / f"{stem}.txt"
        if p.exists():
            shutil.move(str(p), str(dst_dir / p.name))


def _load_keep_file(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    out: set[str] = set()
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description="Cull images without visible peduncle.")
    ap.add_argument("--dataset", default=str(REPO_ROOT / "data" / "ripe_peduncle_closeup_dataset"))
    ap.add_argument("--rejected-dirname", default="rejected_no_peduncle")
    ap.add_argument("--keep-file", default="", help="If set, keep ONLY stems listed in this file.")
    ap.add_argument("--min-green-ratio", type=float, default=0.03)
    ap.add_argument("--min-stem-span-frac", type=float, default=0.28)
    ap.add_argument("--min-thin-aspect", type=float, default=1.8)
    ap.add_argument("--max-noise-std", type=float, default=13.0)
    ap.add_argument("--max-red-blobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    images_dir = root / "images"
    labels_dir = root / "labels"
    rejected_root = root / str(args.rejected_dirname)
    keep_only = _load_keep_file(str(args.keep_file).strip() or None)

    kept: List[str] = []
    rejected: List[Dict[str, object]] = []

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem

        if keep_only is not None and stem not in keep_only:
            rejected.append({"stem": stem, "reason": "not_in_visual_whitelist"})
            if not args.dry_run:
                _move_pair(stem, root, rejected_root, ["images", "labels", "labels_peduncle", "meta", "bbox_vis"])
            continue

        if keep_only is not None and stem in keep_only:
            kept.append(stem)
            continue

        ped_lbl_path = root / "labels_peduncle" / f"{stem}.txt"
        if ped_lbl_path.exists() and ped_lbl_path.read_text(encoding="utf-8", errors="ignore").strip():
            kept.append(stem)
            continue

        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.exists():
            rejected.append({"stem": stem, "reason": "missing_label"})
            continue

        parsed = _parse_berry_label(lbl_path.read_text(encoding="utf-8", errors="ignore"))
        if parsed is None:
            rejected.append({"stem": stem, "reason": "bad_label"})
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            rejected.append({"stem": stem, "reason": "broken_image"})
            continue

        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if _noise_std(gray_full) > float(args.max_noise_std):
            rejected.append({"stem": stem, "reason": "too_noisy"})
            if not args.dry_run:
                _move_pair(stem, root, rejected_root, ["images", "labels", "labels_peduncle", "meta", "bbox_vis"])
            continue
        if _count_red_blobs(bgr) > int(args.max_red_blobs):
            rejected.append({"stem": stem, "reason": "multiple_berries"})
            if not args.dry_run:
                _move_pair(stem, root, rejected_root, ["images", "labels", "labels_peduncle", "meta", "bbox_vis"])
            continue

        ih, iw = bgr.shape[:2]
        berry_xyxy = _yolo_to_xyxy(*parsed, iw, ih)
        verdict = assess_peduncle_visible(
            bgr,
            berry_xyxy,
            min_green_ratio=float(args.min_green_ratio),
            min_stem_span_frac=float(args.min_stem_span_frac),
            min_thin_aspect=float(args.min_thin_aspect),
        )
        row = {"stem": stem, **asdict(verdict)}
        if verdict.passed:
            kept.append(stem)
        else:
            rejected.append(row)
            if not args.dry_run:
                _move_pair(
                    stem,
                    root,
                    rejected_root,
                    ["images", "labels", "labels_peduncle", "meta", "bbox_vis"],
                )

    report = {
        "dataset": str(root),
        "rejected_dir": str(rejected_root),
        "kept": len(kept),
        "rejected": len(rejected),
        "dry_run": bool(args.dry_run),
        "keep_file": str(args.keep_file) if str(args.keep_file).strip() else None,
        "rejected_samples": rejected,
    }
    out_report = root / "reports" / "peduncle_cull_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"kept": len(kept), "rejected": len(rejected)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
