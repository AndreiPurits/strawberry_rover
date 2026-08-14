"""Exclude gripper claws via fixed bottom search slots."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

RectNorm = Tuple[float, float, float, float]

_CLAW_SLOT_CENTERS = (0.36, 0.66)
_CLAW_SLOT_HALF_WIDTH = 0.13
_CLAW_PAD = 0.015
_BRIDGE_SKIP_FRAC = 0.035
_MIN_WIDTH_FRAC = 0.04
_MAX_WIDTH_FRAC = 0.32
_MIN_HEIGHT_ROI_FRAC = 0.12


def _norm_rects_to_pixels(
    rects: Sequence[RectNorm], width: int, height: int
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rects:
        xa = int(max(0.0, min(1.0, x0)) * width)
        ya = int(max(0.0, min(1.0, y0)) * height)
        xb = int(max(0.0, min(1.0, x1)) * width)
        yb = int(max(0.0, min(1.0, y1)) * height)
        if xb > xa and yb > ya:
            out.append((xa, ya, xb, yb))
    return out


def _claw_in_slot(
    roi: np.ndarray,
    *,
    y0: int,
    frame_h: int,
    frame_w: int,
    slot_center: float,
    slot_half_width: float,
    dark_threshold: int,
    bridge_skip: int,
    min_area: float,
    min_width_frac: float,
    max_width_frac: float,
    pad: float,
    bottom_frac: float,
) -> Optional[RectNorm]:
    x_lo = max(0, int((slot_center - slot_half_width) * frame_w))
    x_hi = min(frame_w, int((slot_center + slot_half_width) * frame_w))
    if x_hi <= x_lo + 4:
        return None
    sub = roi[:, x_lo:x_hi]
    dark = (sub < int(dark_threshold)).astype(np.uint8) * 255
    if bridge_skip > 0 and dark.shape[0] > bridge_skip:
        dark = dark[:-bridge_skip, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    dark = cv2.erode(dark, kernel, iterations=1)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Tuple[float, int, int, int, int]] = None
    roi_h = roi.shape[0]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < roi_h * _MIN_HEIGHT_ROI_FRAC:
            continue
        width_frac = bw / frame_w
        if width_frac < min_width_frac or width_frac > max_width_frac:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, bw, bh)
    if best is None:
        return None
    _, x, y, bw, bh = best
    xa = x_lo + x
    xb = xa + bw
    ya = y0 + y
    yb = min(frame_h, ya + bh + bridge_skip)
    return (
        round(max(0.0, xa / frame_w - pad), 4),
        round(max(0.0, bottom_frac - 0.02, ya / frame_h - pad), 4),
        round(min(1.0, xb / frame_w + pad), 4),
        1.0,
    )


def detect_claw_exclude_regions(
    gray: np.ndarray,
    *,
    dark_threshold: int = 48,
    bottom_frac: float = 0.62,
    min_area: float = 250.0,
    slot_centers: Sequence[float] = _CLAW_SLOT_CENTERS,
    slot_half_width: float = _CLAW_SLOT_HALF_WIDTH,
) -> List[RectNorm]:
    """Find two gripper claws in fixed bottom slots; return normalized exclude rects."""
    height, width = gray.shape[:2]
    y0 = int(height * max(0.0, min(1.0, bottom_frac)))
    if y0 >= height:
        return []
    roi = gray[y0:, :]
    bridge_skip = max(4, int(height * _BRIDGE_SKIP_FRAC))
    regions: List[RectNorm] = []
    for slot_center in slot_centers:
        rect = _claw_in_slot(
            roi,
            y0=y0,
            frame_h=height,
            frame_w=width,
            slot_center=float(slot_center),
            slot_half_width=float(slot_half_width),
            dark_threshold=dark_threshold,
            bridge_skip=bridge_skip,
            min_area=min_area,
            min_width_frac=_MIN_WIDTH_FRAC,
            max_width_frac=_MAX_WIDTH_FRAC,
            pad=_CLAW_PAD,
            bottom_frac=bottom_frac,
        )
        if rect is not None:
            regions.append(rect)
    return sorted(regions, key=lambda r: r[0])


def blackout_exclude_regions(
    bgr: np.ndarray,
    regions: Sequence[RectNorm],
    *,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if not regions:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    for xa, ya, xb, yb in _norm_rects_to_pixels(regions, w, h):
        out[ya:yb, xa:xb] = color
    return out


def _bbox_norm(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> RectNorm:
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def _bbox_iou(a: RectNorm, b: RectNorm) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def detection_overlaps_exclude(
    det: dict,
    regions: Sequence[RectNorm],
    width: int,
    height: int,
    *,
    min_iou: float = 0.08,
) -> bool:
    x1, y1, x2, y2 = float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
    cxn = ((x1 + x2) * 0.5) / width
    cyn = ((y1 + y2) * 0.5) / height
    det_box = _bbox_norm(x1, y1, x2, y2, width, height)
    for rx0, ry0, rx1, ry1 in regions:
        if rx0 <= cxn <= rx1 and ry0 <= cyn <= ry1:
            return True
        if _bbox_iou(det_box, (rx0, ry0, rx1, ry1)) >= min_iou:
            return True
    return False


def filter_detections_exclude_regions(
    detections: Sequence[dict],
    regions: Sequence[RectNorm],
    width: int,
    height: int,
) -> List[dict]:
    if not regions:
        return list(detections)
    return [d for d in detections if not detection_overlaps_exclude(d, regions, width, height)]
