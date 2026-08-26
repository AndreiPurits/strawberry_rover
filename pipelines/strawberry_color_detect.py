"""Lightweight strawberry bbox when YOLO weights are unavailable (bench fallback)."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _gripper_mask(h: int, w: int) -> np.ndarray:
    """Mask gripper + cables at bottom (not the shelf berry in HOME2)."""
    m = np.zeros((h, w), dtype=np.uint8)
    # Bottom band: gripper claws + pad + cable entry.
    cv2.rectangle(m, (0, int(h * 0.55)), (w, h), 255, -1)
    # Left cable bundle.
    cv2.rectangle(m, (0, int(h * 0.35)), (int(w * 0.32), h), 255, -1)
    return m


def detect_red_berry_bboxes(
    bgr: np.ndarray,
    *,
    min_area_px: int = 40,
    max_area_frac: float = 0.04,
) -> List[Tuple[int, int, int, int, float]]:
    """Return list of (x1,y1,x2,y2,score) for red strawberry-like blobs."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Red (two wraps) — tighter saturation/value for real berry red.
    m1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 100, 80), (180, 255, 255))
    red = cv2.bitwise_or(m1, m2)
    # Exclude yellow-green gripper pad.
    yellow_green = cv2.inRange(hsv, (30, 50, 50), (90, 255, 255))
    mask = cv2.bitwise_and(red, cv2.bitwise_not(yellow_green))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(_gripper_mask(h, w)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    max_area = int(h * w * max_area_frac)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int, int, int, int, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 6 or bh < 6:
            continue
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 2.8:
            continue
        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / max(1.0, peri * peri)
        if circ < 0.25:
            continue
        roi = bgr[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue
        b, g, r = cv2.split(roi)
        mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        # Must be visibly red, not dark cable/sheath.
        if mr < 90 or mr < mg * 1.15 or mr < mb * 1.15:
            continue
        fill = area / max(1.0, bw * bh)
        # HOME2: berry on shelf — upper/mid frame preferred.
        cy = y + bh * 0.5
        y_bonus = 0.12 if cy < h * 0.55 else -0.1
        score = min(0.95, 0.40 + 0.20 * fill + y_bonus + 0.001 * (mr - mg))
        out.append((x, y, x + bw, y + bh, score))
    out.sort(key=lambda t: t[4], reverse=True)
    return out[:5]
