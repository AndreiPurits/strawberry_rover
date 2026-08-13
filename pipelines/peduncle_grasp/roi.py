"""Close-up ROI relative to calyx (not large legacy crop)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]
BBoxXYXY = Tuple[float, float, float, float]


@dataclass
class CloseupROI:
    x1: int
    y1: int
    x2: int
    y2: int
    crop_bgr: np.ndarray
    # map ROI pixel -> full frame: full = roi + (x1,y1)
    origin_xy: Tuple[int, int]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_full(self, x: float, y: float) -> Point2D:
        return (x + self.origin_xy[0], y + self.origin_xy[1])

    def to_roi(self, x: float, y: float) -> Point2D:
        return (x - self.origin_xy[0], y - self.origin_xy[1])

    def bbox_full(self) -> BBoxXYXY:
        return (float(self.x1), float(self.y1), float(self.x2), float(self.y2))


def build_closeup_roi(
    bgr: np.ndarray,
    calyx_xy: Point2D,
    berry_height_px: float,
    berry_center: Point2D,
    cfg_roi: Dict,
) -> CloseupROI:
    """ROI anchored on calyx: calyx in lower portion, limited stem zone above along outward axis."""
    h, w = bgr.shape[:2]
    bh = max(8.0, float(berry_height_px))
    width = bh * float(cfg_roi.get("width_berry_scale", 1.4))
    above = bh * float(cfg_roi.get("height_above_calyx_scale", 0.55))
    below = bh * float(cfg_roi.get("height_below_calyx_scale", 0.35))
    min_sz = int(cfg_roi.get("min_size_px", 96))
    max_sz = int(cfg_roi.get("max_size_px", 640))

    # outward direction: berry center -> calyx (stem exits this way)
    vx = calyx_xy[0] - berry_center[0]
    vy = calyx_xy[1] - berry_center[1]
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-3:
        ux, uy = 0.0, -1.0  # default image-up only if degenerate
    else:
        ux, uy = vx / n, vy / n

    # local frame: u along outward, v orthogonal
    # ROI center slightly outward from calyx
    cx = calyx_xy[0] + ux * (above - below) * 0.5
    cy = calyx_xy[1] + uy * (above - below) * 0.5
    half_w = 0.5 * width
    half_h = 0.5 * (above + below)

    # axis-aligned ROI containing the oriented window (safe, simple)
    rad = ((half_w) ** 2 + (half_h) ** 2) ** 0.5
    x1 = int(np.floor(cx - rad))
    y1 = int(np.floor(cy - rad))
    x2 = int(np.ceil(cx + rad))
    y2 = int(np.ceil(cy + rad))

    # enforce min/max box size
    bw = max(1, x2 - x1)
    bh_box = max(1, y2 - y1)
    side = max(bw, bh_box, min_sz)
    side = min(side, max_sz)
    mx = int(round(0.5 * (x1 + x2)))
    my = int(round(0.5 * (y1 + y2)))
    x1 = mx - side // 2
    y1 = my - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # clip to image
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)
    if x2c - x1c < min_sz // 2 or y2c - y1c < min_sz // 2:
        # expand toward image center
        x1c = max(0, min(w - min_sz, mx - min_sz // 2))
        y1c = max(0, min(h - min_sz, my - min_sz // 2))
        x2c = min(w, x1c + min_sz)
        y2c = min(h, y1c + min_sz)

    crop = bgr[y1c:y2c, x1c:x2c].copy()
    return CloseupROI(x1=x1c, y1=y1c, x2=x2c, y2=y2c, crop_bgr=crop, origin_xy=(x1c, y1c))
