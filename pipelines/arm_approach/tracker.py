"""Lock a berry across frames: spatial score, jump limit, ROI-first, hold."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import BerryLock

BBox = Dict[str, float]


def bbox_center(det: BBox) -> Tuple[float, float]:
    return (float(det["x1"]) + float(det["x2"])) * 0.5, (float(det["y1"]) + float(det["y2"])) * 0.5


def score_detection(
    det: BBox,
    last_px: Optional[float],
    last_py: Optional[float],
    *,
    spatial_penalty: float = 0.0045,
    roi_bonus: float = 0.12,
    roi_radius_px: float = 130.0,
) -> float:
    conf = float(det.get("conf", 0))
    if last_px is None or last_py is None:
        return conf
    cx, cy = bbox_center(det)
    dist = float(np.hypot(cx - last_px, cy - last_py))
    score = conf - spatial_penalty * dist
    if dist <= roi_radius_px:
        score += roi_bonus
    return score


def pick_detection(
    detections: List[BBox],
    last_px: Optional[float],
    last_py: Optional[float],
    *,
    max_jump_px: float = 140.0,
    min_conf: float = 0.25,
    spatial_penalty: float = 0.0045,
    roi_bonus: float = 0.12,
) -> Optional[BBox]:
    if not detections:
        return None
    if last_px is None or last_py is None:
        return max(detections, key=lambda d: float(d.get("conf", 0)))
    near = [
        d
        for d in detections
        if float(np.hypot(bbox_center(d)[0] - last_px, bbox_center(d)[1] - last_py)) <= max_jump_px
    ]
    pool = near if near else detections
    best = max(
        pool,
        key=lambda d: score_detection(
            d, last_px, last_py, spatial_penalty=spatial_penalty, roi_bonus=roi_bonus
        ),
    )
    if not near and float(best.get("conf", 0)) < min_conf:
        return None
    return best


def roi_from_lock(
    width: int,
    height: int,
    px: float,
    py: float,
    *,
    pad_px: float = 150.0,
) -> Optional[Tuple[int, int, int, int]]:
    pad = int(pad_px)
    cx = int(round(px))
    cy = int(round(py))
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(width, cx + pad)
    y2 = min(height, cy + pad)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return x1, y1, x2, y2


class TargetTracker:
    def __init__(
        self,
        *,
        roi_pad_px: float = 150.0,
        hold_max: int = 14,
        max_jump_px: float = 140.0,
        min_conf: float = 0.25,
    ) -> None:
        self.last_px: Optional[float] = None
        self.last_py: Optional[float] = None
        self.last_depth: Optional[float] = None
        self.last_conf: float = 0.0
        self.last_bbox: Optional[BBox] = None
        self.lost_streak = 0
        self.hold_streak = 0
        self.roi_pad_px = roi_pad_px
        self.hold_max = hold_max
        self.max_jump_px = max_jump_px
        self.min_conf = min_conf

    def update(self, det: Optional[BBox], *, depth_m: Optional[float] = None) -> bool:
        if det is None:
            self.lost_streak += 1
            self.hold_streak += 1
            return False
        self.last_px, self.last_py = bbox_center(det)
        self.last_bbox = dict(det)
        self.last_conf = float(det.get("conf", 0))
        if depth_m is not None:
            self.last_depth = float(depth_m)
        self.lost_streak = 0
        self.hold_streak = 0
        return True

    def select(self, detections: List[BBox], *, depth_m: Optional[float] = None) -> Optional[BBox]:
        det = pick_detection(
            detections,
            self.last_px,
            self.last_py,
            max_jump_px=self.max_jump_px,
            min_conf=self.min_conf,
        )
        self.update(det, depth_m=depth_m)
        return det

    @property
    def can_hold(self) -> bool:
        return self.last_px is not None and self.hold_streak < self.hold_max

    def as_lock(self) -> Optional[BerryLock]:
        if self.last_px is None or self.last_py is None:
            return None
        bbox = None
        if self.last_bbox is not None:
            bbox = (
                float(self.last_bbox["x1"]),
                float(self.last_bbox["y1"]),
                float(self.last_bbox["x2"]),
                float(self.last_bbox["y2"]),
            )
        return BerryLock(
            px=float(self.last_px),
            py=float(self.last_py),
            depth_m=float(self.last_depth or 0.0),
            conf=float(self.last_conf),
            bbox=bbox,
        )
