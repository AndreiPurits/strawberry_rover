"""Canonical, fail-closed berry depth from one synchronized RGB-D frame.

The plastic berry can be hollow in depth: the bbox centre may look through to
the background.  Use the nearest well-supported depth cluster inside the bbox
instead of an arbitrary offset pixel or the median of foreground+background.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass
class BerryDepthSample:
    frame_id: str
    stamp_s: float
    stamp_ros: Optional[Tuple[int, int]]
    bbox: Sequence[float]
    center_px: Tuple[int, int]
    raw_center_depth_m: Optional[float]
    median_bbox_depth_m: Optional[float]
    mask_depth_m: Optional[float]
    canonical_depth_m: Optional[float]
    canonical_source: str
    cluster_support: int
    valid_bbox_depth: int

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bbox"] = [round(float(v), 3) for v in self.bbox]
        for key in (
            "raw_center_depth_m",
            "median_bbox_depth_m",
            "mask_depth_m",
            "canonical_depth_m",
        ):
            if out[key] is not None:
                out[key] = round(float(out[key]), 6)
        return out


def _finite_depth(values: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a) & (a >= min_m) & (a <= max_m)]


def canonical_berry_depth(
    depth_m: np.ndarray,
    bbox: BBox,
    *,
    frame_id: str = "",
    stamp_s: float = 0.0,
    stamp_ros: Optional[Tuple[int, int]] = None,
    min_m: float = 0.05,
    max_m: float = 2.5,
    bin_m: float = 0.01,
    cluster_half_width_m: float = 0.018,
    min_support: int = 12,
    min_support_fraction: float = 0.05,
) -> BerryDepthSample:
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    cx = max(0, min(w - 1, int(round(0.5 * (x1 + x2)))))
    cy = max(0, min(h - 1, int(round(0.5 * (y1 + y2)))))

    raw = float(depth_m[cy, cx])
    raw = raw if np.isfinite(raw) and min_m <= raw <= max_m else None
    valid = _finite_depth(depth_m[iy1:iy2, ix1:ix2], min_m, max_m)
    bbox_med = float(np.median(valid)) if valid.size else None
    canonical = None
    support = 0
    source = "no_supported_bbox_cluster"
    if valid.size:
        lo = np.floor(float(valid.min()) / bin_m) * bin_m
        hi = np.ceil(float(valid.max()) / bin_m) * bin_m + bin_m
        counts, edges = np.histogram(valid, bins=np.arange(lo, hi + 0.5 * bin_m, bin_m))
        required = max(int(min_support), int(np.ceil(valid.size * min_support_fraction)))
        for idx, count in enumerate(counts):
            if int(count) < required:
                continue
            center = 0.5 * (float(edges[idx]) + float(edges[idx + 1]))
            cluster = valid[np.abs(valid - center) <= cluster_half_width_m]
            if cluster.size >= required:
                canonical = float(np.median(cluster))
                support = int(cluster.size)
                source = "nearest_supported_bbox_cluster"
                break

    return BerryDepthSample(
        frame_id=str(frame_id),
        stamp_s=float(stamp_s),
        stamp_ros=stamp_ros,
        bbox=tuple(float(v) for v in bbox),
        center_px=(cx, cy),
        raw_center_depth_m=raw,
        median_bbox_depth_m=bbox_med,
        mask_depth_m=None,
        canonical_depth_m=canonical,
        canonical_source=source,
        cluster_support=support,
        valid_bbox_depth=int(valid.size),
    )


def depth_consistency(
    control_depth_m: Optional[float],
    displayed_depth_m: Optional[float],
    *,
    max_relative: float = 0.10,
    max_absolute_m: float = 0.020,
) -> Tuple[bool, str]:
    if control_depth_m is None or displayed_depth_m is None:
        return False, "missing_control_or_display_depth"
    control = float(control_depth_m)
    shown = float(displayed_depth_m)
    delta = abs(control - shown)
    relative = delta / max(abs(control), 1e-9)
    # STOP only when either configured discrepancy threshold is exceeded.
    ok = delta <= max_absolute_m and relative <= max_relative
    return ok, (
        f"delta_m={delta:.6f},relative={relative:.6f},"
        f"limits={max_absolute_m:.3f}m/{max_relative:.0%}"
    )
