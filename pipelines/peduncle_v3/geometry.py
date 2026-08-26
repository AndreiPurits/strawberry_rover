"""OBB / calyx geometry helpers for peduncle v3 features."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def poly_aabb(pts: Sequence[Point2D]) -> BBox:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + bb - inter + 1e-9))


def obb_axis_ends(points: Sequence[Point2D]) -> Tuple[Point2D, Point2D, float]:
    """Opposite edge midpoints along the long axis → (end0, end1, length)."""
    if len(points) != 4:
        raise ValueError("need 4 OBB corners")
    mids: List[Point2D] = []
    for i in range(4):
        j = (i + 1) % 4
        mids.append(((points[i][0] + points[j][0]) * 0.5, (points[i][1] + points[j][1]) * 0.5))
    best_i, best_j, best_d = 0, 2, -1.0
    for i in range(4):
        j = (i + 2) % 4
        d = math.hypot(mids[i][0] - mids[j][0], mids[i][1] - mids[j][1])
        if d > best_d:
            best_d = d
            best_i, best_j = i, j
    return mids[best_i], mids[best_j], float(best_d)


def choose_base_tip(end0: Point2D, end1: Point2D, calyx: Point2D) -> Tuple[Point2D, Point2D, float]:
    d0 = math.hypot(end0[0] - calyx[0], end0[1] - calyx[1])
    d1 = math.hypot(end1[0] - calyx[0], end1[1] - calyx[1])
    if d0 <= d1:
        return end0, end1, d0
    return end1, end0, d1


def direction_score(berry_center: Point2D, calyx: Point2D, base: Point2D, tip: Point2D) -> float:
    v1 = (calyx[0] - berry_center[0], calyx[1] - berry_center[1])
    v2 = (tip[0] - base[0], tip[1] - base[1])
    n1 = math.hypot(*v1) or 1.0
    n2 = math.hypot(*v2) or 1.0
    u1 = (v1[0] / n1, v1[1] / n1)
    u2 = (v2[0] / n2, v2[1] / n2)
    cos = u1[0] * u2[0] + u1[1] * u2[1]
    return clamp01(0.5 * (cos + 1.0))


def contact_score(base: Point2D, calyx: Point2D, berry_height: float) -> float:
    d = math.hypot(base[0] - calyx[0], base[1] - calyx[1])
    return float(math.exp(-d / max(1.0, 0.20 * berry_height)))


def work_point_along_stem(
    base: Point2D,
    tip: Point2D,
    berry_height: float,
    *,
    offset_berry_scale: float = 0.22,
) -> Point2D:
    """Safe standoff / cut proxy: from base toward tip (peduncle_grasp cut logic)."""
    axis = (tip[0] - base[0], tip[1] - base[1])
    length = math.hypot(*axis) or 1.0
    ux, uy = axis[0] / length, axis[1] / length
    offset = max(1.0, min(float(offset_berry_scale) * berry_height, 0.85 * length))
    return (base[0] + ux * offset, base[1] + uy * offset)


def build_focus_roi(
    berry_bbox: BBox,
    image_wh: Tuple[int, int],
    *,
    width_scale: float = 1.6,
    height_above_scale: float = 0.85,
    height_below_scale: float = 0.45,
    min_size_px: int = 160,
    max_size_px: int = 640,
    pad_px: int = 8,
) -> Tuple[int, int, int, int]:
    """Crop around berry + stem zone above (full-frame coords)."""
    x1, y1, x2, y2 = berry_bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    calyx_y = y1 + 0.12 * bh
    half_w = 0.5 * width_scale * bw
    rx1 = cx - half_w
    rx2 = cx + half_w
    ry1 = calyx_y - height_above_scale * bh
    ry2 = calyx_y + height_below_scale * bh
    side = min(float(max_size_px), max(float(min_size_px), max(rx2 - rx1, ry2 - ry1)))
    rcx = 0.5 * (rx1 + rx2)
    rcy = 0.5 * (ry1 + ry2)
    img_w, img_h = image_wh
    x1i = int(max(0, min(img_w - 1, round(rcx - 0.5 * side) - pad_px)))
    y1i = int(max(0, min(img_h - 1, round(rcy - 0.5 * side) - pad_px)))
    x2i = int(max(x1i + 1, min(img_w, round(rcx + 0.5 * side) + pad_px)))
    y2i = int(max(y1i + 1, min(img_h, round(rcy + 0.5 * side) + pad_px)))
    return x1i, y1i, x2i, y2i
