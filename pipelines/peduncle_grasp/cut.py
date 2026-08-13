"""Cut point selection along peduncle axis from calyx/base."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from . import CutProposal, PeduncleCandidate

Point2D = Tuple[float, float]


def point_in_poly(x: float, y: float, poly: Sequence[Point2D]) -> bool:
    # ray casting
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def propose_cut(
    cand: PeduncleCandidate,
    calyx: Point2D,
    berry_height: float,
    berry_mask: Optional[np.ndarray],
    cfg_cut: dict,
    *,
    depth: Optional[np.ndarray] = None,
) -> CutProposal:
    """Cut near calyx along axis; clamp to OBB; clearance vs berry mask."""
    base, tip = cand.base, cand.tip
    axis = (tip[0] - base[0], tip[1] - base[1])
    length = math.hypot(*axis) or 1.0
    ux, uy = axis[0] / length, axis[1] / length

    offset_px = cfg_cut.get("offset_px")
    offset_mm = cfg_cut.get("offset_mm")
    if offset_mm is not None and depth is not None:
        # without camera intrinsics, approximate mm→px via berry size proxy is unsafe;
        # mark and fall back to scale. Real mm requires calib (next stage).
        offset = float(cfg_cut.get("offset_berry_scale", 0.22)) * berry_height
        conf = 0.55
        reason = "OFFSET_MM_UNAVAILABLE_NO_CALIB"
    elif offset_px is not None:
        offset = float(offset_px)
        conf = 0.7
        reason = "OFFSET_PX"
    else:
        offset = float(cfg_cut.get("offset_berry_scale", 0.22)) * berry_height
        conf = 0.75
        reason = "OFFSET_BERRY_SCALE"

    offset = max(1.0, min(offset, 0.85 * length))
    # start from base (near calyx), move toward tip
    px = base[0] + ux * offset
    py = base[1] + uy * offset

    # clamp into OBB polygon if outside
    if not point_in_poly(px, py, cand.points):
        # binary search along axis for last inside point
        lo, hi = 0.0, offset
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            qx, qy = base[0] + ux * mid, base[1] + uy * mid
            if point_in_poly(qx, qy, cand.points):
                lo = mid
            else:
                hi = mid
        px, py = base[0] + ux * lo, base[1] + uy * lo
        reason = reason + "+CLAMPED_TO_OBB"

    clearance_r = float(cfg_cut.get("clearance_radius_berry_scale", 0.12)) * berry_height
    clearance_ok = True
    if berry_mask is not None and berry_mask.size > 0:
        h, w = berry_mask.shape[:2]
        yy, xx = np.ogrid[-int(clearance_r) : int(clearance_r) + 1, -int(clearance_r) : int(clearance_r) + 1]
        disk = xx * xx + yy * yy <= clearance_r * clearance_r
        cx, cy = int(round(px)), int(round(py))
        y0, y1 = cy - disk.shape[0] // 2, cy + disk.shape[0] // 2 + 1
        x0, x1 = cx - disk.shape[1] // 2, cx + disk.shape[1] // 2 + 1
        # clip
        my0, my1 = max(0, y0), min(h, y1)
        mx0, mx1 = max(0, x0), min(w, x1)
        if my1 > my0 and mx1 > mx0:
            sub = berry_mask[my0:my1, mx0:mx1] > 0
            # corresponding disk slice
            dy0, dx0 = my0 - y0, mx0 - x0
            dsub = disk[dy0 : dy0 + sub.shape[0], dx0 : dx0 + sub.shape[1]]
            if np.any(sub & dsub):
                clearance_ok = False
                reason = reason + "+CLEARANCE_HITS_BERRY"
                conf *= 0.4

    return CutProposal(
        point_2d=(float(px), float(py)),
        point_3d=None,
        confidence=float(conf if clearance_ok else conf * 0.5),
        clearance_ok=bool(clearance_ok),
        reason=reason,
    )
