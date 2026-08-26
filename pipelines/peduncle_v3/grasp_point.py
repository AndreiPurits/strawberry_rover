"""Grasp point on target stem centerline (reuse peduncle_grasp cut geometry, grasp margins)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry import work_point_along_stem

Point2D = Tuple[float, float]


def point_in_poly(x: float, y: float, poly: Sequence[Point2D]) -> bool:
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


@dataclass
class GraspPointResult:
    ok: bool
    reason: str
    point_2d: Optional[Point2D] = None
    base: Optional[Point2D] = None
    tip: Optional[Point2D] = None
    offset_px: float = 0.0
    length_px: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "point_2d": None if self.point_2d is None else [round(self.point_2d[0], 1), round(self.point_2d[1], 1)],
            "base": None if self.base is None else [round(self.base[0], 1), round(self.base[1], 1)],
            "tip": None if self.tip is None else [round(self.tip[0], 1), round(self.tip[1], 1)],
            "offset_px": round(self.offset_px, 1),
            "length_px": round(self.length_px, 1),
        }


def propose_grasp_point(
    polygon: Sequence[Point2D],
    base: Point2D,
    tip: Point2D,
    calyx: Point2D,
    berry_height: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> GraspPointResult:
    """Pick grasp on stem axis: above calyx, clear of attachment and tip end.

    Reuses peduncle_grasp axis / work_point idea with grasp-specific margins.
    Does NOT invent stem beyond the visible OBB.
    """
    cfg = cfg or {}
    axis = (tip[0] - base[0], tip[1] - base[1])
    length = math.hypot(*axis)
    if length < float(cfg.get("min_stem_length_px", 12.0)):
        return GraspPointResult(ok=False, reason="NO_GRASP_POINT_STEM_TOO_SHORT", base=base, tip=tip, length_px=length)

    ux, uy = axis[0] / length, axis[1] / length
    # Margins: prefer berry-relative, but never exceed stem geometry on close-up crops.
    min_from_calyx_bh = float(cfg.get("min_offset_berry_scale", 0.28)) * max(1.0, berry_height)
    min_from_calyx = min(
        max(min_from_calyx_bh, float(cfg.get("min_offset_px", 8.0))),
        float(cfg.get("max_min_offset_frac", 0.35)) * length,
    )
    tip_margin = max(
        float(cfg.get("tip_margin_scale", 0.18)) * length,
        float(cfg.get("tip_margin_px", 6.0)),
    )
    tip_margin = min(tip_margin, 0.40 * length)
    prefer_bh = float(cfg.get("offset_berry_scale", 0.32)) * max(1.0, berry_height)
    prefer = min(prefer_bh, float(cfg.get("prefer_frac", 0.45)) * length)
    max_offset = max(1.0, length - tip_margin)
    if min_from_calyx >= max_offset:
        # Last resort: midpoint of visible safe segment if any length remains.
        if length < float(cfg.get("min_stem_length_px", 12.0)) * 1.2:
            return GraspPointResult(
                ok=False,
                reason="NO_GRASP_POINT_NO_SAFE_SEGMENT",
                base=base,
                tip=tip,
                length_px=length,
            )
        min_from_calyx = 0.25 * length
        max_offset = 0.75 * length
        prefer = 0.45 * length
    offset = min(max(prefer, min_from_calyx), max_offset)
    px = base[0] + ux * offset
    py = base[1] + uy * offset

    # Clamp into OBB if needed (same idea as peduncle_grasp cut).
    if len(polygon) == 4 and not point_in_poly(px, py, polygon):
        lo, hi = min_from_calyx, offset
        found = False
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            qx, qy = base[0] + ux * mid, base[1] + uy * mid
            if point_in_poly(qx, qy, polygon):
                lo = mid
                found = True
            else:
                hi = mid
        if not found:
            return GraspPointResult(
                ok=False,
                reason="NO_GRASP_POINT_OUTSIDE_OBB",
                base=base,
                tip=tip,
                length_px=length,
            )
        px, py = base[0] + ux * lo, base[1] + uy * lo
        offset = lo

    # Reject if still too close to calyx (would grab berry/calyx).
    d_calyx = math.hypot(px - calyx[0], py - calyx[1])
    if d_calyx < float(cfg.get("min_calyx_clearance_berry_scale", 0.18)) * max(1.0, berry_height):
        return GraspPointResult(
            ok=False,
            reason="NO_GRASP_POINT_TOO_NEAR_CALYX",
            base=base,
            tip=tip,
            length_px=length,
            offset_px=offset,
        )

    return GraspPointResult(
        ok=True,
        reason="OK",
        point_2d=(float(px), float(py)),
        base=base,
        tip=tip,
        offset_px=float(offset),
        length_px=float(length),
    )


def fallback_work_point(base: Point2D, tip: Point2D, berry_height: float, scale: float = 0.22) -> Point2D:
    return work_point_along_stem(base, tip, berry_height, offset_berry_scale=scale)
