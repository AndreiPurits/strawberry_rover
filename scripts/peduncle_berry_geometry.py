#!/usr/bin/env python3
"""Berry-anchored geometry for peduncle OBB train/infer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Canonical crop layout (train == infer)
OUT_W = 640
OUT_H = 640
BERRY_CX_FRAC = 0.50
BERRY_CY_FRAC = 0.62
BERRY_H_FRAC = 0.38

# Inference association
ASSOC_TAU = 0.65  # mask-calyx tip vs peduncle base
CUT_ALPHA = 0.30
MIN_PED_AREA = 16.0


@dataclass
class BerryBox:
    x1: float
    y1: float
    x2: float
    y2: float
    calyx_xy: Optional[Tuple[float, float]] = None  # mask tip if known
    source: str = "aabb"  # aabb | mask

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def calyx(self) -> Tuple[float, float]:
        """Mask tip if set, else top-center of AABB."""
        if self.calyx_xy is not None:
            return (float(self.calyx_xy[0]), float(self.calyx_xy[1]))
        return (self.cx, self.y1)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class CanonTransform:
    scale: float
    tx: float
    ty: float
    out_w: int
    out_h: int

    def M(self) -> np.ndarray:
        return np.array([[self.scale, 0.0, self.tx], [0.0, self.scale, self.ty]], dtype=np.float32)

    def map_xy(self, x: float, y: float) -> Tuple[float, float]:
        return (self.scale * x + self.tx, self.scale * y + self.ty)

    def map_pts(self, pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.map_xy(float(x), float(y)) for x, y in pts]

    def inv_map_xy(self, x: float, y: float) -> Tuple[float, float]:
        s = float(self.scale) if abs(self.scale) > 1e-9 else 1.0
        return ((x - self.tx) / s, (y - self.ty) / s)

    def inv_map_pts(self, pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.inv_map_xy(float(x), float(y)) for x, y in pts]


def parse_berry_yolo_aabb(label_file, w: int, h: int) -> Optional[BerryBox]:
    if not label_file.is_file():
        return None
    best: Optional[BerryBox] = None
    best_area = -1.0
    for raw in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 5:
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, bw, bh = map(float, toks[1:5])
        except Exception:
            continue
        if cid != 0:
            continue
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = BerryBox(x1, y1, x2, y2)
    return best


def parse_peduncle_obb(label_file, w: int, h: int) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    if not label_file.is_file():
        return out
    for raw in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 9:
            continue
        try:
            vals = list(map(float, toks[1:9]))
        except Exception:
            continue
        pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]
        out.append(pts)
    return out


def make_canon_transform(
    berry: BerryBox,
    *,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    berry_h_frac: float = BERRY_H_FRAC,
    berry_cx_frac: float = BERRY_CX_FRAC,
    berry_cy_frac: float = BERRY_CY_FRAC,
) -> CanonTransform:
    scale = (berry_h_frac * out_h) / berry.h
    tx = berry_cx_frac * out_w - scale * berry.cx
    ty = berry_cy_frac * out_h - scale * berry.cy
    return CanonTransform(scale=float(scale), tx=float(tx), ty=float(ty), out_w=out_w, out_h=out_h)


def warp_bgr(bgr: np.ndarray, tfm: CanonTransform) -> np.ndarray:
    return cv2.warpAffine(
        bgr,
        tfm.M(),
        (tfm.out_w, tfm.out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def transform_berry(berry: BerryBox, tfm: CanonTransform) -> BerryBox:
    x1, y1 = tfm.map_xy(berry.x1, berry.y1)
    x2, y2 = tfm.map_xy(berry.x2, berry.y2)
    calyx = None
    if berry.calyx_xy is not None:
        calyx = tfm.map_xy(berry.calyx_xy[0], berry.calyx_xy[1])
    return BerryBox(
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
        calyx_xy=calyx,
        source=berry.source,
    )


def berry_from_mask(
    mask_u8: np.ndarray,
    *,
    aabb_hint: Optional[BerryBox] = None,
    min_pixels: int = 80,
) -> Optional[BerryBox]:
    """Build berry box + calyx tip from segmentation mask (optionally gated by AABB)."""
    if mask_u8 is None or mask_u8.size == 0:
        return None
    m = mask_u8 > 0
    if aabb_hint is not None:
        H, W = m.shape[:2]
        x1 = max(0, int(aabb_hint.x1) - int(0.08 * aabb_hint.w))
        y1 = max(0, int(aabb_hint.y1) - int(0.12 * aabb_hint.h))
        x2 = min(W, int(aabb_hint.x2) + int(0.08 * aabb_hint.w))
        y2 = min(H, int(aabb_hint.y2) + int(0.08 * aabb_hint.h))
        gate = np.zeros_like(m, dtype=bool)
        gate[y1:y2, x1:x2] = True
        m = m & gate
    ys, xs = np.where(m)
    if xs.size < min_pixels:
        return None
    bx1, bx2 = float(xs.min()), float(xs.max())
    by1, by2 = float(ys.min()), float(ys.max())
    bh = max(1.0, by2 - by1)
    # calyx tip: mean of topmost mask band (~exit of stem from berry)
    top_y = float(ys.min())
    band = ys <= top_y + max(1.0, 0.08 * bh)
    calyx = (float(xs[band].mean()), float(ys[band].mean()))
    return BerryBox(bx1, by1, bx2, by2, calyx_xy=calyx, source="mask")


def warp_mask(mask_u8: np.ndarray, tfm: CanonTransform) -> np.ndarray:
    return cv2.warpAffine(
        mask_u8,
        tfm.M(),
        (tfm.out_w, tfm.out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def poly_area(pts: Sequence[Tuple[float, float]]) -> float:
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(arr)))


def clip_poly_to_image(
    pts: Sequence[Tuple[float, float]], w: int, h: int
) -> Optional[List[Tuple[float, float]]]:
    """Keep OBB if mostly inside; clamp corners lightly."""
    clamped = [(max(0.0, min(float(w - 1), x)), max(0.0, min(float(h - 1), y))) for x, y in pts]
    # reject if too many corners were far outside before clamp
    n_out = 0
    for (x, y), (xc, yc) in zip(pts, clamped):
        if abs(x - xc) > 0.05 * w or abs(y - yc) > 0.05 * h:
            n_out += 1
    if n_out >= 3:
        return None
    if poly_area(clamped) < MIN_PED_AREA:
        return None
    return clamped


def obb_to_yolo_line(pts: Sequence[Tuple[float, float]], w: int, h: int, cls: int = 0) -> Optional[str]:
    if len(pts) != 4:
        return None
    coords = []
    for x, y in pts:
        coords.append(max(0.0, min(1.0, float(x) / float(w))))
        coords.append(max(0.0, min(1.0, float(y) / float(h))))
    if poly_area([(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)]) < MIN_PED_AREA:
        return None
    return f"{cls} " + " ".join(f"{c:.6f}" for c in coords)


def berry_to_yolo_line(berry: BerryBox, w: int, h: int, cls: int = 0) -> str:
    xc = berry.cx / float(w)
    yc = berry.cy / float(h)
    bw = berry.w / float(w)
    bh = berry.h / float(h)
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def aabb_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def poly_aabb(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def poly_centroid(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class PeduncleAssoc:
    points: List[Tuple[float, float]]
    conf: float
    calyx_end: Tuple[float, float]
    tip: Tuple[float, float]
    cut_point: Tuple[float, float]
    dist_norm: float
    score: float
    alignment: float


def associate_peduncle(
    candidates: Sequence[Tuple[List[Tuple[float, float]], float]],
    berry: BerryBox,
    *,
    tau: float = ASSOC_TAU,
    cut_alpha: float = CUT_ALPHA,
    max_berry_iou: float = 0.35,
) -> Optional[PeduncleAssoc]:
    """Pick best peduncle OBB relative to berry calyx / geometry."""
    C = berry.calyx
    s = berry.h
    berry_c = (berry.cx, berry.cy)
    # preferred outward direction: from berry center toward calyx (up / out of berry)
    out_vec = (C[0] - berry_c[0], C[1] - berry_c[1])
    out_norm = math.hypot(out_vec[0], out_vec[1]) or 1.0
    out_u = (out_vec[0] / out_norm, out_vec[1] / out_norm)

    best: Optional[PeduncleAssoc] = None
    for pts, conf in candidates:
        if len(pts) != 4:
            continue
        # calyx_end = corner/endpoint of long axis nearest to C
        # approximate long axis via PCA or max pairwise distance
        best_pair = (0, 1)
        best_d = -1.0
        for i in range(4):
            for j in range(i + 1, 4):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d > best_d:
                    best_d = d
                    best_pair = (i, j)
        a, b = pts[best_pair[0]], pts[best_pair[1]]
        # use midpoints of short sides for more stable ends: take the two vertices of long edges
        # simpler: ends are the two points of the longest pair; refine with edge midpoints of OBB
        # For a thin OBB, longest-pair vertices are opposite corners — bad.
        # Better: edge midpoints, pick pair of opposite edge midpoints with max separation.
        mids = []
        for i in range(4):
            j = (i + 1) % 4
            mids.append(((pts[i][0] + pts[j][0]) * 0.5, (pts[i][1] + pts[j][1]) * 0.5))
        end_i, end_j = 0, 2
        best_mid_d = -1.0
        for i in range(4):
            j = (i + 2) % 4
            d = math.hypot(mids[i][0] - mids[j][0], mids[i][1] - mids[j][1])
            if d > best_mid_d:
                best_mid_d = d
                end_i, end_j = i, j
        e0, e1 = mids[end_i], mids[end_j]
        d0 = math.hypot(e0[0] - C[0], e0[1] - C[1])
        d1 = math.hypot(e1[0] - C[0], e1[1] - C[1])
        if d0 <= d1:
            calyx_end, tip = e0, e1
            dist = d0
        else:
            calyx_end, tip = e1, e0
            dist = d1
        dist_norm = dist / s
        if dist_norm > tau:
            continue

        cent = poly_centroid(pts)
        # peduncle centroid should be above calyx (smaller y) or at least not deep inside berry
        if cent[1] > berry.cy + 0.15 * s:
            continue
        iou = aabb_iou(poly_aabb(pts), berry.as_xyxy())
        if iou > max_berry_iou:
            continue

        axis = (tip[0] - calyx_end[0], tip[1] - calyx_end[1])
        axis_n = math.hypot(axis[0], axis[1]) or 1.0
        axis_u = (axis[0] / axis_n, axis[1] / axis_n)
        alignment = axis_u[0] * out_u[0] + axis_u[1] * out_u[1]
        # allow sideways stems: require not pointing strongly into berry
        into_berry = axis_u[0] * (-out_u[0]) + axis_u[1] * (-out_u[1])
        if into_berry > 0.55:
            continue
        cos_pos = max(0.0, alignment)
        score = float(conf) * math.exp(-dist_norm) * (0.35 + 0.65 * cos_pos)
        cut = (
            (1.0 - cut_alpha) * calyx_end[0] + cut_alpha * tip[0],
            (1.0 - cut_alpha) * calyx_end[1] + cut_alpha * tip[1],
        )
        cand = PeduncleAssoc(
            points=list(pts),
            conf=float(conf),
            calyx_end=calyx_end,
            tip=tip,
            cut_point=cut,
            dist_norm=dist_norm,
            score=score,
            alignment=alignment,
        )
        if best is None or cand.score > best.score:
            best = cand
    return best
