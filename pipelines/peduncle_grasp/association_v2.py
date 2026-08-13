"""Association v2: calyx-anchored peduncle scoring (does not replace legacy)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import PeduncleCandidate, PeduncleCandidateFeatures

Point2D = Tuple[float, float]


def obb_axis_ends(points: Sequence[Point2D]) -> Tuple[Point2D, Point2D, float]:
    """Return (end0, end1, length) from opposite edge midpoints (long axis)."""
    if len(points) != 4:
        raise ValueError("need 4 OBB corners")
    mids = []
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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def direction_score(
    berry_center: Point2D,
    calyx: Point2D,
    base: Point2D,
    tip: Point2D,
) -> float:
    """berry → calyx → peduncle continuation. Cross stems score low but not hard-fail alone."""
    v1 = (calyx[0] - berry_center[0], calyx[1] - berry_center[1])
    v2 = (tip[0] - base[0], tip[1] - base[1])
    n1 = math.hypot(*v1) or 1.0
    n2 = math.hypot(*v2) or 1.0
    u1 = (v1[0] / n1, v1[1] / n1)
    u2 = (v2[0] / n2, v2[1] / n2)
    cos = u1[0] * u2[0] + u1[1] * u2[1]
    return _clamp01(0.5 * (cos + 1.0))  # map [-1,1] → [0,1]


def contact_score(
    base: Point2D,
    calyx: Point2D,
    berry_height: float,
    calyx_mask: Optional[np.ndarray],
) -> Optional[float]:
    """Proximity/contact with calyx mask or point. None if no signal beyond distance (always have distance)."""
    d = math.hypot(base[0] - calyx[0], base[1] - calyx[1])
    # soft contact from point distance
    s = math.exp(-d / max(1.0, 0.20 * berry_height))
    if calyx_mask is not None and calyx_mask.size > 0:
        # morphological contact: distance transform from base to mask
        m = (calyx_mask > 0).astype(np.uint8)
        if m.any():
            # dilate calyx
            k = max(1, int(0.06 * berry_height))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            dil = cv2.dilate(m, kernel)
            bx, by = int(round(base[0])), int(round(base[1]))
            h, w = dil.shape[:2]
            if 0 <= bx < w and 0 <= by < h and dil[by, bx] > 0:
                return max(s, 0.95)
            # nearest mask pixel
            ys, xs = np.where(m > 0)
            if xs.size:
                dd = np.hypot(xs - base[0], ys - base[1]).min()
                s2 = math.exp(-float(dd) / max(1.0, 0.20 * berry_height))
                return max(s, s2)
    return s


def poly_aabb(pts: Sequence[Point2D]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_iou(a, b) -> float:
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


def score_candidate(
    points: Sequence[Point2D],
    conf: float,
    berry_center: Point2D,
    berry_bbox: Tuple[float, float, float, float],
    berry_height: float,
    calyx: Point2D,
    calyx_mask: Optional[np.ndarray],
    cfg_assoc: dict,
    *,
    mode: str = "close",
) -> Optional[PeduncleCandidate]:
    end0, end1, length = obb_axis_ends(points)
    base, tip, dist = choose_base_tip(end0, end1, calyx)
    d_norm = dist / max(1.0, berry_height)
    tau = float(cfg_assoc["close_tau"] if mode == "close" else cfg_assoc["far_tau"])
    if d_norm > tau:
        return None
    len_norm = length / max(1.0, berry_height)
    if len_norm > float(cfg_assoc.get("max_obb_length_berry_scale", 0.7)):
        return None
    if float(conf) < float(cfg_assoc.get("min_model_conf", 0.15)):
        return None

    iou = aabb_iou(poly_aabb(points), berry_bbox)
    if iou > float(cfg_assoc.get("max_berry_iou", 0.4)):
        return None

    # into-berry hard check
    out = (calyx[0] - berry_center[0], calyx[1] - berry_center[1])
    on = math.hypot(*out) or 1.0
    ou = (out[0] / on, out[1] / on)
    axis = (tip[0] - base[0], tip[1] - base[1])
    an = math.hypot(*axis) or 1.0
    au = (axis[0] / an, axis[1] / an)
    into = au[0] * (-ou[0]) + au[1] * (-ou[1])
    if into > float(cfg_assoc.get("into_berry_max", 0.45)):
        return None

    dir_s = direction_score(berry_center, calyx, base, tip)
    contact = contact_score(base, calyx, berry_height, calyx_mask)
    prox = math.exp(-d_norm / max(1e-6, tau))
    conf_s = _clamp01(float(conf))
    crossing = iou  # soft penalty for overlapping berry heavily

    feats = PeduncleCandidateFeatures(
        model_confidence=conf_s,
        base_to_calyx_distance=float(dist),
        normalized_base_distance=float(d_norm),
        calyx_contact=contact,
        direction_consistency=dir_s,
        continuity=None,  # unavailable v1
        depth_consistency=None,
        leaf_occlusion=None,
        crossing_penalty=float(crossing),
        obb_length_norm=float(len_norm),
    )

    # weighted score with renormalization over available terms
    terms = []
    weights = []
    mapping = [
        ("w_confidence", conf_s),
        ("w_proximity", prox),
        ("w_contact", contact),
        ("w_direction", dir_s),
        ("w_continuity", None),
        ("w_depth", None),
    ]
    for key, val in mapping:
        w = float(cfg_assoc.get(key, 0.0))
        if w <= 0 or val is None:
            continue
        terms.append(float(val))
        weights.append(w)
    # penalties
    pen = 0.0
    w_cross = float(cfg_assoc.get("w_crossing", 0.0))
    if w_cross > 0 and feats.crossing_penalty is not None:
        pen += w_cross * float(feats.crossing_penalty)
    w_occ = float(cfg_assoc.get("w_occlusion", 0.0))
    if w_occ > 0 and feats.leaf_occlusion is not None:
        pen += w_occ * float(feats.leaf_occlusion)

    wsum = sum(weights) or 1.0
    score = sum(t * w for t, w in zip(terms, weights)) / wsum - pen
    if score < float(cfg_assoc.get("min_score", 0.28)):
        return None

    return PeduncleCandidate(
        points=list(points),
        conf=float(conf),
        base=base,
        tip=tip,
        features=feats,
        score=float(score),
        available_weight_sum=float(wsum),
    )


def associate_candidates(
    candidates: Sequence[Tuple[List[Point2D], float]],
    berry_center: Point2D,
    berry_bbox: Tuple[float, float, float, float],
    berry_height: float,
    calyx: Point2D,
    calyx_mask: Optional[np.ndarray],
    cfg_assoc: dict,
    *,
    mode: str = "close",
) -> Tuple[List[PeduncleCandidate], Optional[PeduncleCandidate]]:
    scored: List[PeduncleCandidate] = []
    for pts, conf in candidates:
        if len(pts) != 4:
            continue
        c = score_candidate(
            pts,
            conf,
            berry_center,
            berry_bbox,
            berry_height,
            calyx,
            calyx_mask,
            cfg_assoc,
            mode=mode,
        )
        if c is not None:
            scored.append(c)
    scored.sort(key=lambda c: c.score, reverse=True)
    best = scored[0] if scored else None
    return scored, best
