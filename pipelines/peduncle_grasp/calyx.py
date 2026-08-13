"""Calyx estimation: model adapter stub + geometry fallback (PCA tip)."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from . import CalyxEstimate

Point2D = Tuple[float, float]


class CalyxDetector(Protocol):
    def estimate(
        self,
        bgr: np.ndarray,
        berry_bbox: Tuple[float, float, float, float],
        berry_mask: Optional[np.ndarray],
    ) -> Optional[CalyxEstimate]:
        ...


class NullCalyxDetector:
    """Placeholder until a trained calyx model exists."""

    def estimate(self, bgr, berry_bbox, berry_mask) -> Optional[CalyxEstimate]:
        return None


def estimate_calyx_geometry(
    berry_bbox: Tuple[float, float, float, float],
    berry_mask: Optional[np.ndarray],
    *,
    tip_band_frac: float = 0.12,
    geometry_confidence: float = 0.35,
) -> CalyxEstimate:
    """Estimate calyx without assuming image 'up' = stem.

    Uses PCA major axis of the berry mask (or bbox ellipse) and picks the
    narrower/end that is farther from the berry mass centroid along that axis
    as the stem-exit proxy. Falls back carefully when mask is missing.
    """
    x1, y1, x2, y2 = berry_bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = max(1.0, y2 - y1)
    bw = max(1.0, x2 - x1)

    if berry_mask is not None and berry_mask.size > 0:
        m = berry_mask > 0
        ys, xs = np.where(m)
        if xs.size >= 40:
            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            mean = pts.mean(axis=0)
            centered = pts - mean
            # PCA
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, int(np.argmax(eigvals))]
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            proj = centered @ axis
            p_min, p_max = float(proj.min()), float(proj.max())
            # ends
            end_a = mean + axis * p_min
            end_b = mean + axis * p_max

            def local_width(sign: float) -> float:
                span = max(p_max - p_min, 1.0)
                if sign < 0:
                    near = proj <= p_min + tip_band_frac * span
                else:
                    near = proj >= p_max - tip_band_frac * span
                if not np.any(near):
                    return float("inf")
                orth = centered[near] - np.outer((centered[near] @ axis), axis)
                return float(np.linalg.norm(orth, axis=1).mean())

            wa = local_width(-1.0)
            wb = local_width(+1.0)
            tip = end_a if wa <= wb else end_b
            # small calyx mask band around tip
            calyx_mask = np.zeros_like(berry_mask, dtype=np.uint8)
            r = max(2, int(0.08 * max(bh, bw)))
            cv2.circle(calyx_mask, (int(round(tip[0])), int(round(tip[1]))), r, 255, -1)
            calyx_mask = cv2.bitwise_and(calyx_mask, (berry_mask > 0).astype(np.uint8) * 255)
            return CalyxEstimate(
                point_2d=(float(tip[0]), float(tip[1])),
                confidence=float(geometry_confidence),
                source="geometry",
                mask=calyx_mask,
            )

    # AABB fallback: prefer top of box but mark low confidence / unknown orientation risk
    tip = (cx, y1)
    return CalyxEstimate(
        point_2d=(float(tip[0]), float(tip[1])),
        confidence=float(geometry_confidence) * 0.7,
        source="geometry",
        mask=None,
    )


def resolve_calyx(
    bgr: np.ndarray,
    berry_bbox: Tuple[float, float, float, float],
    berry_mask: Optional[np.ndarray],
    cfg: dict,
    detector: Optional[CalyxDetector] = None,
) -> CalyxEstimate:
    calyx_cfg = cfg.get("calyx") or {}
    if calyx_cfg.get("use_model") and detector is not None:
        est = detector.estimate(bgr, berry_bbox, berry_mask)
        if est is not None and est.confidence > 0:
            return est
    if calyx_cfg.get("geometry_fallback", True):
        return estimate_calyx_geometry(
            berry_bbox,
            berry_mask,
            tip_band_frac=float(calyx_cfg.get("tip_band_frac", 0.12)),
            geometry_confidence=float(calyx_cfg.get("geometry_confidence", 0.35)),
        )
    cx = 0.5 * (berry_bbox[0] + berry_bbox[2])
    cy = 0.5 * (berry_bbox[1] + berry_bbox[3])
    return CalyxEstimate(point_2d=(cx, cy), confidence=0.0, source="unknown", mask=None)
