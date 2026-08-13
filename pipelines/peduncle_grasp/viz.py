"""Debug visualization for peduncle grasp v1."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import PerceptionResult
from .roi import CloseupROI


def draw_debug(
    bgr: np.ndarray,
    result: PerceptionResult,
    roi: Optional[CloseupROI] = None,
    raw_cands=None,
) -> np.ndarray:
    vis = bgr.copy()
    berry = result.berry
    if berry is None:
        cv2.putText(vis, result.reason, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    x1, y1, x2, y2 = map(int, berry.bbox)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(vis, f"id={berry.track_id}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    if berry.mask is not None:
        overlay = vis.copy()
        overlay[berry.mask > 0] = (0.65 * overlay[berry.mask > 0] + np.array([0, 90, 0])).astype(np.uint8)
        vis = overlay

    if berry.calyx is not None:
        cx, cy = map(int, berry.calyx.point_2d)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"calyx {berry.calyx.source} {berry.calyx.confidence:.2f}",
            (cx + 6, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        if berry.calyx.mask is not None:
            cnts, _ = cv2.findContours(
                (berry.calyx.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, cnts, -1, (0, 200, 255), 1)

    if roi is not None:
        cv2.rectangle(vis, (roi.x1, roi.y1), (roi.x2, roi.y2), (255, 200, 0), 1)

    if raw_cands:
        for pts, conf in raw_cands:
            poly = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
            cv2.polylines(vis, [poly], True, (90, 90, 90), 1)

    for i, c in enumerate(berry.peduncle_candidates):
        color = (0, 140, 255) if c is berry.selected_peduncle else (120, 120, 120)
        thick = 2 if c is berry.selected_peduncle else 1
        poly = np.array([[int(p[0]), int(p[1])] for p in c.points], dtype=np.int32)
        cv2.polylines(vis, [poly], True, color, thick)
        cv2.circle(vis, (int(c.base[0]), int(c.base[1])), 4, (255, 180, 0), -1)
        cv2.putText(
            vis,
            f"#{i} s={c.score:.2f} d={c.features.normalized_base_distance:.2f}",
            (int(c.base[0]), int(c.base[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    if berry.cut is not None:
        px, py = map(int, berry.cut.point_2d)
        col = (0, 0, 255) if berry.cut.clearance_ok else (0, 128, 255)
        cv2.circle(vis, (px, py), 7, col, -1)

    status = f"{result.state.value} | {result.status.value} | {result.reason}"
    cv2.putText(vis, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis
