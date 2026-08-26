"""Overlay drawing for peduncle v3 perception."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .association import AssociationOutcome

Point2D = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def draw_peduncle_v3_overlay(
    bgr: np.ndarray,
    *,
    berry_bbox: Optional[BBox] = None,
    calyx_xy: Optional[Point2D] = None,
    calyx_conf: float = 0.0,
    association: Optional[AssociationOutcome] = None,
    state: str = "",
    roi: Optional[Tuple[int, int, int, int]] = None,
    grasp_xy: Optional[Point2D] = None,
) -> np.ndarray:
    out = bgr.copy()
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1)
    if berry_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in berry_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, "target berry", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    if calyx_xy is not None:
        cx, cy = int(round(calyx_xy[0])), int(round(calyx_xy[1]))
        cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)
        cv2.circle(out, (cx, cy), 10, (0, 180, 255), 2)
        cv2.putText(
            out,
            f"calyx {calyx_conf:.2f}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    decision = association.decision if association else "?"
    if association is not None:
        for cand in association.candidates:
            pts = np.array([[int(round(x)), int(round(y))] for x, y in cand.polygon], dtype=np.int32)
            is_tgt = bool(cand.selected and association.decision == "TARGET")
            color = (0, 0, 255) if is_tgt else (255, 160, 0)
            thickness = 3 if is_tgt else 1
            cv2.polylines(out, [pts], True, color, thickness)
            label = f"{'TARGET' if is_tgt else 'cand'} {cand.score:.2f}/{cand.det_conf:.2f}"
            bx, by = int(round(cand.base[0])), int(round(cand.base[1]))
            cv2.putText(out, label, (bx, max(14, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if is_tgt:
                wx, wy = int(round(cand.work_point[0])), int(round(cand.work_point[1]))
                cv2.drawMarker(out, (wx, wy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    if grasp_xy is not None:
        gx, gy = int(round(grasp_xy[0])), int(round(grasp_xy[1]))
        cv2.drawMarker(out, (gx, gy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(out, "grasp", (gx + 8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    banner = f"{state} | {decision}"
    if association is not None:
        banner += f" | {association.reason} thr={association.threshold:.2f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def overlay_from_result_dict(bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Best-effort redraw from serialized pipeline result."""
    berry = result.get("berry_bbox")
    calyx = result.get("calyx") or {}
    calyx_xy = None
    if calyx.get("x") is not None and calyx.get("y") is not None:
        calyx_xy = (float(calyx["x"]), float(calyx["y"]))
    # Rebuild a lightweight AssociationOutcome-like view via draw args only
    assoc_raw = result.get("association") or {}
    from .association import AssociationOutcome, StemCandidate

    cands = []
    for c in assoc_raw.get("candidates") or []:
        cands.append(
            StemCandidate(
                polygon=[(float(p[0]), float(p[1])) for p in c.get("polygon") or []],
                det_conf=float(c.get("det_conf") or 0),
                base=tuple(c.get("base") or (0.0, 0.0)),  # type: ignore[arg-type]
                tip=tuple(c.get("tip") or (0.0, 0.0)),  # type: ignore[arg-type]
                work_point=tuple((c.get("work_point") or c.get("base") or (0.0, 0.0))),  # type: ignore[arg-type]
                features=dict(c.get("features") or {}),
                score=float(c.get("score") or 0),
                selected=bool(c.get("selected")),
            )
        )
    selected = None
    sel = assoc_raw.get("selected")
    if sel:
        selected = StemCandidate(
            polygon=[(float(p[0]), float(p[1])) for p in sel.get("polygon") or []],
            det_conf=float(sel.get("det_conf") or 0),
            base=tuple(sel.get("base") or (0.0, 0.0)),  # type: ignore[arg-type]
            tip=tuple(sel.get("tip") or (0.0, 0.0)),  # type: ignore[arg-type]
            work_point=tuple(sel.get("work_point") or (0.0, 0.0)),  # type: ignore[arg-type]
            features=dict(sel.get("features") or {}),
            score=float(sel.get("score") or 0),
            selected=True,
        )
    outcome = AssociationOutcome(
        decision=str(assoc_raw.get("decision") or "NO_ASSOC"),
        reason=str(assoc_raw.get("reason") or ""),
        threshold=float(assoc_raw.get("threshold") or 0.1),
        candidates=cands,
        selected=selected,
    )
    roi = result.get("roi")
    return draw_peduncle_v3_overlay(
        bgr,
        berry_bbox=tuple(berry) if berry else None,  # type: ignore[arg-type]
        calyx_xy=calyx_xy,
        calyx_conf=float(calyx.get("conf") or 0),
        association=outcome,
        state=str(result.get("state") or ""),
        roi=tuple(roi) if roi else None,  # type: ignore[arg-type]
    )
