"""Draw RoArm approach target overlay on stereo BGR frames (hub MJPEG + preview)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _scale_xy(
    px: float,
    py: float,
    *,
    src_w: float,
    src_h: float,
    dst_w: int,
    dst_h: int,
) -> Tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return int(px), int(py)
    return (
        int(round(px * dst_w / src_w)),
        int(round(py * dst_h / src_h)),
    )


def draw_approach_overlay_bgr(bgr: np.ndarray, approach: Optional[Dict[str, Any]]) -> np.ndarray:
    """Return BGR image with target crosshair, anchor circle, and HUD."""
    if bgr is None or bgr.size == 0:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    src_w = float(approach.get("image_w") or w) if approach else float(w)
    src_h = float(approach.get("image_h") or h) if approach else float(h)

    cx, cy = _scale_xy(src_w * 0.5, src_h * 0.5, src_w=src_w, src_h=src_h, dst_w=w, dst_h=h)
    cv2.line(out, (cx - 18, cy), (cx + 18, cy), (80, 220, 255), 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 18), (cx, cy + 18), (80, 220, 255), 1, cv2.LINE_AA)

    if not approach:
        return out

    tol_px = float(approach.get("track_tolerance_px") or 95.0)
    tol_r = max(6, int(round(tol_px * w / max(src_w, 1.0))))

    # Dim "crowd" of detected corners so the locked one stands out.
    corner_idx = int(approach.get("corner_idx", -1))
    for cand in (approach.get("candidates") or [])[:60]:
        try:
            ci = int(cand[0])
            cpx, cpy = float(cand[1]), float(cand[2])
        except (TypeError, ValueError, IndexError):
            continue
        gx, gy = _scale_xy(cpx, cpy, src_w=src_w, src_h=src_h, dst_w=w, dst_h=h)
        if ci == corner_idx:
            continue  # drawn specially below
        cv2.circle(out, (gx, gy), 3, (140, 140, 140), 1, cv2.LINE_AA)

    anchor_px = approach.get("anchor_px")
    anchor_py = approach.get("anchor_py")
    if anchor_px is not None and anchor_py is not None:
        ax, ay = _scale_xy(
            float(anchor_px),
            float(anchor_py),
            src_w=src_w,
            src_h=src_h,
            dst_w=w,
            dst_h=h,
        )
        cv2.circle(out, (ax, ay), tol_r, (255, 180, 120), 1, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 4, (255, 180, 120), -1, cv2.LINE_AA)

    track_px = approach.get("track_px")
    track_py = approach.get("track_py")
    if track_px is not None and track_py is not None:
        tx, ty = _scale_xy(
            float(track_px),
            float(track_py),
            src_w=src_w,
            src_h=src_h,
            dst_w=w,
            dst_h=h,
        )
        cv2.circle(out, (tx, ty), 5, (220, 120, 255), 1, cv2.LINE_AA)

    px = approach.get("px")
    py = approach.get("py")
    valid = bool(approach.get("valid")) and px is not None and py is not None
    if valid:
        x, y = _scale_xy(float(px), float(py), src_w=src_w, src_h=src_h, dst_w=w, dst_h=h)
        # Distinct diamond + double ring so the locked corner never blends in.
        diamond = np.array([[x, y - 22], [x + 22, y], [x, y + 22], [x - 22, y]], dtype=np.int32)
        cv2.polylines(out, [diamond], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(out, [diamond], True, (154, 255, 61), 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 15, (154, 255, 61), 2, cv2.LINE_AA)
        cv2.line(out, (x - 26, y), (x + 26, y), (154, 255, 61), 1, cv2.LINE_AA)
        cv2.line(out, (x, y - 26), (x, y + 26), (154, 255, 61), 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"LOCK #{corner_idx}" if corner_idx >= 0 else "TARGET"
        ts = approach.get("template_score")
        if ts is not None:
            label += f" {float(ts):.2f}"
        cv2.putText(out, label, (x + 18, max(14, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x + 18, max(14, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (154, 255, 61), 1, cv2.LINE_AA)
        hud = []
        if approach.get("depth_m") is not None:
            hud.append(f"z {float(approach['depth_m']):.2f}m")
        if approach.get("cam_err_m") is not None:
            hud.append(f"d {float(approach['cam_err_m']):+.2f}m")
        src = str(approach.get("source") or "").replace("plane:", "")
        if src:
            hud.append(src[:18])
        for i, line in enumerate(hud[:3]):
            cv2.putText(
                out,
                line,
                (x + 16, y + 12 + i * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (154, 255, 61),
                1,
                cv2.LINE_AA,
            )
    else:
        msg = str(approach.get("status_text") or approach.get("reject_reason") or "no target")[:42]
        cv2.putText(
            out,
            msg,
            (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 120, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def annotate_jpeg_bytes(jpeg_bytes: bytes, approach: Optional[Dict[str, Any]], *, quality: int = 72) -> bytes:
    """Decode JPEG, draw overlay, re-encode."""
    if not jpeg_bytes or not approach:
        return jpeg_bytes
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jpeg_bytes
    annotated = draw_approach_overlay_bgr(bgr, approach)
    ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return jpeg_bytes
    return enc.tobytes()
