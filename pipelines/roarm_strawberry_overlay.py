"""Draw strawberry detector bboxes on stereo BGR frames (hub MJPEG)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def draw_strawberry_overlay_bgr(bgr: np.ndarray, overlay: Optional[Dict[str, Any]]) -> np.ndarray:
    """Return BGR image with strawberry bounding boxes only (no grid/corner HUD)."""
    if bgr is None or bgr.size == 0 or not overlay:
        return bgr
    dets: List[dict] = list(overlay.get("detections") or [])
    if not dets:
        return bgr

    out = bgr.copy()
    h, w = out.shape[:2]
    src_w = float(overlay.get("image_w") or w)
    src_h = float(overlay.get("image_h") or h)
    sx = w / max(1.0, src_w)
    sy = h / max(1.0, src_h)

    for det in dets[:20]:
        try:
            x1 = int(round(float(det["x1"]) * sx))
            y1 = int(round(float(det["y1"]) * sy))
            x2 = int(round(float(det["x2"]) * sx))
            y2 = int(round(float(det["y2"]) * sy))
            conf = float(det.get("conf", 0))
        except (TypeError, ValueError, KeyError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        source = str(det.get("source") or "yolo")
        color = (40, 220, 40) if source == "yolo" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        depth_m = det.get("depth_m")
        if depth_m is not None:
            label = f"{source} {conf:.2f} {float(depth_m):.2f}m"
        else:
            label = f"{source} {conf:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def annotate_jpeg_bytes(jpeg_bytes: bytes, overlay: Optional[Dict[str, Any]], *, quality: int = 72) -> bytes:
    if not jpeg_bytes or not overlay or not overlay.get("detections"):
        return jpeg_bytes
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jpeg_bytes
    annotated = draw_strawberry_overlay_bgr(bgr, overlay)
    ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return jpeg_bytes
    return enc.tobytes()
