"""Strawberry detector preview for hub stereo overlay (read-only, no arm motion)."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_cache: Dict[str, Any] = {}
_last_at: float = 0.0
_log: list[str] = []
_detector = None
_detector_mode = "none"  # yolo | color
_manifest: Optional[dict] = None


def _weights_available(path: str) -> bool:
    return Path(path).is_file()


def _get_detector_mode() -> str:
    global _detector_mode
    if _detector_mode != "none":
        return _detector_mode
    weights = _detector_weights()
    if _weights_available(weights):
        _detector_mode = "yolo"
    else:
        _detector_mode = "missing"
        _push_log(f"YOLO weights missing: {weights}")
    return _detector_mode


def last_strawberry_overlay() -> Dict[str, Any]:
    return dict(_cache) if _cache else {}


def _push_log(line: str, max_lines: int = 12) -> None:
    global _log
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {line}"
    if _log and _log[-1] == entry:
        return
    _log.append(entry)
    if len(_log) > max_lines:
        _log = _log[-max_lines:]


def _load_manifest() -> dict:
    global _manifest
    if _manifest is not None:
        return _manifest
    p = _REPO / "models" / "weights_manifest.json"
    try:
        _manifest = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _manifest = {}
    return _manifest


def _detector_weights() -> str:
    from pipelines.strawberry_ensemble import default_production_config

    cfg = default_production_config(_REPO)
    return cfg.detector_weights


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector
    from pipelines.strawberry_ensemble import YoloDetector

    weights = _detector_weights()
    _detector = YoloDetector(
        weights_path=weights,
        device="cuda",
        imgsz=480,
        conf=0.35,
        iou=0.6,
        max_det=8,
    )
    _push_log(f"detector loaded: {Path(weights).name}")
    return _detector


def _decode_stereo_jpeg(st: dict):
    import cv2
    import numpy as np

    raw = base64.b64decode(st["jpeg_b64"])
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def _run_detector(bgr) -> List[dict]:
    mode = _get_detector_mode()
    if mode != "yolo":
        return []
    t0 = time.perf_counter()
    det = _get_detector()
    dets = det.infer(bgr)
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(d.detector_conf, 3)})
    ms = (time.perf_counter() - t0) * 1000.0
    _push_log(f"detect(yolo): {len(out)} berry(s) in {ms:.0f}ms")
    return out


def collect_roarm_strawberry_preview(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
    *,
    interval_s: float = 0.8,
) -> Dict[str, Any]:
    """Return strawberry detections for hub overlay + status."""
    global _last_at, _cache

    now = time.monotonic()
    if _cache and (now - _last_at) < interval_s:
        return {**dict(_cache), "log": list(_log)}

    base = local_web.rstrip("/")
    st = fetch_json(f"{base}/api/perception/stereo_camera", 0.8) or {}
    if not st.get("ok") or not st.get("jpeg_b64"):
        reason = st.get("reason") or st.get("error") or "no_stereo_frame"
        _push_log(f"камера: {reason}")
        _last_at = now
        _cache = {
            "valid": False,
            "count": 0,
            "detections": [],
            "status": "no_frame",
            "status_text": "Нет кадра стерео",
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}

    try:
        bgr = _decode_stereo_jpeg(st)
        if bgr is None:
            raise RuntimeError("jpeg_decode_failed")
        h, w = bgr.shape[:2]
        mode = _get_detector_mode()
        detections = _run_detector(bgr)
        if mode != "yolo":
            status_text = "Нет весов детектора (скопируйте runs/ со старого Orin)"
            status = "no_weights"
        elif detections:
            status_text = f"Клубника: {len(detections)}"
            status = "ok"
        else:
            status_text = "Клубника не найдена"
            status = "empty"
        _cache = {
            "valid": len(detections) > 0,
            "count": len(detections),
            "detections": detections,
            "image_w": w,
            "image_h": h,
            "status": status,
            "status_text": status_text,
            "updated_at": time.time(),
        }
        _last_at = now
        return {**_cache, "log": list(_log)}
    except Exception as exc:
        _push_log(f"detect error: {exc}")
        _last_at = now
        _cache = {
            "valid": False,
            "count": 0,
            "detections": [],
            "status": "error",
            "status_text": str(exc),
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}
