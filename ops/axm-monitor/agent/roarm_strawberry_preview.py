"""Strawberry detector preview for hub stereo overlay (read-only, no arm motion)."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Jetson: pip torchvision ops are ABI-incompatible with JetPack torch.
try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    pass

_cache: Dict[str, Any] = {}
_last_at: float = 0.0
_log: list[str] = []
_detector = None
_detector_mode = "none"  # yolo | missing
_ros_provider = None
_manifest: Optional[dict] = None
_perc_cfg = None


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


def _perception_cfg():
    global _perc_cfg
    if _perc_cfg is not None:
        return _perc_cfg
    from pipelines.roarm_perception import PerceptionConfig

    _perc_cfg = PerceptionConfig()
    return _perc_cfg


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


def _get_ros_provider():
    global _ros_provider
    if _ros_provider is not None:
        return _ros_provider
    from pipelines.ros_rgb_depth import Ros2RgbDepthProvider

    for rgb_topic, depth_topic in (
        ("/stereo_camera/color/image_rect_raw", "/stereo_camera/depth/image_rect_raw"),
        ("/stereo_camera/color/image_raw", "/stereo_camera/depth/image_raw"),
    ):
        try:
            provider = Ros2RgbDepthProvider(
                rgb_topic=rgb_topic,
                depth_topic=depth_topic,
                sync_slop_s=0.15,
            )
            provider.open(camera_info_topic="/stereo_camera/color/camera_info")
            _ros_provider = provider
            _push_log(f"stereo ROS: {rgb_topic}")
            return _ros_provider
        except Exception:
            continue
    return None


def _decode_stereo_jpeg(st: dict):
    import cv2
    import numpy as np

    raw = base64.b64decode(st["jpeg_b64"])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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


def _scale_depth_to_bgr(depth_m, bgr_shape: Tuple[int, int, int]):
    import cv2
    import numpy as np

    if depth_m is None:
        return None
    bh, bw = bgr_shape[:2]
    dh, dw = depth_m.shape[:2]
    if (dh, dw) == (bh, bw):
        return depth_m
    return cv2.resize(depth_m, (bw, bh), interpolation=cv2.INTER_NEAREST)


def _attach_depth(detections: List[dict], depth_m, *, rgb_w: int, rgb_h: int) -> None:
    from pipelines.roarm_perception import sample_depth_median

    cfg = _perception_cfg()
    depth_aligned = _scale_depth_to_bgr(depth_m, (rgb_h, rgb_w, 3))
    for det in detections:
        cx = int(round((det["x1"] + det["x2"]) * 0.5))
        cy = int(round((det["y1"] + det["y2"]) * 0.5))
        det["px"] = cx
        det["py"] = cy
        if depth_aligned is None:
            det["depth_m"] = None
            continue
        depth_val, reject = sample_depth_median(
            depth_aligned, cx, cy, cfg.depth_median_radius, cfg
        )
        det["depth_m"] = round(depth_val, 3) if depth_val is not None else None
        if reject:
            det["depth_reject"] = reject


def _status_from_detections(detections: List[dict], mode: str) -> Tuple[str, str]:
    if mode != "yolo":
        return "no_weights", "Нет весов детектора"
    if not detections:
        return "empty", "Клубника не найдена"
    parts = [f"Клубника: {len(detections)}"]
    for i, d in enumerate(detections[:3]):
        z = d.get("depth_m")
        if z is not None:
            parts.append(f"#{i + 1} {z:.2f}m conf {d['conf']:.2f}")
        else:
            parts.append(f"#{i + 1} conf {d['conf']:.2f}")
    return "ok", " · ".join(parts)


def _build_overlay(bgr, depth_m) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    mode = _get_detector_mode()
    detections = _run_detector(bgr)
    _attach_depth(detections, depth_m, rgb_w=w, rgb_h=h)
    status, status_text = _status_from_detections(detections, mode)
    with_depth = sum(1 for d in detections if d.get("depth_m") is not None)
    if detections and depth_m is not None:
        _push_log(f"depth: {with_depth}/{len(detections)} bbox with z")
    elif detections and depth_m is None:
        _push_log("depth: no aligned depth frame")
    return {
        "valid": len(detections) > 0,
        "count": len(detections),
        "detections": detections,
        "image_w": w,
        "image_h": h,
        "depth_ok": depth_m is not None,
        "status": status,
        "status_text": status_text,
        "updated_at": time.time(),
    }


def update_strawberry_overlay_from_jpeg(
    jpeg_bytes: bytes,
    *,
    min_interval_s: float = 0.35,
) -> Dict[str, Any]:
    """Run detector on the same JPEG bytes sent to hub MJPEG (keeps bbox aligned)."""
    global _last_at, _cache

    now = time.monotonic()
    if _cache and (now - _last_at) < min_interval_s:
        return dict(_cache)

    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return dict(_cache) if _cache else {}

    depth_m = None
    try:
        provider = _get_ros_provider()
        if provider is not None:
            pair = provider.read(timeout_s=0.35)
            if pair is not None:
                depth_m = pair.depth_m
    except Exception:
        depth_m = None

    try:
        _cache = _build_overlay(bgr, depth_m)
        _last_at = now
    except Exception as exc:
        _push_log(f"detect error: {exc}")
        _cache = {
            "valid": False,
            "count": 0,
            "detections": [],
            "status": "error",
            "status_text": str(exc),
            "updated_at": time.time(),
        }
        _last_at = now
    return dict(_cache)


def collect_roarm_strawberry_preview(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
    *,
    interval_s: float = 0.8,
) -> Dict[str, Any]:
    """Return strawberry detections for hub telemetry (uses cache if fresh)."""
    global _last_at, _cache

    now = time.monotonic()
    if _cache and (now - _last_at) < interval_s:
        return {**dict(_cache), "log": list(_log)}

    # Prefer synchronized RGB+depth from ROS; fallback to HTTP JPEG (depth may be missing).
    bgr = None
    depth_m = None
    try:
        provider = _get_ros_provider()
        if provider is not None:
            pair = provider.read(timeout_s=1.2)
            if pair is not None:
                bgr = pair.rgb_bgr
                depth_m = pair.depth_m
    except Exception:
        pass

    if bgr is None:
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
        bgr = _decode_stereo_jpeg(st)

    try:
        if bgr is None:
            raise RuntimeError("frame_decode_failed")
        _cache = _build_overlay(bgr, depth_m)
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
