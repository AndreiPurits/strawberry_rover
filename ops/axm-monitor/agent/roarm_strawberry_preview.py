"""Strawberry detector preview for hub stereo overlay (read-only, no arm motion)."""

from __future__ import annotations

import base64
import json
import sys
import threading
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
_classifier = None
_detector_mode = "none"  # yolo | missing
_ros_provider = None
_manifest: Optional[dict] = None
_perc_cfg = None
_hub_tracker = None


def _get_hub_tracker():
    global _hub_tracker
    if _hub_tracker is None:
        from pipelines.roarm_strawberry_target import StrawberryTargetTracker

        _hub_tracker = StrawberryTargetTracker()
    return _hub_tracker


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


def _production_config():
    from pipelines.strawberry_ensemble import default_production_config

    return default_production_config(_REPO)


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector
    from pipelines.strawberry_ensemble import YoloDetector

    cfg = _production_config()
    weights = cfg.detector_weights
    _detector = YoloDetector(
        weights_path=weights,
        device="cuda",
        imgsz=480,
        conf=0.28,
        iou=0.6,
        max_det=8,
    )
    _push_log(f"detector loaded: {Path(weights).name}")
    return _detector


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    from pipelines.strawberry_ensemble import RipenessClassifier

    cfg = _production_config()
    if not _weights_available(cfg.classifier_weights):
        _push_log(f"classifier weights missing: {cfg.classifier_weights}")
        return None
    _classifier = RipenessClassifier(cfg.classifier_weights, device="cuda")
    _push_log(f"classifier loaded: {Path(cfg.classifier_weights).name}")
    return _classifier


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


def _clamp_bbox(det: dict, w: int, h: int, *, pad_frac: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
    try:
        x1 = float(det["x1"])
        y1 = float(det["y1"])
        x2 = float(det["x2"])
        y2 = float(det["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_frac
    py = bh * pad_frac
    x1i = max(0, min(w - 1, int(round(x1 - px))))
    y1i = max(0, min(h - 1, int(round(y1 - py))))
    x2i = max(0, min(w - 1, int(round(x2 + px))))
    y2i = max(0, min(h - 1, int(round(y2 + py))))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


def _classify_detections(bgr, detections: List[dict]) -> None:
    """Attach ripeness_class/classifier_conf to each detection."""
    if not detections:
        return
    h, w = bgr.shape[:2]
    classifier = _get_classifier()
    crops: List[Any] = []
    crop_meta: List[Tuple[int, int, int, int, dict]] = []
    for det in detections:
        bbox = _clamp_bbox(det, w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        crop_meta.append((x1, y1, x2, y2, det))

    if classifier is not None and crops:
        try:
            for (_, _, _, _, det), (cls_name, cls_conf) in zip(crop_meta, classifier.infer_crops_bgr(crops)):
                det["ripeness_class"] = str(cls_name)
                det["classifier_conf"] = round(float(cls_conf), 3)
        except Exception as exc:
            _push_log(f"classifier error: {str(exc)[:80]}")


def _run_detector(bgr) -> List[dict]:
    mode = _get_detector_mode()
    if mode != "yolo":
        return []
    t0 = time.perf_counter()
    det = _get_detector()
    from pipelines.roarm_strawberry_target import detect_strawberries_in_frame

    tracker = _get_hub_tracker()
    out = [dict(d) for d in detect_strawberries_in_frame(bgr, det, tracker)]
    _classify_detections(bgr, out)
    if out:
        tracker.update(max(out, key=lambda d: float(d.get("conf", 0.0))))
    ms = (time.perf_counter() - t0) * 1000.0
    src = "roi" if any(d.get("from_roi") for d in out) else "full"
    yolo_n = sum(1 for d in out if d.get("source") == "yolo")
    color_n = sum(1 for d in out if d.get("source") == "color")
    _push_log(f"detect({src}): {len(out)} berry in {ms:.0f}ms (yolo={yolo_n} color={color_n})")
    return out


def _letterbox_array(arr, max_w: int, max_h: int):
    """Match rover_web_interface ros_bridge._encode_jpeg_bgr_fit letterbox."""
    import cv2
    import numpy as np

    src_h, src_w = arr.shape[:2]
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_NEAREST if arr.ndim == 2 else cv2.INTER_AREA
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    meta = {
        "scale": scale,
        "x0": (max_w - new_w) // 2,
        "y0": (max_h - new_h) // 2,
        "content_w": new_w,
        "content_h": new_h,
    }
    if new_w == max_w and new_h == max_h:
        return resized, meta
    if arr.ndim == 2:
        canvas = np.zeros((max_h, max_w), dtype=arr.dtype)
    else:
        canvas = np.zeros((max_h, max_w, arr.shape[2]), dtype=arr.dtype)
    y0, x0 = meta["y0"], meta["x0"]
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, meta


def _hub_content_rect(hub_w: int, hub_h: int, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    scale = min(hub_w / src_w, hub_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x0 = (hub_w - new_w) // 2
    y0 = (hub_h - new_h) // 2
    return x0, y0, new_w, new_h


def _filter_detections_in_padding(
    detections: List[dict],
    hub_w: int,
    hub_h: int,
    *,
    src_w: int = 848,
    src_h: int = 480,
    margin_px: int = 6,
) -> List[dict]:
    """Drop bbox hits in letterbox black bars (hub JPEG != native ROS size)."""
    x0, y0, cw, ch = _hub_content_rect(hub_w, hub_h, src_w, src_h)
    kept: List[dict] = []
    for det in detections:
        cx = (float(det["x1"]) + float(det["x2"])) * 0.5
        cy = (float(det["y1"]) + float(det["y2"])) * 0.5
        if cx < x0 + margin_px or cx >= x0 + cw - margin_px:
            continue
        if cy < y0 + margin_px or cy >= y0 + ch - margin_px:
            continue
        kept.append(det)
    dropped = len(detections) - len(kept)
    if dropped:
        _push_log(f"filter: dropped {dropped} bbox in letterbox padding")
    return kept


def _align_depth_to_hub_bgr(depth_m, hub_w: int, hub_h: int):
    if depth_m is None:
        return None
    dh, dw = depth_m.shape[:2]
    if dw == hub_w and dh == hub_h:
        return depth_m
    aligned, _ = _letterbox_array(depth_m, hub_w, hub_h)
    return aligned


def _fetch_hub_stereo_bgr(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
) -> Optional[Any]:
    base = local_web.rstrip("/")
    st = fetch_json(f"{base}/api/perception/stereo_camera", 0.8) or {}
    if not st.get("ok") or not st.get("jpeg_b64"):
        return None
    return _decode_stereo_jpeg(st)


def _fetch_ros_depth(timeout_s: float = 0.5):
    try:
        provider = _get_ros_provider()
        if provider is None:
            return None
        pair = provider.read(timeout_s=timeout_s)
        if pair is None:
            return None
        return pair.depth_m
    except Exception:
        return None


def _attach_depth(detections: List[dict], depth_m, *, rgb_w: int, rgb_h: int) -> None:
    from pipelines.roarm_perception import sample_depth_median

    cfg = _perception_cfg()
    depth_aligned = depth_m
    if depth_m is not None and depth_m.shape[:2] != (rgb_h, rgb_w):
        depth_aligned = _align_depth_to_hub_bgr(depth_m, rgb_w, rgb_h)
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
        label = str(d.get("ripeness_class") or "berry")
        cls_conf = d.get("classifier_conf")
        cls_txt = f" cls {float(cls_conf):.2f}" if cls_conf is not None else ""
        xy = ""
        if d.get("px") is not None and d.get("py") is not None:
            xy = f" px={int(d['px'])} py={int(d['py'])}"
        if z is not None:
            parts.append(f"#{i + 1} {label} {z:.2f}m conf {d['conf']:.2f}{cls_txt}{xy}")
        else:
            parts.append(f"#{i + 1} {label} conf {d['conf']:.2f}{cls_txt}{xy}")
    return "ok", " · ".join(parts)


def _build_overlay(bgr, depth_m) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    mode = _get_detector_mode()
    detections = _run_detector(bgr)
    detections = _filter_detections_in_padding(detections, w, h)
    depth_hub = _align_depth_to_hub_bgr(depth_m, w, h) if depth_m is not None else None
    _attach_depth(detections, depth_hub, rgb_w=w, rgb_h=h)
    status, status_text = _status_from_detections(detections, mode)
    with_depth = sum(1 for d in detections if d.get("depth_m") is not None)
    if detections and depth_hub is not None:
        _push_log(f"depth: {with_depth}/{len(detections)} bbox with z")
    elif detections and depth_hub is None:
        _push_log("depth: no aligned depth frame")
    return {
        "valid": len(detections) > 0,
        "count": len(detections),
        "detections": detections,
        "image_w": w,
        "image_h": h,
        "depth_ok": depth_hub is not None,
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

    depth_m = _fetch_ros_depth(timeout_s=0.35)

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

    # Detect on the same hub JPEG (640×480 letterbox), not native ROS 848×480.
    bgr = _fetch_hub_stereo_bgr(local_web, fetch_json)
    depth_m = _fetch_ros_depth(timeout_s=0.5)

    if bgr is None:
        _push_log("камера: no_stereo_frame")
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


def start_strawberry_detect_thread(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
    stop_event: threading.Event,
    *,
    interval_s: float = 0.7,
) -> threading.Thread:
    """Background YOLO+depth — keeps camera MJPEG loop fast (annotate only)."""

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                collect_roarm_strawberry_preview(local_web, fetch_json, interval_s=0.0)
            except Exception as exc:
                _push_log(f"bg detect: {exc}")
            stop_event.wait(max(0.25, interval_s))

    t = threading.Thread(target=_loop, daemon=True, name="strawberry-detect")
    t.start()
    return t
