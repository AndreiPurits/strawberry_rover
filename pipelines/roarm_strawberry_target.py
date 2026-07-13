"""Strawberry target measurement on hub JPEG (aligned with fleet-agent overlay)."""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    pass

_HUB_W = 640
_HUB_H = 480
_SRC_W = 848
_SRC_H = 480
_REPO_ROOT = Path(__file__).resolve().parents[1]
_last_exclude_profile: Optional[str] = None


def _brightness_mask_module():
    root = _REPO_ROOT / "src" / "rover_perception"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rover_perception.stereo_brightness_mask import (  # noqa: WPS433
        blackout_exclude_regions,
        default_mask_path,
        filter_detections_exclude_regions,
        resolve_active_exclude_regions,
    )

    return (
        blackout_exclude_regions,
        default_mask_path,
        filter_detections_exclude_regions,
        resolve_active_exclude_regions,
    )


def _exclude_mask_path() -> str:
    return os.environ.get("AXM_STEREO_EXCLUDE_MASK", "").strip() or _brightness_mask_module()[1]()


def _gripper_g_from_env() -> Optional[float]:
    raw = os.environ.get("AXM_GRIPPER_G", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _gripper_g_live() -> Optional[float]:
    """Read gripper joint g from RoArm feedback (preferred over env)."""
    agent_dir = _REPO_ROOT / "ops" / "axm-monitor/agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    try:
        from roarm_proxy import telemetry_snapshot  # noqa: WPS433

        arm = telemetry_snapshot().get("arm") or {}
        g = arm.get("g")
        if g is not None:
            return float(g)
    except Exception:
        pass
    return _gripper_g_from_env()


def _resolve_exclude_for_frame(bgr) -> Tuple[str, List[Tuple[float, float, float, float]]]:
    global _last_exclude_profile
    if os.environ.get("AXM_STEREO_EXCLUDE_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return "disabled", []
    path = _exclude_mask_path()
    if not path or not os.path.isfile(path):
        return "none", []
    _, _, _, resolve_active = _brightness_mask_module()
    forced = os.environ.get("AXM_GRIPPER_MASK_PROFILE", "").strip() or None
    gripper_g = _gripper_g_live()
    profile, regions = resolve_active(
        path,
        gripper_g=gripper_g,
        forced_profile=forced if forced else None,
    )
    if profile != _last_exclude_profile:
        _last_exclude_profile = profile
        gtxt = f"{gripper_g:.3f}" if gripper_g is not None else "?"
        print(f"[berry-mask] profile={profile} g={gtxt} regions={len(regions)}")
    return profile, regions


def _apply_exclude_mask(bgr):
    _, regions = _resolve_exclude_for_frame(bgr)
    if not regions:
        return bgr
    blackout_exclude_regions, _, _, _ = _brightness_mask_module()
    return blackout_exclude_regions(bgr, regions)


def _letterbox_array(arr, max_w: int, max_h: int):
    import cv2

    src_h, src_w = arr.shape[:2]
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_NEAREST if arr.ndim == 2 else cv2.INTER_AREA
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    if new_w == max_w and new_h == max_h:
        return resized
    if arr.ndim == 2:
        canvas = np.zeros((max_h, max_w), dtype=arr.dtype)
    else:
        canvas = np.zeros((max_h, max_w, arr.shape[2]), dtype=arr.dtype)
    y0 = (max_h - new_h) // 2
    x0 = (max_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _hub_content_rect(hub_w: int = _HUB_W, hub_h: int = _HUB_H) -> Tuple[int, int, int, int]:
    scale = min(hub_w / _SRC_W, hub_h / _SRC_H)
    new_w = max(1, int(round(_SRC_W * scale)))
    new_h = max(1, int(round(_SRC_H * scale)))
    x0 = (hub_w - new_w) // 2
    y0 = (hub_h - new_h) // 2
    return x0, y0, new_w, new_h


def _filter_padding(detections: List[dict], margin_px: int = 6) -> List[dict]:
    x0, y0, cw, ch = _hub_content_rect()
    kept = []
    for det in detections:
        cx = (float(det["x1"]) + float(det["x2"])) * 0.5
        cy = (float(det["y1"]) + float(det["y2"])) * 0.5
        if cx < x0 + margin_px or cx >= x0 + cw - margin_px:
            continue
        if cy < y0 + margin_px or cy >= y0 + ch - margin_px:
            continue
        kept.append(det)
    return kept


def _filter_work_band(
    detections: List[dict],
    *,
    x_min: float = 70.0,
    x_max: float = 590.0,
    y_min: float = 55.0,
    y_max: float = 410.0,
) -> List[dict]:
    kept = []
    for det in detections:
        cx, cy = _bbox_center(det)
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            kept.append(det)
    return kept


def _fetch_hub_bgr(local_web: str = "http://127.0.0.1:8080"):
    import cv2

    with urllib.request.urlopen(f"{local_web.rstrip('/')}/api/perception/stereo_camera", timeout=3) as r:
        st = json.loads(r.read())
    if not st.get("ok") or not st.get("jpeg_b64"):
        return None, st
    raw = base64.b64decode(st["jpeg_b64"])
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return bgr, st


def _bbox_center(det: dict) -> Tuple[float, float]:
    return (float(det["x1"]) + float(det["x2"])) * 0.5, (float(det["y1"]) + float(det["y2"])) * 0.5


def _score_detection(
    det: dict,
    last_px: Optional[float],
    last_py: Optional[float],
    *,
    spatial_penalty: float = 0.0045,
    roi_bonus: float = 0.12,
    roi_radius_px: float = 130.0,
) -> float:
    conf = float(det.get("conf", 0))
    if last_px is None or last_py is None:
        return conf
    cx, cy = _bbox_center(det)
    dist = float(np.hypot(cx - last_px, cy - last_py))
    score = conf - spatial_penalty * dist
    if dist <= roi_radius_px:
        score += roi_bonus
    return score


def _pick_detection(
    detections: List[dict],
    last_px: Optional[float],
    last_py: Optional[float],
    *,
    preferred_px: Optional[float] = None,
    preferred_py: Optional[float] = None,
    preferred_max_dist_px: Optional[float] = None,
    max_jump_px: float = 140.0,
) -> Optional[dict]:
    if not detections:
        return None
    if (last_px is None or last_py is None) and preferred_px is not None and preferred_py is not None:
        if preferred_max_dist_px is not None:
            detections = [
                d for d in detections
                if float(np.hypot(_bbox_center(d)[0] - preferred_px, _bbox_center(d)[1] - preferred_py))
                <= preferred_max_dist_px
            ]
            if not detections:
                return None
        return max(
            detections,
            key=lambda d: _score_detection(
                d,
                preferred_px,
                preferred_py,
                spatial_penalty=0.0035,
                roi_bonus=0.08,
                roi_radius_px=170.0,
            ),
        )
    if last_px is None or last_py is None:
        return max(detections, key=lambda d: float(d.get("conf", 0)))
    scored = sorted(
        detections,
        key=lambda d: _score_detection(d, last_px, last_py),
        reverse=True,
    )
    best = scored[0]
    cx, cy = _bbox_center(best)
    near = [
        d for d in detections
        if float(np.hypot(_bbox_center(d)[0] - last_px, _bbox_center(d)[1] - last_py)) <= max_jump_px
    ]
    if near:
        return max(near, key=lambda d: _score_detection(d, last_px, last_py))
    conf_min = 0.40
    cx, cy = _bbox_center(best)
    if cx > 560 or cx < 80 or cy < 70 or cy > 410:
        conf_min = 0.32
    return best if float(best.get("conf", 0)) >= conf_min else None


def _infer_on_bgr(det_model, bgr) -> List[dict]:
    h, w = bgr.shape[:2]
    masked = _apply_exclude_mask(bgr)
    out = []
    for d in det_model.infer(masked):
        x1, y1, x2, y2 = d.bbox_xyxy
        out.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": round(float(d.detector_conf), 3),
            "source": "yolo",
        })
    regions = _resolve_exclude_for_frame(bgr)[1]
    if regions:
        _, _, filter_detections_exclude_regions, _ = _brightness_mask_module()
        out = filter_detections_exclude_regions(out, regions, w, h)
    return out


def _center_distance(a: dict, b: dict) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


def _color_fallback_detections(bgr) -> List[dict]:
    """Detect red berry-like blobs that YOLO misses; used for multi-berry preview."""
    from pipelines.strawberry_color_detect import detect_red_berry_bboxes

    out = []
    for x1, y1, x2, y2, score in detect_red_berry_bboxes(
        bgr,
        min_area_px=int(os.environ.get("AXM_BERRY_COLOR_MIN_AREA", "18")),
        max_area_frac=float(os.environ.get("AXM_BERRY_COLOR_MAX_AREA_FRAC", "0.08")),
    ):
        det = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "conf": round(float(score), 3),
            "color_score": round(float(score), 3),
            "source": "color",
        }
        cx, cy = _bbox_center(det)
        # Bench berries hang in the upper/mid frame; this rejects red gripper/table artifacts.
        if cy > float(os.environ.get("AXM_BERRY_COLOR_MAX_CY", "230")):
            continue
        out.append(det)
    return _filter_work_band(_filter_padding(out))


def _merge_color_fallback(yolo_dets: List[dict], color_dets: List[dict]) -> List[dict]:
    merged = list(yolo_dets)
    overlap_px = float(os.environ.get("AXM_BERRY_COLOR_MERGE_DIST_PX", "45"))
    for cdet in color_dets:
        near = [yd for yd in merged if _center_distance(yd, cdet) <= overlap_px]
        if near:
            for yd in near:
                yd["color_score"] = max(float(yd.get("color_score", 0.0)), float(cdet.get("color_score", 0.0)))
            continue
        merged.append(cdet)
    return sorted(merged, key=lambda d: float(d.get("conf", 0.0)), reverse=True)


def _infer_full_and_roi(det_model, bgr, tracker: Optional["StrawberryTargetTracker"]) -> List[dict]:
    import cv2

    raw = _infer_on_bgr(det_model, bgr)
    if tracker is None or tracker.last_px is None or tracker.last_py is None:
        return raw

    h, w = bgr.shape[:2]
    pad = int(tracker.roi_pad_px)
    cx = int(round(tracker.last_px))
    cy = int(round(tracker.last_py))
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad)
    y2 = min(h, cy + pad)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return raw

    crop = bgr[y1:y2, x1:x2]
    roi_dets = _infer_on_bgr(det_model, crop)
    for det in roi_dets:
        det["x1"] = float(det["x1"]) + x1
        det["x2"] = float(det["x2"]) + x1
        det["y1"] = float(det["y1"]) + y1
        det["y2"] = float(det["y2"]) + y1
        det["from_roi"] = True
    merged = {(
        round(float(d["x1"]), 0),
        round(float(d["y1"]), 0),
        round(float(d["x2"]), 0),
        round(float(d["y2"]), 0),
    ): d for d in raw}
    for d in roi_dets:
        key = (
            round(float(d["x1"]), 0),
            round(float(d["y1"]), 0),
            round(float(d["x2"]), 0),
            round(float(d["y2"]), 0),
        )
        if key not in merged or float(d["conf"]) > float(merged[key]["conf"]):
            merged[key] = d
    return list(merged.values())


class StrawberryTargetTracker:
    def __init__(
        self,
        *,
        roi_pad_px: float = 150.0,
        hold_max: int = 14,
        preferred_px: Optional[float] = None,
        preferred_py: Optional[float] = None,
        preferred_max_dist_px: Optional[float] = None,
        strict_lock: bool = False,
        lock_radius_px: float = 85.0,
    ) -> None:
        self.last_px: Optional[float] = None
        self.last_py: Optional[float] = None
        self.last_bbox: Optional[dict] = None
        self.last_conf: float = 0.0
        self.lost_streak = 0
        self.hold_streak = 0
        self.roi_pad_px = roi_pad_px
        self.hold_max = hold_max
        self.preferred_px = preferred_px
        self.preferred_py = preferred_py
        self.preferred_max_dist_px = preferred_max_dist_px
        self.strict_lock = bool(strict_lock)
        self.lock_radius_px = float(lock_radius_px)
        self.lock_count = 0

    def update(self, det: Optional[dict]) -> bool:
        if det is None:
            self.lost_streak += 1
            self.hold_streak += 1
            return False
        self.last_px, self.last_py = _bbox_center(det)
        self.last_bbox = dict(det)
        self.last_conf = float(det.get("conf", 0))
        self.lost_streak = 0
        self.hold_streak = 0
        self.lock_count += 1
        return True

    def reset_lock(self) -> None:
        self.last_px = None
        self.last_py = None
        self.last_bbox = None
        self.last_conf = 0.0
        self.lost_streak = 0
        self.hold_streak = 0
        self.lock_count = 0

    @property
    def is_lost(self) -> bool:
        return self.lost_streak >= 8 and self.hold_streak >= self.hold_max

    @property
    def can_hold(self) -> bool:
        return self.last_px is not None and self.hold_streak < self.hold_max


_detector = None


def _get_detector(repo_root):
    global _detector
    if _detector is not None:
        return _detector
    from pathlib import Path

    from pipelines.strawberry_ensemble import YoloDetector, default_production_config

    cfg = default_production_config(Path(repo_root))
    _detector = YoloDetector(cfg.detector_weights, "cuda", 480, 0.28, 0.6, 8)
    return _detector


def detect_strawberry_in_frame(
    bgr,
    det_model,
    tracker: Optional[StrawberryTargetTracker],
) -> Optional[dict]:
    """Stable single-target pick: choose the current highest-confidence berry."""
    dets = detect_strawberries_in_frame(bgr, det_model, tracker)
    if tracker and tracker.strict_lock and tracker.last_px is not None and tracker.last_py is not None:
        radius = float(tracker.lock_radius_px)
        dets = [
            d for d in dets
            if float(np.hypot(_bbox_center(d)[0] - tracker.last_px, _bbox_center(d)[1] - tracker.last_py)) <= radius
        ]
    return _pick_detection(
        dets,
        tracker.last_px if tracker else None,
        tracker.last_py if tracker else None,
        preferred_px=tracker.preferred_px if tracker else None,
        preferred_py=tracker.preferred_py if tracker else None,
        preferred_max_dist_px=tracker.preferred_max_dist_px if tracker else None,
    )


def detect_strawberries_in_frame(
    bgr,
    det_model,
    tracker: Optional[StrawberryTargetTracker] = None,
) -> List[dict]:
    """Return all visible berry candidates for overlay; includes color fallback."""
    raw_dets = _infer_full_and_roi(det_model, bgr, tracker)
    yolo_dets = _filter_work_band(_filter_padding(raw_dets))
    color_dets = _color_fallback_detections(bgr)
    return _merge_color_fallback(yolo_dets, color_dets)


def measure_strawberry_hub(
    *,
    repo_root,
    local_web: str = "http://127.0.0.1:8080",
    depth_m_native=None,
    tracker: Optional[StrawberryTargetTracker] = None,
    frames: int = 3,
) -> Optional[Tuple[float, float, float, float, Tuple[int, int]]]:
    """Return median (px, py, depth_m, conf, image_wh) on hub JPEG coordinates."""
    from pipelines.roarm_perception import PerceptionConfig, sample_depth_median

    det_model = _get_detector(repo_root)
    perc = PerceptionConfig()
    pxs, pys, ds, confs = [], [], [], []
    wh = (_HUB_W, _HUB_H)

    for _ in range(max(1, frames)):
        bgr, st = _fetch_hub_bgr(local_web)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        wh = (w, h)
        det = detect_strawberry_in_frame(bgr, det_model, tracker)
        if det is None and tracker and tracker.can_hold and depth_m_native is not None:
            cx = int(round(tracker.last_px))
            cy = int(round(tracker.last_py))
            depth_hub = _letterbox_array(depth_m_native, w, h)
            depth_val, _ = sample_depth_median(depth_hub, cx, cy, perc.depth_median_radius, perc)
            if depth_val is not None:
                pxs.append(cx)
                pys.append(cy)
                ds.append(depth_val)
                confs.append(max(0.2, tracker.last_conf * 0.85))
                tracker.hold_streak += 1
            continue
        if det is None:
            if tracker:
                tracker.lost_streak += 1
                tracker.hold_streak += 1
            continue
        cx = int(round((det["x1"] + det["x2"]) * 0.5))
        cy = int(round((det["y1"] + det["y2"]) * 0.5))
        depth_val = None
        if depth_m_native is not None:
            depth_hub = _letterbox_array(depth_m_native, w, h)
            depth_val, _ = sample_depth_median(depth_hub, cx, cy, perc.depth_median_radius, perc)
        if depth_val is None:
            continue
        pxs.append(cx)
        pys.append(cy)
        ds.append(depth_val)
        confs.append(float(det["conf"]))
        if tracker:
            tracker.update(det)

    if not pxs:
        if tracker:
            tracker.lost_streak += 1
        return None
    return (
        float(np.median(pxs)),
        float(np.median(pys)),
        float(np.median(ds)),
        float(np.median(confs)),
        wh,
    )
