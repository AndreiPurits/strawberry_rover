"""Stereo grid / plane target preview for hub (read-only perception, no arm motion)."""

from __future__ import annotations

import base64
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_cache: Dict[str, Any] = {}
_last_at: float = 0.0
_log: list[str] = []
_plane_tracker = None
_grid_tracker = None
_template_tracker = None
_ros_provider = None
_learn_cfg: Optional[dict] = None


def last_approach_overlay() -> Dict[str, Any]:
    """Latest overlay for MJPEG annotation (fleet-agent camera loop)."""
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


def _load_learn_cfg() -> dict:
    global _learn_cfg
    if _learn_cfg is not None:
        return _learn_cfg
    path = _REPO / "config" / "roarm_learn.yaml"
    if path.is_file():
        try:
            _learn_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            _learn_cfg = {}
    else:
        _learn_cfg = {}
    return _learn_cfg


def _perception_cfg():
    from pipelines.roarm_perception import PerceptionConfig

    learn = _load_learn_cfg()
    perc = learn.get("perception") or {}
    return PerceptionConfig(
        target_camera_distance_m=float(perc.get("target_camera_distance_m", 0.11)),
        track_max_jump_px=float(perc.get("track_max_jump_px", 110.0)),
        track_lost_max_frames=int(perc.get("track_lost_max_frames", 6)),
        use_template_lock=bool(perc.get("use_template_lock", True)),
        template_patch_px=int(perc.get("template_patch_px", 64)),
        template_search_radius_px=int(perc.get("template_search_radius_px", 48)),
        template_reacquire_radius_px=int(perc.get("template_reacquire_radius_px", 140)),
        template_min_score=float(perc.get("template_min_score", 0.52)),
        template_spatial_penalty=float(perc.get("template_spatial_penalty", 0.45)),
        template_max_jump_px=float(perc.get("template_max_jump_px", 40.0)),
    )


def _plane_cfg():
    from pipelines.roarm_plane_perception import PlanePerceptionConfig

    learn = _load_learn_cfg()
    return PlanePerceptionConfig.from_yaml(learn.get("perception") or {})


def _track_tolerance_px() -> float:
    learn = _load_learn_cfg()
    perc = learn.get("perception") or {}
    return float(perc.get("corner_track_tolerance_px", 95.0))


def _plane_model_from_meta(meta: dict, depth_m: float):
    from pipelines.roarm_plane_perception import PlaneModel
    import numpy as np

    p = meta.get("plane") if isinstance(meta, dict) else None
    if isinstance(p, dict) and isinstance(p.get("normal"), list) and len(p["normal"]) == 3:
        return PlaneModel(
            normal=np.array(p["normal"], dtype=np.float64),
            d=float(p.get("d_m", depth_m)),
            rmse_m=float(p.get("rmse_m", 0)),
            inlier_frac=float(p.get("inlier_frac", 1)),
        )
    return PlaneModel(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        d=float(depth_m),
        rmse_m=0.0,
        inlier_frac=1.0,
    )


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
            t0 = time.time()
            while provider.get_intrinsics() is None and (time.time() - t0) < 1.5:
                time.sleep(0.05)
            _ros_provider = provider
            return _ros_provider
        except Exception:
            continue
    return None


def _preview_from_ros_template() -> Optional[Dict[str, Any]]:
    """Template lock on one corner — local search near last (px, py)."""
    global _template_tracker
    try:
        from pipelines.roarm_template_track import TemplateCornerTracker, build_template_target

        provider = _get_ros_provider()
        if provider is None:
            return None
        pair = provider.read(timeout_s=2.0)
        intr = provider.get_intrinsics()
        h, w = pair.rgb_bgr.shape[:2]
        perc = _perception_cfg()
        if _template_tracker is None:
            _template_tracker = TemplateCornerTracker.from_config(perc)
        target = build_template_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc,
            _template_tracker,
            intrinsics=intr,
        )
        # Detect the "crowd" of corners for the GUI so the locked one stands out.
        candidates = []
        try:
            from pipelines.roarm_perception import detect_grid_corners

            corners, _kind = detect_grid_corners(pair.rgb_bgr, perc)
            if corners is not None:
                for i, c in enumerate(corners.reshape(-1, 2)[:60]):
                    candidates.append([i, float(c[0]), float(c[1])])
        except Exception:
            candidates = []
        tracking = {
            "anchor_px": _template_tracker.anchor_px,
            "anchor_py": _template_tracker.anchor_py,
            "track_px": _template_tracker.track_px,
            "track_py": _template_tracker.track_py,
            "template_score": round(_template_tracker.last_score, 3),
            "mode": "template",
        }
        return _target_to_overlay(
            target,
            {"tracking": tracking, "candidates": candidates},
            w,
            h,
            track_tolerance_px=_track_tolerance_px(),
            tracking=tracking,
        )
    except Exception:
        return None


def _preview_from_ros_plane() -> Optional[Dict[str, Any]]:
    """Stereo RGB+depth plane target with persistent tracker (fleet-agent on Orin)."""
    global _plane_tracker
    try:
        from pipelines.roarm_plane_perception import PlaneTracker, build_plane_target

        provider = _get_ros_provider()
        if provider is None:
            return None
        pair = provider.read(timeout_s=2.0)
        intr = provider.get_intrinsics()
        h, w = pair.rgb_bgr.shape[:2]
        perc = _perception_cfg()
        plane_cfg = _plane_cfg()
        if _plane_tracker is None:
            _plane_tracker = PlaneTracker(max_lost=int(perc.track_lost_max_frames))
        plane_kw = _plane_tracker.tracking_kwargs()
        forced = None if _plane_tracker.locked else None
        if _plane_tracker.locked:
            forced = None
        prev_plane = None
        if _plane_tracker.locked and _plane_tracker.normal is not None:
            from pipelines.roarm_plane_perception import PlaneModel

            prev_plane = PlaneModel(
                normal=_plane_tracker.normal,
                d=_plane_tracker.d,
                rmse_m=0.0,
                inlier_frac=1.0,
            )
        target, meta = build_plane_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc,
            plane_cfg,
            intrinsics=intr,
            forced_corner_idx=forced,
            prev_plane=prev_plane,
            **plane_kw,
        )
        plane = _plane_model_from_meta(meta, target.depth_m if target.valid else 0.0)
        _plane_tracker.note_valid(target, plane if target.valid else None)
        if _plane_tracker.is_lost:
            _plane_tracker = PlaneTracker(max_lost=int(perc.track_lost_max_frames))
        tracking = _plane_tracker.tracking_kwargs()
        return _target_to_overlay(
            target,
            meta,
            w,
            h,
            track_tolerance_px=_track_tolerance_px(),
            tracking=tracking,
        )
    except Exception:
        return None


def _target_to_overlay(
    target,
    meta: dict,
    w: int,
    h: int,
    *,
    track_tolerance_px: float,
    tracking: Optional[dict] = None,
) -> Dict[str, Any]:
    if target.valid:
        status = "target_locked"
        status_text = (
            f"Цель: {target.source} · depth {target.depth_m:.2f} m · "
            f"cam_err {target.cam_err_m:+.2f} m"
        )
        _push_log(status_text)
    else:
        status = "searching"
        status_text = f"Поиск: {target.reject_reason or 'no target'}"
        _push_log(status_text)

    tracking = tracking or {}
    out = {
        "valid": target.valid,
        "status": status,
        "status_text": status_text,
        "px": target.px,
        "py": target.py,
        "anchor_px": tracking.get("anchor_px"),
        "anchor_py": tracking.get("anchor_py"),
        "track_px": tracking.get("track_px"),
        "track_py": tracking.get("track_py"),
        "image_w": w,
        "image_h": h,
        "u": round(target.u, 4),
        "v": round(target.v, 4),
        "depth_m": round(target.depth_m, 4) if target.valid else None,
        "cam_err_m": round(target.cam_err_m, 4),
        "source": target.source,
        "reject_reason": target.reject_reason or "",
        "corner_idx": target.corner_idx,
        "candidates": meta.get("candidates") or [],
        "template_score": tracking.get("template_score"),
        "plane": meta.get("plane"),
        "plane_mode": meta.get("plane_mode"),
        "track_tolerance_px": track_tolerance_px,
        "updated_at": time.time(),
    }
    return out


def collect_roarm_approach_preview(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
    *,
    interval_s: float = 0.35,
) -> Dict[str, Any]:
    """Return target overlay + status for hub telemetry."""
    global _last_at, _cache, _grid_tracker

    now = time.monotonic()
    if _cache and (now - _last_at) < interval_s:
        out = dict(_cache)
        out["log"] = list(_log)
        return out

    try:
        from pipelines.roarm_live_approach import read_live_approach

        live = read_live_approach(max_age_s=12.0)
        if live:
            live = dict(live)
            live["log"] = list(_log)
            _cache = {k: v for k, v in live.items() if k != "log"}
            _last_at = now
            return live
    except Exception:
        pass

    ros_overlay = _preview_from_ros_template()
    if ros_overlay is None:
        ros_overlay = _preview_from_ros_plane()
    if ros_overlay is not None:
        _cache = ros_overlay
        _last_at = now
        return {**_cache, "log": list(_log)}

    base = local_web.rstrip("/")
    st = fetch_json(f"{base}/api/perception/stereo_camera", 0.8) or {}
    if not st.get("ok") or not st.get("jpeg_b64"):
        reason = st.get("reason") or st.get("error") or "no_stereo_frame"
        _push_log(f"камера: {reason}")
        _last_at = now
        _cache = {
            "valid": False,
            "status": "no_frame",
            "status_text": "Нет кадра стерео",
            "reject_reason": reason,
            "track_tolerance_px": _track_tolerance_px(),
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}

    try:
        import cv2
        import numpy as np

        from pipelines.roarm_perception import TargetTracker, build_grid_target

        raw = base64.b64decode(st["jpeg_b64"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("jpeg_decode_failed")

        h, w = bgr.shape[:2]
        perc_cfg = _perception_cfg()
        if _grid_tracker is None:
            _grid_tracker = TargetTracker(
                max_jump_px=perc_cfg.track_max_jump_px,
                max_lost=perc_cfg.track_lost_max_frames,
            )
        target = build_grid_target(
            bgr,
            None,
            perc_cfg,
            **_grid_tracker.tracking_kwargs(),
        )
        _grid_tracker.note_valid(target)
        if _grid_tracker.is_lost:
            _grid_tracker = TargetTracker(
                max_jump_px=perc_cfg.track_max_jump_px,
                max_lost=perc_cfg.track_lost_max_frames,
            )
        _cache = _target_to_overlay(
            target,
            {},
            w,
            h,
            track_tolerance_px=_track_tolerance_px(),
            tracking=_grid_tracker.tracking_kwargs(),
        )
        _last_at = now
        return {**_cache, "log": list(_log)}
    except Exception as exc:
        _push_log(f"perception error: {exc}")
        _last_at = now
        _cache = {
            "valid": False,
            "status": "error",
            "status_text": str(exc),
            "reject_reason": str(exc),
            "track_tolerance_px": _track_tolerance_px(),
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}
