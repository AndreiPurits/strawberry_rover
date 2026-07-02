"""Stereo grid target preview for hub (read-only perception, no arm motion)."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_cache: Dict[str, Any] = {}
_last_at: float = 0.0
_log: list[str] = []


def _push_log(line: str, max_lines: int = 12) -> None:
    global _log
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {line}"
    if _log and _log[-1] == entry:
        return
    _log.append(entry)
    if len(_log) > max_lines:
        _log = _log[-max_lines:]


def collect_roarm_approach_preview(
    local_web: str,
    fetch_json: Callable[[str, float], Optional[dict]],
    *,
    interval_s: float = 2.0,
) -> Dict[str, Any]:
    """Return target overlay + status for hub telemetry."""
    global _last_at, _cache

    now = time.monotonic()
    if _cache and (now - _last_at) < interval_s:
        out = dict(_cache)
        out["log"] = list(_log)
        return out

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
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}

    try:
        import cv2
        import numpy as np

        from pipelines.roarm_perception import PerceptionConfig, build_grid_target

        raw = base64.b64decode(st["jpeg_b64"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("jpeg_decode_failed")

        h, w = bgr.shape[:2]
        perc_cfg = PerceptionConfig(target_camera_distance_m=0.11)
        target = build_grid_target(bgr, None, perc_cfg)

        if target.valid:
            status = "target_locked"
            status_text = (
                f"Цель: {target.source} · depth {target.depth_m:.2f} m · "
                f"cam_err {target.cam_err_m:+.2f} m"
            )
            _push_log(status_text)
        else:
            status = "searching"
            status_text = f"Поиск сетки: {target.reject_reason or 'no_grid'}"
            _push_log(status_text)

        _cache = {
            "valid": target.valid,
            "status": status,
            "status_text": status_text,
            "px": target.px,
            "py": target.py,
            "image_w": w,
            "image_h": h,
            "u": round(target.u, 4),
            "v": round(target.v, 4),
            "depth_m": round(target.depth_m, 4) if target.valid else None,
            "cam_err_m": round(target.cam_err_m, 4),
            "source": target.source,
            "reject_reason": target.reject_reason or "",
            "corner_idx": target.corner_idx,
            "updated_at": time.time(),
        }
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
            "updated_at": time.time(),
        }
        return {**_cache, "log": list(_log)}
