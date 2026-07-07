"""Shared live approach overlay state (Orin script → fleet-agent → hub)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_APPROACH_PATH = REPO_ROOT / "runs/roarm_learn/live_approach.json"


def write_live_approach(
    *,
    target: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    status: str = "running",
    status_text: str = "",
    attempt: int = 1,
    track_tolerance_px: float = 95.0,
    image_wh: Optional[list] = None,
) -> None:
    """Atomically publish target overlay for hub MJPEG canvas."""
    LIVE_APPROACH_PATH.parent.mkdir(parents=True, exist_ok=True)
    t = target or {}
    wh = image_wh or (meta or {}).get("image_wh") or [848, 480]
    payload = {
        "valid": bool(t.get("valid")),
        "status": status,
        "status_text": status_text,
        "px": t.get("px"),
        "py": t.get("py"),
        "anchor_px": (tracking or {}).get("anchor_px"),
        "anchor_py": (tracking or {}).get("anchor_py"),
        "track_px": (tracking or {}).get("final_track_px") or (tracking or {}).get("track_px"),
        "track_py": (tracking or {}).get("final_track_py") or (tracking or {}).get("track_py"),
        "image_w": int(wh[0]) if wh else 848,
        "image_h": int(wh[1]) if len(wh) > 1 else 480,
        "u": t.get("u"),
        "v": t.get("v"),
        "depth_m": t.get("depth_m"),
        "cam_err_m": t.get("cam_err_m"),
        "source": t.get("source", ""),
        "reject_reason": t.get("reject_reason", ""),
        "corner_idx": t.get("corner_idx", -1),
        "candidates": (meta or {}).get("candidates") or [],
        "template_score": (tracking or {}).get("template_score"),
        "plane": (meta or {}).get("plane"),
        "plane_mode": (meta or {}).get("plane_mode"),
        "track_tolerance_px": float(track_tolerance_px),
        "attempt": int(attempt),
        "updated_at": time.time(),
    }
    tmp = LIVE_APPROACH_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LIVE_APPROACH_PATH)


def read_live_approach(*, max_age_s: float = 6.0) -> Optional[Dict[str, Any]]:
    if not LIVE_APPROACH_PATH.is_file():
        return None
    age = time.time() - LIVE_APPROACH_PATH.stat().st_mtime
    if age > max_age_s:
        return None
    try:
        return json.loads(LIVE_APPROACH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_live_approach() -> None:
    try:
        if LIVE_APPROACH_PATH.is_file():
            LIVE_APPROACH_PATH.unlink()
    except OSError:
        pass
