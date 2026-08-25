"""Manual plastic_lab snap: fresh RGB + crop + meta. No arm motion."""
from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "peduncle_plastic_lab" / "manual_capture"
MAX_RGB_AGE_S = 1.5


def _next_shot_id(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in out_dir.glob("plastic_*.json"):
        try:
            n = int(p.stem.split("_")[1])
            best = max(best, n)
        except (IndexError, ValueError):
            continue
    return best + 1


def _fetch_stereo_fresh(local_web: str = "http://127.0.0.1:8080", *, attempts: int = 4) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    """Return (bgr, stamp, error). Rejects stale frames (age > MAX_RGB_AGE_S)."""
    url = f"{local_web.rstrip('/')}/api/perception/stereo_camera"
    last_err = "no_rgb"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            last_err = f"fetch_failed:{exc}"
            time.sleep(0.08)
            continue
        if not data or not data.get("ok") or not data.get("jpeg_b64"):
            last_err = "stereo_camera_empty"
            time.sleep(0.08)
            continue
        stamp = data.get("stamp") or data.get("timestamp")
        try:
            stamp_f = float(stamp) if stamp is not None else None
        except (TypeError, ValueError):
            stamp_f = None
        now = time.time()
        # ROS stamps can be sim-time; also accept wall clock fields
        age = None
        if stamp_f is not None:
            # If stamp looks like ROS epoch far from wall clock, skip age gate
            if abs(now - stamp_f) < 3600:
                age = now - stamp_f
            else:
                wall = data.get("wall_stamp") or data.get("recv_stamp")
                if wall is not None:
                    try:
                        age = now - float(wall)
                    except (TypeError, ValueError):
                        age = None
        if age is not None and age > MAX_RGB_AGE_S:
            last_err = f"stale_rgb_age={age:.2f}s"
            time.sleep(0.12)
            continue
        raw = base64.b64decode(data["jpeg_b64"])
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            last_err = "jpeg_decode_failed"
            continue
        return bgr, stamp_f if stamp_f is not None else now, None
    return None, None, last_err


def _detect_berry(bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        from pipelines.roarm_strawberry_target import (
            StrawberryTargetTracker,
            _get_detector,
            detect_strawberry_in_frame,
            min_berry_conf,
        )
    except Exception:
        return None
    tracker = StrawberryTargetTracker(strict_lock=False, lock_radius_px=160)
    det_model = _get_detector(REPO)
    det = detect_strawberry_in_frame(bgr, det_model, tracker)
    if det is None:
        return None
    conf = float(det.get("conf", 0))
    if conf < float(min_berry_conf()) * 0.5:  # allow slightly softer for lab snaps
        # still keep if any reasonable detection
        if conf < 0.25:
            return None
    x1, y1, x2, y2 = float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "px": 0.5 * (x1 + x2),
        "py": 0.5 * (y1 + y2),
        "conf": conf,
    }


def _sample_depth(px: float, py: float) -> Optional[float]:
    try:
        from pipelines.roarm_perception import PerceptionConfig, sample_depth_median
        from pipelines.ros_rgb_depth import Ros2RgbDepthProvider

        provider = Ros2RgbDepthProvider(
            rgb_topic="/stereo_camera/color/image_rect_raw",
            depth_topic="/stereo_camera/depth/image_rect_raw",
            sync_slop_s=0.2,
        )
        provider.open(camera_info_topic="/stereo_camera/color/camera_info")
        pair = provider.read(timeout_s=1.2)
        if pair is None or pair.depth_m is None:
            return None
        from pipelines.roarm_strawberry_target import _letterbox_array

        # hub JPEG is letterboxed 640x480; depth should match
        h, w = 480, 640
        depth_hub = _letterbox_array(pair.depth_m, w, h)
        perc = PerceptionConfig()
        d, _ = sample_depth_median(depth_hub, int(px), int(py), perc.depth_median_radius, perc)
        return float(d) if d is not None else None
    except Exception:
        return None


def _focus_crop(bgr: np.ndarray, berry: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, list]:
    h, w = bgr.shape[:2]
    if berry and berry.get("bbox_xyxy"):
        bbox = tuple(float(v) for v in berry["bbox_xyxy"])
    else:
        # center crop fallback
        side = min(w, h) // 2
        cx, cy = w // 2, h // 2
        bbox = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
    try:
        from pipelines.peduncle_v3.geometry import build_focus_roi

        roi = build_focus_roi(bbox, (w, h))  # type: ignore[arg-type]
    except Exception:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 40
        roi = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    rx1, ry1, rx2, ry2 = roi
    return bgr[ry1:ry2, rx1:rx2].copy(), [int(rx1), int(ry1), int(rx2), int(ry2)]


def count_shots(out_dir: Optional[Path] = None) -> int:
    d = Path(out_dir or DEFAULT_OUT)
    if not d.is_dir():
        return 0
    return len(list(d.glob("plastic_*.json")))


def take_manual_snap(
    *,
    joints: Optional[Dict[str, float]] = None,
    out_dir: Optional[Path] = None,
    local_web: str = "http://127.0.0.1:8080",
) -> Dict[str, Any]:
    """Capture-only. Never moves the arm."""
    out = Path(out_dir or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)

    bgr, stamp, err = _fetch_stereo_fresh(local_web)
    if bgr is None:
        return {"ok": False, "error": err or "no_fresh_rgb", "count": count_shots(out)}

    berry = _detect_berry(bgr)
    depth = None
    if berry is not None:
        depth = _sample_depth(berry["px"], berry["py"])

    crop, roi = _focus_crop(bgr, berry)
    shot_n = _next_shot_id(out)
    shot_id = f"plastic_{shot_n:04d}"
    full_path = out / f"{shot_id}_full.jpg"
    crop_path = out / f"{shot_id}_crop.jpg"
    meta_path = out / f"{shot_id}.json"
    cv2.imwrite(str(full_path), bgr)
    cv2.imwrite(str(crop_path), crop)

    meta = {
        "shot_id": shot_id,
        "ts": time.time(),
        "rgb_stamp": stamp,
        "full_path": str(full_path.relative_to(REPO)) if str(full_path).startswith(str(REPO)) else str(full_path),
        "crop_path": str(crop_path.relative_to(REPO)) if str(crop_path).startswith(str(REPO)) else str(crop_path),
        "image_wh": [int(bgr.shape[1]), int(bgr.shape[0])],
        "joints": joints or {},
        "berry_bbox_xyxy": (berry or {}).get("bbox_xyxy"),
        "berry_px": (berry or {}).get("px"),
        "berry_py": (berry or {}).get("py"),
        "berry_conf": (berry or {}).get("conf"),
        "depth_m": depth,
        "peduncle_roi_xyxy": roi,
        "domain": "plastic_lab",
        "capture_mode": "manual_button",
        "berry_found": berry is not None,
    }
    try:
        meta["full_path"] = str(full_path.resolve().relative_to(REPO.resolve()))
        meta["crop_path"] = str(crop_path.resolve().relative_to(REPO.resolve()))
    except Exception:
        pass
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ok, buf = cv2.imencode(".jpg", crop)
    crop_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    return {
        "ok": True,
        "shot_id": shot_id,
        "full_path": meta["full_path"],
        "crop_path": meta["crop_path"],
        "count": count_shots(out),
        "berry_found": berry is not None,
        "berry_px": meta.get("berry_px"),
        "berry_py": meta.get("berry_py"),
        "berry_conf": meta.get("berry_conf"),
        "depth_m": depth,
        "crop_jpeg_b64": crop_b64,
    }
