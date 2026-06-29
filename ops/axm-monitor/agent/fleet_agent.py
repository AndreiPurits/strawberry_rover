#!/usr/bin/env python3
"""Orin fleet agent: telemetry + remote control via rover.axm.tech hub."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from gnss_reader import gnss_snapshot, start_gnss_reader
from ntrip_client import ntrip_configured
from mega_client import port_busy, port_exists, probe_mega, send_command, twist_to_pwm
from roarm_proxy import execute_rpc, roarm_enabled, telemetry_snapshot as roarm_telemetry

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tools", "rover_mega"))
from tank_drive import ui_to_cmd_vel, ui_to_pwm, ui_to_tracks  # noqa: E402

_DRIVE_MODE = "joystick"
_last_lidar_stamp: Optional[float] = None
_last_lidar_arc: Dict[str, Any] = {}
_last_drive_at: float = 0.0
_last_drive_linear_x: float = 0.0
_motors_active: bool = False
_lidar_stop_latched_fwd: bool = False
_lidar_stop_latched_rev: bool = False
_lidar_clear_streak_fwd: int = 0
_lidar_clear_streak_rev: int = 0
_lidar_guard_lock = threading.Lock()
_lidar_override_until: float = 0.0
_session_active: bool = False
_last_stereo_camera_stamp: Optional[float] = None
_last_stereo_telemetry: Dict[str, Any] = {}
_stereo_post_times: List[float] = []
_stereo_stream_fps_ema: Optional[float] = None
_last_camera_stamp: Optional[float] = None
_last_heartbeat_rtt_ms: Optional[float] = None
_last_hub_ok_at: float = 0.0
_hub_ok_lock = threading.Lock()

WATCHDOG_TIMEOUT_S = 0.5
ROVER_WIDTH_M = 0.25
LIDAR_GUARD_M = 0.40
LIDAR_OVERRIDE_HOLD_S = 0.55
try:
    LIDAR_GUARD_RELEASE_M = float(os.environ.get("AXM_LIDAR_GUARD_RELEASE_M", str(LIDAR_GUARD_M + 0.08)))
except ValueError:
    LIDAR_GUARD_RELEASE_M = LIDAR_GUARD_M + 0.08
try:
    LIDAR_GUARD_CLEAR_FRAMES = max(1, int(os.environ.get("AXM_LIDAR_GUARD_CLEAR_FRAMES", "3")))
except ValueError:
    LIDAR_GUARD_CLEAR_FRAMES = 3
TELEMETRY_INTERVAL_S = 5.0
KEEPALIVE_INTERVAL_S = 2.0
COMMAND_POLL_S = 0.1
HUB_LINK_GRACE_S = 45.0
LIDAR_ARC_MAX_AGE_S = 0.8


def _drive_signs() -> Tuple[float, float]:
    try:
        fs = float(_env("AXM_FORWARD_SIGN", "-1"))
    except ValueError:
        fs = -1.0
    try:
        ts = float(_env("AXM_TURN_SIGN", "1"))
    except ValueError:
        ts = 1.0
    return fs, ts


def _remap_ui_drive(forward: float, turn: float) -> Tuple[float, float]:
    """UI W/S + A/D -> cmd_vel. Turn opposes tracks (tank skid-steer)."""
    fs, ts = _drive_signs()
    return ui_to_cmd_vel(forward, turn, forward_sign=fs, turn_sign=ts)


def _apply_lidar_guard_twist(
    linear_x: float, angular_z: float, override: bool
) -> Tuple[float, float, Dict[str, Any], bool]:
    """Block physical forward/backward motion from LiDAR latch (after drive remap)."""
    guard = _current_lidar_guard()
    lx = float(linear_x)
    az = float(angular_z)
    if lx > 0:
        if bool(guard.get("latched_forward")) and not override:
            return 0.0, az, guard, True
        return lx, az, guard, False
    if override:
        return lx, az, guard, False
    if lx < 0:
        if bool(guard.get("latched_backward")):
            return 0.0, az, guard, True
        return lx, az, guard, False
    return lx, az, guard, False


def _camera_fps() -> float:
    raw = _env("AXM_CAMERA_FPS", "30")
    try:
        fps = float(raw)
    except ValueError:
        fps = 30.0
    return fps if fps > 0 else 30.0


def _hub_camera_fps() -> float:
    """JPEG upload rate to hub for front camera."""
    raw = _env("AXM_HUB_CAMERA_FPS", "12")
    try:
        fps = float(raw)
    except ValueError:
        fps = 12.0
    return max(0.5, min(fps, 15.0))


def _camera_stream_loop_interval() -> float:
    """Loop cadence — must be fast enough for stereo hub_fps."""
    front_iv = 1.0 / _hub_camera_fps()
    if not _hub_stereo_enabled():
        return front_iv
    return min(front_iv, 1.0 / _hub_stereo_fps())


def _hub_stereo_enabled() -> bool:
    return _env("AXM_HUB_STEREO", "true").lower() not in ("0", "false", "no")


def _hub_stereo_fps() -> float:
    raw = _env("AXM_HUB_STEREO_FPS", "10")
    try:
        fps = float(raw)
    except ValueError:
        fps = 10.0
    return max(0.25, min(fps, 10.0))


def _touch_hub_ok() -> None:
    global _last_hub_ok_at
    with _hub_ok_lock:
        _last_hub_ok_at = time.monotonic()


def _hub_link_ok() -> bool:
    with _hub_ok_lock:
        return (time.monotonic() - _last_hub_ok_at) <= HUB_LINK_GRACE_S


def _guard_half_angle_rad() -> float:
    """Half-angle of forward guard cone: rover width at guard distance."""
    raw = _env("AXM_LIDAR_GUARD_HALF_ANGLE_DEG", "")
    if raw:
        try:
            return math.radians(max(5.0, min(float(raw), 89.0)))
        except ValueError:
            pass
    return math.atan((ROVER_WIDTH_M / 2.0) / LIDAR_GUARD_M)


def _guard_full_angle_deg() -> float:
    return round(math.degrees(_guard_half_angle_rad()) * 2.0, 1)


def _note_lidar_override(active: bool) -> None:
    global _lidar_override_until
    if active:
        _lidar_override_until = time.monotonic() + LIDAR_OVERRIDE_HOLD_S


def _lidar_override_live() -> bool:
    return time.monotonic() < _lidar_override_until


def _forward_guard_cone(lidar_arc: Dict[str, Any]) -> Tuple[Optional[float], List[int]]:
    """Min distance and sector indices strictly inside the forward guard cone."""
    half_rad = _guard_half_angle_rad()
    idx = _sector_indices_in_cone(lidar_arc, 0.0, half_rad)
    return _min_dist_from_indices(lidar_arc, idx), idx


def _sector_indices_in_cone(lidar_arc: Dict[str, Any], center_rad: float, half_angle_rad: float) -> List[int]:
    sectors = lidar_arc.get("sectors") or []
    n = len(sectors)
    if n == 0:
        return []
    fov_deg = float(lidar_arc.get("fov_deg") or 160.0)
    fov_half = math.radians(fov_deg / 2.0)
    sector_width = math.radians(fov_deg) / n
    indices: List[int] = []
    for i in range(n):
        sec_center = -fov_half + (i + 0.5) * sector_width
        delta = math.atan2(math.sin(sec_center - center_rad), math.cos(sec_center - center_rad))
        if abs(delta) <= (half_angle_rad + sector_width * 0.5):
            indices.append(i)
    return indices


def _min_dist_from_indices(lidar_arc: Dict[str, Any], indices: List[int]) -> Optional[float]:
    sectors = lidar_arc.get("sectors") or []
    if not indices:
        return None
    dists: List[float] = []
    for i in indices:
        if i < 0 or i >= len(sectors):
            continue
        d = sectors[i].get("dist_m")
        if d is not None:
            dists.append(float(d))
    return min(dists) if dists else None


def _lidar_guard_state(lidar_arc: Dict[str, Any]) -> Dict[str, Any]:
    half_rad = _guard_half_angle_rad()
    half_deg = math.degrees(half_rad)
    fov_deg = float(lidar_arc.get("fov_deg") or 160.0)
    stamp = lidar_arc.get("stamp")
    now = time.time()
    try:
        age_s = (now - float(stamp)) if stamp is not None else None
    except Exception:
        age_s = None
    lidar_stale = age_s is None or age_s > float(
        _env("AXM_LIDAR_ARC_MAX_AGE_S", str(LIDAR_ARC_MAX_AGE_S))
    )
    m_cone, idx_fwd = _forward_guard_cone(lidar_arc)
    sectors = lidar_arc.get("sectors") or []
    center_idx = len(sectors) // 2 if sectors else 0
    center_d = None
    if sectors and center_idx in idx_fwd and 0 <= center_idx < len(sectors):
        raw_center = sectors[center_idx].get("dist_m")
        if raw_center is not None:
            center_d = float(raw_center)
    # STOP only from points inside the ~40° forward cone; all other sectors ignored.
    blocked_fwd = (m_cone is not None) and (m_cone < LIDAR_GUARD_M)
    blocked_rev = False
    return {
        "active": blocked_fwd,
        "active_forward": blocked_fwd,
        "active_backward": blocked_rev,
        "min_forward_m": round(m_cone, 3) if m_cone is not None else None,
        "min_backward_m": None,
        "threshold_m": LIDAR_GUARD_M,
        "release_m": round(LIDAR_GUARD_RELEASE_M, 3),
        "rover_width_m": ROVER_WIDTH_M,
        "guard_half_angle_deg": round(half_deg, 1),
        "guard_full_angle_deg": round(2.0 * half_deg, 1),
        "sector_indices": idx_fwd,
        "rear_sector_indices": [],
        "forward_data_ok": (m_cone is not None) and (not lidar_stale),
        "forward_cone_ok": m_cone is not None,
        "center_forward_m": round(center_d, 3) if center_d is not None else None,
        "lidar_stale": lidar_stale,
        "lidar_age_s": round(age_s, 3) if age_s is not None else None,
    }


def _apply_lidar_latches(guard: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hysteresis/latch so guard state does not flap on noisy scans."""
    global _lidar_stop_latched_fwd, _lidar_stop_latched_rev
    global _lidar_clear_streak_fwd, _lidar_clear_streak_rev

    m_center = guard.get("center_forward_m")
    m_rev = guard.get("min_backward_m")

    with _lidar_guard_lock:
        if guard.get("active_forward"):
            _lidar_stop_latched_fwd = True
            _lidar_clear_streak_fwd = 0
        elif (
            guard.get("forward_data_ok")
            and (m_center is not None)
            and (float(m_center) >= LIDAR_GUARD_RELEASE_M)
        ):
            _lidar_clear_streak_fwd += 1
            if _lidar_clear_streak_fwd >= LIDAR_GUARD_CLEAR_FRAMES:
                _lidar_stop_latched_fwd = False
        else:
            _lidar_clear_streak_fwd = 0

        if guard.get("active_backward"):
            _lidar_stop_latched_rev = True
            _lidar_clear_streak_rev = 0
        elif m_rev is None:
            # No rear data in front-only FOV modes: never block reverse by stale rear latch.
            _lidar_stop_latched_rev = False
            _lidar_clear_streak_rev = 0
        elif float(m_rev) >= LIDAR_GUARD_RELEASE_M:
            _lidar_clear_streak_rev += 1
            if _lidar_clear_streak_rev >= LIDAR_GUARD_CLEAR_FRAMES:
                _lidar_stop_latched_rev = False
        else:
            _lidar_clear_streak_rev = 0

        latched_fwd = _lidar_stop_latched_fwd
        latched_rev = _lidar_stop_latched_rev
        clear_fwd = _lidar_clear_streak_fwd
        clear_rev = _lidar_clear_streak_rev

    guard["latched_forward"] = latched_fwd
    guard["latched_backward"] = latched_rev
    guard["clear_streak_forward"] = clear_fwd
    guard["clear_streak_backward"] = clear_rev
    guard["active_effective_forward"] = latched_fwd
    guard["active_effective_backward"] = latched_rev
    # UI / telemetry STOP is forward-only; reverse guard still blocks backward motion.
    guard["active_effective"] = latched_fwd
    guard["active"] = latched_fwd
    return guard


def _current_lidar_guard() -> Dict[str, Any]:
    return _apply_lidar_latches(_lidar_guard_state(_last_lidar_arc))


def _apply_lidar_guard(fwd: float, override: bool) -> Tuple[float, Dict[str, Any], bool]:
    """Legacy UI-forward guard (prefer _apply_lidar_guard_twist after remap)."""
    lx, _az, guard, blocked = _apply_lidar_guard_twist(
        _remap_ui_drive(fwd, 0.0)[0], 0.0, override
    )
    if fwd > 0 and blocked:
        return 0.0, guard, True
    if fwd < 0 and blocked:
        return 0.0, guard, True
    return fwd, guard, False


def _refresh_lidar_arc(local_web: str) -> bool:
    """Pull latest /scan sectors from local ros_bridge (fast path for guard)."""
    global _last_lidar_stamp, _last_lidar_arc
    base = local_web.rstrip("/")
    arc = _fetch_json(f"{base}/api/perception/lidar_arc", 0.15) or {}
    if not arc.get("ok"):
        return False
    arc_data = {k: v for k, v in arc.items() if k != "ok"}
    _last_lidar_arc = arc_data
    stamp = arc_data.get("stamp")
    if stamp != _last_lidar_stamp:
        _last_lidar_stamp = stamp
    return True


def _stereo_stream_fps() -> Optional[float]:
    global _stereo_stream_fps_ema
    with _hub_ok_lock:
        times = list(_stereo_post_times)
        ema = _stereo_stream_fps_ema
    if len(times) < 2:
        return round(ema, 1) if ema is not None else None
    span = times[-1] - times[0]
    if span <= 0:
        return round(ema, 1) if ema is not None else None
    instant = (len(times) - 1) / span
    if ema is None:
        ema = instant
    else:
        ema = 0.65 * ema + 0.35 * instant
    with _hub_ok_lock:
        _stereo_stream_fps_ema = ema
    return round(ema, 1)


def _note_stereo_post() -> None:
    global _stereo_post_times
    now = time.monotonic()
    with _hub_ok_lock:
        _stereo_post_times.append(now)
        if len(_stereo_post_times) > 20:
            _stereo_post_times = _stereo_post_times[-20:]


def _refresh_stereo_stats(base: str) -> None:
    """Pull brightness/tuning from local web — not only on hub POST."""
    global _last_stereo_telemetry
    if not _hub_stereo_enabled():
        return
    st = _fetch_json(f"{base}/api/perception/stereo_camera", 0.35) or {}
    if not st.get("ok"):
        return
    merged = dict(_last_stereo_telemetry)
    merged.update(
        {
            "hub_fps": _hub_stereo_fps(),
            "stream_fps": _stereo_stream_fps(),
            "brightness_mean": st.get("brightness_mean"),
            "brightness_ok": st.get("brightness_ok"),
            "brightness": st.get("brightness"),
            "gamma": st.get("gamma"),
            "tuning": st.get("tuning"),
            "trial_brightness": st.get("trial_brightness"),
            "trial_gamma": st.get("trial_gamma"),
            "last_trial_mean": st.get("last_trial_mean"),
            "trials_total": st.get("trials_total"),
            "target_min": st.get("target_min"),
            "target_max": st.get("target_max"),
        }
    )
    _last_stereo_telemetry = merged


def _collect_perception(local_web: str) -> Dict[str, Any]:
    base = local_web.rstrip("/")
    out: Dict[str, Any] = {}
    if _refresh_lidar_arc(base):
        out["lidar_arc"] = dict(_last_lidar_arc)
    out["lidar_guard"] = _current_lidar_guard()
    _refresh_stereo_stats(base)
    if _last_stereo_telemetry:
        out["stereo"] = dict(_last_stereo_telemetry)
    return out


def _link_status(rtt_ms: Optional[float], camera_age_ms: Optional[float]) -> str:
    if rtt_ms is None:
        return "red"
    if rtt_ms < 150 and (camera_age_ms is None or camera_age_ms < 300):
        return "green"
    if rtt_ms < 400 and (camera_age_ms is None or camera_age_ms < 800):
        return "yellow"
    return "red"


def _camera_age_ms() -> Optional[float]:
    if _last_camera_stamp is None:
        return None
    return round((time.time() - float(_last_camera_stamp)) * 1000.0, 1)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _fetch_json(url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json_retry(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 8.0,
    attempts: int = 8,
    base_delay_s: float = 0.12,
) -> Dict[str, Any]:
    """Retry POST for flaky WiFi / VPS routes (heartbeat must get through)."""
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return _post_json(url, payload, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(base_delay_s * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("post_retry_failed")


def _post_bytes_retry(
    url: str,
    data: bytes,
    *,
    headers: Dict[str, str],
    timeout: float = 6.0,
    attempts: int = 3,
) -> None:
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(0.08 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def _post_stereo_camera_frame(
    hub_url: str,
    rover_id: str,
    token: str,
    jpeg_bytes: bytes,
    stamp: float,
) -> None:
    global _last_stereo_camera_stamp
    q = urllib.parse.urlencode({"rover_id": rover_id, "token": token})
    url = f"{hub_url.rstrip('/')}/api/agents/stereo_camera_frame?{q}"
    _post_bytes_retry(
        url,
        jpeg_bytes,
        headers={"Content-Type": "image/jpeg", "X-Frame-Stamp": str(stamp)},
        timeout=5.0,
        attempts=2,
    )
    _last_stereo_camera_stamp = float(stamp)


def _post_camera_frame(
    hub_url: str,
    rover_id: str,
    token: str,
    jpeg_bytes: bytes,
    stamp: float,
) -> None:
    global _last_camera_stamp
    q = urllib.parse.urlencode({"rover_id": rover_id, "token": token})
    url = f"{hub_url.rstrip('/')}/api/agents/camera_frame?{q}"
    _post_bytes_retry(
        url,
        jpeg_bytes,
        headers={"Content-Type": "image/jpeg", "X-Frame-Stamp": str(stamp)},
        timeout=5.0,
        attempts=2,
    )
    _last_camera_stamp = float(stamp)


def _camera_stream_loop(
    hub_url: str,
    rover_id: str,
    token: str,
    local_web: str,
    stop_event: threading.Event,
) -> None:
    global _last_stereo_telemetry
    interval = _camera_stream_loop_interval()
    stereo_interval = 1.0 / _hub_stereo_fps()
    last_stereo_at = 0.0
    base = local_web.rstrip("/")
    stereo = _hub_stereo_enabled()
    while not stop_event.is_set():
        if not _hub_link_ok():
            stop_event.wait(interval)
            continue
        try:
            health = _fetch_json(f"{base}/api/health", 0.3) or {}
            if not health.get("bridge_active"):
                stop_event.wait(interval)
                continue
            cam = _fetch_json(f"{base}/api/perception/front_camera?hub=1", 0.4) or {}
            if cam.get("ok") and cam.get("jpeg_b64"):
                jpeg = base64.b64decode(cam["jpeg_b64"])
                if len(jpeg) <= 120_000:
                    stamp = float(cam.get("stamp") or time.time())
                    _post_camera_frame(hub_url, rover_id, token, jpeg, stamp)
                    _touch_hub_ok()
            if stereo and (time.monotonic() - last_stereo_at) >= stereo_interval:
                st = _fetch_json(f"{base}/api/perception/stereo_camera", 0.4) or {}
                if st.get("ok") and st.get("jpeg_b64"):
                    sjpeg = base64.b64decode(st["jpeg_b64"])
                    if len(sjpeg) <= 120_000:
                        sstamp = float(st.get("stamp") or time.time())
                        _post_stereo_camera_frame(hub_url, rover_id, token, sjpeg, sstamp)
                        _note_stereo_post()
                        _refresh_stereo_stats(base)
                        last_stereo_at = time.monotonic()
        except Exception:
            pass
        stop_event.wait(interval)


def _parse_arduino(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], dict):
            return raw["data"]
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return {}


def _mega_status_dict(arduino_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Mega STATUS fields for the hub dashboard."""
    mega: Dict[str, Any] = {}
    status_raw = arduino_data.get("status")
    if isinstance(status_raw, str) and status_raw.startswith("{"):
        try:
            mega = json.loads(status_raw)
        except json.JSONDecodeError:
            pass
    elif isinstance(status_raw, dict):
        mega = dict(status_raw)

    sensor_keys = (
        "armed",
        "fl_us",
        "fr_us",
        "rl_us",
        "rr_us",
        "left_us",
        "right_us",
        "left_power_pct",
        "right_power_pct",
        "current_a0",
        "current_d0",
        "current_d22",
        "temp_c",
        "humidity_pct",
        "vibration_d24",
        "dht_ok",
    )
    for key in sensor_keys:
        if key in arduino_data and arduino_data[key] is not None:
            mega[key] = arduino_data[key]

    left = int(mega.get("left_us") or 1500)
    right = int(mega.get("right_us") or 1500)

    def track_pct(us: int) -> float:
        return max(-100.0, min(100.0, (us - 1500) / 5.0))

    def signed_power(us: int, power: Any) -> float:
        if power is None:
            return track_pct(us)
        p = float(power)
        if us > 1500:
            return p
        if us < 1500:
            return -p
        return 0.0

    left_pct = signed_power(left, mega.get("left_power_pct"))
    right_pct = signed_power(right, mega.get("right_power_pct"))
    linear_pct = (left_pct + right_pct) / 2.0

    return {
        "connected": arduino_data.get("connected"),
        "armed": mega.get("armed") if mega.get("armed") is not None else arduino_data.get("mega_armed"),
        "mega_armed": mega.get("armed") if mega.get("armed") is not None else arduino_data.get("mega_armed"),
        "web_started": arduino_data.get("web_started"),
        "web_manual": arduino_data.get("web_manual"),
        "fl_us": mega.get("fl_us"),
        "fr_us": mega.get("fr_us"),
        "rl_us": mega.get("rl_us"),
        "rr_us": mega.get("rr_us"),
        "left_us": left,
        "right_us": right,
        "left_power_pct": mega.get("left_power_pct"),
        "right_power_pct": mega.get("right_power_pct"),
        "left_pct": round(left_pct, 1),
        "right_pct": round(right_pct, 1),
        "linear_pct": round(linear_pct, 1),
        "speed_mps": round(abs(linear_pct) * 0.015, 2),
        "current_a0": mega.get("current_a0"),
        "current_d0": mega.get("current_d0"),
        "current_d22": mega.get("current_d22"),
        "temp_c": mega.get("temp_c"),
        "humidity_pct": mega.get("humidity_pct"),
        "vibration_d24": mega.get("vibration_d24"),
        "dht_ok": mega.get("dht_ok"),
    }


def collect_telemetry(local_web: str, mega_port: str) -> Dict[str, Any]:
    base = local_web.rstrip("/")
    health = _fetch_json(f"{base}/api/health") or {}
    arduino_api = _fetch_json(f"{base}/api/arduino/status") or {}
    arduino_data = _parse_arduino(arduino_api.get("data") or arduino_api.get("raw") or health.get("arduino"))

    connected = arduino_data.get("connected")
    source = "local_web" if health.get("bridge_active") else "none"

    if connected is not True:
        if port_exists(mega_port) and port_busy(mega_port):
            connected = True
            source = "serial_locked"
        else:
            probe = probe_mega(mega_port)
            if probe.get("connected"):
                connected = True
            if probe.get("status"):
                arduino_data["status"] = probe["status"]
            if probe.get("armed") is not None:
                arduino_data["armed"] = probe["armed"]
            arduino_data = {**arduino_data, **{k: v for k, v in probe.items() if k != "status"}}
            if source == "none" or not health.get("bridge_active"):
                source = str(probe.get("source") or "serial_probe")
            elif connected:
                source = "local_web+mega_probe"

    arduino_data["connected"] = connected
    mega = _mega_status_dict(arduino_data)
    perception = _collect_perception(base) if health.get("bridge_active") else {}
    rtk = gnss_snapshot()

    cam_age = _camera_age_ms()
    rtt = _last_heartbeat_rtt_ms
    return {
        "hostname": socket.gethostname(),
        "agent": "orin",
        "mega_port": mega_port,
        "arduino_connected": connected,
        "armed": mega.get("armed"),
        "bridge_active": health.get("bridge_active"),
        "health_ok": health.get("ok"),
        "telemetry_source": source,
        "drive_mode": _DRIVE_MODE,
        "session_active": _session_active,
        "mega": mega,
        "perception": perception,
        "rtk": rtk,
        "roarm": roarm_telemetry(),
        "link": {
            "rtt_ms": rtt,
            "camera_age_ms": cam_age,
            "status": _link_status(rtt, cam_age),
        },
    }


def _local_web_post(local_web: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{local_web.rstrip('/')}{path}"
    if payload is None:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode())
    return _post_json(url, payload)


def _drive_local(local_web: str, left: float, right: float) -> Dict[str, Any]:
    return _local_web_post(
        local_web,
        "/api/control/tracks",
        {"left": left, "right": right, "source": "fleet"},
    )


def _drive_local_twist(local_web: str, linear_x: float, angular_z: float) -> Dict[str, Any]:
    return _local_web_post(
        local_web,
        "/api/control/command",
        {
            "command": {"linear_x": linear_x, "angular_z": angular_z},
            "source": "fleet",
        },
    )


def _drive_serial(mega_port: str, linear_x: float, angular_z: float) -> Dict[str, Any]:
    fl, fr, rl, rr = twist_to_pwm(linear_x, angular_z)
    line = f"M FL={fl} FR={fr} RL={rl} RR={rr}"
    return send_command(mega_port, line)


def _touch_drive(linear_x: float, angular_z: float) -> None:
    global _last_drive_at, _last_drive_linear_x, _motors_active
    _last_drive_at = time.monotonic()
    _last_drive_linear_x = float(linear_x)
    _motors_active = abs(linear_x) > 1e-4 or abs(angular_z) > 1e-4


def _stop_motors(local_web: str, prefer_web: bool, mega_port: str) -> None:
    global _motors_active, _last_drive_linear_x
    if prefer_web:
        _drive_local(local_web, 0.0, 0.0)
    else:
        send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
    _motors_active = False
    _last_drive_linear_x = 0.0


def _watchdog_tick(local_web: str, prefer_web: bool, mega_port: str) -> None:
    if not _session_active:
        return
    if _motors_active and (time.monotonic() - _last_drive_at) > WATCHDOG_TIMEOUT_S:
        _stop_motors(local_web, prefer_web, mega_port)


def execute_command(
    cmd: Dict[str, Any],
    *,
    local_web: str,
    mega_port: str,
    prefer_web: bool,
) -> Dict[str, Any]:
    global _DRIVE_MODE, _session_active
    action = str(cmd.get("action", "")).strip().lower()
    params = cmd.get("params") or {}
    cmd_id = cmd.get("id")

    result: Dict[str, Any] = {"id": cmd_id, "action": action, "ok": False}

    try:
        if action == "set_drive_mode":
            mode = str(params.get("mode", "joystick")).lower()
            if mode not in ("joystick", "auto"):
                result["error"] = "invalid_mode"
                return result
            _DRIVE_MODE = mode
            if mode == "joystick":
                _touch_drive(0.0, 0.0)
                _motors_active = False
                if prefer_web:
                    _drive_local(local_web, 0.0, 0.0)
                result.update({"ok": True, "drive_mode": mode})
            else:
                _session_active = False
                _motors_active = False
                if prefer_web:
                    _local_web_post(local_web, "/api/control/stop")
                    _local_web_post(local_web, "/api/control/mode", {"mode": "auto"})
                else:
                    send_command(mega_port, "DISARM")
                    send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
                result.update({"ok": True, "drive_mode": mode})

        elif action == "session_start":
            if _DRIVE_MODE != "joystick":
                result["error"] = "not_in_joystick_mode"
                return result
            _session_active = True
            _touch_drive(0.0, 0.0)
            if prefer_web:
                # Pulse stop→start so Mega driver always sees started edge and re-ARMs.
                _local_web_post(local_web, "/api/control/stop")
                _local_web_post(local_web, "/api/control/start")
                _local_web_post(local_web, "/api/control/mode", {"mode": "manual"})
                result.update({"ok": True, "via": "local_web"})
            else:
                r = send_command(mega_port, "ARM")
                result.update({"ok": r.get("ok"), "via": "serial", "response": r.get("response")})

        elif action in ("session_stop", "disarm"):
            _session_active = False
            _stop_motors(local_web, prefer_web, mega_port)
            if prefer_web:
                _local_web_post(local_web, "/api/control/stop")
                result.update({"ok": True, "via": "local_web"})
            else:
                r = send_command(mega_port, "DISARM")
                result.update({"ok": r.get("ok"), "via": "serial", "response": r.get("response")})

        elif action == "arm":
            if _DRIVE_MODE != "joystick":
                result["error"] = "not_in_joystick_mode"
                return result
            if prefer_web:
                _local_web_post(local_web, "/api/control/start")
                _local_web_post(local_web, "/api/control/mode", {"mode": "manual"})
                result.update({"ok": True, "via": "local_web"})
            else:
                r = send_command(mega_port, "ARM")
                result.update({"ok": r.get("response") == "OK", "via": "serial", "response": r.get("response")})

        elif action in ("drive", "command"):
            if _DRIVE_MODE != "joystick":
                result["error"] = "auto_mode_active"
                return result
            if not _session_active:
                result["error"] = "session_not_active"
                return result
            _refresh_lidar_arc(local_web)
            scale = max(0.08, min(1.0, float(params.get("speed_scale", 1.0))))
            fwd = float(params.get("forward", params.get("linear_x", 0.0))) * scale
            turn = float(params.get("turn", params.get("angular_z", 0.0))) * scale
            lidar_override = bool(params.get("lidar_override"))
            _note_lidar_override(lidar_override)
            fwd, guard, guard_blocked = _apply_lidar_guard(fwd, lidar_override)
            fs, ts = _drive_signs()
            l_tr, r_tr = ui_to_tracks(fwd, turn, forward_sign=fs, turn_sign=ts)
            fl, fr, _, _ = ui_to_pwm(fwd, turn, forward_sign=fs, turn_sign=ts)
            if abs(fwd) > 1e-3 or abs(turn) > 1e-3 or lidar_override or guard_blocked:
                print(
                    "[fleet-agent] DRIVE "
                    f"ui_fwd={round(fwd,3)} ui_turn={round(turn,3)} "
                    f"tracks L={round(l_tr,3)} R={round(r_tr,3)} "
                    f"pwm FL={fl} FR={fr} override={lidar_override} blocked={guard_blocked}"
                )
            result["lidar_guard"] = guard
            if guard_blocked and not lidar_override:
                result["lidar_blocked"] = True
                _stop_motors(local_web, prefer_web, mega_port)
                _touch_drive(0.0, 0.0)
                result.update({"ok": True, "via": "guard_stop", "forward": fwd, "turn": turn})
                print(
                    "[fleet-agent] LIDAR_GUARD_STOP "
                    f"fwd={guard.get('min_forward_m')}m back={guard.get('min_backward_m')}m "
                    f"threshold={guard.get('threshold_m')}m"
                )
                return result
            phys_fwd = (l_tr + r_tr) * 0.5
            _touch_drive(phys_fwd, 0.0)
            if prefer_web:
                _drive_local(local_web, l_tr, r_tr)
                result.update({
                    "ok": True,
                    "via": "local_web_tracks",
                    "forward": fwd,
                    "turn": turn,
                    "left_track": l_tr,
                    "right_track": r_tr,
                })
            else:
                line = f"M FL={fl} FR={fr} RL={fl} RR={fr}"
                r = send_command(mega_port, line)
                result.update({"ok": r.get("ok"), "via": "serial_pwm", "response": r.get("response")})

        elif action == "stop_drive":
            _touch_drive(0.0, 0.0)
            _motors_active = False
            if prefer_web:
                _drive_local(local_web, 0.0, 0.0)
                result.update({"ok": True, "via": "local_web"})
            else:
                r = send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
                result.update({"ok": r.get("ok"), "via": "serial"})

        elif action == "ping":
            result.update({"ok": True, "pong": True})

        else:
            result["error"] = f"unknown_action:{action}"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def pull_roarm_commands(hub_url: str, rover_id: str, token: str) -> Dict[str, Any]:
    body = {"rover_id": rover_id, "token": token}
    return _post_json_retry(
        f"{hub_url.rstrip('/')}/api/agents/roarm_pull",
        body,
        timeout=4.0,
        attempts=3,
        base_delay_s=0.05,
    )


def post_roarm_result(
    hub_url: str,
    rover_id: str,
    token: str,
    req_id: str,
    result: Dict[str, Any],
) -> None:
    body = {"rover_id": rover_id, "token": token, "id": req_id, "result": result}
    try:
        _post_json_retry(
            f"{hub_url.rstrip('/')}/api/agents/roarm_result",
            body,
            timeout=4.0,
            attempts=3,
            base_delay_s=0.05,
        )
    except Exception:
        pass


def pull_commands(hub_url: str, rover_id: str, token: str) -> Dict[str, Any]:
    body = {"rover_id": rover_id, "token": token}
    return _post_json_retry(
        f"{hub_url.rstrip('/')}/api/agents/pull_commands",
        body,
        timeout=4.0,
        attempts=4,
        base_delay_s=0.05,
    )


def heartbeat(
    hub_url: str,
    rover_id: str,
    token: str,
    payload: Dict[str, Any],
    results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    global _last_heartbeat_rtt_ms
    body: Dict[str, Any] = {
        "rover_id": rover_id,
        "token": token,
        "name": payload.get("name"),
        "telemetry": payload.get("telemetry", {}),
        "meta": payload.get("meta", {}),
    }
    if results:
        body["command_results"] = results
    t0 = time.monotonic()
    resp = _post_json_retry(
        f"{hub_url.rstrip('/')}/api/agents/heartbeat",
        body,
        timeout=12.0,
        attempts=12,
        base_delay_s=0.15,
    )
    _last_heartbeat_rtt_ms = round((time.monotonic() - t0) * 1000.0, 1)
    _touch_hub_ok()
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description="AXM fleet agent for Orin")
    parser.add_argument("--hub-url", default=_env("AXM_HUB_URL", "https://rover.axm.tech"))
    parser.add_argument("--rover-id", default=_env("AXM_ROVER_ID", "rover-01"))
    parser.add_argument("--token", default=_env("AXM_ROVER_TOKEN"))
    parser.add_argument("--name", default=_env("AXM_ROVER_NAME"))
    parser.add_argument("--local-web", default=_env("AXM_LOCAL_WEB", "http://127.0.0.1:8080"))
    parser.add_argument("--mega-port", default=_env("MEGA_PORT", "/dev/ttyUSB1"))
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()

    if not args.token:
        print("ERROR: set AXM_ROVER_TOKEN or --token", file=sys.stderr)
        return 2

    camera_fps = _camera_fps()
    if args.interval is None:
        if _env("AXM_HEARTBEAT_INTERVAL"):
            args.interval = float(_env("AXM_HEARTBEAT_INTERVAL"))
        else:
            args.interval = 1.0 / camera_fps

    name = args.name or args.rover_id
    print(
        f"[fleet-agent] rover={args.rover_id} hub={args.hub_url} "
        f"mega={args.mega_port} interval={args.interval}s camera_fps={camera_fps}"
    )

    cam_stop = threading.Event()
    try:
        rtk_baud = int(_env("RTK_BAUD", "38400"))
    except ValueError:
        rtk_baud = 38400
    rtk_port = _env("RTK_PORT", "/dev/ttyACM0")
    start_gnss_reader(port=rtk_port, baud=rtk_baud)
    gnss_mode = _env("AXM_GNSS_MODE", "gps").strip().lower()
    ntrip_on = ntrip_configured() and gnss_mode == "rtk"
    print(
        f"[fleet-agent] gnss mode={gnss_mode} port={rtk_port} baud={rtk_baud}"
        f" ntrip={'on' if ntrip_on else 'off'}"
    )
    if roarm_enabled():
        print(f"[fleet-agent] roarm enabled ip={_env('ROARM_IP', '192.168.1.87')}")
    cam_thread = threading.Thread(
        target=_camera_stream_loop,
        args=(args.hub_url, args.rover_id, args.token, args.local_web, cam_stop),
        daemon=True,
        name="camera-stream",
    )
    cam_thread.start()

    poll_stop = threading.Event()

    def _lidar_poll_loop() -> None:
        try:
            poll_s = float(_env("AXM_LIDAR_POLL_S", "0.05"))
        except ValueError:
            poll_s = 0.05
        poll_s = max(0.02, min(poll_s, 0.25))
        while not poll_stop.is_set():
            try:
                _refresh_lidar_arc(args.local_web)
            except Exception:
                pass
            poll_stop.wait(poll_s)

    lidar_thread = threading.Thread(target=_lidar_poll_loop, daemon=True, name="lidar-poll")
    lidar_thread.start()

    def _command_poll_loop() -> None:
        poll_s = float(_env("AXM_COMMAND_POLL_S", str(COMMAND_POLL_S)))
        while not poll_stop.is_set():
            if not _hub_link_ok():
                poll_stop.wait(max(0.5, poll_s))
                continue
            try:
                health = _fetch_json(f"{args.local_web.rstrip('/')}/api/health", 0.3) or {}
                prefer_web = bool(health.get("ok") and health.get("bridge_active"))
                resp = pull_commands(args.hub_url, args.rover_id, args.token)
                for cmd in resp.get("commands") or []:
                    execute_command(
                        cmd,
                        local_web=args.local_web,
                        mega_port=args.mega_port,
                        prefer_web=prefer_web,
                    )
                _watchdog_tick(args.local_web, prefer_web, args.mega_port)
                if _session_active and _motors_active and not _lidar_override_live():
                    if _lidar_stop_latched_fwd and _last_drive_linear_x > 1e-4:
                        _stop_motors(args.local_web, prefer_web, args.mega_port)
                    elif _lidar_stop_latched_rev and _last_drive_linear_x < -1e-4:
                        _stop_motors(args.local_web, prefer_web, args.mega_port)
            except Exception:
                pass
            poll_stop.wait(poll_s)

    poll_thread = threading.Thread(target=_command_poll_loop, daemon=True, name="command-poll")
    poll_thread.start()

    def _roarm_poll_loop() -> None:
        if not roarm_enabled():
            return
        poll_s = float(_env("AXM_ROARM_POLL_S", "0.15"))
        exec_q: "queue.Queue[tuple[str, str, str, Dict[str, Any]] | None]" = __import__("queue").Queue()

        def _executor_loop() -> None:
            while True:
                item = exec_q.get()
                if item is None:
                    break
                req_id, op, params = item
                try:
                    result = execute_rpc(op, params)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                post_roarm_result(
                    args.hub_url,
                    args.rover_id,
                    args.token,
                    req_id,
                    result,
                )

        threading.Thread(target=_executor_loop, daemon=True, name="roarm-exec").start()

        def _enqueue(req_id: str, op: str, params: Dict[str, Any]) -> None:
            if op == "status":
                pending = []
                while True:
                    try:
                        pending.append(exec_q.get_nowait())
                    except Exception:
                        break
                for old in pending:
                    if old is None:
                        exec_q.put(None)
                        return
                    if old[1] != "status":
                        exec_q.put(old)
                exec_q.put((req_id, op, params))
                return
            exec_q.put((req_id, op, params))

        while not poll_stop.is_set():
            if not _hub_link_ok():
                poll_stop.wait(max(0.5, poll_s))
                continue
            try:
                resp = pull_roarm_commands(args.hub_url, args.rover_id, args.token)
                for item in resp.get("commands") or []:
                    req_id = str(item.get("id") or "")
                    op = str(item.get("op") or "")
                    params = item.get("params") or {}
                    if not req_id or not op:
                        continue
                    _enqueue(req_id, op, params)
            except Exception:
                pass
            poll_stop.wait(poll_s)

    if roarm_enabled():
        roarm_thread = threading.Thread(target=_roarm_poll_loop, daemon=True, name="roarm-poll")
        roarm_thread.start()

    keepalive_stop = threading.Event()
    keepalive_interval = float(_env("AXM_KEEPALIVE_INTERVAL", str(KEEPALIVE_INTERVAL_S)))

    def _keepalive_loop() -> None:
        """Minimal heartbeat on a slow cadence — keeps ONLINE on flaky uplink."""
        while not keepalive_stop.is_set():
            try:
                payload = {
                    "name": name,
                    "telemetry": {
                        "agent": "orin",
                        "drive_mode": _DRIVE_MODE,
                        "session_active": _session_active,
                        "link": {"rtt_ms": _last_heartbeat_rtt_ms},
                    },
                    "meta": {"agent": "orin", "keepalive": True},
                }
                heartbeat(args.hub_url, args.rover_id, args.token, payload)
            except Exception:
                pass
            keepalive_stop.wait(max(1.0, keepalive_interval))

    keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True, name="keepalive")
    keepalive_thread.start()

    telem_interval = float(_env("AXM_TELEMETRY_INTERVAL", str(TELEMETRY_INTERVAL_S)))

    try:
        while True:
            telemetry = collect_telemetry(args.local_web, args.mega_port)
            health = _fetch_json(f"{args.local_web.rstrip('/')}/api/health") or {}
            prefer_web = bool(health.get("ok") and health.get("bridge_active"))

            payload = {
                "name": name,
                "telemetry": telemetry,
                "meta": {"agent": "orin", "version": "0.5.0", "prefer_web": prefer_web},
            }

            try:
                resp = heartbeat(args.hub_url, args.rover_id, args.token, payload)
                # Commands also drained by poll thread; heartbeat may return duplicates rarely — ok.

                mega = telemetry.get("mega") or {}
                link = telemetry.get("link") or {}
                guard = (telemetry.get("perception") or {}).get("lidar_guard") or {}
                print(
                    f"[fleet-agent] ok mode={_DRIVE_MODE} mega={mega.get('connected')} "
                    f"speed={mega.get('speed_mps')} "
                    f"rtt={link.get('rtt_ms')}ms guard={guard.get('active')}"
                )
            except urllib.error.HTTPError as exc:
                print(f"[fleet-agent] HTTP {exc.code}: {exc.read().decode()[:200]}", file=__import__("sys").stderr)
            except Exception as exc:
                print(f"[fleet-agent] error: {exc}", file=__import__("sys").stderr)

            time.sleep(max(0.05, min(telem_interval, 60.0)))
    finally:
        keepalive_stop.set()
        poll_stop.set()
        cam_stop.set()
        keepalive_thread.join(timeout=2.0)
        cam_thread.join(timeout=2.0)
        poll_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
