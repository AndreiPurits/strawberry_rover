#!/usr/bin/env python3
"""Orin fleet agent: telemetry + remote control via rover.axm.tech hub."""
from __future__ import annotations

import argparse
import base64
import json
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
from mega_client import port_busy, port_exists, probe_mega, send_command, twist_to_pwm

_DRIVE_MODE = "joystick"
_last_lidar_stamp: Optional[float] = None
_last_lidar_arc: Dict[str, Any] = {}
_last_drive_at: float = 0.0
_motors_active: bool = False
_lidar_stop_latched: bool = False
_session_active: bool = False
_last_stereo_camera_stamp: Optional[float] = None
_last_camera_stamp: Optional[float] = None
_last_heartbeat_rtt_ms: Optional[float] = None
_last_hub_ok_at: float = 0.0
_hub_ok_lock = threading.Lock()

WATCHDOG_TIMEOUT_S = 0.5
LIDAR_GUARD_M = 0.2
TELEMETRY_INTERVAL_S = 5.0
KEEPALIVE_INTERVAL_S = 2.0
COMMAND_POLL_S = 0.25
HUB_LINK_GRACE_S = 45.0

# Gecoma motor shield: cmd_vel axes swapped vs nominal (do not rewire).
def _remap_ui_drive(forward: float, turn: float) -> Tuple[float, float]:
    """UI forward/turn -> ROS linear_x, angular_z (Gecoma axes swapped)."""
    return -turn, forward


def _camera_fps() -> float:
    raw = _env("AXM_CAMERA_FPS", "30")
    try:
        fps = float(raw)
    except ValueError:
        fps = 30.0
    return fps if fps > 0 else 30.0


def _hub_camera_fps() -> float:
    """JPEG upload rate to hub — keep very low on flaky uplink."""
    raw = _env("AXM_HUB_CAMERA_FPS", "2")
    try:
        fps = float(raw)
    except ValueError:
        fps = 2.0
    return max(0.5, min(fps, 5.0))


def _hub_stereo_enabled() -> bool:
    return _env("AXM_HUB_STEREO", "false").lower() in ("1", "true", "yes")


def _touch_hub_ok() -> None:
    global _last_hub_ok_at
    with _hub_ok_lock:
        _last_hub_ok_at = time.monotonic()


def _hub_link_ok() -> bool:
    with _hub_ok_lock:
        return (time.monotonic() - _last_hub_ok_at) <= HUB_LINK_GRACE_S


def _forward_min_dist(lidar_arc: Dict[str, Any]) -> Optional[float]:
    sectors = lidar_arc.get("sectors") or []
    n = len(sectors)
    if n == 0:
        return None
    center = n // 2
    dists: List[float] = []
    for i in range(max(0, center - 1), min(n, center + 2)):
        d = sectors[i].get("dist_m")
        if d is not None:
            dists.append(float(d))
    return min(dists) if dists else None


def _lidar_guard_state(lidar_arc: Dict[str, Any]) -> Dict[str, Any]:
    m = _forward_min_dist(lidar_arc)
    blocked = m is not None and m < LIDAR_GUARD_M
    return {
        "active": blocked,
        "min_forward_m": round(m, 3) if m is not None else None,
        "threshold_m": LIDAR_GUARD_M,
    }


def _apply_lidar_guard(fwd: float, override: bool) -> Tuple[float, Dict[str, Any]]:
    """Block forward when obstacle ahead; latch stop until override."""
    global _lidar_stop_latched
    guard = _lidar_guard_state(_last_lidar_arc)
    if override:
        _lidar_stop_latched = False
        return fwd, guard
    if guard["active"] and fwd > 0:
        _lidar_stop_latched = True
        return 0.0, guard
    if _lidar_stop_latched and fwd > 0:
        return 0.0, guard
    if fwd <= 0:
        _lidar_stop_latched = False
    return fwd, guard


def _collect_perception(local_web: str) -> Dict[str, Any]:
    global _last_lidar_stamp, _last_lidar_arc
    base = local_web.rstrip("/")
    out: Dict[str, Any] = {}

    arc = _fetch_json(f"{base}/api/perception/lidar_arc", 0.5) or {}
    if arc.get("ok"):
        arc_data = {k: v for k, v in arc.items() if k != "ok"}
        _last_lidar_arc = arc_data
        stamp = arc_data.get("stamp")
        if stamp != _last_lidar_stamp:
            _last_lidar_stamp = stamp
        out["lidar_arc"] = arc_data

    out["lidar_guard"] = _lidar_guard_state(_last_lidar_arc)
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
    interval = 1.0 / _hub_camera_fps()
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
            if stereo:
                st = _fetch_json(f"{base}/api/perception/stereo_camera", 0.4) or {}
                if st.get("ok") and st.get("jpeg_b64"):
                    sjpeg = base64.b64decode(st["jpeg_b64"])
                    if len(sjpeg) <= 120_000:
                        sstamp = float(st.get("stamp") or time.time())
                        _post_stereo_camera_frame(hub_url, rover_id, token, sjpeg, sstamp)
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
    status_raw = arduino_data.get("status")
    mega: Dict[str, Any] = {}
    if isinstance(status_raw, str) and status_raw.startswith("{"):
        try:
            mega = json.loads(status_raw)
        except json.JSONDecodeError:
            pass
    for key in (
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
    ):
        if key in arduino_data and key not in mega:
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
        "armed": mega.get("armed"),
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


def _drive_local(local_web: str, linear_x: float, angular_z: float) -> Dict[str, Any]:
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
    global _last_drive_at, _motors_active
    _last_drive_at = time.monotonic()
    _motors_active = abs(linear_x) > 1e-4 or abs(angular_z) > 1e-4


def _stop_motors(local_web: str, prefer_web: bool, mega_port: str) -> None:
    global _motors_active
    if prefer_web:
        _drive_local(local_web, 0.0, 0.0)
    else:
        send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
    _motors_active = False


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
                # Joystick tab alone does not arm motors — only session_start does.
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
            scale = max(0.08, min(1.0, float(params.get("speed_scale", 1.0))))
            fwd = float(params.get("forward", params.get("linear_x", 0.0))) * scale
            turn = float(params.get("turn", params.get("angular_z", 0.0))) * scale
            lidar_override = bool(params.get("lidar_override"))
            fwd, guard = _apply_lidar_guard(fwd, lidar_override)
            if _lidar_stop_latched and not lidar_override:
                result["lidar_blocked"] = True
                _stop_motors(local_web, prefer_web, mega_port)
            result["lidar_guard"] = guard
            lx, az = _remap_ui_drive(fwd, turn)
            _touch_drive(lx, az)
            if prefer_web:
                _drive_local(local_web, lx, az)
                result.update({"ok": True, "via": "local_web", "forward": fwd, "turn": turn, "linear_x": lx, "angular_z": az})
            else:
                r = _drive_serial(mega_port, lx, az)
                result.update({"ok": r.get("ok"), "via": "serial", "response": r.get("response")})

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
    parser.add_argument("--mega-port", default=_env("MEGA_PORT", "/dev/ttyUSB0"))
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
    start_gnss_reader(port=_env("RTK_PORT", "/dev/ttyACM0"))
    cam_thread = threading.Thread(
        target=_camera_stream_loop,
        args=(args.hub_url, args.rover_id, args.token, args.local_web, cam_stop),
        daemon=True,
        name="camera-stream",
    )
    cam_thread.start()

    poll_stop = threading.Event()

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
                if _lidar_stop_latched and _session_active and _motors_active:
                    _stop_motors(args.local_web, prefer_web, args.mega_port)
            except Exception:
                pass
            poll_stop.wait(poll_s)

    poll_thread = threading.Thread(target=_command_poll_loop, daemon=True, name="command-poll")
    poll_thread.start()

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
