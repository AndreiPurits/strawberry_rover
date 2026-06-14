#!/usr/bin/env python3
"""Orin fleet agent: telemetry + remote control via rover.axm.tech hub."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from mega_client import port_busy, port_exists, probe_mega, send_command, twist_to_pwm

_DRIVE_MODE = "joystick"


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
        "left_us",
        "right_us",
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

    left_pct = track_pct(left)
    right_pct = track_pct(right)
    linear_pct = (left_pct + right_pct) / 2.0

    return {
        "connected": arduino_data.get("connected"),
        "armed": mega.get("armed"),
        "left_us": left,
        "right_us": right,
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
        "mega": mega,
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


def execute_command(
    cmd: Dict[str, Any],
    *,
    local_web: str,
    mega_port: str,
    prefer_web: bool,
) -> Dict[str, Any]:
    global _DRIVE_MODE
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
                if prefer_web:
                    _local_web_post(local_web, "/api/control/start")
                    _local_web_post(local_web, "/api/control/mode", {"mode": "manual"})
                else:
                    send_command(mega_port, "ARM")
                result.update({"ok": True, "drive_mode": mode})
            else:
                if prefer_web:
                    _local_web_post(local_web, "/api/control/mode", {"mode": "auto"})
                    _drive_local(local_web, 0.0, 0.0)
                else:
                    send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
                result.update({"ok": True, "drive_mode": mode, "note": "auto_stub"})

        elif action == "session_start":
            if _DRIVE_MODE != "joystick":
                result["error"] = "not_in_joystick_mode"
                return result
            if prefer_web:
                _local_web_post(local_web, "/api/control/start")
                _local_web_post(local_web, "/api/control/mode", {"mode": "manual"})
                result.update({"ok": True, "via": "local_web"})
            else:
                r = send_command(mega_port, "ARM")
                result.update({"ok": r.get("ok"), "via": "serial", "response": r.get("response")})

        elif action in ("session_stop", "disarm"):
            if prefer_web:
                _local_web_post(local_web, "/api/control/stop")
                result.update({"ok": True, "via": "local_web"})
            else:
                send_command(mega_port, "M FL=1500 FR=1500 RL=1500 RR=1500")
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
            lx = float(params.get("linear_x", 0.0))
            az = float(params.get("angular_z", 0.0))
            if prefer_web:
                _drive_local(local_web, lx, az)
                result.update({"ok": True, "via": "local_web", "linear_x": lx, "angular_z": az})
            else:
                r = _drive_serial(mega_port, lx, az)
                result.update({"ok": r.get("ok"), "via": "serial", "response": r.get("response")})

        elif action == "stop_drive":
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


def heartbeat(
    hub_url: str,
    rover_id: str,
    token: str,
    payload: Dict[str, Any],
    results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "rover_id": rover_id,
        "token": token,
        "name": payload.get("name"),
        "telemetry": payload.get("telemetry", {}),
        "meta": payload.get("meta", {}),
    }
    if results:
        body["command_results"] = results
    return _post_json(f"{hub_url.rstrip('/')}/api/agents/heartbeat", body)


def main() -> int:
    parser = argparse.ArgumentParser(description="AXM fleet agent for Orin")
    parser.add_argument("--hub-url", default=_env("AXM_HUB_URL", "https://rover.axm.tech"))
    parser.add_argument("--rover-id", default=_env("AXM_ROVER_ID", "rover-01"))
    parser.add_argument("--token", default=_env("AXM_ROVER_TOKEN"))
    parser.add_argument("--name", default=_env("AXM_ROVER_NAME"))
    parser.add_argument("--local-web", default=_env("AXM_LOCAL_WEB", "http://127.0.0.1:8080"))
    parser.add_argument("--mega-port", default=_env("MEGA_PORT", "/dev/ttyUSB0"))
    parser.add_argument("--interval", type=float, default=float(_env("AXM_HEARTBEAT_INTERVAL", "0.05")))
    args = parser.parse_args()

    if not args.token:
        print("ERROR: set AXM_ROVER_TOKEN or --token", file=sys.stderr)
        return 2

    name = args.name or args.rover_id
    print(
        f"[fleet-agent] rover={args.rover_id} hub={args.hub_url} "
        f"mega={args.mega_port} interval={args.interval}s"
    )

    while True:
        telemetry = collect_telemetry(args.local_web, args.mega_port)
        health = _fetch_json(f"{args.local_web.rstrip('/')}/api/health") or {}
        prefer_web = bool(health.get("ok") and health.get("bridge_active"))

        payload = {
            "name": name,
            "telemetry": telemetry,
            "meta": {"agent": "orin", "version": "0.3.0", "prefer_web": prefer_web},
        }

        try:
            resp = heartbeat(args.hub_url, args.rover_id, args.token, payload)
            commands = resp.get("commands") or []
            results: List[Dict[str, Any]] = []
            for cmd in commands:
                print(f"[fleet-agent] cmd {cmd.get('action')} id={cmd.get('id')}")
                results.append(
                    execute_command(
                        cmd,
                        local_web=args.local_web,
                        mega_port=args.mega_port,
                        prefer_web=prefer_web,
                    )
                )
            if results:
                telemetry = collect_telemetry(args.local_web, args.mega_port)
                payload["telemetry"] = telemetry
                heartbeat(args.hub_url, args.rover_id, args.token, payload, results=results)

            mega = telemetry.get("mega") or {}
            print(
                f"[fleet-agent] ok mode={_DRIVE_MODE} mega={mega.get('connected')} "
                f"speed={mega.get('speed_mps')} cmds={len(commands)}"
            )
        except urllib.error.HTTPError as exc:
            print(f"[fleet-agent] HTTP {exc.code}: {exc.read().decode()[:200]}", file=__import__("sys").stderr)
        except Exception as exc:
            print(f"[fleet-agent] error: {exc}", file=__import__("sys").stderr)

        time.sleep(max(0.05, min(float(args.interval), 60.0)))


if __name__ == "__main__":
    raise SystemExit(main())
