#!/usr/bin/env python3
"""Push rover telemetry from Orin to AXM fleet hub (outbound HTTPS)."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _fetch_json(url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def collect_telemetry(local_web: str, mega_port: str) -> Dict[str, Any]:
    health = _fetch_json(f"{local_web.rstrip('/')}/api/health") or {}
    arduino_raw = health.get("arduino")
    arduino_data: Dict[str, Any] = {}
    if isinstance(arduino_raw, str):
        try:
            arduino_data = json.loads(arduino_raw)
        except json.JSONDecodeError:
            arduino_data = {"raw": arduino_raw}
    elif isinstance(arduino_raw, dict):
        arduino_data = arduino_raw

    return {
        "hostname": socket.gethostname(),
        "local_web_url": local_web,
        "mega_port": mega_port,
        "arduino_connected": arduino_data.get("connected"),
        "armed": arduino_data.get("armed"),
        "health_ok": health.get("ok"),
        "bridge_active": health.get("bridge_active"),
        "arduino": arduino_data,
    }


def post_heartbeat(hub_url: str, rover_id: str, token: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(
        {
            "rover_id": rover_id,
            "token": token,
            "name": payload.get("name"),
            "telemetry": payload.get("telemetry", {}),
            "meta": payload.get("meta", {}),
        }
    ).encode()
    req = urllib.request.Request(
        f"{hub_url.rstrip('/')}/api/agents/heartbeat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"heartbeat failed: HTTP {resp.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AXM fleet agent for Orin")
    parser.add_argument("--hub-url", default=_env("AXM_HUB_URL", "https://rover.axm.tech"))
    parser.add_argument("--rover-id", default=_env("AXM_ROVER_ID", "rover-01"))
    parser.add_argument("--token", default=_env("AXM_ROVER_TOKEN"))
    parser.add_argument("--name", default=_env("AXM_ROVER_NAME"))
    parser.add_argument("--local-web", default=_env("AXM_LOCAL_WEB", "http://127.0.0.1:8080"))
    parser.add_argument("--mega-port", default=_env("MEGA_PORT", "/dev/ttyUSB0"))
    parser.add_argument("--interval", type=float, default=float(_env("AXM_HEARTBEAT_INTERVAL", "10")))
    args = parser.parse_args()

    if not args.token:
        print("ERROR: set AXM_ROVER_TOKEN or --token", file=sys.stderr)
        return 2

    name = args.name or args.rover_id
    print(f"[fleet-agent] rover={args.rover_id} hub={args.hub_url} interval={args.interval}s")

    while True:
        telemetry = collect_telemetry(args.local_web, args.mega_port)
        payload = {
            "name": name,
            "telemetry": telemetry,
            "meta": {"agent": "orin", "version": "0.1.0"},
        }
        try:
            post_heartbeat(args.hub_url, args.rover_id, args.token, payload)
            print(f"[fleet-agent] heartbeat ok armed={telemetry.get('armed')}")
        except urllib.error.HTTPError as exc:
            print(f"[fleet-agent] HTTP {exc.code}: {exc.read().decode()[:200]}", file=sys.stderr)
        except Exception as exc:
            print(f"[fleet-agent] error: {exc}", file=sys.stderr)
        time.sleep(max(3.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
