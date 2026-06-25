"""Execute RoArm RPC from hub queue (runs on Orin, local LAN only)."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from roarm_client import RoArmClient, RoArmClientError

_client: Optional[RoArmClient] = None
_last_probe: float = 0.0
_last_status: Dict[str, Any] = {
    "enabled": False,
    "reachable": False,
    "ip": "",
    "error": None,
    "updated_at": None,
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def roarm_enabled() -> bool:
    return _env("ROARM_ENABLED", "true").lower() not in ("0", "false", "no")


def _client_ip() -> str:
    return _env("ROARM_IP", "192.168.1.87")


def _get_client() -> RoArmClient:
    global _client
    ip = _client_ip()
    if _client is None or _client.ip != ip:
        _client = RoArmClient(ip=ip, timeout_sec=float(_env("ROARM_TIMEOUT_S", "5")))
    return _client


def probe_status(force: bool = False) -> Dict[str, Any]:
    """Read-only status probe for telemetry (no motion)."""
    global _last_probe, _last_status
    if not roarm_enabled():
        _last_status = {
            "enabled": False,
            "reachable": False,
            "ip": _client_ip(),
            "error": "disabled",
            "updated_at": time.time(),
        }
        return dict(_last_status)
    interval = float(_env("ROARM_PROBE_INTERVAL_S", "10"))
    now = time.time()
    if not force and _last_status.get("updated_at") and (now - _last_probe) < interval:
        return dict(_last_status)
    _last_probe = now
    snap: Dict[str, Any] = {
        "enabled": True,
        "reachable": False,
        "ip": _client_ip(),
        "error": None,
        "updated_at": now,
        "arm": None,
    }
    try:
        _url, status = _get_client().get_status()
        snap["reachable"] = True
        snap["arm"] = status
    except RoArmClientError as exc:
        snap["error"] = str(exc)
    except Exception as exc:
        snap["error"] = str(exc)
    _last_status = snap
    return dict(_last_status)


def telemetry_snapshot() -> Dict[str, Any]:
    return probe_status(force=False)


def execute_rpc(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not roarm_enabled():
        return {"ok": False, "error": "roarm_disabled"}
    client = _get_client()
    try:
        if op == "status":
            url, status = client.get_status()
            return {"ok": True, "url": url, "status": status}
        if op == "home":
            url, resp = client.home()
            return {"ok": True, "url": url, "response": resp}
        if op == "gripper_open":
            url, resp = client.gripper_open()
            return {"ok": True, "url": url, "response": resp}
        if op == "gripper_close":
            url, resp = client.gripper_close()
            return {"ok": True, "url": url, "response": resp}
        if op == "torque_on":
            url, resp = client.torque(True)
            return {"ok": True, "url": url, "response": resp}
        if op == "torque_off":
            url, resp = client.torque(False)
            return {"ok": True, "url": url, "response": resp}
        if op == "move_xyz":
            url, resp = client.move_xyz(
                float(params.get("x", 0)),
                float(params.get("y", 0)),
                float(params.get("z", 0)),
                float(params.get("t", 0)),
                float(params.get("r", 0)),
                float(params.get("g", 0)),
                float(params.get("spd", 0.25)),
            )
            return {"ok": True, "url": url, "response": resp}
        return {"ok": False, "error": f"unknown_op:{op}"}
    except RoArmClientError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
