"""Execute RoArm RPC from hub queue (runs on Orin, local LAN only)."""

from __future__ import annotations

import os
import threading
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
_last_feedback: Dict[str, Any] = {}
_last_feedback_ts: float = 0.0
_rpc_inflight: int = 0
_inflight_lock = threading.Lock()


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


def reset_client() -> None:
    """Drop cached HTTP client (after RoArm power-cycle or IP change)."""
    global _client, _last_probe, _last_status, _last_feedback, _last_feedback_ts
    _client = None
    _last_probe = 0.0
    _last_feedback = {}
    _last_feedback_ts = 0.0
    _last_status = {
        "enabled": roarm_enabled(),
        "reachable": False,
        "ip": _client_ip(),
        "error": "reset",
        "updated_at": time.time(),
    }


def _move_timeout() -> float:
    return float(_env("ROARM_MOVE_TIMEOUT_S", "30"))


def _joint_move_timeout() -> float:
    return float(_env("ROARM_JOINT_MOVE_TIMEOUT_S", "8"))


def _grip_timeout() -> float:
    return float(_env("ROARM_GRIP_RPC_TIMEOUT_S", "20"))


def _rpc_busy() -> bool:
    with _inflight_lock:
        return _rpc_inflight > 0


def _rpc_begin() -> None:
    global _rpc_inflight
    with _inflight_lock:
        _rpc_inflight += 1


def _rpc_end() -> None:
    global _rpc_inflight
    with _inflight_lock:
        _rpc_inflight = max(0, _rpc_inflight - 1)


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
    connect_timeout = float(_env("ROARM_CONNECT_TIMEOUT_S", "3"))
    now = time.time()
    if not force and _last_status.get("updated_at") and (now - _last_probe) < interval:
        return dict(_last_status)
    if _rpc_busy():
        return dict(_last_status)
    _last_probe = now
    snap: Dict[str, Any] = {
        "enabled": True,
        "reachable": False,
        "tcp_open": False,
        "ip": _client_ip(),
        "error": None,
        "updated_at": now,
        "arm": None,
    }
    client = _get_client()
    try:
        _url, status = client.servo_feedback(timeout_sec=connect_timeout)
        snap["reachable"] = True
        snap["tcp_open"] = True
        snap["arm"] = status
        global _last_feedback, _last_feedback_ts
        _last_feedback = status if isinstance(status, dict) else {}
        _last_feedback_ts = now
    except RoArmClientError as exc:
        snap["error"] = str(exc)
        snap["tcp_open"] = client.tcp_open(timeout_sec=connect_timeout)
    except Exception as exc:
        snap["error"] = str(exc)
        snap["tcp_open"] = client.tcp_open(timeout_sec=connect_timeout)
    _last_status = snap
    return dict(_last_status)


def telemetry_snapshot() -> Dict[str, Any]:
    return probe_status(force=False)


def execute_rpc(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not roarm_enabled():
        return {"ok": False, "error": "roarm_disabled"}
    client = _get_client()
    _rpc_begin()
    try:
        if op == "status":
            url, status = client.servo_feedback(timeout_sec=8.0)
            global _last_feedback, _last_feedback_ts
            if isinstance(status, dict):
                _last_feedback = status
                _last_feedback_ts = time.time()
            return {"ok": True, "url": url, "status": status, "feedback": status}
        if op == "home":
            url, resp = client.home(timeout_sec=10.0)
            return {"ok": True, "url": url, "response": resp}
        if op == "home_joints":
            url, resp = client.joints_rad_ctrl(
                base=float(params.get("base", 0)),
                shoulder=float(params.get("shoulder", 0)),
                elbow=float(params.get("elbow", 1.57)),
                wrist=float(params.get("wrist", 0)),
                roll=float(params.get("roll", 0)),
                hand=float(params.get("hand", 3.14)),
                spd=float(params.get("spd", 0)),
                acc=float(params.get("acc", 10)),
                timeout_sec=_move_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "home_joints_staged":
            from roarm_staged_move import move_joints_staged

            url, resp = move_joints_staged(client, params)
            return {"ok": True, "url": url, "response": resp, "staged": True}
        if op == "joint_move":
            url, resp = client.joint_control(
                int(params.get("joint", 1)),
                float(params.get("rad", 0)),
                spd=float(params.get("spd", 0)),
                acc=float(params.get("acc", 10)),
                timeout_sec=_joint_move_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "joints_move":
            url, resp = client.joints_rad_ctrl(
                base=float(params.get("base", 0)),
                shoulder=float(params.get("shoulder", 0)),
                elbow=float(params.get("elbow", 1.57)),
                wrist=float(params.get("wrist", 0)),
                roll=float(params.get("roll", 0)),
                hand=float(params.get("hand", 3.14)),
                spd=float(params.get("spd", 0)),
                acc=float(params.get("acc", 10)),
                timeout_sec=_move_timeout(),
            )
            return {"ok": True, "url": url, "response": resp, "t102": True}
        if op == "feedback":
            url, fb = client.servo_feedback(timeout_sec=8.0)
            return {"ok": True, "url": url, "feedback": fb}
        if op == "set_servo_middle":
            url, resp = client.set_servo_middle(timeout_sec=10.0)
            return {"ok": True, "url": url, "response": resp}
        if op == "gripper_open":
            url, resp = client.gripper_open()
            return {"ok": True, "url": url, "response": resp}
        if op == "gripper_ctrl":
            url, resp = client.gripper_ctrl(
                float(params.get("cmd", 2.2)),
                spd=float(params.get("spd", 0)),
                acc=float(params.get("acc", 0)),
                timeout_sec=_grip_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "dynamic_torque_limits":
            url, resp = client.dynamic_torque_limits(
                mode=int(params.get("mode", 0)),
                b=int(params.get("b", 1000)),
                s=int(params.get("s", 1000)),
                e=int(params.get("e", 1000)),
                t=int(params.get("t", 1000)),
                r=int(params.get("r", 1000)),
                g=int(params.get("g", 300)),
                timeout_sec=_grip_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "gripper_close":
            url, resp = client.gripper_close(timeout_sec=_grip_timeout())
            return {"ok": True, "url": url, "response": resp, "force": True}
        if op == "gripper_close_force":
            url, resp = client.gripper_close_force(timeout_sec=_grip_timeout())
            return {"ok": True, "url": url, "response": resp, "force": True}
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
                timeout_sec=_move_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "move_xyz_direct":
            url, resp = client.move_xyz_direct(
                float(params.get("x", 0)),
                float(params.get("y", 0)),
                float(params.get("z", 0)),
                float(params.get("t", 0)),
                float(params.get("r", 0)),
                float(params.get("g", 0)),
                timeout_sec=_move_timeout(),
            )
            return {"ok": True, "url": url, "response": resp}
        if op == "sequence_start":
            from roarm_sequence_runner import sequence_start

            steps = params.get("steps") or []
            if not isinstance(steps, list):
                return {"ok": False, "error": "steps_must_be_list"}
            return sequence_start(client, steps)
        if op == "sequence_stop":
            from roarm_sequence_runner import sequence_stop

            return sequence_stop()
        if op == "sequence_status":
            from roarm_sequence_runner import sequence_status

            return {"ok": True, **sequence_status()}
        if op == "plastic_manual_snap":
            # Capture-only: never moves joints / approach / grasp.
            import sys
            from pathlib import Path as _Path

            repo = _Path(__file__).resolve().parents[3]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from pipelines.peduncle_v3.plastic_manual_snap import take_manual_snap

            joints = None
            try:
                _url, fb = client.servo_feedback(timeout_sec=5.0)
                if isinstance(fb, dict):
                    joints = {
                        "base": float(fb.get("b", fb.get("base", 0))),
                        "shoulder": float(fb.get("s", fb.get("shoulder", 0))),
                        "elbow": float(fb.get("e", fb.get("elbow", 0))),
                        "wrist": float(fb.get("t", fb.get("wrist", 0))),
                        "roll": float(fb.get("r", fb.get("roll", 0))),
                        "hand": float(fb.get("g", fb.get("hand", 0))),
                    }
            except Exception:
                joints = None
            out = take_manual_snap(joints=joints)
            return out
        if op == "plastic_manual_count":
            import sys
            from pathlib import Path as _Path

            repo = _Path(__file__).resolve().parents[3]
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from pipelines.peduncle_v3.plastic_manual_snap import count_shots

            return {"ok": True, "count": count_shots()}
        return {"ok": False, "error": f"unknown_op:{op}"}
    except RoArmClientError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        _rpc_end()


def _repo_root():
    from pathlib import Path as _Path

    return _Path(__file__).resolve().parents[3]


def load_named_home_joints(pose_name: str) -> Dict[str, float]:
    """Load joint pose from config/roarm_home_joints.yaml (e.g. DOM_FINAL)."""
    import yaml

    path = _repo_root() / "config" / "roarm_home_joints.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pose = cfg.get(str(pose_name).strip().upper())
    if not isinstance(pose, dict):
        raise KeyError(f"unknown_home_pose:{pose_name}")
    return {
        "base": float(pose.get("base", 0)),
        "shoulder": float(pose.get("shoulder", 0)),
        "elbow": float(pose.get("elbow", 1.57)),
        "wrist": float(pose.get("wrist", 0)),
        "roll": float(pose.get("roll", 0)),
        "hand": float(pose.get("hand", 3.14)),
    }


def startup_home_pose_name() -> Optional[str]:
    """ROARM_STARTUP_HOME: pose name (default DOM_FINAL) or off/0/false to disable."""
    raw = _env("ROARM_STARTUP_HOME", "DOM_FINAL")
    if not raw or raw.lower() in ("0", "false", "no", "off", "none", "disable", "disabled"):
        return None
    return raw.strip().upper()


def go_named_home_staged(pose_name: str, *, acc: float = 12.0) -> Dict[str, Any]:
    joints = load_named_home_joints(pose_name)
    params = {**joints, "spd": 0.0, "acc": float(acc)}
    return execute_rpc("home_joints_staged", params)


def start_roarm_startup_home_thread() -> Optional[threading.Thread]:
    """On Orin boot / fleet-agent start: staged move to DOM_FINAL (or ROARM_STARTUP_HOME)."""
    pose = startup_home_pose_name()
    if not pose or not roarm_enabled():
        return None

    delay_s = float(_env("ROARM_STARTUP_HOME_DELAY_S", "4"))
    retries = max(1, int(float(_env("ROARM_STARTUP_HOME_RETRIES", "6"))))
    retry_s = float(_env("ROARM_STARTUP_HOME_RETRY_S", "5"))

    def _run() -> None:
        print(f"[fleet-agent] roarm startup home → {pose} (delay={delay_s:.1f}s)")
        if delay_s > 0:
            time.sleep(delay_s)
        last_err = "unreachable"
        for attempt in range(1, retries + 1):
            try:
                st = probe_status(force=True)
                if not st.get("reachable"):
                    last_err = str(st.get("error") or "not_reachable")
                    print(f"[fleet-agent] startup home wait {attempt}/{retries}: {last_err}")
                    time.sleep(retry_s)
                    continue
                out = go_named_home_staged(pose)
                if out.get("ok"):
                    print(f"[fleet-agent] startup home OK → {pose}")
                    return
                last_err = str(out.get("error") or "move_failed")
                print(f"[fleet-agent] startup home fail {attempt}/{retries}: {last_err}")
            except Exception as exc:
                last_err = str(exc)
                print(f"[fleet-agent] startup home error {attempt}/{retries}: {last_err}")
            time.sleep(retry_s)
        print(f"[fleet-agent] startup home ABORT → {pose}: {last_err}")

    t = threading.Thread(target=_run, daemon=True, name="roarm-startup-home")
    t.start()
    return t
