"""Staged joint motion: base first, then shoulder/elbow, then wrist/gripper."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

from roarm_client import RoArmClient

# (joint_id, param_key)
STAGED_ORDER: Tuple[Tuple[int, str], ...] = (
    (1, "base"),
    (2, "shoulder"),
    (3, "elbow"),
    (4, "wrist"),
    (5, "roll"),
    (6, "hand"),
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _pause_sec() -> float:
    return float(_env("ROARM_STAGED_PAUSE_S", "0.9"))


def _move_timeout() -> float:
    return float(_env("ROARM_MOVE_TIMEOUT_S", "30"))


def joints_from_params(params: Dict[str, Any]) -> Dict[str, float]:
    return {
        "base": float(params.get("base", 0)),
        "shoulder": float(params.get("shoulder", 0)),
        "elbow": float(params.get("elbow", 1.57)),
        "wrist": float(params.get("wrist", 0)),
        "roll": float(params.get("roll", 0)),
        "hand": float(params.get("hand", 3.14)),
    }


def _pause_after_joint(joint_key: str, target_rad: float, current_rad: float | None) -> float:
    """Sleep after T:101 so the servo can move before the next joint."""
    base = float(_env("ROARM_STAGED_PAUSE_S", "0.9"))
    if current_rad is None:
        return base
    delta = abs(float(target_rad) - float(current_rad))
    # ~1.2 rad/s effective; clamp 0.35–2.5 s
    return min(2.5, max(0.35, base * 0.5 + delta / 1.2))


def move_joints_staged(
    client: RoArmClient,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Move one joint at a time in safe order (base → shoulder → elbow → …)."""
    joints = joints_from_params(params)
    spd = float(params.get("spd", 0))
    acc = float(params.get("acc", 10))
    per_joint_timeout = min(_move_timeout(), float(_env("ROARM_STAGED_JOINT_TIMEOUT_S", "12")))
    last_url = ""
    last_resp = ""
    feedback: Dict[str, float] = {}
    try:
        _url, fb = client.servo_feedback(timeout_sec=8.0)
        if isinstance(fb, dict):
            for jid, key in STAGED_ORDER:
                v = fb.get(key if key != "base" else "b")
                if v is not None:
                    feedback[key] = float(v)
    except Exception:
        pass

    for joint_id, key in STAGED_ORDER:
        target = joints[key]
        current = feedback.get(key)
        url, resp = client.joint_control(
            joint_id,
            target,
            spd=spd,
            acc=acc,
            timeout_sec=per_joint_timeout,
        )
        last_url, last_resp = url, resp
        time.sleep(_pause_after_joint(key, target, current))
        feedback[key] = target
    return last_url, last_resp
