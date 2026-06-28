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
    return float(_env("ROARM_STAGED_PAUSE_S", "1.4"))


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


def move_joints_staged(
    client: RoArmClient,
    params: Dict[str, Any],
) -> Tuple[str, str]:
    """Move one joint at a time in safe order (base → shoulder → elbow → …)."""
    joints = joints_from_params(params)
    spd = float(params.get("spd", 0))
    acc = float(params.get("acc", 10))
    timeout = _move_timeout()
    pause = _pause_sec()
    last_url = ""
    last_resp = ""
    for joint_id, key in STAGED_ORDER:
        url, resp = client.joint_control(
            joint_id,
            joints[key],
            spd=spd,
            acc=acc,
            timeout_sec=timeout,
        )
        last_url, last_resp = url, resp
        time.sleep(pause)
    return last_url, last_resp
