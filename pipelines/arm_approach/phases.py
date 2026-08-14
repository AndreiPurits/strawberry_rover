"""CENTER then REACH phase logic (command deltas only, no actuators)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def next_center_reach_step(
    px: float,
    py: float,
    depth_m: float,
    *,
    aim_px: float,
    aim_py: float,
    phase: str,
    cfg: Dict[str, Any],
    dpx_per_base: float = 900.0,
    ddepth_per_extend: float = -0.25,
) -> Tuple[str, Dict[str, float]]:
    """Return (next_phase, delta_q). delta_q is a planned increment, not a command."""
    pcfg = cfg.get("phases") or cfg
    center_tol = float(pcfg.get("center_tol_px", 28.0))
    standoff = float(pcfg.get("standoff_m", 0.12))
    depth_floor = float(pcfg.get("depth_floor_m", 0.02))
    max_step = float(pcfg.get("max_step_rad", 0.08))
    reach_step = float(pcfg.get("reach_step_rad", 0.06))
    aim_mode = str(pcfg.get("aim_mode", "lock_y"))

    err_x = aim_px - px
    err_y = aim_py - py
    derr = depth_m - standoff
    stop_floor = standoff + depth_floor
    if aim_mode == "lock_y":
        centered = abs(err_x) <= center_tol
    else:
        centered = float(np.hypot(err_x, err_y)) <= center_tol

    next_phase = phase
    if phase == "CENTER" and centered:
        next_phase = "STANDOFF" if derr <= 0.02 else "REACH"
    elif phase == "REACH" and not centered and abs(err_x) > center_tol * 1.8:
        next_phase = "CENTER"
    elif phase == "REACH" and derr <= 0.02:
        next_phase = "STANDOFF"

    dq = {"base": 0.0, "shoulder": 0.0, "elbow": 0.0}
    if next_phase == "STANDOFF":
        return next_phase, dq
    if next_phase == "CENTER" or (next_phase == "REACH" and abs(err_x) > center_tol):
        if abs(err_x) > 2.0:
            dq["base"] = float(np.clip(err_x / max(abs(dpx_per_base), 200.0), -max_step, max_step))
        return next_phase, dq

    dd = ddepth_per_extend if abs(ddepth_per_extend) > 1e-3 else -0.25
    ds = float(np.clip((-derr) / dd, -reach_step, reach_step))
    pred = depth_m + ds * dd
    if pred < stop_floor - 1e-3:
        ds = float(np.clip((stop_floor - depth_m) / dd, -reach_step, reach_step))
    dq["shoulder"] = ds
    dq["elbow"] = -ds
    return next_phase, dq
