#!/usr/bin/env python3
"""Geometry of eye-in-hand one-shot approach (demo metric, Jacobian residual, tip→cam)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from pipelines.arm_approach.cartesian import cartesian_step, fit_tip_camera_map
from pipelines.arm_approach.planner import (
    demo_distance,
    nearest_demo,
    predict_berry,
    solve_delta_with_prior,
)
from pipelines.arm_approach.phases import next_center_reach_step

__all__ = [
    "berry_error",
    "cartesian_step",
    "demo_distance",
    "fit_tip_camera_map",
    "nearest_demo",
    "next_center_reach_step",
    "predict_berry",
    "solve_delta_with_prior",
]


def berry_error(
    berry: Dict[str, Any],
    target: Dict[str, Any],
    *,
    standoff_m: float = 0.12,
) -> np.ndarray:
    """Image/depth residual of a lock relative to a desired image+standoff."""
    return np.array(
        [
            float(target.get("px", 320.0)) - float(berry.get("px", 320.0)),
            float(target.get("py", 240.0)) - float(berry.get("py", 240.0)),
            float(target.get("depth_m", standoff_m)) - float(berry.get("depth_m", standoff_m)),
        ],
        dtype=np.float64,
    )


def identity_tip_camera_map() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def clamp_standoff_depth(depth_m: float, *, depth_min_m: float = 0.10, depth_max_m: float = 0.17) -> bool:
    return depth_min_m <= float(depth_m) <= depth_max_m


def reproject_pixel(
    cam_xyz_mm: Sequence[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[Tuple[float, float]]:
    z = float(cam_xyz_mm[2])
    if z <= 1e-6:
        return None
    px = fx * float(cam_xyz_mm[0]) / z + cx
    py = fy * float(cam_xyz_mm[1]) / z + cy
    return px, py
