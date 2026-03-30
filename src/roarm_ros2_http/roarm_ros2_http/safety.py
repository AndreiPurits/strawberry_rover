"""Safety checks for Cartesian targets before sending motion commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple


@dataclass(frozen=True)
class WorkspaceLimits:
    """Axis and radial limits for safe Cartesian targets."""

    x_min: float = 0.05
    x_max: float = 0.35
    y_min: float = -0.20
    y_max: float = 0.20
    z_min: float = 0.02
    z_max: float = 0.30
    min_planar_radius: float = 0.08
    max_planar_radius: float = 0.32


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to the [low, high] range."""
    return max(low, min(high, value))


def clamp_xyz(x: float, y: float, z: float, limits: WorkspaceLimits) -> Tuple[float, float, float]:
    """Clamp x, y, z into workspace box limits."""
    return (
        clamp(x, limits.x_min, limits.x_max),
        clamp(y, limits.y_min, limits.y_max),
        clamp(z, limits.z_min, limits.z_max),
    )


def validate_xyz(x: float, y: float, z: float, limits: WorkspaceLimits) -> Tuple[bool, str]:
    """Validate Cartesian target against box and radial constraints."""
    if not (limits.x_min <= x <= limits.x_max):
        return False, f"x={x:.3f} outside [{limits.x_min:.3f}, {limits.x_max:.3f}]"
    if not (limits.y_min <= y <= limits.y_max):
        return False, f"y={y:.3f} outside [{limits.y_min:.3f}, {limits.y_max:.3f}]"
    if not (limits.z_min <= z <= limits.z_max):
        return False, f"z={z:.3f} outside [{limits.z_min:.3f}, {limits.z_max:.3f}]"

    planar_radius = sqrt(x * x + y * y)
    if planar_radius < limits.min_planar_radius:
        return (
            False,
            f"planar radius {planar_radius:.3f} below min {limits.min_planar_radius:.3f}",
        )
    if planar_radius > limits.max_planar_radius:
        return (
            False,
            f"planar radius {planar_radius:.3f} above max {limits.max_planar_radius:.3f}",
        )
    return True, "ok"


def sanitize_target(
    x: float,
    y: float,
    z: float,
    limits: WorkspaceLimits,
    reject_if_unsafe: bool = True,
) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """Validate target and optionally clamp instead of rejecting."""
    ok, reason = validate_xyz(x, y, z, limits)
    if ok:
        return (x, y, z), "ok"

    clamped = clamp_xyz(x, y, z, limits)
    if reject_if_unsafe:
        return None, reason
    return clamped, f"{reason}; clamped to {clamped}"

