"""Shared RoArm joint limits for planning and safety checks."""
from __future__ import annotations

from typing import Dict, Tuple

# Hard limits are the software envelope for commands. The elbow lower bound was
# widened after operator-confirmed test4 motion reached ~0.275 rad.
HARD_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-1.57, 1.57),
    "shoulder": (-1.60, 1.60),
    "elbow": (0.20, 3.20),
}

# Soft limits are planning preferences. A target outside soft but inside hard is
# allowed, but should be logged because it means the planner is near an edge.
SOFT_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-1.25, 1.25),
    "shoulder": (-1.35, 1.45),
    "elbow": (0.35, 3.05),
}


def clamp_joint(name: str, value: float, *, hard: bool = True) -> float:
    limits = HARD_JOINT_LIMITS if hard else SOFT_JOINT_LIMITS
    lo, hi = limits.get(name, (-3.14, 3.14))
    return max(lo, min(hi, float(value)))


def near_hard_limit(name: str, value: float, *, margin: float = 0.05) -> bool:
    lo, hi = HARD_JOINT_LIMITS.get(name, (-3.14, 3.14))
    return float(value) <= lo + margin or float(value) >= hi - margin
