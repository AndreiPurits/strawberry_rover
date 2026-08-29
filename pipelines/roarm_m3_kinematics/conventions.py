"""Canonical RoArm-M3 geometry, joint order, units, and safety envelopes.

The pose model uses the exact conventions exposed by Waveshare firmware T:105:

    q = [base, shoulder, elbow, wrist, roll]  # radians
    pose = [x_mm, y_mm, z_mm, pitch_rad, roll_rad]

``hand`` is the gripper opening command and is deliberately not part of q.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

JOINT_NAMES: Tuple[str, ...] = ("base", "shoulder", "elbow", "wrist", "roll")
POSE_NAMES: Tuple[str, ...] = ("x_mm", "y_mm", "z_mm", "pitch", "roll")


@dataclass(frozen=True)
class Geometry:
    """Official firmware link-vector representation, in millimetres/radians."""

    l2_a_mm: float = 236.82
    l2_b_mm: float = 30.00
    l3_a_mm: float = 144.49
    l3_b_mm: float = 0.0
    tool_a_mm: float = 171.67
    tool_b_mm: float = 13.69

    @property
    def l2_mm(self) -> float:
        return math.hypot(self.l2_a_mm, self.l2_b_mm)

    @property
    def l3_mm(self) -> float:
        return math.hypot(self.l3_a_mm, self.l3_b_mm)

    @property
    def tool_mm(self) -> float:
        return math.hypot(self.tool_a_mm, self.tool_b_mm)

    @property
    def d2_rad(self) -> float:
        return math.atan2(self.l2_b_mm, self.l2_a_mm)

    @property
    def d3_rad(self) -> float:
        return math.atan2(self.l3_b_mm, self.l3_a_mm)

    @property
    def tool_offset_rad(self) -> float:
        return math.atan2(self.tool_b_mm, self.tool_a_mm)

    @property
    def max_reach_mm(self) -> float:
        return self.l2_mm + self.l3_mm + self.tool_mm


GEOMETRY = Geometry()


# Raw command saturation observed in the official RoArm-M3 firmware. The elbow
# servo mapping clamps to approximately [0, pi]. These are model/mechanism
# limits, not permission to move at the edge.
FIRMWARE_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-math.pi, math.pi),
    "shoulder": (-math.pi / 2.0, math.pi / 2.0),
    "elbow": (0.0, math.pi),
    "wrist": (-math.pi / 2.0, math.pi / 2.0),
    "roll": (-math.pi, math.pi),
}

# Conservative operational envelope for the new stack. It is the intersection
# of firmware limits and already-used project limits. The replacement arm's
# DOM_FINAL was physically established at roll=1.89 rad, so only the positive
# roll envelope is extended; the unvalidated negative side stays at -pi/2.
HARD_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-1.57, 1.57),
    "shoulder": (-math.pi / 2.0, math.pi / 2.0),
    "elbow": (0.20, math.pi),
    "wrist": (-math.pi / 2.0, math.pi / 2.0),
    "roll": (-math.pi / 2.0, 2.20),
}

# Planning preference only. Hard-limit checks always remain authoritative.
SOFT_LIMITS: Dict[str, Tuple[float, float]] = {
    "base": (-1.25, 1.25),
    "shoulder": (-1.35, 1.45),
    "elbow": (0.35, 3.05),
    "wrist": (-1.30, 1.30),
    "roll": (-1.30, 2.05),
}


@dataclass(frozen=True)
class JointVector:
    base: float
    shoulder: float
    elbow: float
    wrist: float
    roll: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.base, self.shoulder, self.elbow, self.wrist, self.roll],
            dtype=np.float64,
        )

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(JOINT_NAMES, map(float, self.as_array())))


@dataclass(frozen=True)
class Pose5:
    x_mm: float
    y_mm: float
    z_mm: float
    pitch: float
    roll: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.x_mm, self.y_mm, self.z_mm, self.pitch, self.roll],
            dtype=np.float64,
        )

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(POSE_NAMES, map(float, self.as_array())))


JointLike = Union[JointVector, Sequence[float], Mapping[str, float], np.ndarray]
PoseLike = Union[Pose5, Sequence[float], Mapping[str, float], np.ndarray]


def as_joint_vector(q: JointLike) -> JointVector:
    if isinstance(q, JointVector):
        return q
    if isinstance(q, Mapping):
        return JointVector(*(float(q[name]) for name in JOINT_NAMES))
    values = np.asarray(q, dtype=np.float64).reshape(-1)
    if values.size != len(JOINT_NAMES):
        raise ValueError(f"q must contain {len(JOINT_NAMES)} values in {JOINT_NAMES} order")
    return JointVector(*map(float, values))


def as_pose5(pose: PoseLike) -> Pose5:
    if isinstance(pose, Pose5):
        return pose
    if isinstance(pose, Mapping):
        # Accept exact canonical names plus firmware feedback names.
        return Pose5(
            x_mm=float(pose.get("x_mm", pose.get("x"))),
            y_mm=float(pose.get("y_mm", pose.get("y"))),
            z_mm=float(pose.get("z_mm", pose.get("z"))),
            pitch=float(pose.get("pitch", pose.get("tit"))),
            roll=float(pose.get("roll", pose.get("r"))),
        )
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.size != len(POSE_NAMES):
        raise ValueError(f"pose must contain {len(POSE_NAMES)} values in {POSE_NAMES} order")
    return Pose5(*map(float, values))


def limits_for(kind: str = "hard") -> Dict[str, Tuple[float, float]]:
    key = kind.lower()
    if key == "hard":
        return HARD_LIMITS
    if key == "soft":
        return SOFT_LIMITS
    if key == "firmware":
        return FIRMWARE_LIMITS
    raise ValueError("limit kind must be 'soft', 'hard', or 'firmware'")


def within_limits(q: JointLike, kind: str = "hard", *, atol: float = 1e-12) -> bool:
    values = as_joint_vector(q).as_array()
    limits = limits_for(kind)
    return bool(
        np.isfinite(values).all()
        and all(
            limits[name][0] - atol <= value <= limits[name][1] + atol
            for name, value in zip(JOINT_NAMES, values)
        )
    )


def joint_limit_margins(q: JointLike, kind: str = "hard") -> Dict[str, float]:
    values = as_joint_vector(q).as_array()
    limits = limits_for(kind)
    return {
        name: float(min(value - limits[name][0], limits[name][1] - value))
        for name, value in zip(JOINT_NAMES, values)
    }
