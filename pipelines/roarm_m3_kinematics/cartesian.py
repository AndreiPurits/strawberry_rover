"""Deadband-aware Cartesian planning on top of canonical RoArm-M3 FK/IK."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pipelines.roarm_calibration.se3 import RigidTransform
from pipelines.roarm_calibration.urdf_fk import base_to_link5

from .conventions import JOINT_NAMES, JointLike, JointVector, Pose5, as_joint_vector, joint_limit_margins
from .model import fk, ik, pose_error


EMPIRICAL_GATES_MM: Dict[str, float] = {
    "coarse": 15.0,
    "visual": 8.0,
    "final": 7.0,
}
JOINT_DEADBAND_RAD: Dict[str, float] = {
    "base": 0.003,
    "shoulder": 0.025,
    "elbow": 0.012,
    "wrist": 0.006,
    "roll": 0.003,
}
MIN_MEANINGFUL_XY_MM = 3.0
MIN_MEANINGFUL_Z_MM = 6.0
MAX_JACOBIAN_CONDITION = 35.0
MIN_HARD_LIMIT_MARGIN_RAD = 0.10


@dataclass(frozen=True)
class CartesianPlan:
    status: str
    reason: str
    stage: str
    seed_q: JointVector
    target: Pose5
    raw_q: Optional[JointVector]
    executable_q: Optional[JointVector]
    suppressed_joints: tuple
    fk_round_trip_error: tuple
    executable_error: tuple
    jacobian_condition: Optional[float]
    min_hard_margin_rad: Optional[float]

    @property
    def ok(self) -> bool:
        return self.status == "PASS" and self.executable_q is not None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "stage": self.stage,
            "seed_q": self.seed_q.as_dict(),
            "target": self.target.as_dict(),
            "raw_q": None if self.raw_q is None else self.raw_q.as_dict(),
            "executable_q": None if self.executable_q is None else self.executable_q.as_dict(),
            "suppressed_joints": list(self.suppressed_joints),
            "fk_round_trip_error": list(self.fk_round_trip_error),
            "executable_error": list(self.executable_error),
            "jacobian_condition": self.jacobian_condition,
            "min_hard_margin_rad": self.min_hard_margin_rad,
        }


def pixel_depth_to_camera_mm(
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: Sequence[float],
) -> np.ndarray:
    """Deproject a rectified color pixel using (fx, fy, cx, cy)."""

    fx, fy, cx, cy = map(float, intrinsics)
    if depth_m <= 0.0 or min(fx, fy) <= 0.0:
        raise ValueError("positive depth and focal lengths are required")
    z_mm = float(depth_m) * 1000.0
    return np.asarray(
        [(float(u_px) - cx) * z_mm / fx, (float(v_px) - cy) * z_mm / fy, z_mm],
        dtype=np.float64,
    )


def point_camera_to_base_mm(
    point_camera_mm: Sequence[float],
    q: JointLike,
    transform_link5_camera: RigidTransform,
) -> np.ndarray:
    base_camera = base_to_link5(as_joint_vector(q).as_array()).then(transform_link5_camera)
    return base_camera.transform_point(
        np.asarray(point_camera_mm, dtype=np.float64), frame=transform_link5_camera.child
    )


def standoff_target_pose(
    point_camera_mm: Sequence[float],
    point_base_mm: Sequence[float],
    q: JointLike,
    transform_link5_camera: RigidTransform,
    *,
    standoff_mm: float,
    pitch: Optional[float] = None,
    roll: Optional[float] = None,
) -> Pose5:
    if standoff_mm < 0.0:
        raise ValueError("standoff_mm must be non-negative")
    seed = as_joint_vector(q)
    current = fk(seed)
    camera_point = np.asarray(point_camera_mm, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(camera_point))
    if norm <= 1e-9:
        raise ValueError("camera point cannot be at the optical origin")
    base_camera = base_to_link5(seed.as_array()).then(transform_link5_camera)
    approach_base = base_camera.matrix[:3, :3] @ (camera_point / norm)
    target_xyz = np.asarray(point_base_mm, dtype=np.float64).reshape(3) - standoff_mm * approach_base
    return Pose5(
        *map(float, target_xyz),
        current.pitch if pitch is None else float(pitch),
        current.roll if roll is None else float(roll),
    )


def _failed(seed: JointVector, target: Pose5, stage: str, reason: str) -> CartesianPlan:
    zeros = (0.0,) * 5
    return CartesianPlan(
        "FAIL", reason, stage, seed, target, None, None, tuple(), zeros, zeros, None, None
    )


def plan_cartesian_pose(
    target: Pose5,
    seed_q: JointLike,
    *,
    stage: str,
    requested_delta: Optional[Sequence[float]] = None,
    max_joint_step_rad: Optional[float] = None,
    deadband_rad: Mapping[str, float] = JOINT_DEADBAND_RAD,
) -> CartesianPlan:
    stage_key = str(stage).lower()
    if stage_key not in EMPIRICAL_GATES_MM:
        raise ValueError("stage must be coarse, visual, or final")
    seed = as_joint_vector(seed_q)
    if requested_delta is not None:
        delta = np.asarray(requested_delta, dtype=np.float64).reshape(5)
        xy = float(np.linalg.norm(delta[:2]))
        if 0.0 < xy < MIN_MEANINGFUL_XY_MM:
            return _failed(seed, target, stage_key, "XY_BELOW_MEANINGFUL_STEP")
        if 0.0 < abs(float(delta[2])) < MIN_MEANINGFUL_Z_MM:
            return _failed(seed, target, stage_key, "Z_BELOW_MEANINGFUL_STEP")

    result = ik(target, seed)
    if not result.ok:
        return _failed(seed, target, stage_key, result.reason)
    selected = result.selected
    assert selected is not None
    raw_q = selected.q
    raw_delta = raw_q.as_array() - seed.as_array()
    if max_joint_step_rad is not None and float(np.max(np.abs(raw_delta))) > max_joint_step_rad:
        return _failed(seed, target, stage_key, "JOINT_STEP_GATE")
    if selected.jacobian_condition > MAX_JACOBIAN_CONDITION:
        return _failed(seed, target, stage_key, "JACOBIAN_GATE")
    margins = joint_limit_margins(raw_q, "hard")
    min_margin = float(min(margins.values()))
    if min_margin < MIN_HARD_LIMIT_MARGIN_RAD:
        return _failed(seed, target, stage_key, "HARD_LIMIT_MARGIN_GATE")

    executable_values = raw_q.as_array().copy()
    suppressed = []
    for index, name in enumerate(JOINT_NAMES):
        if 0.0 < abs(raw_delta[index]) < float(deadband_rad[name]):
            executable_values[index] = seed.as_array()[index]
            suppressed.append(name)
    executable_q = JointVector(*map(float, executable_values))
    raw_error = tuple(map(float, pose_error(fk(raw_q), target)))
    executable_error = tuple(map(float, pose_error(fk(executable_q), target)))
    if float(np.linalg.norm(np.asarray(executable_error[:3]))) > EMPIRICAL_GATES_MM[stage_key]:
        return CartesianPlan(
            "FAIL",
            "EXECUTABLE_FK_GATE",
            stage_key,
            seed,
            target,
            raw_q,
            executable_q,
            tuple(suppressed),
            raw_error,
            executable_error,
            selected.jacobian_condition,
            min_margin,
        )
    return CartesianPlan(
        "PASS",
        "OK",
        stage_key,
        seed,
        target,
        raw_q,
        executable_q,
        tuple(suppressed),
        raw_error,
        executable_error,
        selected.jacobian_condition,
        min_margin,
    )
