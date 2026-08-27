"""FK/IK for the CAD-derived fixed grasp TCP on RoArm-M3 link5."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipelines.roarm_calibration.urdf_fk import base_to_grasp

from .conventions import HARD_LIMITS, JOINT_NAMES, JointLike, JointVector, Pose5, PoseLike, as_joint_vector, as_pose5, within_limits
from .model import IK_NO_SOLUTION, pose_error


@dataclass(frozen=True)
class GraspIKResult:
    status: str
    target: Pose5
    q: Optional[JointVector]
    residual: tuple[float, float, float, float, float]
    iterations: int
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == "OK" and self.q is not None


def grasp_fk(q: JointLike) -> Pose5:
    """Return grasp origin XYZ and approach-axis pitch/roll."""

    joints = as_joint_vector(q)
    transform = base_to_grasp(joints.as_array())
    rotation = transform.rotation
    approach = rotation[:, 0]
    closing = rotation[:, 1]
    pitch = math.atan2(-float(approach[2]), math.hypot(float(approach[0]), float(approach[1])))
    base = joints.base
    closing_zero = np.asarray([-math.sin(base), math.cos(base), 0.0])
    normal_zero = np.cross(approach, closing_zero)
    normal_zero /= np.linalg.norm(normal_zero)
    roll = math.atan2(float(closing @ normal_zero), float(closing @ closing_zero))
    return Pose5(*map(float, transform.translation_mm), pitch, roll)


def grasp_jacobian(q: JointLike, *, step_rad: float = 1e-6) -> np.ndarray:
    """Central finite-difference 5x5 Jacobian for the grasp TCP."""

    center = as_joint_vector(q).as_array()
    result = np.empty((5, 5), dtype=np.float64)
    for column in range(5):
        plus, minus = center.copy(), center.copy()
        plus[column] += step_rad
        minus[column] -= step_rad
        result[:, column] = pose_error(grasp_fk(minus), grasp_fk(plus)) / (2.0 * step_rad)
    return result


def grasp_ik(
    target: PoseLike,
    seed_q: JointLike,
    *,
    max_iterations: int = 80,
    translation_scale_mm: float = 200.0,
    damping: float = 1e-3,
    max_joint_step_rad: float = 0.15,
) -> GraspIKResult:
    """Seeded damped numerical IK for ``[grasp XYZ, pitch, roll]``."""

    desired = as_pose5(target)
    q = as_joint_vector(seed_q).as_array().copy()
    if not within_limits(q, "hard"):
        return GraspIKResult(IK_NO_SOLUTION, desired, None, (math.inf,) * 5, 0, "SEED_LIMIT")
    scale = np.asarray([1.0 / translation_scale_mm] * 3 + [1.0, 1.0])
    for iteration in range(max_iterations + 1):
        residual = pose_error(grasp_fk(q), desired)
        if np.linalg.norm(residual[:3]) <= 1e-5 and np.linalg.norm(residual[3:]) <= 1e-7:
            solution = JointVector(*map(float, q))
            return GraspIKResult("OK", desired, solution, tuple(map(float, residual)), iteration, "OK")
        if iteration == max_iterations:
            break
        jacobian = scale[:, None] * grasp_jacobian(q)
        task = scale * residual
        normal = jacobian @ jacobian.T + (damping * damping) * np.eye(5)
        try:
            delta = jacobian.T @ np.linalg.solve(normal, task)
        except np.linalg.LinAlgError:
            break
        largest = float(np.max(np.abs(delta)))
        if largest > max_joint_step_rad:
            delta *= max_joint_step_rad / largest
        q += delta
        for index, name in enumerate(JOINT_NAMES):
            lower, upper = HARD_LIMITS[name]
            q[index] = float(np.clip(q[index], lower, upper))
        if not np.isfinite(q).all():
            break
    residual = pose_error(grasp_fk(q), desired)
    return GraspIKResult(
        IK_NO_SOLUTION,
        desired,
        None,
        tuple(map(float, residual)),
        max_iterations,
        "NO_CONVERGENCE",
    )
