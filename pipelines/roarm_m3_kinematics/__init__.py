"""Opt-in canonical kinematics for Waveshare RoArm-M3.

This package intentionally does not replace legacy ``pipelines.roarm_kinematics``.
Phase A contains no servo/RPC transport and cannot move the arm.
"""
from .conventions import (
    FIRMWARE_LIMITS,
    GEOMETRY,
    HARD_LIMITS,
    JOINT_NAMES,
    POSE_NAMES,
    SOFT_LIMITS,
    Geometry,
    JointVector,
    Pose5,
    as_joint_vector,
    as_pose5,
    joint_limit_margins,
    within_limits,
)
from .model import (
    IK_NO_SOLUTION,
    IKCandidate,
    IKResult,
    angular_distance,
    damped_pseudoinverse,
    fk,
    ik,
    jacobian,
    jacobian_condition,
    pose_error,
    scaled_jacobian,
    select_ik_solution,
    wrap_to_pi,
)

__all__ = [
    "FIRMWARE_LIMITS",
    "GEOMETRY",
    "HARD_LIMITS",
    "IK_NO_SOLUTION",
    "IKCandidate",
    "IKResult",
    "JOINT_NAMES",
    "POSE_NAMES",
    "SOFT_LIMITS",
    "Geometry",
    "JointVector",
    "Pose5",
    "angular_distance",
    "as_joint_vector",
    "as_pose5",
    "damped_pseudoinverse",
    "fk",
    "ik",
    "jacobian",
    "jacobian_condition",
    "joint_limit_margins",
    "pose_error",
    "scaled_jacobian",
    "select_ik_solution",
    "within_limits",
    "wrap_to_pi",
]
