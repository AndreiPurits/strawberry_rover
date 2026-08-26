"""Full SE(3) FK copied from the official RoArm-M3 Xacro chain.

The chain deliberately starts at ``base_link`` (not the URDF ``world`` frame,
whose only difference is a fixed 70.1 mm display offset). Joint order and
angles are the canonical project convention:

    [base, shoulder, elbow, wrist, roll]
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from pipelines.roarm_m3_kinematics.conventions import as_joint_vector

from .se3 import RigidTransform, rotation_about_axis, rotation_from_rpy


def _origin(parent: str, child: str, xyz_m: Sequence[float], rpy: Sequence[float]) -> RigidTransform:
    return RigidTransform.from_rt(
        parent,
        child,
        rotation_from_rpy(*rpy),
        1000.0 * np.asarray(xyz_m, dtype=np.float64),
    )


def _joint(parent: str, child: str, xyz_m: Sequence[float], rpy: Sequence[float], q: float) -> RigidTransform:
    origin = _origin(parent, parent + "__joint_origin", xyz_m, rpy)
    rotation = RigidTransform.from_rt(
        parent + "__joint_origin",
        child,
        rotation_about_axis([0.0, 0.0, 1.0], q),
        [0.0, 0.0, 0.0],
    )
    return origin.then(rotation)


def base_to_link5(q: Sequence[float]) -> RigidTransform:
    base, shoulder, elbow, wrist, roll = as_joint_vector(q).as_array()
    transforms = [
        _joint("base_link", "link1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], base),
        _joint("link1", "link2", [0.0, 0.0, 0.051959], [-1.5708, -1.5708, 0.0], shoulder),
        _joint("link2", "link3", [0.236815, 0.030002, 0.0], [0.0, 0.0, 1.5708], elbow),
        _joint("link3", "link4", [0.0, -0.144586, 0.0], [0.0, 0.0, 0.0], wrist),
        _joint("link4", "link5", [0.015147, -0.053653, 0.0], [1.5708, 1.5708, 0.0], roll),
    ]
    result = transforms[0]
    for transform in transforms[1:]:
        result = result.then(transform)
    return result


def base_to_hand_tcp(q: Sequence[float]) -> RigidTransform:
    fixed = _origin(
        "link5", "hand_tcp", [0.0, 0.0, 0.115428], [1.5708, -1.5708, 0.0]
    )
    return base_to_link5(q).then(fixed)
