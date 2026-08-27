"""Full SE(3) FK copied from the official RoArm-M3 Xacro chain.

The chain deliberately starts at ``base_link`` (not the URDF ``world`` frame,
whose only difference is a fixed 70.1 mm display offset). Joint order and
angles are the canonical project convention:

    [base, shoulder, elbow, wrist, roll]
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .se3 import RigidTransform, rotation_about_axis, rotation_from_rpy


# Official Xacro hand_tcp is [0, 0, 115.428] mm.  The official STL contact
# faces at closed hand=0 are x=-10.025849 and x=-5.983416 mm in link5, so the
# actual jaw mid-plane is x=-8.0046325 mm.  Keep the official longitudinal TCP
# and orientation, but put the grasp origin in that measured mid-plane.
GRASP_CENTER_LINK5_MM = np.asarray([-8.0046325, 0.0, 115.428], dtype=np.float64)
GRASP_RPY_LINK5_RAD = np.asarray([1.5708, -1.5708, 0.0], dtype=np.float64)


def _joint_values(q: Sequence[float]) -> np.ndarray:
    if hasattr(q, "as_array"):
        values = np.asarray(q.as_array(), dtype=np.float64).reshape(-1)
    elif isinstance(q, Mapping):
        values = np.asarray([q[name] for name in ("base", "shoulder", "elbow", "wrist", "roll")], dtype=np.float64)
    else:
        values = np.asarray(q, dtype=np.float64).reshape(-1)
    if values.size != 5 or not np.isfinite(values).all():
        raise ValueError("q must contain five finite pose joints")
    return values


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
    base, shoulder, elbow, wrist, roll = _joint_values(q)
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


def link5_to_grasp() -> RigidTransform:
    """Fixed CAD grasp frame at the center between the closed jaw faces.

    Frame axes follow the official ``hand_tcp`` orientation:
    ``+X`` is the approach axis (link5 ``+Z``), ``+Y`` is the jaw-closing
    direction (link5 ``-X``), and ``+Z`` completes the right-handed frame
    (link5 ``-Y``).
    """

    return RigidTransform.from_rt(
        "link5",
        "grasp",
        rotation_from_rpy(*GRASP_RPY_LINK5_RAD),
        GRASP_CENTER_LINK5_MM,
    )


def base_to_grasp(q: Sequence[float]) -> RigidTransform:
    return base_to_link5(q).then(link5_to_grasp())
