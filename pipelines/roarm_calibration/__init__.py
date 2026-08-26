"""RoArm-M3 eye-in-hand calibration primitives.

All rigid transforms use ``T_parent_child`` notation and map points from the
child frame into the parent frame. Translation units are millimetres.
"""

from .handeye import CalibrationResult, HandEyeObservation, solve_handeye
from .se3 import RigidTransform
from .urdf_fk import base_to_hand_tcp, base_to_link5
from .validation import KnownPointObservation, validate_known_points

__all__ = [
    "CalibrationResult",
    "HandEyeObservation",
    "RigidTransform",
    "KnownPointObservation",
    "base_to_hand_tcp",
    "base_to_link5",
    "solve_handeye",
    "validate_known_points",
]
