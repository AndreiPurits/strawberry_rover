"""Named, validated SE(3) transforms for the RoArm calibration stack."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


def _as_rotation_matrix(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-7):
        raise ValueError("rotation determinant must be +1")
    return matrix


def rotation_angle_rad(rotation: Sequence[Sequence[float]]) -> float:
    matrix = _as_rotation_matrix(rotation)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_about_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = vector / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray(
        [
            [c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s],
            [y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s],
            [z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RigidTransform:
    """A transform mapping child-frame coordinates into a parent frame."""

    parent: str
    child: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if not self.parent or not self.child:
            raise ValueError("parent and child frame names are required")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("invalid homogeneous transform last row")
        _as_rotation_matrix(matrix[:3, :3])
        object.__setattr__(self, "matrix", matrix.copy())

    @classmethod
    def identity(cls, frame: str) -> "RigidTransform":
        return cls(frame, frame, np.eye(4, dtype=np.float64))

    @classmethod
    def from_rt(
        cls,
        parent: str,
        child: str,
        rotation: Sequence[Sequence[float]],
        translation_mm: Sequence[float],
    ) -> "RigidTransform":
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _as_rotation_matrix(rotation)
        translation = np.asarray(translation_mm, dtype=np.float64).reshape(3)
        if not np.isfinite(translation).all():
            raise ValueError("translation must be finite")
        matrix[:3, 3] = translation
        return cls(parent, child, matrix)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3].copy()

    @property
    def translation_mm(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    def inverse(self) -> "RigidTransform":
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        return RigidTransform.from_rt(
            self.child, self.parent, rotation.T, -rotation.T @ translation
        )

    def then(self, child_transform: "RigidTransform") -> "RigidTransform":
        """Compose ``T_parent_selfchild @ T_selfchild_grandchild``."""
        if self.child != child_transform.parent:
            raise ValueError(
                "frame mismatch: {} != {}".format(self.child, child_transform.parent)
            )
        return RigidTransform(
            self.parent,
            child_transform.child,
            self.matrix @ child_transform.matrix,
        )

    def transform_point(self, point_mm: Sequence[float], *, frame: str) -> np.ndarray:
        if frame != self.child:
            raise ValueError("point frame {} does not match {}".format(frame, self.child))
        point = np.asarray(point_mm, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            raise ValueError("point must be finite")
        return self.rotation @ point + self.translation_mm

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent": self.parent,
            "child": self.child,
            "units": "mm",
            "matrix": self.matrix.tolist(),
        }
