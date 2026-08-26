"""Independent physical known-point validation for a solved hand-eye transform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .se3 import RigidTransform


@dataclass(frozen=True)
class KnownPointObservation:
    observation_id: str
    transform_base_link5: RigidTransform
    point_camera_mm: np.ndarray
    point_base_expected_mm: np.ndarray

    def __post_init__(self) -> None:
        if self.transform_base_link5.parent != "base_link" or self.transform_base_link5.child != "link5":
            raise ValueError("expected T_base_link5")
        for name in ("point_camera_mm", "point_base_expected_mm"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.isfinite(value).all():
                raise ValueError("{} must be finite".format(name))
            object.__setattr__(self, name, value)


def validate_known_points(
    transform_link5_camera: RigidTransform,
    observations: Sequence[KnownPointObservation],
    *,
    max_error_mm: float,
    min_observations: int = 3,
) -> Dict[str, object]:
    if transform_link5_camera.parent != "link5":
        raise ValueError("expected T_link5_camera")
    if max_error_mm <= 0.0:
        raise ValueError("max_error_mm must be an explicit positive acceptance limit")
    rows = []
    for item in observations:
        base_camera = item.transform_base_link5.then(transform_link5_camera)
        predicted = base_camera.transform_point(
            item.point_camera_mm, frame=transform_link5_camera.child
        )
        error_vector = predicted - item.point_base_expected_mm
        rows.append(
            {
                "id": item.observation_id,
                "predicted_base_mm": predicted.tolist(),
                "expected_base_mm": item.point_base_expected_mm.tolist(),
                "error_vector_mm": error_vector.tolist(),
                "error_norm_mm": float(np.linalg.norm(error_vector)),
            }
        )
    errors = np.asarray([row["error_norm_mm"] for row in rows], dtype=np.float64)
    enough = len(rows) >= int(min_observations)
    passed = bool(enough and np.all(errors <= float(max_error_mm)))
    return {
        "status": "PASS" if passed else "FAIL",
        "KIN_GATE": "PASS" if passed else "FAIL",
        "count": len(rows),
        "min_observations": int(min_observations),
        "max_error_limit_mm": float(max_error_mm),
        "max_error_mm": float(np.max(errors)) if errors.size else None,
        "mean_error_mm": float(np.mean(errors)) if errors.size else None,
        "observations": rows,
        "reason": (
            "OK"
            if passed
            else "INSUFFICIENT_VALIDATION_OBSERVATIONS"
            if not enough
            else "KNOWN_POINT_ERROR_EXCEEDED"
        ),
    }
