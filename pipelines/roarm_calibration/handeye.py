"""Eye-in-hand solver for a fixed calibration board and RoArm ``link5``."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .se3 import RigidTransform, rotation_angle_rad


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class HandEyeObservation:
    observation_id: str
    transform_base_link5: RigidTransform
    transform_camera_board: RigidTransform
    split: str = "train"
    reprojection_rmse_px: Optional[float] = None

    def __post_init__(self) -> None:
        if self.transform_base_link5.parent != "base_link" or self.transform_base_link5.child != "link5":
            raise ValueError("expected T_base_link5")
        if (
            self.transform_camera_board.parent != "stereo_camera_color_optical_frame"
            or self.transform_camera_board.child != "calibration_board"
        ):
            raise ValueError("expected T_stereo_camera_color_optical_frame_calibration_board")
        if self.split not in ("train", "validation"):
            raise ValueError("split must be train or validation")


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    method: Optional[str]
    transform_link5_camera: Optional[RigidTransform]
    report: Dict[str, object]

    @property
    def ok(self) -> bool:
        return self.status == "SOLVED" and self.transform_link5_camera is not None


def _rotation_mean(rotations: Sequence[np.ndarray]) -> np.ndarray:
    accumulator = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _singular, vt = np.linalg.svd(accumulator)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _board_consistency(
    observations: Sequence[HandEyeObservation],
    transform_link5_camera: RigidTransform,
) -> Dict[str, object]:
    estimates = [
        obs.transform_base_link5.then(transform_link5_camera).then(obs.transform_camera_board)
        for obs in observations
    ]
    if not estimates:
        return {"count": 0, "translation_mm": _stats([]), "rotation_deg": _stats([])}
    translation_mean = np.mean([item.translation_mm for item in estimates], axis=0)
    rotation_mean = _rotation_mean([item.rotation for item in estimates])
    translation_errors = [
        float(np.linalg.norm(item.translation_mm - translation_mean)) for item in estimates
    ]
    rotation_errors = [
        math.degrees(rotation_angle_rad(rotation_mean.T @ item.rotation)) for item in estimates
    ]
    return {
        "count": len(estimates),
        "translation_mm": _stats(translation_errors),
        "rotation_deg": _stats(rotation_errors),
        "estimated_T_base_board_mean": RigidTransform.from_rt(
            "base_link", "calibration_board", rotation_mean, translation_mean
        ).to_dict(),
    }


def _motion_diversity(observations: Sequence[HandEyeObservation]) -> Dict[str, object]:
    rotation_vectors: List[np.ndarray] = []
    translations: List[np.ndarray] = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            relative = observations[i].transform_base_link5.inverse().then(
                observations[j].transform_base_link5
            )
            rvec, _ = cv2.Rodrigues(relative.rotation)
            rotation_vectors.append(rvec.reshape(3))
            translations.append(relative.translation_mm)
    rotation_array = np.asarray(rotation_vectors, dtype=np.float64).reshape(-1, 3)
    translation_array = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    rotation_sv = np.linalg.svd(rotation_array, compute_uv=False) if len(rotation_array) else np.zeros(3)
    translation_sv = np.linalg.svd(translation_array, compute_uv=False) if len(translation_array) else np.zeros(3)
    pair_angles = [float(np.linalg.norm(value)) for value in rotation_array]
    pair_distances = [float(np.linalg.norm(value)) for value in translation_array]
    return {
        "pair_count": int(len(rotation_array)),
        "rotation_vector_singular_values_rad": rotation_sv.tolist(),
        "translation_singular_values_mm": translation_sv.tolist(),
        "pair_rotation_deg": _stats([math.degrees(value) for value in pair_angles]),
        "pair_translation_mm": _stats(pair_distances),
        "rotation_rank": int(np.linalg.matrix_rank(rotation_array)) if len(rotation_array) else 0,
        "translation_rank": int(np.linalg.matrix_rank(translation_array)) if len(translation_array) else 0,
    }


def _solve_method(
    observations: Sequence[HandEyeObservation], method: int
) -> RigidTransform:
    rotations_gripper_to_base = [obs.transform_base_link5.rotation for obs in observations]
    translations_gripper_to_base = [obs.transform_base_link5.translation_mm for obs in observations]
    rotations_target_to_camera = [obs.transform_camera_board.rotation for obs in observations]
    translations_target_to_camera = [obs.transform_camera_board.translation_mm for obs in observations]
    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method,
    )
    return RigidTransform.from_rt(
        "link5", "stereo_camera_color_optical_frame", rotation, np.asarray(translation).reshape(3)
    )


def solve_handeye(
    observations: Sequence[HandEyeObservation],
    *,
    min_train: int = 8,
    min_validation: int = 3,
) -> CalibrationResult:
    train = [item for item in observations if item.split == "train"]
    validation = [item for item in observations if item.split == "validation"]
    base_report: Dict[str, object] = {
        "model": "eye-in-hand",
        "solved_transform": "T_link5_camera",
        "units": "mm",
        "observations": {"train": len(train), "validation": len(validation)},
        "requirements": {"min_train": min_train, "min_validation": min_validation},
        "motion_diversity_train": _motion_diversity(train),
        "independent_known_point_validation": "NOT_RUN",
        "KIN_GATE": "FAIL",
    }
    if len(train) < min_train or len(validation) < min_validation:
        base_report["reason"] = "INSUFFICIENT_CALIBRATION_OBSERVATIONS"
        return CalibrationResult("INSUFFICIENT_OBSERVATIONS", None, None, base_report)

    candidates: Dict[str, Dict[str, object]] = {}
    transforms: Dict[str, RigidTransform] = {}
    for name, method in METHODS.items():
        try:
            transform = _solve_method(train, method)
            train_metrics = _board_consistency(train, transform)
            validation_metrics = _board_consistency(validation, transform)
            candidates[name] = {
                "status": "OK",
                "transform": transform.to_dict(),
                "train": train_metrics,
                "validation": validation_metrics,
            }
            transforms[name] = transform
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            candidates[name] = {"status": "FAILED", "error": str(exc)}
    if not transforms:
        base_report["reason"] = "ALL_HANDEYE_METHODS_FAILED"
        base_report["methods"] = candidates
        return CalibrationResult("SOLVER_FAILED", None, None, base_report)

    # Select without looking at the held-out set; validation remains a genuine
    # check against merely fitting the poses used by the solver. It is still not
    # the independent physical known-point validation required before motion.
    def score(name: str) -> Tuple[float, float]:
        metrics = candidates[name]["train"]
        return (
            float(metrics["translation_mm"]["median"]),
            float(metrics["rotation_deg"]["median"]),
        )

    selected_name = min(transforms, key=score)
    selected = transforms[selected_name]
    base_report.update(
        {
            "selected_method": selected_name,
            "methods": candidates,
            "T_link5_camera": selected.to_dict(),
            "reason": "SOLVED_NOT_PHYSICALLY_VALIDATED",
        }
    )
    return CalibrationResult("SOLVED", selected_name, selected, base_report)
