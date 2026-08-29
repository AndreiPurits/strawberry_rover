"""Fail-closed stereo obstacle gate for RoArm joint-space moves.

The current RGB-D cloud is treated as a static local map.  A dominant plane
(wall/board) is fitted outside the selected target ROI, then the predicted
camera and grasp-TCP path is checked against that plane.  This is a last-line
safety interlock; it does not replace FK/IK, joint limits, or an E-stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from pipelines.roarm_calibration.se3 import RigidTransform
from pipelines.roarm_calibration.urdf_fk import base_to_grasp, base_to_link5
from .conventions import JointLike, as_joint_vector, within_limits


@dataclass(frozen=True)
class StereoSafetyResult:
    status: str
    reason: str
    start_clearance_mm: Optional[float] = None
    target_clearance_mm: Optional[float] = None
    swept_clearance_mm: Optional[float] = None
    plane_inlier_ratio: Optional[float] = None
    plane_rmse_mm: Optional[float] = None
    valid_depth_fraction: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "start_clearance_mm": self.start_clearance_mm,
            "target_clearance_mm": self.target_clearance_mm,
            "swept_clearance_mm": self.swept_clearance_mm,
            "plane_inlier_ratio": self.plane_inlier_ratio,
            "plane_rmse_mm": self.plane_rmse_mm,
            "valid_depth_fraction": self.valid_depth_fraction,
        }


def feedback_motion_health(q: JointLike, torque_flags: Optional[Sequence[int]] = None) -> StereoSafetyResult:
    """Reject sentinel/out-of-limit feedback before any physical command."""
    try:
        joints = as_joint_vector(q)
    except Exception:
        return StereoSafetyResult("FAIL", "INVALID_JOINT_FEEDBACK")
    if not within_limits(joints, "hard"):
        return StereoSafetyResult("FAIL", "JOINT_FEEDBACK_OUT_OF_LIMITS")
    if torque_flags is not None:
        flags = tuple(int(value) for value in torque_flags)
        if len(flags) != 6 or any(value not in (0, 1) for value in flags):
            return StereoSafetyResult("FAIL", "INVALID_TORQUE_FEEDBACK")
        # A fully de-energized bus together with a motion request is never
        # silently accepted.  The caller may explicitly enable torque only
        # after the rest of the safety gates have passed.
        if not any(flags):
            return StereoSafetyResult("FAIL", "ALL_TORQUE_OFF")
    return StereoSafetyResult("PASS", "HEALTHY_FEEDBACK")


def _depth_points(
    depth_m: np.ndarray,
    intrinsics: Sequence[float],
    *,
    target_roi: Optional[Tuple[int, int, int, int]],
    stride: int,
) -> Tuple[np.ndarray, float]:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        return np.empty((0, 3)), 0.0
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth[::stride, ::stride]
    valid = np.isfinite(zs) & (zs >= 0.08) & (zs <= 1.50)
    # The lower band is dominated by the eye-in-hand gripper and cable.
    valid &= ys < int(0.82 * h)
    if target_roi is not None:
        x1, y1, x2, y2 = target_roi
        pad = max(8, int(0.04 * max(h, w)))
        valid &= ~(
            (xs >= x1 - pad) & (xs <= x2 + pad)
            & (ys >= y1 - pad) & (ys <= y2 + pad)
        )
    fraction = float(np.count_nonzero(valid)) / max(1, valid.size)
    if not np.any(valid):
        return np.empty((0, 3)), fraction
    fx, fy, cx, cy = map(float, intrinsics)
    z_mm = zs[valid] * 1000.0
    x_mm = (xs[valid] - cx) * z_mm / fx
    y_mm = (ys[valid] - cy) * z_mm / fy
    return np.column_stack((x_mm, y_mm, z_mm)), fraction


def fit_dominant_plane(
    points_mm: np.ndarray,
    *,
    iterations: int = 180,
    inlier_threshold_mm: float = 10.0,
) -> Tuple[np.ndarray, float, float, float]:
    """Return camera-facing plane ``n, d, inlier_ratio, rmse``.

    The signed clearance of a camera-frame point is ``n @ p + d``; the camera
    origin is forced onto the positive side.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if len(points) < 80:
        raise ValueError("INSUFFICIENT_DEPTH_POINTS")
    rng = np.random.RandomState(17)
    best = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal /= norm
        d = -float(normal @ sample[0])
        residual = np.abs(points @ normal + d)
        count = int(np.count_nonzero(residual <= inlier_threshold_mm))
        if count > best_count:
            best_count, best = count, residual <= inlier_threshold_mm
    if best is None or best_count < 60:
        raise ValueError("NO_DOMINANT_PLANE")
    cloud = points[best]
    centroid = cloud.mean(axis=0)
    _, _, vh = np.linalg.svd(cloud - centroid, full_matrices=False)
    normal = vh[-1]
    d = -float(normal @ centroid)
    if d < 0.0:
        normal, d = -normal, -d
    residual = points @ normal + d
    inliers = np.abs(residual) <= inlier_threshold_mm
    ratio = float(np.count_nonzero(inliers)) / len(points)
    rmse = float(np.sqrt(np.mean(np.square(residual[inliers]))))
    return normal, d, ratio, rmse


def stereo_motion_gate(
    depth_m: np.ndarray,
    intrinsics: Sequence[float],
    q_start: JointLike,
    q_target: JointLike,
    transform_link5_camera: RigidTransform,
    *,
    target_roi: Optional[Tuple[int, int, int, int]] = None,
    minimum_clearance_mm: float = 25.0,
    samples: int = 25,
    roll_zero_offset_rad: float = 0.0,
) -> StereoSafetyResult:
    """Check the predicted camera/tool path against the observed wall plane."""
    start = as_joint_vector(q_start)
    target = as_joint_vector(q_target)
    if not within_limits(start, "hard"):
        return StereoSafetyResult("FAIL", "START_JOINTS_OUT_OF_LIMITS")
    if not within_limits(target, "hard"):
        return StereoSafetyResult("FAIL", "TARGET_JOINTS_OUT_OF_LIMITS")
    points, valid_fraction = _depth_points(
        depth_m, intrinsics, target_roi=target_roi, stride=8
    )
    if valid_fraction < 0.18:
        return StereoSafetyResult(
            "FAIL", "DEPTH_COVERAGE_LOW", valid_depth_fraction=valid_fraction
        )
    try:
        normal, d, ratio, rmse = fit_dominant_plane(points)
    except ValueError as exc:
        return StereoSafetyResult(
            "FAIL", str(exc), valid_depth_fraction=valid_fraction
        )
    if ratio < 0.28 or rmse > 8.0:
        return StereoSafetyResult(
            "FAIL", "PLANE_QUALITY_LOW", plane_inlier_ratio=ratio,
            plane_rmse_mm=rmse, valid_depth_fraction=valid_fraction,
        )

    def physical_values(values: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(values, dtype=np.float64).copy()
        adjusted[4] -= float(roll_zero_offset_rad)
        return adjusted

    base_camera0 = base_to_link5(physical_values(start.as_array())).then(transform_link5_camera)
    camera0_base = base_camera0.inverse()
    clearances = []
    start_values, target_values = start.as_array(), target.as_array()
    # Camera origin, TCP, and intermediate tool-shaft points are checked.
    for alpha in np.linspace(0.0, 1.0, max(2, int(samples))):
        q = physical_values(start_values + alpha * (target_values - start_values))
        base_link5 = base_to_link5(q)
        base_camera = base_link5.then(transform_link5_camera)
        base_grasp = base_to_grasp(q)
        origins_base = [base_camera.translation_mm]
        for fraction in (0.35, 0.70, 1.0):
            origins_base.append(
                base_link5.translation_mm
                + fraction * (base_grasp.translation_mm - base_link5.translation_mm)
            )
        for point_base in origins_base:
            point_camera0 = camera0_base.transform_point(point_base, frame="base_link")
            clearances.append(float(normal @ point_camera0 + d))
    per_step = 4
    start_clearance = min(clearances[:per_step])
    target_clearance = min(clearances[-per_step:])
    swept_clearance = min(clearances)
    common = dict(
        start_clearance_mm=start_clearance,
        target_clearance_mm=target_clearance,
        swept_clearance_mm=swept_clearance,
        plane_inlier_ratio=ratio,
        plane_rmse_mm=rmse,
        valid_depth_fraction=valid_fraction,
    )
    # Always permit a monotonic retreat from an already-too-close condition.
    if start_clearance < minimum_clearance_mm and target_clearance > start_clearance + 3.0:
        return StereoSafetyResult("PASS", "RETREAT_FROM_OBSTACLE", **common)
    if swept_clearance < minimum_clearance_mm:
        return StereoSafetyResult("FAIL", "OBSTACLE_CLEARANCE_GATE", **common)
    return StereoSafetyResult("PASS", "CLEAR", **common)
