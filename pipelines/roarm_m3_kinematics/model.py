"""Canonical 5-pose-DOF kinematics for Waveshare RoArm-M3."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .conventions import (
    GEOMETRY,
    HARD_LIMITS,
    JOINT_NAMES,
    SOFT_LIMITS,
    Geometry,
    JointLike,
    JointVector,
    Pose5,
    PoseLike,
    as_joint_vector,
    as_pose5,
    joint_limit_margins,
    within_limits,
)

IK_NO_SOLUTION = "IK_NO_SOLUTION"


def wrap_to_pi(angle: float) -> float:
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi and angle > 0 else wrapped


def angular_distance(a: float, b: float) -> float:
    return wrap_to_pi(float(a) - float(b))


@dataclass(frozen=True)
class IKCandidate:
    q: JointVector
    radial_branch: str
    elbow_branch: str
    fk_residual: Tuple[float, float, float, float, float]
    hard_ok: bool
    soft_ok: bool
    min_hard_margin_rad: float
    jacobian_condition: float
    score: float = math.inf


@dataclass(frozen=True)
class IKResult:
    status: str
    target: Pose5
    candidates: Tuple[IKCandidate, ...]
    selected: Optional[IKCandidate]
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == "OK" and self.selected is not None


def fk(q: JointLike, *, geometry: Geometry = GEOMETRY) -> Pose5:
    """Firmware-equivalent FK for the edge/tool origin reported by T:105.

    Translation is in millimetres. Pitch and roll are in radians. No base-height
    offset is added because T:105 uses the shoulder/base kinematic origin.
    """

    j = as_joint_vector(q)
    b, s, e, t, r = j.as_array()
    phi2 = s + geometry.d2_rad
    phi3 = s + e + geometry.d3_rad
    phi_tool = s + e + t + geometry.tool_offset_rad

    rho = (
        geometry.l2_mm * math.sin(phi2)
        + geometry.l3_mm * math.sin(phi3)
        + geometry.tool_mm * math.sin(phi_tool)
    )
    z_mm = (
        geometry.l2_mm * math.cos(phi2)
        + geometry.l3_mm * math.cos(phi3)
        + geometry.tool_mm * math.cos(phi_tool)
    )
    return Pose5(
        x_mm=rho * math.cos(b),
        y_mm=rho * math.sin(b),
        z_mm=z_mm,
        pitch=s + e + t - math.pi / 2.0,
        roll=r,
    )


def pose_error(current: PoseLike, target: PoseLike) -> np.ndarray:
    """Return target-current in canonical task order.

    Translation errors are millimetres; orientation errors are wrapped radians.
    """

    cur = as_pose5(current).as_array()
    tgt = as_pose5(target).as_array()
    err = tgt - cur
    err[3] = angular_distance(tgt[3], cur[3])
    err[4] = angular_distance(tgt[4], cur[4])
    return err


def jacobian(q: JointLike, *, geometry: Geometry = GEOMETRY) -> np.ndarray:
    """Return physical 5x5 J for [x,y,z,pitch,roll] wrt canonical q.

    The first three rows are mm/rad; the last two rows are rad/rad.
    """

    b, s, e, t, _r = as_joint_vector(q).as_array()
    phi2 = s + geometry.d2_rad
    phi3 = s + e + geometry.d3_rad
    phi_tool = s + e + t + geometry.tool_offset_rad

    rho = (
        geometry.l2_mm * math.sin(phi2)
        + geometry.l3_mm * math.sin(phi3)
        + geometry.tool_mm * math.sin(phi_tool)
    )
    rho_s = (
        geometry.l2_mm * math.cos(phi2)
        + geometry.l3_mm * math.cos(phi3)
        + geometry.tool_mm * math.cos(phi_tool)
    )
    rho_e = geometry.l3_mm * math.cos(phi3) + geometry.tool_mm * math.cos(phi_tool)
    rho_t = geometry.tool_mm * math.cos(phi_tool)
    z_s = -rho
    z_e = -geometry.l3_mm * math.sin(phi3) - geometry.tool_mm * math.sin(phi_tool)
    z_t = -geometry.tool_mm * math.sin(phi_tool)

    cb, sb = math.cos(b), math.sin(b)
    return np.asarray(
        [
            [-rho * sb, cb * rho_s, cb * rho_e, cb * rho_t, 0.0],
            [rho * cb, sb * rho_s, sb * rho_e, sb * rho_t, 0.0],
            [0.0, z_s, z_e, z_t, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def scaled_jacobian(
    q_or_jacobian: Union[JointLike, np.ndarray],
    *,
    translation_scale_mm: float = GEOMETRY.max_reach_mm,
) -> np.ndarray:
    """Dimensionless task-scaled Jacobian for conditioning diagnostics."""

    if translation_scale_mm <= 0:
        raise ValueError("translation_scale_mm must be positive")
    if isinstance(q_or_jacobian, np.ndarray) and q_or_jacobian.shape == (5, 5):
        j = np.asarray(q_or_jacobian, dtype=np.float64)
    else:
        j = jacobian(q_or_jacobian)
    scale = np.diag([1.0 / translation_scale_mm] * 3 + [1.0, 1.0])
    return scale @ j


def jacobian_condition(
    q_or_jacobian: Union[JointLike, np.ndarray],
    *,
    translation_scale_mm: float = GEOMETRY.max_reach_mm,
) -> float:
    return float(
        np.linalg.cond(
            scaled_jacobian(q_or_jacobian, translation_scale_mm=translation_scale_mm)
        )
    )


def damped_pseudoinverse(
    j: np.ndarray,
    *,
    damping: float = 1e-3,
    translation_scale_mm: Optional[float] = None,
) -> np.ndarray:
    """Damped least-squares pseudoinverse.

    When ``translation_scale_mm`` is supplied, task translation is normalized
    before damping and the returned matrix still maps an unscaled canonical
    task delta [mm, mm, mm, rad, rad] to joint radians.
    """

    matrix = np.asarray(j, dtype=np.float64)
    if matrix.shape != (5, 5):
        raise ValueError("expected a 5x5 Jacobian")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    if translation_scale_mm is None:
        scale = np.eye(5, dtype=np.float64)
    else:
        if translation_scale_mm <= 0:
            raise ValueError("translation_scale_mm must be positive")
        scale = np.diag([1.0 / translation_scale_mm] * 3 + [1.0, 1.0])
    js = scale @ matrix
    u, singular, vt = np.linalg.svd(js, full_matrices=False)
    gains = singular / (singular * singular + float(damping) ** 2)
    return (vt.T * gains) @ u.T @ scale


def _candidate_score(
    q: JointVector,
    seed_q: Optional[JointVector],
    *,
    condition: float,
) -> float:
    values = q.as_array()
    if seed_q is None:
        # Neutral reference is the centre of the soft envelope.
        seed = np.asarray(
            [(SOFT_LIMITS[name][0] + SOFT_LIMITS[name][1]) / 2.0 for name in JOINT_NAMES],
            dtype=np.float64,
        )
    else:
        seed = seed_q.as_array()
    delta = values - seed
    delta[0] = angular_distance(values[0], seed[0])
    delta[4] = angular_distance(values[4], seed[4])
    weights = np.asarray([1.0, 1.2, 1.0, 0.8, 0.5], dtype=np.float64)
    travel = float(np.sum(weights * delta * delta))

    soft_penalty = 0.0
    for name, value in zip(JOINT_NAMES, values):
        lo, hi = SOFT_LIMITS[name]
        if value < lo:
            soft_penalty += 100.0 * (lo - value) ** 2 + 10.0
        elif value > hi:
            soft_penalty += 100.0 * (value - hi) ** 2 + 10.0

    margins = joint_limit_margins(q, "hard")
    min_margin = max(1e-9, min(margins.values()))
    edge_penalty = 0.01 / min_margin
    condition_penalty = math.inf if not math.isfinite(condition) else 0.01 * math.log1p(condition)
    return travel + soft_penalty + edge_penalty + condition_penalty


def select_ik_solution(
    candidates: Iterable[IKCandidate],
    seed_q: Optional[JointLike] = None,
) -> Optional[IKCandidate]:
    """Select a hard-valid, continuous, limit-aware candidate."""

    seed = as_joint_vector(seed_q) if seed_q is not None else None
    # Exact rank loss is never a selectable IK solution. Near-singular finite
    # candidates remain visible for diagnostics and are ordered behind better
    # conditioned alternatives; a later physical gate must set its measured
    # finite-condition threshold.
    valid = [
        candidate
        for candidate in candidates
        if candidate.hard_ok and math.isfinite(candidate.jacobian_condition)
    ]
    if not valid:
        return None
    rescored = [
        IKCandidate(
            q=c.q,
            radial_branch=c.radial_branch,
            elbow_branch=c.elbow_branch,
            fk_residual=c.fk_residual,
            hard_ok=c.hard_ok,
            soft_ok=c.soft_ok,
            min_hard_margin_rad=c.min_hard_margin_rad,
            jacobian_condition=c.jacobian_condition,
            score=_candidate_score(c.q, seed, condition=c.jacobian_condition),
        )
        for c in valid
    ]
    return min(rescored, key=lambda candidate: candidate.score)


def ik(
    target_pose: PoseLike,
    seed_q: Optional[JointLike] = None,
    *,
    geometry: Geometry = GEOMETRY,
    residual_tolerance_mm: float = 1e-6,
    residual_tolerance_rad: float = 1e-9,
) -> IKResult:
    """Closed-form IK with radial and elbow branches plus explicit failure."""

    target = as_pose5(target_pose)
    target_values = target.as_array()
    if not np.isfinite(target_values).all():
        return IKResult(IK_NO_SOLUTION, target, tuple(), None, "non_finite_target")

    x_mm, y_mm, z_mm, pitch, roll = target_values
    radial_abs = math.hypot(x_mm, y_mm)
    base_primary = math.atan2(y_mm, x_mm) if radial_abs > 1e-12 else 0.0
    radial_branches: List[Tuple[str, float, float]] = [("forward", base_primary, radial_abs)]
    if radial_abs > 1e-12:
        radial_branches.append(
            ("backward", wrap_to_pi(base_primary + math.pi), -radial_abs)
        )

    candidates: List[IKCandidate] = []
    seen = set()
    for radial_name, base, rho in radial_branches:
        rho_w = rho - geometry.tool_mm * math.cos(pitch + geometry.tool_offset_rad)
        z_w = z_mm + geometry.tool_mm * math.sin(pitch + geometry.tool_offset_rad)
        denominator = 2.0 * geometry.l2_mm * geometry.l3_mm
        cosine = (
            rho_w * rho_w
            + z_w * z_w
            - geometry.l2_mm * geometry.l2_mm
            - geometry.l3_mm * geometry.l3_mm
        ) / denominator
        if cosine < -1.0 - 1e-10 or cosine > 1.0 + 1e-10:
            continue
        cosine = max(-1.0, min(1.0, cosine))
        elbow_abs = math.acos(cosine)
        for elbow_name, relative_elbow in (("positive", elbow_abs), ("negative", -elbow_abs)):
            phi2 = math.atan2(rho_w, z_w) - math.atan2(
                geometry.l3_mm * math.sin(relative_elbow),
                geometry.l2_mm + geometry.l3_mm * math.cos(relative_elbow),
            )
            shoulder = wrap_to_pi(phi2 - geometry.d2_rad)
            elbow = wrap_to_pi(relative_elbow + geometry.d2_rad - geometry.d3_rad)
            wrist = wrap_to_pi(pitch + math.pi / 2.0 - shoulder - elbow)
            candidate_q = JointVector(
                base=wrap_to_pi(base),
                shoulder=shoulder,
                elbow=elbow,
                wrist=wrist,
                roll=wrap_to_pi(roll),
            )
            key = tuple(round(v, 12) for v in candidate_q.as_array())
            if key in seen:
                continue
            seen.add(key)

            residual = pose_error(fk(candidate_q, geometry=geometry), target)
            residual_tuple = tuple(map(float, residual))
            if (
                np.max(np.abs(residual[:3])) > residual_tolerance_mm
                or np.max(np.abs(residual[3:])) > residual_tolerance_rad
            ):
                continue
            hard_ok = within_limits(candidate_q, "hard")
            margins = joint_limit_margins(candidate_q, "hard")
            candidates.append(
                IKCandidate(
                    q=candidate_q,
                    radial_branch=radial_name,
                    elbow_branch=elbow_name,
                    fk_residual=residual_tuple,
                    hard_ok=hard_ok,
                    soft_ok=within_limits(candidate_q, "soft"),
                    min_hard_margin_rad=float(min(margins.values())),
                    jacobian_condition=jacobian_condition(candidate_q),
                )
            )

    selected = select_ik_solution(candidates, seed_q)
    if selected is None:
        if not candidates:
            reason = "unreachable_geometry"
        elif not any(candidate.hard_ok for candidate in candidates):
            reason = "all_solutions_outside_hard_limits"
        else:
            reason = "all_hard_valid_solutions_singular"
        return IKResult(IK_NO_SOLUTION, target, tuple(candidates), None, reason)
    return IKResult("OK", target, tuple(candidates), selected, "selected_minimum_cost_solution")
