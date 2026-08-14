"""One-shot plan: nearest demo prior + local Jacobian residual."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import JOINT_KEYS, ApproachPlan, BerryLock
from .joints import clamp_joint, near_hard_limit

J_INPUTS = ("d_shoulder", "d_elbow", "d_base")
FALLBACK_PX_PER_BASE_RAD = 900.0
FALLBACK_CENTER_PX = 320.0
FALLBACK_BASE_CENTER_GAIN = 1.35
FALLBACK_BASE_CENTER_GAIN_RIGHT = 1.9


def _as_float_dict(d: Dict[str, Any], keys: Tuple[str, ...] = JOINT_KEYS) -> Dict[str, float]:
    return {k: float(d.get(k, 0.0)) for k in keys if k in d}


def delta_between(q0: Dict[str, Any], q1: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(q1.get(k, 0.0)) - float(q0.get(k, 0.0)) for k in JOINT_KEYS}


def apply_delta(q0: Dict[str, Any], dq: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(q0.get(k, 0.0)) for k in JOINT_KEYS}
    for k in JOINT_KEYS:
        out[k] = clamp_joint(k, float(q0.get(k, 0.0)) + float(dq.get(k, 0.0)))
    return out


def demo_distance(
    demo: Dict[str, Any],
    q_start: Dict[str, float],
    berry: Dict[str, Any],
) -> float:
    dq = demo.get("q_start") or {}
    db = demo.get("berry_start") or {}
    joint_cost = sum((float(q_start.get(k, 0.0)) - float(dq.get(k, 0.0))) ** 2 for k in JOINT_KEYS)
    px_cost = ((float(berry.get("px", 320.0)) - float(db.get("px", 320.0))) / 180.0) ** 2
    py_cost = ((float(berry.get("py", 240.0)) - float(db.get("py", 240.0))) / 160.0) ** 2
    depth_cost = ((float(berry.get("depth_m", 0.35)) - float(db.get("depth_m", 0.35))) / 0.20) ** 2
    return joint_cost + px_cost + py_cost + depth_cost


def nearest_demo(
    learned: Dict[str, Any],
    q_start: Dict[str, float],
    berry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    demos = [d for d in learned.get("demos") or [] if d.get("success")]
    if not demos:
        return None
    return min(demos, key=lambda d: demo_distance(d, q_start, berry))


def load_local_jacobian(learned: Dict[str, Any]) -> Optional[np.ndarray]:
    block = learned.get("local_jacobian") or {}
    matrix = block.get("matrix") if isinstance(block, dict) else None
    if not matrix:
        return None
    if block.get("quality_ok") is not True:
        return None
    arr = np.array(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    return arr


def solve_delta_with_prior(
    error: np.ndarray,
    prior: np.ndarray,
    jacobian: Optional[np.ndarray],
    *,
    prior_weight: float = 1.0,
    pixel_norm: float = 160.0,
    py_norm: float = 140.0,
    depth_norm: float = 0.08,
    correction_clip_rad: float = 0.35,
) -> np.ndarray:
    if jacobian is None:
        return prior
    w = np.diag([1.0 / pixel_norm, 1.0 / py_norm, 1.0 / depth_norm])
    a = np.vstack([w @ jacobian, np.sqrt(prior_weight) * np.eye(3)])
    b = np.concatenate([w @ error, np.sqrt(prior_weight) * prior])
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return prior + np.clip(sol - prior, -correction_clip_rad, correction_clip_rad)


def predict_berry(
    berry: Dict[str, Any],
    dq_plan: np.ndarray,
    jacobian: Optional[np.ndarray],
) -> Dict[str, float]:
    px = float(berry.get("px", 320.0))
    py = float(berry.get("py", 240.0))
    depth = float(berry.get("depth_m", 0.35))
    if jacobian is not None:
        pred = np.array([px, py, depth], dtype=np.float64) + jacobian @ dq_plan
        px, py, depth = float(pred[0]), float(pred[1]), float(pred[2])
    return {"px": px, "py": py, "depth_m": depth}


def plan_one_shot(
    q_start: Dict[str, Any],
    berry_lock: Dict[str, Any],
    learned: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> ApproachPlan:
    one = (cfg or {}).get("one_shot") or {}
    standoff_m = float(one.get("standoff_m", 0.12))
    prior_weight = float(one.get("prior_weight", 1.4))
    q0 = _as_float_dict(q_start)
    demo = nearest_demo(learned, q0, berry_lock)
    demo_start_px: Optional[float] = None
    if demo:
        dq_demo = np.array(
            [
                float(demo.get("delta_q", {}).get("shoulder", 0.0)),
                float(demo.get("delta_q", {}).get("elbow", 0.0)),
                float(demo.get("delta_q", {}).get("base", 0.0)),
            ],
            dtype=np.float64,
        )
        target = demo.get("berry_success") or {}
        demo_start = demo.get("berry_start") or {}
        if demo_start.get("px") is not None:
            demo_start_px = float(demo_start.get("px"))
        target_px = float(target.get("px", berry_lock.get("px", 320.0)))
        target_py = float(target.get("py", berry_lock.get("py", 240.0)))
        target_depth = float(target.get("depth_m", standoff_m))
        reason = f"nearest_demo:{demo.get('source', 'unknown')}"
    else:
        dq_demo = np.zeros(3, dtype=np.float64)
        target_px = 320.0
        target_py = 240.0
        target_depth = standoff_m
        reason = "no_demo_center_standoff"

    error = np.array(
        [
            target_px - float(berry_lock.get("px", 320.0)),
            target_py - float(berry_lock.get("py", 240.0)),
            target_depth - float(berry_lock.get("depth_m", standoff_m)),
        ],
        dtype=np.float64,
    )
    jacobian = load_local_jacobian(learned)
    dq_s, dq_e, dq_b = solve_delta_with_prior(
        error,
        dq_demo,
        jacobian,
        prior_weight=prior_weight,
        pixel_norm=float(one.get("pixel_norm", 160.0)),
        py_norm=float(one.get("py_norm", 140.0)),
        depth_norm=float(one.get("depth_norm", 0.08)),
        correction_clip_rad=float(one.get("correction_clip_rad", 0.35)),
    )
    adjustments: List[str] = []
    if jacobian is None and demo_start_px is not None:
        current_px = float(berry_lock.get("px", demo_start_px))
        px_per_base = float(one.get("fallback_px_per_base_rad", FALLBACK_PX_PER_BASE_RAD))
        center_px = float(one.get("fallback_center_px", FALLBACK_CENTER_PX))
        default_gain = FALLBACK_BASE_CENTER_GAIN_RIGHT if current_px > center_px else FALLBACK_BASE_CENTER_GAIN
        centered_dq_b = (center_px - current_px) / max(1.0, abs(px_per_base))
        centered_dq_b *= default_gain
        dq_b = float(np.clip(centered_dq_b, -0.35, 0.35))
        adjustments.append("fallback_base_center")
    delta_q = {"base": float(dq_b), "shoulder": float(dq_s), "elbow": float(dq_e)}
    q_target = apply_delta(q0, delta_q)
    dq_plan = np.array([delta_q["shoulder"], delta_q["elbow"], delta_q["base"]], dtype=np.float64)
    prediction = predict_berry(berry_lock, dq_plan, jacobian)
    warnings = [f"{name}_near_hard_limit" for name in JOINT_KEYS if near_hard_limit(name, q_target[name])]
    lock = BerryLock(
        px=float(berry_lock.get("px", 0.0)),
        py=float(berry_lock.get("py", 0.0)),
        depth_m=float(berry_lock.get("depth_m", 0.0)),
        conf=float(berry_lock.get("conf", 0.0)),
    )
    return ApproachPlan(
        q_start=q0,
        q_target=q_target,
        delta_q=delta_q,
        berry_lock=lock,
        prediction=prediction,
        target_image={"px": target_px, "py": target_py, "depth_m": target_depth},
        reason=reason + (" + local_jacobian_projection" if jacobian is not None else " + prior_only"),
        confidence=0.85 if demo else 0.35,
        warnings=warnings,
        adjustments=adjustments,
    )
