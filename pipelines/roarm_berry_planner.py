"""Open-loop RoArm berry planner and learned-demo schema."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipelines.roarm_joint_limits import HARD_JOINT_LIMITS, SOFT_JOINT_LIMITS, clamp_joint, near_hard_limit
from pipelines.roarm_kinematics import JointState

SCHEMA = "roarm_berry_learned_v2"
JOINT_KEYS = ("base", "shoulder", "elbow")
ALL_JOINT_KEYS = ("base", "shoulder", "elbow", "wrist", "roll", "hand")
J_INPUTS = ("d_shoulder", "d_elbow", "d_base")
J_OUTPUTS = ("dpx", "dpy", "ddepth_m")
FALLBACK_PX_PER_BASE_RAD = 900.0
FALLBACK_PX_PER_SHOULDER_RAD = 13.0
FALLBACK_PX_PER_ELBOW_RAD = -44.0
FALLBACK_CENTER_PX = 320.0
FALLBACK_BASE_CENTER_GAIN = 0.55


def _as_float_dict(d: Dict[str, Any], keys: Tuple[str, ...] = ALL_JOINT_KEYS) -> Dict[str, float]:
    return {k: float(d.get(k, 0.0)) for k in keys if k in d}


def q_to_dict(q: JointState) -> Dict[str, float]:
    return q.as_dict()


def q_from_dict(d: Dict[str, Any], *, hand_default: float = 1.08) -> JointState:
    return JointState(
        base=float(d.get("base", 0.0)),
        shoulder=float(d.get("shoulder", 0.0)),
        elbow=float(d.get("elbow", 1.57)),
        wrist=float(d.get("wrist", 0.0)),
        roll=float(d.get("roll", 0.0)),
        hand=float(d.get("hand", hand_default)),
    )


def delta_between(q0: Dict[str, Any], q1: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(q1.get(k, 0.0)) - float(q0.get(k, 0.0)) for k in JOINT_KEYS}


def apply_delta(q0: Dict[str, Any], dq: Dict[str, float], *, hand: float = 1.08) -> Dict[str, float]:
    out = dict(q0)
    for k in JOINT_KEYS:
        out[k] = clamp_joint(k, float(q0.get(k, 0.0)) + float(dq.get(k, 0.0)))
    out["wrist"] = float(out.get("wrist", 0.0))
    out["roll"] = float(out.get("roll", 0.0))
    out["hand"] = hand
    return {k: float(out.get(k, 0.0)) for k in ALL_JOINT_KEYS}


def default_learned(label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "schema": SCHEMA,
        "created_at": time.time(),
        "updated_at": time.time(),
        "joint_limits": {
            "hard": {k: list(v) for k, v in HARD_JOINT_LIMITS.items()},
            "soft": {k: list(v) for k, v in SOFT_JOINT_LIMITS.items()},
        },
        "demos": [],
        "local_jacobian": None,
        "episodes": [],
    }


def demo_from_locks(
    label: str,
    start_lock: Dict[str, Any],
    success_lock: Dict[str, Any],
    *,
    source: str = "operator_manual",
) -> Dict[str, Any]:
    q_start = _as_float_dict(start_lock.get("joints") or {})
    q_success = _as_float_dict(success_lock.get("joints") or {})
    if not q_start or not q_success:
        raise ValueError("start and success locks must contain joints")
    berry_start = dict(start_lock.get("berry") or {})
    berry_success = dict(success_lock.get("berry") or {})
    if not berry_start or not berry_success:
        raise ValueError("start and success locks must contain detected berry")
    return {
        "label": label,
        "source": source,
        "created_at": time.time(),
        "q_start": q_start,
        "berry_start": berry_start,
        "q_success": q_success,
        "berry_success": berry_success,
        "delta_q": delta_between(q_start, q_success),
        "success": True,
    }


def migrate_v1(label: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    if doc.get("schema") == SCHEMA:
        return doc
    learned = default_learned(label or str(doc.get("label", "")))
    q_start = doc.get("start_joints") or {}
    berry_start = doc.get("lock_berry") or {}
    q_success = doc.get("success_joints") or {}
    if not q_success and q_start and doc.get("delta_from_start"):
        q_success = apply_delta(q_start, doc["delta_from_start"])
    berry_success = doc.get("berry_after") or {}
    if q_start and berry_start and q_success and berry_success:
        demo = {
            "label": learned["label"],
            "source": "legacy_v1",
            "created_at": float(doc.get("updated_at", time.time())) if isinstance(doc.get("updated_at"), (int, float)) else time.time(),
            "q_start": _as_float_dict(q_start),
            "berry_start": dict(berry_start),
            "q_success": _as_float_dict(q_success),
            "berry_success": dict(berry_success),
            "delta_q": delta_between(q_start, q_success),
            "success": True,
            "legacy_delta_from_start": doc.get("delta_from_start"),
        }
        learned["demos"].append(demo)
        learned["joint_limits"]["elbow_min_observed"] = float(q_success.get("elbow", HARD_JOINT_LIMITS["elbow"][0]))
    return learned


def load_learned(path: Path, *, label: str = "") -> Dict[str, Any]:
    if not path.is_file():
        return default_learned(label or path.stem.replace("_learned", ""))
    doc = json.loads(path.read_text(encoding="utf-8"))
    return migrate_v1(label or str(doc.get("label", "")), doc)


def write_learned(path: Path, doc: Dict[str, Any]) -> None:
    doc["schema"] = SCHEMA
    doc["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def upsert_demo(doc: Dict[str, Any], demo: Dict[str, Any]) -> Dict[str, Any]:
    demos: List[Dict[str, Any]] = list(doc.get("demos") or [])
    demos.append(demo)
    doc["demos"] = demos
    elbow_values = [
        float(d.get("q_success", {}).get("elbow"))
        for d in demos
        if d.get("q_success", {}).get("elbow") is not None
    ]
    if elbow_values:
        doc.setdefault("joint_limits", {})["elbow_min_observed"] = min(elbow_values)
    return doc


def _demo_distance(demo: Dict[str, Any], q_start: Dict[str, float], berry: Dict[str, Any]) -> float:
    dq = demo.get("q_start") or {}
    db = demo.get("berry_start") or {}
    joint_cost = sum((float(q_start.get(k, 0.0)) - float(dq.get(k, 0.0))) ** 2 for k in JOINT_KEYS)
    px_cost = ((float(berry.get("px", 320.0)) - float(db.get("px", 320.0))) / 180.0) ** 2
    py_cost = ((float(berry.get("py", 240.0)) - float(db.get("py", 240.0))) / 160.0) ** 2
    depth_cost = ((float(berry.get("depth_m", 0.35)) - float(db.get("depth_m", 0.35))) / 0.20) ** 2
    return joint_cost + px_cost + py_cost + depth_cost


def nearest_demo(doc: Dict[str, Any], q_start: Dict[str, float], berry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    demos = [d for d in doc.get("demos") or [] if d.get("success")]
    if not demos:
        return None
    return min(demos, key=lambda d: _demo_distance(d, q_start, berry))


def load_local_jacobian(doc: Dict[str, Any], calib_path: Optional[Path] = None) -> Optional[np.ndarray]:
    block = doc.get("local_jacobian") or {}
    matrix = block.get("matrix") if isinstance(block, dict) else None
    if matrix:
        if block.get("quality_ok") is not True:
            return None
        arr = np.array(matrix, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr
    if calib_path and calib_path.is_file():
        cal = json.loads(calib_path.read_text(encoding="utf-8"))
        diag = cal.get("stage1", {}).get("diagnostic_3joint") or {}
        r2_values = [
            float((diag.get(out) or {}).get("r2", 0.0))
            for out in ("dpx", "dpy", "ddepth_m")
        ]
        if min(r2_values or [0.0]) < 0.25:
            return None
        rows = []
        for out in ("dpx", "dpy", "ddepth_m"):
            coef = (diag.get(out) or {}).get("coef")
            if coef and len(coef) == 3:
                rows.append([float(coef[0]), float(coef[1]), float(coef[2])])
        if len(rows) == 3:
            return np.array(rows, dtype=np.float64)
    return None


def solve_delta_with_prior(
    error: np.ndarray,
    prior: np.ndarray,
    jacobian: Optional[np.ndarray],
    *,
    prior_weight: float = 1.0,
) -> np.ndarray:
    if jacobian is None:
        return prior
    # Normalize pixels/depth so one unit is roughly one meaningful correction.
    w = np.diag([1.0 / 160.0, 1.0 / 140.0, 1.0 / 0.08])
    a = np.vstack([w @ jacobian, np.sqrt(prior_weight) * np.eye(3)])
    b = np.concatenate([w @ error, np.sqrt(prior_weight) * prior])
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    # Keep local-J correction from overpowering the demonstrated motion.
    return prior + np.clip(sol - prior, -0.35, 0.35)


def predict_berry(berry: Dict[str, Any], dq_plan: np.ndarray, jacobian: Optional[np.ndarray]) -> Dict[str, float]:
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
    calib_path: Optional[Path] = None,
    standoff_m: float = 0.12,
    prior_weight: float = 1.4,
) -> Dict[str, Any]:
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
    jacobian = load_local_jacobian(learned, calib_path)
    dq_s, dq_e, dq_b = solve_delta_with_prior(error, dq_demo, jacobian, prior_weight=prior_weight)
    adjustments: List[str] = []
    if jacobian is None and demo_start_px is not None:
        current_px = float(berry_lock.get("px", demo_start_px))
        px_per_base = float(learned.get("fallback_px_per_base_rad") or FALLBACK_PX_PER_BASE_RAD)
        center_px = float(learned.get("fallback_center_px") or FALLBACK_CENTER_PX)
        center_gain = float(learned.get("fallback_base_center_gain") or FALLBACK_BASE_CENTER_GAIN)
        px_per_shoulder = float(learned.get("fallback_px_per_shoulder_rad") or FALLBACK_PX_PER_SHOULDER_RAD)
        px_per_elbow = float(learned.get("fallback_px_per_elbow_rad") or FALLBACK_PX_PER_ELBOW_RAD)
        dpx_without_base = px_per_shoulder * float(dq_s) + px_per_elbow * float(dq_e)
        centered_dq_b = (center_px - current_px - dpx_without_base) / max(1.0, abs(px_per_base))
        centered_dq_b *= center_gain
        centered_dq_b = float(np.clip(centered_dq_b, -0.35, 0.35))
        dq_b = centered_dq_b
        adjustments.append(f"fallback_base_center_px={center_px:.0f}")
        adjustments.append(f"fallback_base_center_gain={center_gain:.2f}")
        adjustments.append(f"fallback_base_dq={centered_dq_b:+.3f}")
    delta_q = {"base": float(dq_b), "shoulder": float(dq_s), "elbow": float(dq_e)}
    q_target = apply_delta(q0, delta_q)
    dq_plan = np.array([delta_q["shoulder"], delta_q["elbow"], delta_q["base"]], dtype=np.float64)
    prediction = predict_berry(berry_lock, dq_plan, jacobian)
    warnings = [
        f"{name}_near_hard_limit"
        for name in JOINT_KEYS
        if near_hard_limit(name, q_target[name])
    ]
    return {
        "schema": "roarm_berry_plan_v1",
        "q_start": q0,
        "berry_lock": dict(berry_lock),
        "q_target": q_target,
        "delta_q": delta_q,
        "prediction": prediction,
        "confidence": 0.85 if demo else 0.35,
        "reason": reason + (" + local_jacobian_projection" if jacobian is not None else " + prior_only"),
        "target_image": {"px": target_px, "py": target_py, "depth_m": target_depth},
        "warnings": warnings,
        "adjustments": adjustments,
    }


def verify_success(
    berry_before: Dict[str, Any],
    berry_after: Optional[Dict[str, Any]],
    q_start: Dict[str, Any],
    q_target: Dict[str, Any],
    q_end: Dict[str, Any],
    *,
    expected_px: Optional[float] = None,
    expected_py: Optional[float] = None,
    min_conf: float = 0.5,
    depth_min_m: float = 0.10,
    depth_max_m: float = 0.17,
) -> Tuple[bool, Dict[str, Any]]:
    reasons: List[str] = []
    if not berry_after:
        return False, {"reasons": ["no_berry_after"]}
    conf = float(berry_after.get("conf", 0.0))
    px0, py0, d0 = float(berry_before.get("px", 0.0)), float(berry_before.get("py", 0.0)), float(berry_before.get("depth_m", 9.0))
    px1, py1, d1 = float(berry_after.get("px", 0.0)), float(berry_after.get("py", 0.0)), float(berry_after.get("depth_m", 9.0))
    in_frame = 45.0 <= px1 <= 600.0 and 60.0 <= py1 <= 390.0
    depth_ok = depth_min_m <= d1 <= depth_max_m
    progress_ok = d1 < d0 - min(0.08, max(0.025, (d0 - 0.16) * 0.35))
    py_ok = 140.0 <= py1 <= 340.0
    px_ok = True if expected_px is None else abs(px1 - expected_px) <= 95.0
    py_target_ok = True if expected_py is None else abs(py1 - expected_py) <= 95.0
    image_changed = abs(px1 - px0) + abs(py1 - py0) >= 24.0
    depth_changed = d1 < d0 - 0.06
    joint_motion = max(abs(float(q_target.get(k, 0.0)) - float(q_start.get(k, 0.0))) for k in JOINT_KEYS)
    joint_err = max(abs(float(q_end.get(k, 0.0)) - float(q_target.get(k, 0.0))) for k in JOINT_KEYS)
    if conf < min_conf:
        reasons.append("low_conf")
    if not in_frame:
        reasons.append("out_of_frame")
    if not depth_ok:
        reasons.append("depth_not_in_standoff")
    if not progress_ok:
        reasons.append("insufficient_depth_progress")
    if not py_ok:
        reasons.append("py_out_of_work_band")
    if not px_ok or not py_target_ok:
        reasons.append("far_from_expected_image_target")
    if joint_motion > 0.45 and not image_changed and not depth_changed:
        reasons.append("suspicious_no_image_motion")
    if joint_err > 0.12:
        reasons.append("joint_target_not_reached")
    return not reasons, {
        "reasons": reasons,
        "conf": conf,
        "in_frame": in_frame,
        "depth_ok": depth_ok,
        "progress_ok": progress_ok,
        "py_ok": py_ok,
        "image_changed": image_changed,
        "depth_changed": depth_changed,
        "joint_err_rad": round(joint_err, 4),
    }
