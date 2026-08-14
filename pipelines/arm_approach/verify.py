"""Post-move standoff verification (vision/geometry only)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import JOINT_KEYS, VerifyReport


def verify_success(
    berry_before: Dict[str, Any],
    berry_after: Optional[Dict[str, Any]],
    q_start: Dict[str, Any],
    q_target: Dict[str, Any],
    q_end: Dict[str, Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    expected_px: Optional[float] = None,
    expected_py: Optional[float] = None,
) -> Tuple[bool, VerifyReport]:
    vcfg = (cfg or {}).get("verify") or {}
    min_conf = float(vcfg.get("min_conf", 0.70))
    depth_min_m = float(vcfg.get("depth_min_m", 0.10))
    depth_max_m = float(vcfg.get("depth_max_m", 0.17))
    fx0, fx1 = vcfg.get("frame_x") or [45.0, 600.0]
    fy0, fy1 = vcfg.get("frame_y") or [60.0, 390.0]
    py0, py1 = vcfg.get("work_py") or [140.0, 340.0]
    expected_px_tol = float(vcfg.get("expected_px_tol", 95.0))
    min_image_shift = float(vcfg.get("min_image_shift_px", 24.0))
    min_depth_drop = float(vcfg.get("min_depth_drop_m", 0.06))
    joint_motion_thr = float(vcfg.get("joint_motion_suspicious_rad", 0.45))
    joint_err_max = float(vcfg.get("joint_err_max_rad", 0.12))

    reasons = []
    if not berry_after:
        report = VerifyReport(ok=False, reasons=["no_berry_after"])
        return False, report
    conf = float(berry_after.get("conf", 0.0))
    px0 = float(berry_before.get("px", 0.0))
    py_b = float(berry_before.get("py", 0.0))
    d0 = float(berry_before.get("depth_m", 9.0))
    px1 = float(berry_after.get("px", 0.0))
    py_a = float(berry_after.get("py", 0.0))
    d1 = float(berry_after.get("depth_m", 9.0))
    in_frame = float(fx0) <= px1 <= float(fx1) and float(fy0) <= py_a <= float(fy1)
    depth_ok = depth_min_m <= d1 <= depth_max_m
    progress_ok = d1 < d0 - min(0.08, max(0.025, (d0 - 0.16) * 0.35))
    py_ok = float(py0) <= py_a <= float(py1)
    px_ok = True if expected_px is None else abs(px1 - expected_px) <= expected_px_tol
    py_target_ok = True if expected_py is None else abs(py_a - expected_py) <= expected_px_tol
    image_changed = abs(px1 - px0) + abs(py_a - py_b) >= min_image_shift
    depth_changed = d1 < d0 - min_depth_drop
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
    if joint_motion > joint_motion_thr and not image_changed and not depth_changed:
        reasons.append("suspicious_no_image_motion")
    if joint_err > joint_err_max:
        reasons.append("joint_target_not_reached")
    report = VerifyReport(
        ok=not reasons,
        reasons=reasons,
        conf=conf,
        in_frame=in_frame,
        depth_ok=depth_ok,
        progress_ok=progress_ok,
    )
    return report.ok, report
