"""Peduncle FOV framing: keep berry in frame with headroom for calyx/stem.

CENTER (stage1 Jacobian) for X/Y only. REACH only when depth is truly outside
the standoff gate. If approach already left berry in FOV with headroom and
depth in gate → SKIP (no motion).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipelines.roarm_joint_limits import clamp_joint
from pipelines.roarm_kinematics import JointState, KinematicsConfig, explicit_reach_delta

REPO = Path(__file__).resolve().parents[2]
CALIB = REPO / "runs/roarm_kinematics/calibration.json"
Point2D = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _status(stage: str, **kw: Any) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    print(f"[STATUS] stage={stage}" + (f" {extra}" if extra else ""), flush=True)


def framing_cfg(v3_cfg: Dict[str, Any]) -> Dict[str, Any]:
    focus = v3_cfg.get("focus_roi") or {}
    fr = dict(v3_cfg.get("framing") or {})
    fr.setdefault("target_nx", 0.50)
    fr.setdefault("target_ny", 0.58)  # below center → headroom above berry
    fr.setdefault("tol_nx", 0.08)
    fr.setdefault("tol_ny", 0.08)
    fr.setdefault("edge_margin_px", 12.0)
    fr.setdefault("headroom_berry_scale", float(focus.get("height_above_scale", 0.85)))
    fr.setdefault("headroom_min_px", 40.0)
    fr.setdefault("side_pad_berry_scale", float(focus.get("width_scale", 1.6)) * 0.15)
    fr.setdefault("max_center_iters", 6)
    fr.setdefault("max_step_rad", 0.10)
    fr.setdefault("center_gain", 0.90)
    fr.setdefault("max_move_px", 160.0)
    fr.setdefault("settle_s", 0.28)
    fr.setdefault("lookup_iters", 3)
    fr.setdefault("depth_aim_m", 0.14)
    fr.setdefault("depth_slack_m", 0.030)
    return fr


def _load_stage1(calib_path: Path = CALIB) -> Dict[str, Any]:
    if not calib_path.is_file():
        return {}
    doc = json.loads(calib_path.read_text(encoding="utf-8"))
    return dict(doc.get("stage1") or {})


def _detect_same_berry(
    bgr: np.ndarray,
    tracker: Any,
    preferred_px: float,
    preferred_py: float,
    *,
    max_dist_px: float = 180.0,
    provider: Any = None,
    allow_global: bool = True,
) -> Optional[Dict[str, Any]]:
    """Re-find berry for framing: prefer real detector bbox, ignore sticky edge holds."""
    from scripts.roarm_strawberry_approach import _get_detector
    from pipelines.roarm_strawberry_target import (
        detect_strawberries_in_frame,
        _bbox_center,
        min_berry_conf,
    )

    h, w = bgr.shape[:2]
    det_model = _get_detector(REPO)
    prev_strict = getattr(tracker, "strict_lock", False)
    prev_last = (getattr(tracker, "last_px", None), getattr(tracker, "last_py", None))
    try:
        tracker.strict_lock = False
        dets = detect_strawberries_in_frame(bgr, det_model, tracker)
    except Exception:
        dets = []
    finally:
        tracker.strict_lock = prev_strict

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for d in dets:
        if float(d.get("conf", 0.0)) < min_berry_conf():
            continue
        cx, cy = _bbox_center(d)
        x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        edge_pen = 0.0
        margin = 20.0
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            edge_pen = 80.0
        dist_pref = float(np.hypot(cx - preferred_px, cy - preferred_py))
        dist_frame = float(np.hypot(cx - 0.5 * w, cy - 0.58 * h))
        score = dist_pref + 0.35 * dist_frame + edge_pen - 40.0 * float(d.get("conf", 0.0))
        scored.append(
            (
                score,
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "px": float(cx),
                    "py": float(cy),
                    "conf": float(d.get("conf", 0.0)),
                    "bbox": (x1, y1, x2, y2),
                    "source": "detector",
                },
            )
        )

    if scored:
        scored.sort(key=lambda t: t[0])
        best = scored[0][1]
        if allow_global or float(np.hypot(best["px"] - preferred_px, best["py"] - preferred_py)) <= max_dist_px:
            depth = _sample_depth(provider, best["px"], best["py"]) if provider is not None else None
            best["depth_m"] = depth
            try:
                tracker.reanchor(float(best["px"]), float(best["py"]))
                tracker.last_conf = float(best["conf"])
            except Exception:
                pass
            return best

    if prev_last[0] is not None:
        try:
            tracker.last_px, tracker.last_py = prev_last
        except Exception:
            pass
    return None


def _exploratory_lookup_error(frame_w: int, frame_h: int, fr: Dict[str, Any]) -> Tuple[float, float]:
    """Assume berry is above FOV; command CENTER toward framing target from top-center."""
    target_px = float(fr["target_nx"]) * frame_w
    target_py = float(fr["target_ny"]) * frame_h
    fake_cx = target_px
    fake_cy = 8.0
    return target_px - fake_cx, target_py - fake_cy


def _sample_depth(provider: Any, px: float, py: float) -> Optional[float]:
    from pipelines.roarm_perception import PerceptionConfig, sample_depth_median

    try:
        pair = provider.read(timeout_s=1.5)
    except Exception:
        return None
    if pair is None or pair.depth_m is None:
        return None
    d, _ = sample_depth_median(
        pair.depth_m, int(round(px)), int(round(py)), radius=5, cfg=PerceptionConfig()
    )
    return float(d) if d is not None else None


def _depth_in_gate(depth: Optional[float], depth_min_m: float, depth_max_m: float) -> bool:
    return depth is not None and depth_min_m <= float(depth) <= depth_max_m


def _resolve_input_depth(
    sampled: Optional[float],
    depth_hint: Optional[float],
    depth_min_m: float,
    depth_max_m: float,
) -> Tuple[Optional[float], str]:
    """Prefer approach-verified depth_hint when it is already in the standoff gate.

    Framing's fresh median can spike (e.g. 0.127→0.194) and falsely trip REACH.
    """
    hint_ok = _depth_in_gate(depth_hint, depth_min_m, depth_max_m)
    sample_ok = _depth_in_gate(sampled, depth_min_m, depth_max_m)
    if hint_ok and sample_ok:
        # Keep fresher sample when both valid.
        return float(sampled), "sample_and_hint_in_gate"
    if hint_ok:
        return float(depth_hint), "approach_hint_in_gate"
    if sample_ok:
        return float(sampled), "sample_in_gate"
    if sampled is not None:
        return float(sampled), "sample_out_of_gate"
    if depth_hint is not None:
        return float(depth_hint), "hint_out_of_gate"
    return None, "no_depth"


def evaluate_peduncle_fov(
    berry: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    fr: Dict[str, Any],
) -> Dict[str, Any]:
    """Check full berry in frame + calyx/stem headroom ROI inside FOV."""
    x1, y1, x2, y2 = berry["bbox"]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    margin = float(fr["edge_margin_px"])
    headroom = max(float(fr["headroom_min_px"]), bh * float(fr["headroom_berry_scale"]))
    side = max(8.0, bw * float(fr["side_pad_berry_scale"]))

    reasons: List[str] = []
    if x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin:
        reasons.append("berry_near_edge")
    stem_top = y1 - headroom
    stem_left = x1 - side
    stem_right = x2 + side
    if stem_top < margin:
        reasons.append("calyx_roi_out_top")
    if stem_left < margin or stem_right > frame_w - margin:
        reasons.append("calyx_roi_out_side")

    target_px = float(fr["target_nx"]) * frame_w
    target_py = float(fr["target_ny"]) * frame_h
    tol_x = float(fr["tol_nx"]) * frame_w
    tol_y = float(fr["tol_ny"]) * frame_h
    cx, cy = float(berry["px"]), float(berry["py"])
    err_x = target_px - cx
    err_y = target_py - cy
    on_target = abs(err_x) <= tol_x and abs(err_y) <= tol_y

    ok = not reasons and on_target
    return {
        "ok": ok,
        "fov_ok": not reasons,
        "on_target": on_target,
        "reasons": reasons,
        "berry_bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "berry_center": [round(cx, 1), round(cy, 1)],
        "target_px": round(target_px, 1),
        "target_py": round(target_py, 1),
        "err_px": round(err_x, 1),
        "err_py": round(err_y, 1),
        "headroom_px": round(headroom, 1),
        "stem_roi": [
            round(stem_left, 1),
            round(stem_top, 1),
            round(stem_right, 1),
            round(y1, 1),
        ],
        "frame_wh": [int(frame_w), int(frame_h)],
    }


def _center_joint_step(
    q: JointState,
    err_x: float,
    err_y: float,
    stage1: Dict[str, Any],
    fr: Dict[str, Any],
) -> Tuple[Optional[JointState], Dict[str, Any]]:
    """One CENTER step via stage1 control Jacobian / Jc_inv (base + pitch)."""
    info: Dict[str, Any] = {"mechanism": "stage1_center"}
    gain = float(fr["center_gain"])
    max_px = float(fr["max_move_px"])
    max_step = float(fr["max_step_rad"])
    want_dpx = float(np.clip(err_x * gain, -max_px, max_px))
    want_dpy = float(np.clip(err_y * gain, -max_px, max_px))

    db = ds = 0.0
    jc = stage1.get("Jc_inv_rows")
    ctrl = stage1.get("control_jacobian_px_per_rad") or {}
    if jc and len(jc) == 2:
        inv = np.array(jc, dtype=np.float64)
        dq = inv @ np.array([want_dpx, want_dpy], dtype=np.float64)
        db, ds = float(dq[0]), float(dq[1])
        info["via"] = "Jc_inv"
    else:
        dpx_base = float((ctrl.get("base") or {}).get("dpx") or 600.0)
        dpy_pitch = float((ctrl.get("pitch") or {}).get("dpy") or -650.0)
        db = want_dpx / max(200.0, abs(dpx_base))
        ds = want_dpy / max(20.0, abs(dpy_pitch))
        if dpy_pitch > 0:
            ds = -ds
        info["via"] = "control_jacobian"

    db = float(np.clip(db, -max_step, max_step))
    ds = float(np.clip(ds, -max_step, max_step))
    info["delta_base"] = round(db, 4)
    info["delta_pitch"] = round(ds, 4)
    info["want_dpx"] = round(want_dpx, 1)
    info["want_dpy"] = round(want_dpy, 1)
    if abs(db) + abs(ds) < 1e-4:
        info["skipped"] = "zero_step"
        return None, info

    q2 = JointState(
        base=clamp_joint("base", float(q.base) + db),
        shoulder=clamp_joint("shoulder", float(q.shoulder) + ds),
        elbow=clamp_joint("elbow", float(q.elbow) - ds),
        wrist=float(q.wrist),
        roll=float(q.roll),
        hand=float(q.hand),
    )
    return q2, info


def _draw_framing_overlay(bgr: np.ndarray, berry: Dict[str, Any], fov: Dict[str, Any]) -> np.ndarray:
    out = bgr.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in berry["bbox"]]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cx, cy = int(round(berry["px"])), int(round(berry["py"]))
    cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
    tx, ty = int(round(fov["target_px"])), int(round(fov["target_py"]))
    cv2.drawMarker(out, (tx, ty), (0, 200, 255), cv2.MARKER_CROSS, 18, 2)
    if fov.get("stem_roi"):
        sx1, sy1, sx2, sy2 = [int(round(v)) for v in fov["stem_roi"]]
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (255, 128, 0), 1)
    label = "FOV_OK" if fov.get("fov_ok") else ",".join(fov.get("reasons") or ["off_target"])
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def _fresh_lock_depth(
    *,
    provider: Any,
    tracker: Any,
    preferred_px: float,
    preferred_py: float,
    fetch_bgr: Callable[[], Optional[np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[float]]:
    bgr = fetch_bgr()
    if bgr is None:
        return None, None, None
    berry = _detect_same_berry(bgr, tracker, preferred_px, preferred_py, provider=provider)
    if berry is None:
        return bgr, None, None
    depth = berry.get("depth_m")
    if depth is None:
        depth = _sample_depth(provider, berry["px"], berry["py"])
        berry["depth_m"] = depth
    return bgr, berry, depth


def run_peduncle_framing(
    *,
    provider: Any,
    tracker: Any,
    execute_rpc: Callable,
    move_t102_smooth: Callable,
    read_q: Callable,
    q_now: Any,
    preferred_px: float,
    preferred_py: float,
    depth_hint: Optional[float],
    v3_cfg: Dict[str, Any],
    log_dir: Path,
    label: str,
    attempt_idx: int,
    depth_max_m: float,
    depth_min_m: float,
    spd: float = 0.08,
    acc: float = 5.0,
    fetch_bgr: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Frame berry for calyx/stem FOV. Returns ok + updated lock; saves before/after frames."""
    fr = framing_cfg(v3_cfg)
    stage1 = _load_stage1()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out: Dict[str, Any] = {
        "ok": False,
        "executed_center": False,
        "executed_micro_reach": False,
        "framing_action": "skip",
        "cfg": {
            "target_nx": fr["target_nx"],
            "target_ny": fr["target_ny"],
            "tol_nx": fr["tol_nx"],
            "tol_ny": fr["tol_ny"],
        },
    }

    def _bgr() -> Optional[np.ndarray]:
        if fetch_bgr is not None:
            return fetch_bgr()
        try:
            from scripts.roarm_strawberry_approach import _fetch_hub_bgr

            bgr, _ = _fetch_hub_bgr("http://127.0.0.1:8080")
            if bgr is not None:
                return bgr
        except Exception:
            pass
        try:
            pair = provider.read(timeout_s=1.5)
            return pair.rgb_bgr if pair is not None else None
        except Exception:
            return None

    bgr0 = _bgr()
    if bgr0 is None:
        out["reason"] = "no_frame"
        _status("PEDUNCLE_FOV_INPUT", ok=0, reason="no_frame")
        _status("FRAMING_ACTION", action="skip", reason="no_frame")
        _status("PEDUNCLE_FOV_FINAL", ok=0, reason="no_frame")
        return out

    h0, w0 = bgr0.shape[:2]
    berry0 = _detect_same_berry(bgr0, tracker, preferred_px, preferred_py, provider=provider)
    q = read_q(execute_rpc) or q_now
    if not hasattr(q, "as_dict"):
        q = JointState(**{k: float(q.get(k, 0)) for k in ("base", "shoulder", "elbow", "wrist", "roll", "hand")})
    q_before = JointState(
        base=float(q.base),
        shoulder=float(q.shoulder),
        elbow=float(q.elbow),
        wrist=float(q.wrist),
        roll=float(q.roll),
        hand=float(q.hand),
    )

    # Berry missing → exploratory CENTER lookup (XY only), then re-evaluate.
    if berry0 is None:
        _status("PEDUNCLE_FOV_INPUT", ok=0, reason="no_berry_in_frame_lookup")
        _status("FRAMING_ACTION", action="center", reason="lookup_no_berry")
        before_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_before.jpg"
        cv2.imwrite(str(before_path), bgr0)
        out["before_path"] = str(before_path)
        out["before"] = {"berry": None, "fov": {"ok": False, "reasons": ["no_berry_in_frame"]}}
        out["framing_action"] = "center"
        for it in range(int(fr.get("lookup_iters", 3))):
            err_x, err_y = _exploratory_lookup_error(w0, h0, fr)
            q_step, step_info = _center_joint_step(q, err_x, err_y, stage1, fr)
            _status(
                "BERRY_RECENTER",
                iter=it,
                mode="lookup",
                via=step_info.get("via"),
                d_b=step_info.get("delta_base"),
                d_pitch=step_info.get("delta_pitch"),
            )
            if q_step is None:
                break
            move_t102_smooth(execute_rpc, q_step, spd=min(spd, 0.10), acc=min(acc, 6.0))
            time.sleep(float(fr["settle_s"]))
            q = read_q(execute_rpc) or q_step
            out["executed_center"] = True
            out.setdefault("center_steps", []).append({**step_info, "mode": "lookup"})
            bgr0, berry0, depth_chk = _fresh_lock_depth(
                provider=provider,
                tracker=tracker,
                preferred_px=preferred_px,
                preferred_py=preferred_py,
                fetch_bgr=_bgr,
            )
            if berry0 is None:
                continue
            if not _depth_in_gate(depth_chk, depth_min_m, depth_max_m):
                move_t102_smooth(execute_rpc, q_before, spd=min(spd, 0.10), acc=min(acc, 6.0))
                time.sleep(float(fr["settle_s"]))
                out["ok"] = False
                out["reason"] = f"depth_out_of_gate_after_center:{depth_chk}"
                out["rollback"] = True
                _status("PEDUNCLE_FOV_FINAL", ok=0, reason=out["reason"], rollback=1)
                return out
            break
        if berry0 is None:
            out["reason"] = "berry_not_in_fov"
            _status("PEDUNCLE_FOV_FINAL", ok=0, reason="berry_not_in_fov")
            after_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_after.jpg"
            if bgr0 is not None:
                cv2.imwrite(str(after_path), bgr0)
                out["after_path"] = str(after_path)
            return out

    sampled0 = berry0.get("depth_m")
    if sampled0 is None:
        sampled0 = _sample_depth(provider, berry0["px"], berry0["py"])
    depth0, depth_src = _resolve_input_depth(sampled0, depth_hint, depth_min_m, depth_max_m)
    berry0["depth_m"] = depth0
    h, w = bgr0.shape[:2]
    fov0 = evaluate_peduncle_fov(berry0, w, h, fr)
    out["before"] = {
        "berry": {k: berry0[k] for k in ("px", "py", "conf", "depth_m", "bbox") if k in berry0},
        "fov": fov0,
        "source": berry0.get("source"),
        "depth_source": depth_src,
        "depth_sample": sampled0,
        "depth_hint": depth_hint,
    }
    overlay0 = _draw_framing_overlay(bgr0, berry0, fov0)
    before_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_before.jpg"
    cv2.imwrite(str(before_path), overlay0)
    out["before_path"] = str(before_path)

    _status(
        "PEDUNCLE_FOV_INPUT",
        bbox=",".join(str(int(round(v))) for v in berry0["bbox"]),
        headroom=round(float(fov0["headroom_px"]), 1),
        depth=round(float(depth0), 3) if depth0 is not None else None,
        depth_src=depth_src,
        fov_ok=int(bool(fov0["fov_ok"])),
        reasons=",".join(fov0["reasons"]) or "none",
        cx=f"{berry0['px']:.0f}",
        cy=f"{berry0['py']:.0f}",
    )

    berry = berry0
    fov = fov0
    depth = depth0
    preferred_px, preferred_py = float(berry["px"]), float(berry["py"])
    depth_ok0 = _depth_in_gate(depth0, depth_min_m, depth_max_m)

    # --- Decision: SKIP if FOV+headroom OK and depth already in gate ---
    if fov0["fov_ok"] and depth_ok0:
        out["ok"] = True
        out["reason"] = "skip_fov_depth_ok" if not fov0.get("on_target") else "skip_on_target"
        out["framing_action"] = "skip"
        out["berry"] = out["before"]["berry"]
        out["bbox"] = list(berry0["bbox"])
        out["after"] = out["before"]
        out["after_path"] = str(before_path)
        _status("FRAMING_ACTION", action="skip", reason=out["reason"])
        _status(
            "PEDUNCLE_FOV_FINAL",
            ok=1,
            cx=f"{berry0['px']:.0f}",
            cy=f"{berry0['py']:.0f}",
            depth=round(float(depth0), 3),
            action="skip",
            reasons=out["reason"],
        )
        return out

    # --- REACH only when depth is truly outside the gate (never for X/Y) ---
    if depth is not None and not depth_ok0:
        out["framing_action"] = "reach"
        cam_err = float(depth) - float(fr["depth_aim_m"])
        cfg = KinematicsConfig(max_step_rad=0.08)
        ds, de = explicit_reach_delta(cam_err, cfg)
        q_m = JointState(
            base=float(q.base),
            shoulder=clamp_joint("shoulder", float(q.shoulder) + float(ds)),
            elbow=clamp_joint("elbow", float(q.elbow) + float(de)),
            wrist=float(q.wrist),
            roll=float(q.roll),
            hand=float(q.hand),
        )
        _status(
            "FRAMING_ACTION",
            action="reach",
            reason=f"depth_out_of_gate:{round(float(depth), 3)}",
            d_s=round(ds, 4),
            d_e=round(de, 4),
        )
        move_t102_smooth(execute_rpc, q_m, spd=min(spd, 0.10), acc=min(acc, 6.0))
        time.sleep(float(fr["settle_s"]))
        q = read_q(execute_rpc) or q_m
        out["executed_micro_reach"] = True
        bgr, berry_n, depth_n = _fresh_lock_depth(
            provider=provider,
            tracker=tracker,
            preferred_px=preferred_px,
            preferred_py=preferred_py,
            fetch_bgr=_bgr,
        )
        if berry_n is None:
            move_t102_smooth(execute_rpc, q_before, spd=min(spd, 0.10), acc=min(acc, 6.0))
            time.sleep(float(fr["settle_s"]))
            out["ok"] = False
            out["reason"] = "berry_lost_after_reach"
            out["rollback"] = True
            _status("PEDUNCLE_FOV_FINAL", ok=0, reason=out["reason"], rollback=1)
            return out
        berry = berry_n
        depth = depth_n
        preferred_px, preferred_py = float(berry["px"]), float(berry["py"])
        if bgr is not None:
            h, w = bgr.shape[:2]
            fov = evaluate_peduncle_fov(berry, w, h, fr)
            bgr0 = bgr
        if not _depth_in_gate(depth, depth_min_m, depth_max_m):
            move_t102_smooth(execute_rpc, q_before, spd=min(spd, 0.10), acc=min(acc, 6.0))
            time.sleep(float(fr["settle_s"]))
            out["ok"] = False
            out["reason"] = f"depth_out_of_gate_after_reach:{depth}"
            out["rollback"] = True
            _status("PEDUNCLE_FOV_FINAL", ok=0, reason=out["reason"], rollback=1, depth=depth)
            return out
        # After REACH, if FOV still bad → CENTER below; else finish.
        if fov.get("fov_ok"):
            out["ok"] = True
            out["reason"] = "ok_after_reach"
            out["berry"] = {k: berry[k] for k in ("px", "py", "conf", "depth_m", "bbox")}
            out["bbox"] = list(berry["bbox"])
            out["after"] = {"berry": out["berry"], "fov": fov}
            overlay_f = _draw_framing_overlay(bgr0, berry, fov)
            after_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_after.jpg"
            cv2.imwrite(str(after_path), overlay_f)
            out["after_path"] = str(after_path)
            _status(
                "PEDUNCLE_FOV_FINAL",
                ok=1,
                cx=f"{berry['px']:.0f}",
                cy=f"{berry['py']:.0f}",
                depth=round(float(depth), 3) if depth is not None else None,
                action="reach",
                reasons=out["reason"],
            )
            return out

    # --- CENTER for X/Y / headroom only (depth already in gate or after failed soft FOV) ---
    need_center = not fov.get("fov_ok", False)
    if need_center:
        out["framing_action"] = "center"
        _status(
            "FRAMING_ACTION",
            action="center",
            reason=",".join(fov.get("reasons") or ["fov_fail"]),
            err_x=fov.get("err_px"),
            err_y=fov.get("err_py"),
        )
        for it in range(int(fr["max_center_iters"])):
            if fov.get("fov_ok"):
                break
            q_step, step_info = _center_joint_step(
                q, float(fov["err_px"]), float(fov["err_py"]), stage1, fr
            )
            _status(
                "BERRY_RECENTER",
                iter=it,
                mode="center",
                via=step_info.get("via"),
                d_b=step_info.get("delta_base"),
                d_pitch=step_info.get("delta_pitch"),
                err_x=fov["err_px"],
                err_y=fov["err_py"],
            )
            if q_step is None:
                out["reason"] = step_info.get("skipped") or "center_zero"
                break
            move_t102_smooth(execute_rpc, q_step, spd=min(spd, 0.10), acc=min(acc, 6.0))
            time.sleep(float(fr["settle_s"]))
            q = read_q(execute_rpc) or q_step
            out["executed_center"] = True
            out.setdefault("center_steps", []).append(step_info)

            bgr, berry_n, depth_n = _fresh_lock_depth(
                provider=provider,
                tracker=tracker,
                preferred_px=preferred_px,
                preferred_py=preferred_py,
                fetch_bgr=_bgr,
            )
            if berry_n is None or bgr is None:
                move_t102_smooth(execute_rpc, q_before, spd=min(spd, 0.10), acc=min(acc, 6.0))
                time.sleep(float(fr["settle_s"]))
                out["ok"] = False
                out["reason"] = "berry_lost_after_center"
                out["rollback"] = True
                _status("PEDUNCLE_FOV_FINAL", ok=0, reason=out["reason"], rollback=1)
                return out
            if not _depth_in_gate(depth_n, depth_min_m, depth_max_m):
                move_t102_smooth(execute_rpc, q_before, spd=min(spd, 0.10), acc=min(acc, 6.0))
                time.sleep(float(fr["settle_s"]))
                out["ok"] = False
                out["reason"] = f"depth_out_of_gate_after_center:{depth_n}"
                out["rollback"] = True
                _status("PEDUNCLE_FOV_FINAL", ok=0, reason=out["reason"], rollback=1)
                return out
            berry = berry_n
            depth = depth_n
            preferred_px, preferred_py = float(berry["px"]), float(berry["py"])
            h, w = bgr.shape[:2]
            fov = evaluate_peduncle_fov(berry, w, h, fr)
            bgr0 = bgr
    elif not depth_ok0:
        # Depth was out of gate but REACH path already handled / aborted above.
        pass
    else:
        # Soft off-target with FOV+depth OK should have skipped earlier.
        _status("FRAMING_ACTION", action="skip", reason="fov_ok_soft_target")

    # Final capture
    bgr_f, berry_f, depth_f = _fresh_lock_depth(
        provider=provider,
        tracker=tracker,
        preferred_px=preferred_px,
        preferred_py=preferred_py,
        fetch_bgr=_bgr,
    )
    if bgr_f is None:
        out["reason"] = "no_frame_final"
        _status("PEDUNCLE_FOV_FINAL", ok=0, reason="no_frame")
        return out
    if berry_f is None:
        out["reason"] = "berry_lost_final"
        after_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_after.jpg"
        cv2.imwrite(str(after_path), bgr_f)
        out["after_path"] = str(after_path)
        _status("PEDUNCLE_FOV_FINAL", ok=0, reason="berry_lost")
        return out
    berry = berry_f
    depth = depth_f if depth_f is not None else berry.get("depth_m")
    berry["depth_m"] = depth
    h, w = bgr_f.shape[:2]
    fov = evaluate_peduncle_fov(berry, w, h, fr)
    overlay_f = _draw_framing_overlay(bgr_f, berry, fov)
    after_path = log_dir / f"{label}_framing{attempt_idx}_{ts}_after.jpg"
    cv2.imwrite(str(after_path), overlay_f)
    out["after_path"] = str(after_path)
    out["after"] = {
        "berry": {k: berry[k] for k in ("px", "py", "conf", "depth_m", "bbox")},
        "fov": fov,
    }

    depth_ok = _depth_in_gate(depth, depth_min_m, depth_max_m)
    if not fov["fov_ok"]:
        out["ok"] = False
        out["reason"] = ",".join(fov["reasons"]) or "fov_fail"
    elif not depth_ok:
        out["ok"] = False
        out["reason"] = f"depth_out_of_gate:{depth}"
    else:
        out["ok"] = True
        out["reason"] = "ok" if fov.get("on_target") else "ok_fov_soft_target"

    out["berry"] = out["after"]["berry"]
    out["bbox"] = list(berry["bbox"])
    _status(
        "PEDUNCLE_FOV_FINAL",
        ok=int(bool(out["ok"])),
        cx=f"{berry['px']:.0f}",
        cy=f"{berry['py']:.0f}",
        depth=round(float(depth), 3) if depth is not None else None,
        action=out.get("framing_action"),
        reasons=out.get("reason"),
        center=int(bool(out["executed_center"])),
        reach=int(bool(out["executed_micro_reach"])),
    )
    return out
