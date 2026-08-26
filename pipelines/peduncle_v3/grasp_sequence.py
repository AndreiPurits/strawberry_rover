"""Opt-in peduncle grasp sequence after berry approach (no cut).

HOME → berry approach (existing) → peduncle v3 → temporal → standoff →
grasp approach → gentle close → verify → DONE/ABORT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from pipelines.peduncle_v3 import PeduncleV3Pipeline, StageState
from pipelines.peduncle_v3.pipeline import load_config as load_peduncle_v3_config
from pipelines.peduncle_v3.temporal import TemporalStemVerifier
from pipelines.peduncle_detect import depth_in_peduncle_gate
from pipelines.roarm_perception import PerceptionConfig, pixel_to_camera_xyz, sample_depth_median

Point2D = Tuple[float, float]
REPO = Path(__file__).resolve().parents[2]


def _status(stage: str, **kw: Any) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    line = f"[STATUS] stage={stage}" + (f" {extra}" if extra else "")
    print(line, flush=True)


def _fetch_bgr(provider) -> Optional[np.ndarray]:
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


def _berry_bbox_from_lock(px: float, py: float, frame_w: int) -> Tuple[float, float, float, float]:
    half = max(36.0, 0.08 * float(frame_w))
    return (px - half, py - half, px + half, py + half)


def _sample_grasp_depth(provider, px: float, py: float) -> Tuple[Optional[float], Optional[Tuple[float, float, float]], str]:
    try:
        pair = provider.read(timeout_s=1.5)
    except Exception as exc:
        return None, None, f"no_frame:{exc}"
    if pair is None or pair.depth_m is None:
        return None, None, "no_depth"
    depth, why = sample_depth_median(
        pair.depth_m,
        int(round(px)),
        int(round(py)),
        radius=5,
        cfg=PerceptionConfig(),
    )
    if depth is None:
        return None, None, why or "depth_invalid"
    cam_xyz = None
    try:
        K = provider.get_intrinsics()
    except Exception:
        K = None
    if K:
        cam_xyz = pixel_to_camera_xyz(int(round(px)), int(round(py)), float(depth), K)
    return float(depth), cam_xyz, "ok"


def _save_log(log_dir: Path, label: str, attempt_idx: int, payload: Dict[str, Any], overlay: Optional[np.ndarray]) -> Dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    json_path = log_dir / f"{label}_grasp{attempt_idx}_{ts}.json"
    paths = {"log_path": str(json_path)}
    if overlay is not None:
        overlay_path = log_dir / f"{label}_grasp{attempt_idx}_{ts}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        paths["overlay_path"] = str(overlay_path)
        payload["overlay_path"] = str(overlay_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths


def _gentle_gripper_close(execute_rpc, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Existing T:106 via proxy gripper_ctrl; optional low torque. Never force-close."""
    close_cmd = float(cfg.get("close_cmd", 2.2))
    close_acc = float(cfg.get("close_acc", 8))
    close_spd = float(cfg.get("close_spd", 0))
    torque_g = int(cfg.get("torque_g", 300))
    settle_s = float(cfg.get("settle_s", 0.8))
    info: Dict[str, Any] = {"close_cmd": close_cmd, "torque_g": torque_g}
    try:
        execute_rpc(
            "dynamic_torque_limits",
            {"mode": 0, "g": torque_g, "b": 1000, "s": 1000, "e": 1000, "t": 1000, "r": 1000},
        )
    except Exception as exc:
        info["torque_warn"] = f"{type(exc).__name__}: {exc}"
    execute_rpc("gripper_ctrl", {"cmd": close_cmd, "spd": close_spd, "acc": close_acc})
    time.sleep(settle_s)
    info["ok"] = True
    return info


def _read_hand(execute_rpc) -> Optional[float]:
    try:
        fb = execute_rpc("feedback", {})
        data = fb.get("feedback") if isinstance(fb, dict) else None
        if isinstance(data, dict) and "g" in data:
            return float(data["g"])
    except Exception:
        return None
    return None


def run_peduncle_grasp_sequence(
    *,
    args: Any,
    result: Dict[str, Any],
    provider: Any,
    tracker: Any,
    execute_rpc: Callable,
    plan_one_shot: Callable,
    move_t102_smooth: Callable,
    q_from_dict: Callable,
    read_q: Callable,
    load_or_seed_learned: Callable,
    learned_path: Path,
    label: str,
    attempt_idx: int,
    q_final: Any,
    calib_path: Path,
    gripper_open: float = 1.08,
) -> Dict[str, Any]:
    """Mutates/extends result with peduncle_grasp block. Returns result."""

    v3_cfg = load_peduncle_v3_config()
    grasp_cfg = v3_cfg.get("grasp") or {}
    temporal_cfg = v3_cfg.get("temporal_verification") or {}
    standoff_cfg = v3_cfg.get("standoff") or {}
    log_dir = REPO / str((v3_cfg.get("logging") or {}).get("dir") or "runs/peduncle_v3_runtime")
    dmin = float(v3_cfg.get("depth_min_m", 0.08))
    dmax = float(v3_cfg.get("depth_max_m", 0.17))
    thr = float((v3_cfg.get("association") or {}).get("threshold", 0.1))

    out: Dict[str, Any] = {
        "enabled": True,
        "ran": False,
        "stage": StageState.PEDUNCLE_ACQUIRE.value,
        "abort": False,
    }
    result["peduncle_grasp"] = out

    def abort(reason: str, stage: StageState = StageState.ABORT) -> Dict[str, Any]:
        out["abort"] = True
        out["reason"] = reason
        out["stage"] = stage.value
        result["stage"] = stage.value
        _status(stage.value, reason=reason, abort=1)
        print(f"[grasp] ABORT {reason}", flush=True)
        # Safe open + optional return handled by caller return_home.
        try:
            execute_rpc("gripper_open", {})
        except Exception:
            pass
        return result

    if not result.get("ok"):
        return abort("approach_not_ok")

    berry = result.get("berry_after") or {}
    depth0 = berry.get("depth_m")
    px0, py0 = berry.get("px"), berry.get("py")
    conf0 = berry.get("conf")
    if not depth_in_peduncle_gate(depth0, depth_min_m=dmin, depth_max_m=dmax):
        return abort("depth_out_of_gate")
    if px0 is None or py0 is None:
        return abort("no_berry_lock")

    # Target lock id: freeze preferred pixel for this attempt.
    track_id = int(getattr(args, "_grasp_track_id", 1) or 1)
    args._grasp_track_id = track_id
    try:
        tracker.reanchor(float(px0), float(py0))
    except Exception:
        pass

    # --- FOV framing before calyx/OBB (headroom for stem) ---
    from pipelines.peduncle_v3.framing import run_peduncle_framing

    framing = run_peduncle_framing(
        provider=provider,
        tracker=tracker,
        execute_rpc=execute_rpc,
        move_t102_smooth=move_t102_smooth,
        read_q=read_q,
        q_now=q_final,
        preferred_px=float(px0),
        preferred_py=float(py0),
        depth_hint=float(depth0) if depth0 is not None else None,
        v3_cfg=v3_cfg,
        log_dir=log_dir,
        label=label,
        attempt_idx=attempt_idx,
        depth_max_m=dmax,
        depth_min_m=dmin,
        spd=float(getattr(args, "spd", 0.08)),
        acc=float(getattr(args, "acc", 5.0)),
        fetch_bgr=lambda: _fetch_bgr(provider),
    )
    out["framing"] = framing
    result["peduncle_framing"] = framing
    if not framing.get("ok"):
        return abort(f"framing_failed:{framing.get('reason')}", StageState.ABORT)
    berry_f = framing.get("berry") or {}
    if berry_f.get("px") is not None:
        px0, py0 = float(berry_f["px"]), float(berry_f["py"])
        conf0 = berry_f.get("conf", conf0)
        try:
            tracker.reanchor(float(px0), float(py0))
        except Exception:
            pass
    if berry_f.get("depth_m") is not None:
        depth0 = float(berry_f["depth_m"])
        if not depth_in_peduncle_gate(depth0, depth_min_m=dmin, depth_max_m=dmax):
            return abort("depth_out_of_gate_after_framing")
    framed_bbox = framing.get("bbox")

    pipe = getattr(args, "_peduncle_v3", None)
    if pipe is None:
        pipe = PeduncleV3Pipeline(cfg=v3_cfg)
        args._peduncle_v3 = pipe
    verifier = TemporalStemVerifier(temporal_cfg)

    # --- Settle + multi-frame acquire ---
    settle = float(grasp_cfg.get("pre_acquire_settle_s", 0.35))
    time.sleep(settle)
    frames_needed = int(temporal_cfg.get("required_frames", 3))
    max_tries = int(temporal_cfg.get("max_acquire_frames", frames_needed + 4))
    last_ped = None
    confirmed = False
    confirm_reason = ""
    for i in range(max_tries):
        result["stage"] = StageState.PEDUNCLE_ACQUIRE.value
        out["stage"] = StageState.PEDUNCLE_ACQUIRE.value
        _status(StageState.PEDUNCLE_ACQUIRE.value, frame=i, track=track_id)
        bgr = _fetch_bgr(provider)
        if bgr is None:
            return abort("no_frame")
        if framed_bbox and len(framed_bbox) == 4:
            bbox = (
                float(framed_bbox[0]),
                float(framed_bbox[1]),
                float(framed_bbox[2]),
                float(framed_bbox[3]),
            )
        else:
            bbox = _berry_bbox_from_lock(float(px0), float(py0), bgr.shape[1])
        ped = pipe.process_focused_berry(bgr, bbox, depth_m=depth0, draw_overlay=True)
        last_ped = ped
        out["last_perception"] = ped.as_dict()
        if ped.state == StageState.NO_ASSOC or ped.association is None or ped.association.selected is None:
            return abort(ped.reason or "NO_ASSOC", StageState.NO_ASSOC)
        if ped.state == StageState.NO_GRASP_POINT or not (ped.grasp_point or {}).get("ok"):
            return abort((ped.grasp_point or {}).get("reason") or "NO_GRASP_POINT", StageState.NO_GRASP_POINT)
        sel = ped.association.selected
        gp = ped.grasp_point["point_2d"]
        if ped.calyx_xy is None:
            return abort("NO_CALYX")
        verifier.push(
            track_id=track_id,
            calyx=ped.calyx_xy,
            base=sel.base,
            tip=sel.tip,
            grasp=(float(gp[0]), float(gp[1])),
            score=float(sel.score),
        )
        ok_t, reason_t, n_t = verifier.evaluate(thr)
        confirm_reason = reason_t
        out["temporal"] = {"ok": ok_t, "reason": reason_t, "n": n_t, "frame": i}
        print(f"[grasp] acquire frame={i} decision={ped.association.decision} temporal={reason_t}", flush=True)
        _status(
            StageState.PEDUNCLE_ACQUIRE.value,
            decision=ped.association.decision,
            temporal=reason_t,
            n=n_t,
            score=round(float(sel.score), 3),
        )
        if ok_t:
            confirmed = True
            break
        if reason_t in ("BASE_UNSTABLE", "GRASP_UNSTABLE", "CALYX_UNSTABLE", "ANGLE_UNSTABLE"):
            # Keep trying within max_tries; clear reset already handled on OBB switch.
            continue
        time.sleep(float(temporal_cfg.get("frame_gap_s", 0.12)))

    if not confirmed or last_ped is None:
        return abort(confirm_reason or "TEMPORAL_FAILED")

    result["stage"] = StageState.PEDUNCLE_CONFIRMED.value
    out["stage"] = StageState.PEDUNCLE_CONFIRMED.value
    mean_gp = verifier.mean_grasp()
    grasp_xy = mean_gp or tuple(last_ped.grasp_point["point_2d"])  # type: ignore[index]
    _status(StageState.PEDUNCLE_CONFIRMED.value, grasp_xy=f"{grasp_xy[0]:.0f},{grasp_xy[1]:.0f}")
    depth_g, cam_xyz, dwhy = _sample_grasp_depth(provider, grasp_xy[0], grasp_xy[1])
    out["grasp_point"] = {
        "xy": [round(grasp_xy[0], 1), round(grasp_xy[1], 1)],
        "depth_m": depth_g,
        "camera_xyz_m": [round(v, 4) for v in cam_xyz] if cam_xyz else None,
        "depth_reason": dwhy,
        "from_perception": last_ped.grasp_point,
    }
    if depth_g is None or not (dmin <= float(depth_g) <= float(grasp_cfg.get("max_grasp_depth_m", 0.22))):
        return abort(f"grasp_depth_invalid:{dwhy}")

    _save_log(log_dir, label, attempt_idx, {"phase": "confirmed", **out}, last_ped.overlay_bgr)

    # --- Open gripper, standoff ---
    result["stage"] = StageState.PEDUNCLE_STANDOFF.value
    out["stage"] = StageState.PEDUNCLE_STANDOFF.value
    _status(StageState.PEDUNCLE_STANDOFF.value, depth=round(float(depth_g), 3))
    execute_rpc("gripper_open", {})
    time.sleep(0.15)
    standoff_depth = float(standoff_cfg.get("standoff_depth_m", 0.12))
    # Command a slightly farther depth than current grasp so tip approaches from front.
    lock_depth = max(float(depth_g), standoff_depth)
    q_now = read_q(execute_rpc) or q_final
    learned = load_or_seed_learned(learned_path, label=label)
    try:
        tracker.reanchor(float(grasp_xy[0]), float(grasp_xy[1]))
        plan_so = plan_one_shot(
            q_now.as_dict() if hasattr(q_now, "as_dict") else q_now,
            {
                "px": float(grasp_xy[0]),
                "py": float(grasp_xy[1]),
                "depth_m": float(depth_g),
                "conf": float(last_ped.association.selected.score),
            },
            learned,
            calib_path=calib_path,
            standoff_m=lock_depth,
        )
        q_so = q_from_dict(plan_so["q_target"])
        q_so.hand = gripper_open
        out["standoff_plan"] = {
            "reason": plan_so.get("reason"),
            "prediction": plan_so.get("prediction"),
            "delta_q": plan_so.get("delta_q"),
        }
        if not getattr(args, "dry_run", False):
            print(f"[grasp] PEDUNCLE_STANDOFF px=({grasp_xy[0]:.0f},{grasp_xy[1]:.0f}) d={depth_g:.3f}", flush=True)
            _status(StageState.PEDUNCLE_STANDOFF.value, moving=1, px=f"{grasp_xy[0]:.0f},{grasp_xy[1]:.0f}")
            move_t102_smooth(
                execute_rpc,
                q_so,
                spd=min(float(getattr(args, "spd", 0.15)), float(grasp_cfg.get("standoff_spd", 0.10))),
                acc=min(float(getattr(args, "acc", 10)), float(grasp_cfg.get("standoff_acc", 6.0))),
            )
            out["standoff_executed"] = True
        else:
            out["standoff_executed"] = False
            out["standoff_dry_run"] = True
    except Exception as exc:
        return abort(f"standoff_err:{type(exc).__name__}:{exc}")

    time.sleep(float(grasp_cfg.get("post_standoff_settle_s", 0.4)))

    # --- Re-acquire after motion (never reuse old 3D blindly) ---
    bgr2 = _fetch_bgr(provider)
    if bgr2 is None:
        return abort("no_frame_after_standoff")
    # Keep same berry lock center; do not switch berry.
    bbox2 = _berry_bbox_from_lock(float(px0), float(py0), bgr2.shape[1])
    ped2 = pipe.process_focused_berry(bgr2, bbox2, depth_m=depth_g, draw_overlay=True)
    out["post_standoff_perception"] = ped2.as_dict()
    if ped2.association is None or ped2.association.decision != "TARGET" or ped2.association.selected is None:
        return abort(ped2.reason or "NO_ASSOC_AFTER_STANDOFF", StageState.NO_ASSOC)
    if not (ped2.grasp_point or {}).get("ok"):
        return abort((ped2.grasp_point or {}).get("reason") or "NO_GRASP_POINT", StageState.NO_GRASP_POINT)
    gp2 = ped2.grasp_point["point_2d"]
    depth2, cam2, dwhy2 = _sample_grasp_depth(provider, float(gp2[0]), float(gp2[1]))
    out["grasp_point_updated"] = {
        "xy": [round(float(gp2[0]), 1), round(float(gp2[1]), 1)],
        "depth_m": depth2,
        "camera_xyz_m": [round(v, 4) for v in cam2] if cam2 else None,
        "depth_reason": dwhy2,
    }
    if depth2 is None:
        return abort(f"grasp_depth_after_standoff:{dwhy2}")

    # --- Final small grasp approach ---
    result["stage"] = StageState.GRASP_APPROACH.value
    out["stage"] = StageState.GRASP_APPROACH.value
    _status(StageState.GRASP_APPROACH.value, depth=round(float(depth2), 3))
    approach_depth = float(grasp_cfg.get("final_approach_depth_m", 0.09))
    # Use measured stem depth as lock; planner targets approach_depth standoff.
    q_now2 = read_q(execute_rpc) or q_so
    try:
        tracker.reanchor(float(gp2[0]), float(gp2[1]))
        plan_g = plan_one_shot(
            q_now2.as_dict() if hasattr(q_now2, "as_dict") else q_now2,
            {
                "px": float(gp2[0]),
                "py": float(gp2[1]),
                "depth_m": float(depth2),
                "conf": float(ped2.association.selected.score),
            },
            learned,
            calib_path=calib_path,
            standoff_m=min(float(depth2), approach_depth),
        )
        q_g = q_from_dict(plan_g["q_target"])
        q_g.hand = gripper_open
        out["grasp_approach_plan"] = {
            "reason": plan_g.get("reason"),
            "prediction": plan_g.get("prediction"),
            "delta_q": plan_g.get("delta_q"),
        }
        if not getattr(args, "dry_run", False):
            print(f"[grasp] GRASP_APPROACH px=({gp2[0]:.0f},{gp2[1]:.0f}) d={depth2:.3f} (NO CUT)", flush=True)
            _status(StageState.GRASP_APPROACH.value, moving=1)
            move_t102_smooth(
                execute_rpc,
                q_g,
                spd=min(float(getattr(args, "spd", 0.15)), float(grasp_cfg.get("approach_spd", 0.08))),
                acc=min(float(getattr(args, "acc", 10)), float(grasp_cfg.get("approach_acc", 5.0))),
            )
            out["grasp_approach_executed"] = True
        else:
            out["grasp_approach_executed"] = False
    except Exception as exc:
        return abort(f"grasp_approach_err:{type(exc).__name__}:{exc}")

    time.sleep(float(grasp_cfg.get("pre_close_settle_s", 0.25)))

    # --- Gentle close ---
    result["stage"] = StageState.GRASP_CLOSE.value
    out["stage"] = StageState.GRASP_CLOSE.value
    _status(StageState.GRASP_CLOSE.value)
    g_before = _read_hand(execute_rpc)
    out["hand_before"] = g_before
    if getattr(args, "dry_run", False):
        out["close"] = {"skipped": True, "dry_run": True}
        result["stage"] = StageState.DONE.value
        out["stage"] = StageState.DONE.value
        out["ran"] = True
        return result

    try:
        out["close"] = _gentle_gripper_close(execute_rpc, grasp_cfg)
    except Exception as exc:
        return abort(f"gripper_close_err:{type(exc).__name__}:{exc}")

    # --- Verify ---
    result["stage"] = StageState.GRASP_VERIFY.value
    out["stage"] = StageState.GRASP_VERIFY.value
    _status(StageState.GRASP_VERIFY.value)
    g_after = _read_hand(execute_rpc)
    out["hand_after"] = g_after
    close_cmd = float(grasp_cfg.get("close_cmd", 2.2))
    g_min = float(grasp_cfg.get("verify_g_min", 1.45))
    # Success heuristic: jaw moved from open toward close_cmd (angle-only feedback).
    verify = {"g_before": g_before, "g_after": g_after, "close_cmd": close_cmd}
    if g_after is None:
        status = StageState.GRASP_UNCERTAIN
        verify["reason"] = "NO_FEEDBACK"
    elif g_after < g_min:
        status = StageState.GRASP_FAILED
        verify["reason"] = "JAW_STILL_OPEN"
    elif abs(g_after - close_cmd) > float(grasp_cfg.get("verify_g_tol", 0.55)):
        # Closed but not near command — may be blocked by stem (good) or jammed.
        status = StageState.GRASP_UNCERTAIN
        verify["reason"] = "JAW_OFF_COMMAND"
    else:
        status = StageState.GRASP_OK
        verify["reason"] = "JAW_REACHED_CMD"

    # Visual check (non-blocking for OK/FAILED classification).
    bgr3 = _fetch_bgr(provider)
    if bgr3 is not None:
        ped3 = pipe.process_focused_berry(
            bgr3,
            _berry_bbox_from_lock(float(px0), float(py0), bgr3.shape[1]),
            depth_m=depth2,
            draw_overlay=True,
        )
        verify["post_close_decision"] = ped3.as_dict().get("decision")
        verify["post_close_state"] = ped3.state.value
        out["post_close_perception"] = ped3.as_dict()
        _save_log(log_dir, label, attempt_idx, {"phase": "post_close", "verify": verify, **out}, ped3.overlay_bgr)

    out["verify"] = verify
    out["ran"] = True
    result["stage"] = status.value
    out["stage"] = status.value
    print(f"[grasp] {status.value} verify={verify.get('reason')} g={g_after} (NO CUT — STOP)", flush=True)
    _status(status.value, verify=verify.get("reason"), g=g_after)
    # Explicit STOP: do not yank, cut, or auto-return here unless return_home flag.
    result["stage"] = StageState.DONE.value if status == StageState.GRASP_OK else status.value
    out["final"] = result["stage"]
    _status(out["final"], attempt=attempt_idx)
    _save_log(log_dir, label, attempt_idx, {"phase": "final", **out}, None)
    return result
