#!/usr/bin/env python3
"""One smooth T:102 move: HOME2 → approach target (joint space).

Bench: target from last learned success joints.
Prod (later): target computed from strawberry lock at HOME2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ops/axm-monitor/agent"))

try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    pass

import yaml  # noqa: E402

from pipelines.roarm_kinematics import JointState  # noqa: E402
from pipelines.roarm_berry_planner import (  # noqa: E402
    load_learned,
    plan_one_shot,
    validate_plan_motion,
    verify_success,
    write_learned,
)
from pipelines.peduncle_detect import PeduncleDetector, depth_in_peduncle_gate  # noqa: E402
from pipelines.peduncle_v3 import PeduncleV3Pipeline, StageState  # noqa: E402
from pipelines.peduncle_v3.pipeline import load_config as load_peduncle_v3_config  # noqa: E402
from pipelines.roarm_strawberry_target import min_berry_conf  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from scripts.roarm_jacobian_probe import (  # noqa: E402
    STEREO_DEPTH,
    STEREO_INFO,
    STEREO_RGB,
    _joints_params,
    clamp_joint,
    read_q,
)
from pipelines.roarm_kinematics import (  # noqa: E402
    KinematicsConfig,
    explicit_reach_delta,
)
from scripts.roarm_strawberry_approach import (  # noqa: E402
    StrawberryTargetTracker,
    _load_fleet_env,
    measure,
)

OUT = REPO / "runs/roarm_learn/one_shot_run.json"
LEARNED = REPO / "runs/roarm_learn/manual_success_pose.json"
LAST_RUN = REPO / "runs/roarm_learn/home2_to_manual_run.json"
LOCK_REF = REPO / "runs/roarm_learn/new_position_lock.json"
CALIB = REPO / "runs/roarm_kinematics/calibration.json"
ATTEMPT_DIR = REPO / "runs/roarm_learn/dom_final_attempts"
SUMMARY = REPO / "runs/roarm_learn/dom_final_repeat_summary.json"
PREVIEW_CACHE = REPO / "runs/roarm_learn/strawberry_preview.json"
GRIPPER_OPEN = 1.08


def _status(stage: str, **kw: Any) -> None:
    """Terminal stage marker for live operator monitoring."""
    extra = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    line = f"[STATUS] stage={stage}" + (f" {extra}" if extra else "")
    print(line, flush=True)


def _emit_planner_status(q_start: JointState, berry: Dict[str, Any], plan: Dict[str, Any]) -> None:
    """Full planner dump before any T:102 move."""
    demo = plan.get("demo") or {}
    dq = plan.get("delta_q") or {}
    qt = plan.get("q_target") or {}
    qs = plan.get("q_start") or q_start.as_dict()
    _status(
        "PLAN",
        mode=plan.get("planner_mode"),
        reason=str(plan.get("reason", "")).replace(" ", "_"),
        n_demos=plan.get("n_demos"),
        berry_px=f"{float(berry.get('px', 0)):.0f}",
        berry_py=f"{float(berry.get('py', 0)):.0f}",
        berry_depth=round(float(berry.get("depth_m", 0)), 3),
    )
    _status(
        "PLAN_JOINTS",
        cur_b=round(float(qs.get("base", q_start.base)), 3),
        cur_s=round(float(qs.get("shoulder", q_start.shoulder)), 3),
        cur_e=round(float(qs.get("elbow", q_start.elbow)), 3),
        tgt_b=round(float(qt.get("base", 0)), 3),
        tgt_s=round(float(qt.get("shoulder", 0)), 3),
        tgt_e=round(float(qt.get("elbow", 0)), 3),
        d_b=round(float(dq.get("base", 0)), 4),
        d_s=round(float(dq.get("shoulder", 0)), 4),
        d_e=round(float(dq.get("elbow", 0)), 4),
    )
    if demo:
        dqs = demo.get("q_start") or {}
        dqf = demo.get("q_success") or {}
        _status(
            "PLAN_DEMO",
            id=demo.get("id"),
            source=demo.get("source"),
            dist=demo.get("distance"),
            demo_start_b=round(float(dqs.get("base", 0)), 3) if dqs else None,
            demo_start_s=round(float(dqs.get("shoulder", 0)), 3) if dqs else None,
            demo_start_e=round(float(dqs.get("elbow", 0)), 3) if dqs else None,
            demo_final_b=round(float(dqf.get("base", 0)), 3) if dqf else None,
            demo_final_s=round(float(dqf.get("shoulder", 0)), 3) if dqf else None,
            demo_final_e=round(float(dqf.get("elbow", 0)), 3) if dqf else None,
        )
    else:
        _status("PLAN_DEMO", id="none", source="none", dist="n/a")
    jac_used = bool(plan.get("jacobian_used"))
    _status(
        "PLAN_JAC",
        used=int(jac_used),
        reason=plan.get("jacobian_reason"),
    )
    tgt_img = plan.get("target_image") or {}
    _status(
        "PLAN_DEPTH",
        depth0=round(float(berry.get("depth_m", 0)), 3),
        target_depth=round(float(tgt_img.get("depth_m", 0)), 3),
        expected_d_change=plan.get("expected_depth_change_m"),
        pred_depth=round(float((plan.get("prediction") or {}).get("depth_m", 0)), 3),
    )


# Learned coupled delta HOME2 → success on this bench (rad)
HOME2_TO_SUCCESS_DELTA = {
    "base": -0.003,
    "shoulder": 1.414,
    "elbow": -1.501,
    "wrist": 0.0,
    "roll": 0.005,
}


def q_from_dict(d: dict) -> JointState:
    return JointState(
        base=float(d["base"]),
        shoulder=float(d["shoulder"]),
        elbow=float(d["elbow"]),
        wrist=float(d.get("wrist", 0)),
        roll=float(d.get("roll", 0)),
        hand=float(d.get("hand", GRIPPER_OPEN)),
    )


def q_home2_cfg(cfg: dict) -> JointState:
    h = cfg["HOME2"]
    return JointState(
        base=float(h["base"]),
        shoulder=float(h["shoulder"]),
        elbow=float(h["elbow"]),
        wrist=float(h.get("wrist", 0)),
        roll=float(h.get("roll", 0)),
        hand=GRIPPER_OPEN,
    )


def q_named_cfg(cfg: dict, name: str, *, force_open: bool = False) -> JointState:
    h = cfg[name]
    return JointState(
        base=float(h["base"]),
        shoulder=float(h["shoulder"]),
        elbow=float(h["elbow"]),
        wrist=float(h.get("wrist", 0)),
        roll=float(h.get("roll", 0)),
        hand=GRIPPER_OPEN if force_open else float(h.get("hand", GRIPPER_OPEN)),
    )


def load_target_learned() -> JointState:
    if LAST_RUN.is_file():
        run = json.loads(LAST_RUN.read_text(encoding="utf-8"))
        if run.get("ok") and run.get("joints_end"):
            print("[one] target from last ok run joints_end")
            return q_from_dict(run["joints_end"])
    doc = json.loads(LEARNED.read_text(encoding="utf-8"))
    return q_from_dict(doc["joints"])


def load_target_yaml(cfg: dict) -> JointState:
    return q_from_dict(cfg["MANUAL_SUCCESS"])


DPX_PER_BASE = 611.0
DDEPTH_DSH = -0.14  # m per rad shoulder+


def load_home2_delta(cfg: dict) -> Dict[str, float]:
    """Δjoints HOME2→success from last calibrated run."""
    h = cfg["HOME2"]
    if LAST_RUN.is_file():
        run = json.loads(LAST_RUN.read_text(encoding="utf-8"))
        je = run.get("joints_end")
        if run.get("ok") and je:
            return {
                "base": float(je["base"]) - float(h["base"]),
                "shoulder": float(je["shoulder"]) - float(h["shoulder"]),
                "elbow": float(je["elbow"]) - float(h["elbow"]),
                "wrist": float(je.get("wrist", 0)) - float(h.get("wrist", 0)),
                "roll": float(je.get("roll", 0)) - float(h.get("roll", 0)),
            }
    return dict(HOME2_TO_SUCCESS_DELTA)


def load_label_delta(label: str, cfg: dict) -> Dict[str, float]:
    """Per-bench Δ from {label}_learned.json, else HOME2 calibration."""
    if label:
        lpath = REPO / "runs/roarm_learn" / f"{label}_learned.json"
        if lpath.is_file():
            doc = json.loads(lpath.read_text(encoding="utf-8"))
            d = doc.get("delta_from_start")
            if d and d.get("shoulder") is not None:
                print(f"[one] delta from {lpath.name}")
                return {
                    "base": float(d.get("base", 0)),
                    "shoulder": float(d["shoulder"]),
                    "elbow": float(d["elbow"]),
                    "wrist": float(d.get("wrist", 0)),
                    "roll": float(d.get("roll", 0)),
                }
    return load_home2_delta(cfg)


def load_recover_joints(label: str) -> Optional[JointState]:
    """Start pose for repeat runs: learned start_joints beats lock (lock drifts)."""
    lpath = REPO / "runs/roarm_learn" / f"{label}_learned.json"
    if lpath.is_file():
        doc = json.loads(lpath.read_text(encoding="utf-8"))
        sj = doc.get("start_joints")
        if sj:
            return q_from_dict(sj)
    rpath = REPO / "runs/roarm_learn" / f"{label}_lock.json"
    if rpath.is_file():
        rdoc = json.loads(rpath.read_text(encoding="utf-8"))
        if rdoc.get("joints"):
            return q_from_dict(rdoc["joints"])
    return None


def compute_target_from_delta(
    q_start: JointState,
    berry_lock: dict,
    delta: Dict[str, float],
    *,
    label: str = "",
) -> JointState:
    """Start lock → target joints: fixed Δ + px/depth trims from berry image."""
    ref_lock = {}
    if label:
        lpath = REPO / "runs/roarm_learn" / f"{label}_learned.json"
        if lpath.is_file():
            lb = json.loads(lpath.read_text()).get("lock_berry") or {}
            if lb.get("px") is not None:
                ref_lock = lb
    if not ref_lock:
        label_lock = REPO / "runs/roarm_learn" / f"{label}_lock.json"
        if label and label_lock.is_file():
            lb = json.loads(label_lock.read_text()).get("berry") or {}
            if lb and lb.get("px") is not None:
                ref_lock = lb
    if not ref_lock and LEARNED.is_file():
        ref_lock = json.loads(LEARNED.read_text()).get("lock_at_HOME2") or {}
    if not ref_lock and LOCK_REF.is_file():
        b = json.loads(LOCK_REF.read_text()).get("berry") or {}
        ref_lock = {"px": b.get("px", 310), "py": b.get("py", 72), "depth_m": b.get("depth_m", 0.37)}

    d_sh = float(delta["shoulder"])
    d_el = float(delta["elbow"])
    d_ba = float(delta["base"])

    px = float(berry_lock.get("px", ref_lock.get("px", 310)))
    depth = float(berry_lock.get("depth_m", ref_lock.get("depth_m", 0.37)))
    ref_px = float(ref_lock.get("px", px))
    ref_depth = float(ref_lock.get("depth_m", depth))

    dpx = px - ref_px
    d_ba -= dpx / DPX_PER_BASE * 0.42

    # farther berry at lock → slightly more shoulder forward
    depth_extra = depth - ref_depth
    d_sh += max(-0.06, min(0.06, depth_extra / DDEPTH_DSH * 0.05))
    # far berry → stronger elbow extension (fold elbow more)
    if depth_extra > 0.02:
        d_el -= min(0.22, depth_extra * 0.55)

    q = JointState(
        base=clamp_joint("base", q_start.base + d_ba),
        shoulder=clamp_joint("shoulder", q_start.shoulder + d_sh),
        elbow=clamp_joint("elbow", q_start.elbow + d_el),
        wrist=float(delta.get("wrist", 0)),
        roll=float(delta.get("roll", 0)),
        hand=GRIPPER_OPEN,
    )
    return q


def joint_error(q: JointState, tgt: JointState) -> float:
    keys = ("base", "shoulder", "elbow", "wrist", "roll")
    return max(abs(getattr(q, k) - getattr(tgt, k)) for k in keys)


def est_move_duration(q0: JointState, q1: JointState, *, spd: float, acc: float) -> float:
    keys = ("base", "shoulder", "elbow", "wrist", "roll")
    delta = max(abs(getattr(q1, k) - getattr(q0, k)) for k in keys)
    if spd > 0.01:
        return min(18.0, max(2.8, delta / spd + 1.2))
    return min(18.0, max(3.5, math.sqrt(2 * delta / max(acc, 1.0)) + 1.5))


def move_t102_smooth(
    execute_rpc,
    q: JointState,
    *,
    spd: float,
    acc: float,
) -> None:
    q.hand = GRIPPER_OPEN
    execute_rpc("joints_move", _joints_params(q, acc=acc, spd=spd))


def wait_reach(
    execute_rpc,
    q_tgt: JointState,
    *,
    timeout_s: float,
    tol: float = 0.045,
    stable_s: float = 0.0,
) -> Optional[JointState]:
    """Wait until joints are within tol of target.

    If stable_s > 0, require that condition to hold continuously so we do not
    return on the first crossing while T:102 is still coasting past the target.
    """
    t0 = time.time()
    ok_since: Optional[float] = None
    last: Optional[JointState] = None
    while time.time() - t0 < timeout_s:
        q = read_q(execute_rpc)
        last = q or last
        if q and joint_error(q, q_tgt) <= tol:
            if stable_s <= 0.0:
                return q
            if ok_since is None:
                ok_since = time.time()
            elif time.time() - ok_since >= stable_s:
                return q
        else:
            ok_since = None
        time.sleep(0.12)
    return last if last is not None else read_q(execute_rpc)


def ensure_home_reached(
    execute_rpc,
    q_home: JointState,
    *,
    spd: float,
    acc: float,
    tol: float,
    timeout_s: float,
) -> Tuple[Optional[JointState], float]:
    """Move to home; recover RoArm shoulder sticky regions.

    Known sticky bands: ≈−0.59 (misses DOM −0.69) and high positive after
    approach (direct DOM command ignored). Use waypoints when far above home.
    """
    q_fb = read_q(execute_rpc)
    if q_fb is not None and (q_fb.shoulder - q_home.shoulder) > 0.35:
        # Descend through intermediates (approach → DOM), matching manual recover.
        print(
            f"[one] home from high shoulder fb_s={q_fb.shoulder:.3f}; "
            "waypoint descent → DOM_FINAL"
        )
        for s_wp, e_wp, wait_s in (
            (0.30, 2.20, 4.0),
            (-0.20, 2.50, 5.0),
            (min(q_home.shoulder - 0.12, -0.80), q_home.elbow, 6.0),
        ):
            q_wp = JointState(
                base=q_home.base,
                shoulder=s_wp,
                elbow=e_wp,
                wrist=q_home.wrist,
                roll=q_home.roll,
                hand=GRIPPER_OPEN,
            )
            move_t102_smooth(execute_rpc, q_wp, spd=max(spd, 0.12), acc=max(acc, 5.0))
            wait_reach(execute_rpc, q_wp, timeout_s=wait_s, tol=0.15)

    move_t102_smooth(execute_rpc, q_home, spd=spd, acc=acc)
    q_fb = wait_reach(execute_rpc, q_home, timeout_s=timeout_s, tol=tol)
    q_fb = q_fb or read_q(execute_rpc)
    err = joint_error(q_fb, q_home) if q_fb else float("inf")
    if err <= tol:
        return q_fb, err

    # Shoulder short of DOM_FINAL (feedback more positive than commanded).
    if q_fb is not None and (q_fb.shoulder - q_home.shoulder) > tol:
        nudge_s = min(q_home.shoulder - 0.12, -0.80)
        print(
            f"[one] home shoulder sticky fb_s={q_fb.shoulder:.3f} "
            f"cmd_s={q_home.shoulder:.3f}; nudge s={nudge_s:.3f} then settle"
        )
        q_nudge = JointState(
            base=q_home.base,
            shoulder=nudge_s,
            elbow=q_home.elbow,
            wrist=q_home.wrist,
            roll=q_home.roll,
            hand=GRIPPER_OPEN,
        )
        move_t102_smooth(execute_rpc, q_nudge, spd=max(spd, 0.12), acc=acc)
        wait_reach(execute_rpc, q_nudge, timeout_s=min(8.0, timeout_s), tol=0.12)
        move_t102_smooth(execute_rpc, q_home, spd=spd, acc=acc)
        q_fb = wait_reach(execute_rpc, q_home, timeout_s=min(10.0, timeout_s), tol=tol)
        q_fb = q_fb or read_q(execute_rpc)
        err = joint_error(q_fb, q_home) if q_fb else float("inf")
    return q_fb, err


def approach_command_joints(q_start: JointState, q_end: JointState) -> JointState:
    """Adjust T:102 command so feedback lands on planned q_end.

    Large DOM_FINAL→approach shoulder swings settle ~+0.06 rad above the
    commanded value (cmd 0.80 → fb 0.862; cmd 0.862 → fb 0.936). Working demos
    require end shoulder ≈0.86 — bias the command, do not change the plan target.
    """
    q_cmd = JointState(
        base=q_end.base,
        shoulder=q_end.shoulder,
        elbow=q_end.elbow,
        wrist=q_end.wrist,
        roll=q_end.roll,
        hand=GRIPPER_OPEN,
    )
    d_s = float(q_end.shoulder) - float(q_start.shoulder)
    if d_s > 1.0:
        bias = 0.062
        q_cmd.shoulder = float(q_end.shoulder) - bias
        print(
            f"[one] T102 shoulder cmd bias −{bias:.3f} "
            f"(plan_s={q_end.shoulder:.3f} → cmd_s={q_cmd.shoulder:.3f})"
        )
    return q_cmd


def lock_berry(
    provider,
    tracker,
    *,
    preview_first: bool = False,
    max_depth_m: float = 0.80,
    min_conf: float = 0.0,
    preferred_px: Optional[float] = None,
    preferred_py: Optional[float] = None,
) -> Optional[Tuple]:
    pref_px = preferred_px
    pref_py = preferred_py
    if pref_px is None and hasattr(tracker, "preferred_px"):
        pref_px = tracker.preferred_px
    if pref_py is None and hasattr(tracker, "preferred_py"):
        pref_py = tracker.preferred_py
    if preview_first:
        m = lock_berry_from_preview(
            tracker,
            max_depth_m=max_depth_m,
            min_conf=min_conf,
            preferred_px=pref_px,
            preferred_py=pref_py,
        )
        # Reject stale approach-view cache at DOM_FINAL (depth≪0.35, berry high in frame).
        if m and float(m[2]) >= 0.34 and float(m[1]) >= 100.0:
            return m
        if m:
            print(
                f"[one] preview lock rejected as stale/close-range "
                f"px={m[0]:.0f} py={m[1]:.0f} d={m[2]:.3f}; local detect"
            )
    for _ in range(4):
        m = measure(provider, tracker, "http://127.0.0.1:8080", frames=1)
        if m and float(m[2]) <= max_depth_m:
            if min_conf <= 0.0 or float(m[3]) >= min_conf:
                # Same DOM_FINAL gate: ignore close-range / top-of-frame ghosts.
                if float(m[2]) >= 0.34 and float(m[1]) >= 100.0:
                    return m
        time.sleep(0.12)
    return lock_berry_from_preview(
        tracker,
        max_depth_m=max_depth_m,
        min_conf=min_conf,
        preferred_px=pref_px,
        preferred_py=pref_py,
    )


def _fetch_json(url: str, timeout: float = 4.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _read_preview_cache(*, max_age_s: float = 2.0) -> Optional[dict]:
    try:
        if not PREVIEW_CACHE.is_file():
            return None
        if time.time() - PREVIEW_CACHE.stat().st_mtime > max_age_s:
            return None
        out = json.loads(PREVIEW_CACHE.read_text(encoding="utf-8"))
        return out if isinstance(out, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def lock_berry_from_preview(
    tracker,
    *,
    max_depth_m: float = 0.80,
    min_conf: float = 0.0,
    preferred_px: Optional[float] = None,
    preferred_py: Optional[float] = None,
) -> Optional[Tuple]:
    """Pick a visible berry from fleet-agent warm preview cache."""
    berries = list_visible_berries_from_preview(
        max_depth_m=max_depth_m,
        min_conf=min_conf,
        preferred_px=preferred_px,
        preferred_py=preferred_py,
        pick="nearest_pref" if preferred_px is not None else "nearest_depth",
    )
    if not berries:
        return None
    det = berries[0]
    px = float(det["px"])
    py = float(det["py"])
    depth = float(det["depth_m"])
    conf = float(det["conf"])
    print(
        f"[one] lock from {det.get('_source', 'preview')}: "
        f"{len(berries)} candidate(s), picked px={px:.0f} py={py:.0f}"
    )
    if hasattr(tracker, "update"):
        tracker.update(det)
    wh = det.get("_image_wh") or (640, 480)
    return px, py, depth, conf, (int(wh[0]), int(wh[1]))


def list_visible_berries_from_preview(
    *,
    max_depth_m: float = 0.80,
    min_conf: float = 0.0,
    preferred_px: Optional[float] = None,
    preferred_py: Optional[float] = None,
    pick: str = "all",
) -> List[dict]:
    """Return visible berries from warm preview (sorted left→right for sweep)."""
    try:
        from roarm_strawberry_preview import collect_roarm_strawberry_preview
        from pipelines.roarm_strawberry_target import _bbox_center, filter_berry_confidence

        if pick == "all":
            source = "live_collect"
            out = collect_roarm_strawberry_preview(
                "http://127.0.0.1:8080", _fetch_json, interval_s=0.0
            )
        else:
            out = _read_preview_cache()
            source = "warm_cache"
            if out is None:
                source = "local_fallback"
                out = collect_roarm_strawberry_preview(
                    "http://127.0.0.1:8080", _fetch_json, interval_s=0.0
                )
        dets = filter_berry_confidence(list(out.get("detections") or []), min_conf=min_conf or None)
        wh = (int(out.get("image_w") or 640), int(out.get("image_h") or 480))
        fallback_depth = float(os.environ.get("AXM_BERRY_FALLBACK_DEPTH_M", "0.42"))
        candidates: List[dict] = []
        for d in dets:
            depth_raw = d.get("depth_m")
            if depth_raw is None:
                depth = fallback_depth
            else:
                depth = float(depth_raw)
            if not (0.05 <= depth <= max_depth_m):
                continue
            px = float(d.get("px", _bbox_center(d)[0]))
            py = float(d.get("py", _bbox_center(d)[1]))
            item = dict(d)
            item["px"] = px
            item["py"] = py
            item["depth_m"] = depth
            item["_source"] = source
            item["_image_wh"] = wh
            candidates.append(item)
        if not candidates:
            return []
        candidates.sort(key=lambda d: (float(d["px"]), -float(d.get("conf", 0.0))))

        if pick == "all":
            return candidates

        if pick == "nearest_pref" and preferred_px is not None and preferred_py is not None:
            best = min(
                candidates,
                key=lambda d: (
                    float(math.hypot(float(d["px"]) - preferred_px, float(d["py"]) - preferred_py)),
                    float(d["depth_m"]),
                    -float(d.get("conf", 0.0)),
                ),
            )
            return [best]

        best = min(candidates, key=lambda d: (float(d["depth_m"]), -float(d.get("conf", 0.0))))
        return [best]
    except Exception as exc:
        print(f"[one] preview list failed: {exc}")
        return []


def discover_berry_targets(
    *,
    max_depth_m: float = 0.80,
    min_conf: float = 0.0,
) -> List[Tuple[str, float, float]]:
    berries: List[dict] = []
    for attempt in range(6):
        berries = list_visible_berries_from_preview(
            max_depth_m=max_depth_m,
            min_conf=min_conf,
            pick="all",
        )
        if len(berries) >= 1:
            break
        time.sleep(0.4 if attempt < 2 else 0.8)
    out: List[Tuple[str, float, float]] = []
    for idx, det in enumerate(berries, start=1):
        out.append((f"berry_{idx}", float(det["px"]), float(det["py"])))
    return out


def verify_berry_from_preview(
    tracker,
    *,
    timeout_s: float = 0.25,
    min_conf: float = 0.0,
) -> Optional[Tuple]:
    """Fast post-move check via fleet-agent preview (no local YOLO)."""
    pref_px = getattr(tracker, "last_px", None)
    pref_py = getattr(tracker, "last_py", None)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        berries = list_visible_berries_from_preview(
            min_conf=min_conf,
            preferred_px=pref_px,
            preferred_py=pref_py,
            pick="nearest_pref" if pref_px is not None else "nearest_depth",
        )
        if berries:
            d = berries[0]
            wh = d.get("_image_wh") or (640, 480)
            return (
                float(d["px"]),
                float(d["py"]),
                float(d["depth_m"]),
                float(d["conf"]),
                (int(wh[0]), int(wh[1])),
            )
        time.sleep(0.03)
    return None


def wait_berry_standoff(
    provider,
    tracker,
    *,
    timeout_s: float,
    frames: int = 1,
    depth_min: float = 0.10,
    depth_max: float = 0.17,
    min_conf: float = 0.0,
) -> Optional[Tuple]:
    t0 = time.time()
    best = None
    while time.time() - t0 < timeout_s:
        m = measure(provider, tracker, "http://127.0.0.1:8080", frames=max(1, frames))
        if not m:
            time.sleep(0.05)
            continue
        if min_conf > 0.0 and float(m[3]) < min_conf:
            time.sleep(0.05)
            continue
        if best is None or abs(float(m[2]) - 0.14) < abs(float(best[2]) - 0.14):
            best = m
        if depth_min <= float(m[2]) <= depth_max:
            return m
        time.sleep(0.05)
    return best


def measure_berry_depth_stable(
    provider,
    tracker,
    *,
    n_samples: int = 5,
    frames: int = 1,
    gap_s: float = 0.06,
    min_conf: float = 0.0,
) -> Tuple[Optional[Tuple], Optional[float], Optional[float], List[float]]:
    """Fresh multi-frame depth: returns (meas_with_median_d, raw_depth, median_depth, samples)."""
    samples: List[Tuple] = []
    for _ in range(max(1, int(n_samples))):
        m = measure(provider, tracker, "http://127.0.0.1:8080", frames=max(1, frames))
        if m is not None and (min_conf <= 0.0 or float(m[3]) >= min_conf):
            samples.append(m)
        time.sleep(max(0.0, float(gap_s)))
    if not samples:
        return None, None, None, []
    depths = [float(s[2]) for s in samples]
    raw_d = depths[-1]
    med_d = float(sorted(depths)[len(depths) // 2])
    # Prefer last lock geometry; attach median depth for verify.
    last = samples[-1]
    meas = (float(last[0]), float(last[1]), med_d, float(last[3]), last[4])
    return meas, raw_d, med_d, depths


def maybe_micro_reach_step(
    q: JointState,
    depth_m: float,
    *,
    aim_depth_m: float = 0.14,
    max_step_rad: float = 0.08,
) -> Tuple[Optional[JointState], Dict[str, Any]]:
    """One REACH-style correction via existing explicit_reach_delta (shoulder+/elbow−)."""
    cam_err = float(depth_m) - float(aim_depth_m)
    info: Dict[str, Any] = {
        "mechanism": "explicit_reach_delta",
        "cam_err_m": round(cam_err, 4),
        "aim_depth_m": float(aim_depth_m),
        "max_step_rad": float(max_step_rad),
    }
    if cam_err <= 0.001:
        info["skipped"] = "already_at_or_below_aim"
        return None, info
    cfg = KinematicsConfig(max_step_rad=float(max_step_rad))
    ds, de = explicit_reach_delta(cam_err, cfg)
    info["delta_shoulder"] = round(float(ds), 4)
    info["delta_elbow"] = round(float(de), 4)
    if abs(ds) + abs(de) < 1e-4:
        info["skipped"] = "zero_step"
        return None, info
    q2 = JointState(
        base=float(q.base),
        shoulder=clamp_joint("shoulder", float(q.shoulder) + float(ds)),
        elbow=clamp_joint("elbow", float(q.elbow) + float(de)),
        wrist=float(q.wrist),
        roll=float(q.roll),
        hand=GRIPPER_OPEN,
    )
    info["q_target"] = q2.as_dict()
    return q2, info


def _should_micro_approach(
    berry_after: Dict[str, Any],
    verify: Dict[str, Any],
    *,
    depth_max_m: float,
    slack_m: float = 0.030,
    min_conf: float = 0.0,
) -> Tuple[bool, str]:
    """Micro only if lock holds and depth is slightly above gate (not a new gate)."""
    if not berry_after:
        return False, "no_berry"
    conf = float(berry_after.get("conf", 0.0))
    depth = float(berry_after.get("depth_m", 9.0))
    if conf < min_conf:
        return False, "low_conf"
    if depth <= depth_max_m:
        return False, "already_in_gate"
    if depth > depth_max_m + slack_m:
        return False, f"too_far_for_micro depth={depth:.3f}"
    reasons = list(verify.get("reasons") or [])
    # Allow micro when the only blocking issue is depth_not_in_standoff (3mm-class miss).
    blocking = [r for r in reasons if r != "depth_not_in_standoff"]
    if blocking:
        return False, f"other_verify_fail:{','.join(blocking)}"
    if "depth_not_in_standoff" not in reasons and reasons:
        return False, "unexpected_reasons"
    return True, "slightly_above_standoff"


def load_or_seed_learned(path: Path, *, label: str) -> Dict[str, Any]:
    if path.is_file():
        return load_learned(path, label=label)
    if label == "dom_final":
        seed_path = REPO / "runs/roarm_learn/test4_learned.json"
        if seed_path.is_file():
            doc = load_learned(seed_path, label="test4")
            doc = dict(doc)
            doc["label"] = label
            doc["seeded_from"] = str(seed_path)
            doc["episodes"] = list(doc.get("episodes") or [])
            write_learned(path, doc)
            print(f"[one] seeded learned → {path}")
            return doc
    return load_learned(path, label=label)


def append_episode_to_learned(path: Path, label: str, result: Dict[str, Any]) -> None:
    doc = load_or_seed_learned(path, label=label)
    episodes = list(doc.get("episodes") or [])
    episodes.append(
        {
            "created_at": result["updated_at"],
            "ok": bool(result.get("ok")),
            "episode_path": result.get("episode_path"),
            "berry_before": result.get("berry_before"),
            "berry_after": result.get("berry_after"),
            "joints_start": result.get("joints_start"),
            "joints_target": result.get("joints_target"),
            "joints_end": result.get("joints_end"),
            "verify": result.get("verify"),
            "move_elapsed_s": result.get("move_elapsed_s"),
        }
    )
    doc["episodes"] = episodes[-100:]
    write_learned(path, doc)


def parse_target_sequence(raw: str, default_px: float, default_py: float) -> list[Tuple[str, float, float]]:
    if not raw.strip():
        return [("target", float(default_px), float(default_py))]
    out = []
    for item in raw.split(","):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) == 2:
            name = f"target{len(out) + 1}"
            px, py = parts
        elif len(parts) == 3:
            name, px, py = parts
        else:
            raise ValueError(f"Bad --target-sequence item: {item!r}; expected name:px:py")
        out.append((name or f"target{len(out) + 1}", float(px), float(py)))
    return out


def run_attempt(
    *,
    attempt_idx: int,
    execute_rpc,
    provider,
    tracker,
    q_home: JointState,
    cfg: dict,
    args: argparse.Namespace,
    label: str,
    learned_path: Path,
    target_name: str = "",
    at_home: bool = False,
) -> Dict[str, Any]:
    cycle_t0 = time.time()
    q_home_move = JointState(
        base=q_home.base,
        shoulder=q_home.shoulder,
        elbow=q_home.elbow,
        wrist=q_home.wrist,
        roll=q_home.roll,
        hand=GRIPPER_OPEN,
    )
    print(f"[one] attempt {attempt_idx}/{args.repeat} → {args.home_pose} (open grip)")
    _status("HOME", attempt=attempt_idx, pose=args.home_pose)
    # Feedback after move (logging / reach check). Planning for DOM_FINAL prior_only
    # must use commanded home — demo Δ is relative to that pose; a drifted start
    # overshoots shoulder (e.g. −0.59+1.56→0.97 instead of −0.69+1.56→0.86).
    q_fb: Optional[JointState] = None
    if not args.skip_home and not at_home:
        q_now = read_q(execute_rpc) or q_home_move
        # Full travel from approach → DOM_FINAL (do not cap at return-wait-s=5).
        home_timeout = est_move_duration(
            q_now, q_home_move, spd=args.return_spd, acc=args.return_acc
        ) + 6.0
        home_timeout = max(home_timeout, 14.0)
        home_timeout = min(home_timeout, 28.0)
        q_fb, home_err = ensure_home_reached(
            execute_rpc,
            q_home_move,
            spd=args.return_spd,
            acc=args.return_acc,
            tol=float(args.return_tol),
            timeout_s=home_timeout,
        )
        if home_err > float(args.return_tol):
            print(
                f"[one] WARN still off home err={home_err:.3f} "
                f"fb_s={(q_fb.shoulder if q_fb else float('nan')):.3f} "
                f"cmd_s={q_home_move.shoulder:.3f}"
            )
    else:
        q_fb = read_q(execute_rpc) or q_home_move

    # Always plan DOM_FINAL prior from commanded home (demo Δ is relative to that
    # pose). Feedback after sticky-nudge can be −0.71 and would shift tgt_s down.
    if str(args.home_pose).upper() == "DOM_FINAL" and not args.from_current:
        q_start = JointState(
            base=q_home_move.base,
            shoulder=q_home_move.shoulder,
            elbow=q_home_move.elbow,
            wrist=q_home_move.wrist,
            roll=q_home_move.roll,
            hand=GRIPPER_OPEN,
        )
        if q_fb is not None and abs(q_fb.shoulder - q_start.shoulder) > 0.02:
            print(
                f"[one] plan_start=commanded {args.home_pose} "
                f"(fb s={q_fb.shoulder:.3f})"
            )
    elif (
        q_fb is not None
        and joint_error(q_fb, q_home_move) <= float(args.return_tol)
    ):
        q_start = q_fb
    else:
        q_start = q_fb or q_home_move
    q_start.hand = GRIPPER_OPEN
    # Let stereo settle after waypoint home before locking berry.
    time.sleep(max(0.35, float(args.settle_s) + 0.25))

    print(
        f"[one] START b={q_start.base:.3f} s={q_start.shoulder:.3f} e={q_start.elbow:.3f}"
        + (
            f" (fb s={q_fb.shoulder:.3f})"
            if q_fb is not None and abs(q_fb.shoulder - q_start.shoulder) > 0.02
            else ""
        )
    )
    if target_name:
        print(f"[one] target selection: {target_name} px={args.target_px:.0f} py={args.target_py:.0f}")
    if hasattr(tracker, "preferred_px"):
        tracker.preferred_px = float(args.target_px)
    if hasattr(tracker, "preferred_py"):
        tracker.preferred_py = float(args.target_py)
    if hasattr(tracker, "preferred_max_dist_px"):
        tracker.preferred_max_dist_px = float(args.target_max_dist)
    if args.reset_target_each_attempt and hasattr(tracker, "reset_lock"):
        tracker.reset_lock()
    t_lock = time.time()
    m0 = lock_berry(
        provider,
        tracker,
        preview_first=args.preview_lock,
        max_depth_m=args.max_target_depth,
        min_conf=args.lock_min_conf,
        preferred_px=float(args.target_px),
        preferred_py=float(args.target_py),
    )
    lock_elapsed = time.time() - t_lock
    out_path = REPO / "runs/roarm_learn" / f"one_shot_{label}.json"
    lock_path = REPO / "runs/roarm_learn" / f"{label}_lock.json"
    if not m0:
        result = {
            "ok": False,
            "label": label,
            "home_pose": args.home_pose,
            "target_name": target_name,
            "attempt": attempt_idx,
            "error": "no_berry",
            "lock_elapsed_s": round(lock_elapsed, 3),
            "joints_start": q_start.as_dict(),
            "updated_at": time.time(),
        }
        ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
        episode_path = ATTEMPT_DIR / f"{label}_attempt{attempt_idx}_{int(result['updated_at'])}.json"
        result["episode_path"] = str(episode_path)
        episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        append_episode_to_learned(learned_path, label, result)
        print("[one] FAIL no_berry")
        _status("STOP", reason="no_berry", attempt=attempt_idx)
        return result
    px0, py0, d0, conf0, _ = m0
    print(f"[one] lock px={px0:.0f} py={py0:.0f} d={d0:.3f} conf={conf0:.2f}")
    print(f"[one] lock time={lock_elapsed:.3f}s")
    _status("BERRY_LOCKED", px=f"{px0:.0f}", py=f"{py0:.0f}", depth=round(float(d0), 3), conf=round(float(conf0), 2))

    lock_path.write_text(
        json.dumps(
            {
                "label": label,
                "home_pose": args.home_pose,
                "attempt": attempt_idx,
                "berry": {"px": px0, "py": py0, "depth_m": d0, "conf": conf0},
                "joints": q_start.as_dict(),
                "updated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plan_result: Optional[Dict[str, Any]] = None
    if args.target == "planner":
        learned_doc = load_or_seed_learned(learned_path, label=label)
        plan_result = plan_one_shot(
            q_start.as_dict(),
            {"px": px0, "py": py0, "depth_m": d0, "conf": conf0},
            learned_doc,
            calib_path=CALIB,
        )
        q_end = q_from_dict(plan_result["q_target"])
        if label == "dom_final" and q_end.elbow < args.min_elbow_target:
            q_end.elbow = args.min_elbow_target
            plan_result["q_target"]["elbow"] = q_end.elbow
            # Keep delta consistent with clamped elbow for STATUS / dry-run gate.
            plan_result["delta_q"]["elbow"] = float(q_end.elbow) - float(q_start.elbow)
            plan_result["max_abs_delta_q"] = round(
                max(abs(float(plan_result["delta_q"][k])) for k in ("base", "shoulder", "elbow")),
                4,
            )
            plan_result.setdefault("adjustments", []).append(
                f"elbow_clamped_to_{args.min_elbow_target:.3f}"
            )
        berry_lock = {"px": px0, "py": py0, "depth_m": d0, "conf": conf0}
        _emit_planner_status(q_start, berry_lock, plan_result)
        plan_ok, plan_why = validate_plan_motion(plan_result)
        if plan_ok:
            _status("DRY_RUN_PASS", mode=plan_result.get("planner_mode"), max_dq=plan_result.get("max_abs_delta_q"))
        else:
            _status("DRY_RUN_FAIL", reason=plan_why, mode=plan_result.get("planner_mode"))
            print(f"[one] planner REJECT motion: {plan_why}")
            result = {
                "ok": False,
                "label": label,
                "home_pose": args.home_pose,
                "target_name": target_name,
                "attempt": attempt_idx,
                "error": f"plan_rejected:{plan_why}",
                "plan": plan_result,
                "berry_before": berry_lock,
                "joints_start": q_start.as_dict(),
                "updated_at": time.time(),
            }
            ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
            episode_path = ATTEMPT_DIR / f"{label}_attempt{attempt_idx}_{int(result['updated_at'])}.json"
            result["episode_path"] = str(episode_path)
            episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            append_episode_to_learned(learned_path, label, result)
            return result
        if getattr(args, "plan_only", False):
            _status("PLAN_ONLY_STOP", ok=1)
            print("[one] plan-only: no T:102 move")
            result = {
                "ok": False,
                "label": label,
                "home_pose": args.home_pose,
                "target_name": target_name,
                "attempt": attempt_idx,
                "error": "plan_only",
                "plan": plan_result,
                "berry_before": berry_lock,
                "joints_start": q_start.as_dict(),
                "joints_target": q_end.as_dict(),
                "updated_at": time.time(),
            }
            ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
            episode_path = ATTEMPT_DIR / f"{label}_attempt{attempt_idx}_{int(result['updated_at'])}.json"
            result["episode_path"] = str(episode_path)
            episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        print(
            f"[one] planner {plan_result['reason']} pred="
            f"px={plan_result['prediction']['px']:.0f} "
            f"py={plan_result['prediction']['py']:.0f} "
            f"d={plan_result['prediction']['depth_m']:.3f}"
        )
        for warning in plan_result.get("warnings") or []:
            print(f"[one] WARN {warning}")
    elif args.target == "delta":
        delta = load_label_delta(label, cfg)
        q_end = compute_target_from_delta(
            q_start, {"px": px0, "py": py0, "depth_m": d0}, delta, label=label
        )
    elif args.target in ("learned", "last_run"):
        q_end = load_target_learned()
    else:
        q_end = load_target_yaml(cfg)

    print(f"[one] END   b={q_end.base:.3f} s={q_end.shoulder:.3f} e={q_end.elbow:.3f}")
    est_s = est_move_duration(q_start, q_end, spd=args.spd, acc=args.acc)
    print(f"[one] est ~{est_s:.1f}s → ONE move")
    t_move = time.time()
    print("[one] ▶ ONE smooth T:102 move")
    _status("BERRY_APPROACH", moving=1)
    q_cmd = approach_command_joints(q_start, q_end)
    move_t102_smooth(execute_rpc, q_cmd, spd=args.spd, acc=args.acc)
    q_done = wait_reach(
        execute_rpc,
        q_end,
        timeout_s=est_s + 6.0,
        tol=args.reach_tol,
        stable_s=0.5,
    )
    move_elapsed = time.time() - t_move
    if args.settle_s > 0:
        time.sleep(args.settle_s)

    # Restored pre-peduncle verify: no demo berry_success reanchor (that parked lock at 540,242).
    # Reset tracker lock so verify re-detects (otherwise px/py stay frozen at pre-move lock).
    if hasattr(tracker, "reset_lock"):
        tracker.reset_lock()
    m1 = None
    verify_wait_elapsed = 0.0
    if args.no_verify:
        pass
    elif getattr(args, "verify_preview", False):
        t_verify = time.time()
        m1 = verify_berry_from_preview(
            tracker,
            timeout_s=args.verify_wait_s,
            min_conf=args.verify_min_conf,
        )
        verify_wait_elapsed = time.time() - t_verify
    else:
        t_verify = time.time()
        m1 = wait_berry_standoff(
            provider,
            tracker,
            timeout_s=max(float(args.verify_wait_s), 1.2),
            frames=args.verify_frames,
            min_conf=args.verify_min_conf,
        )
        verify_wait_elapsed = time.time() - t_verify
        # If px/py identical to lock, treat as frozen-track failure and re-measure once.
        if (
            m1 is not None
            and abs(float(m1[0]) - float(px0)) < 1.0
            and abs(float(m1[1]) - float(py0)) < 1.0
        ):
            print("[one] verify px/py frozen at lock; force re-detect")
            if hasattr(tracker, "reset_lock"):
                tracker.reset_lock()
            time.sleep(0.2)
            m1 = wait_berry_standoff(
                provider,
                tracker,
                timeout_s=1.5,
                frames=max(1, int(args.verify_frames)),
                min_conf=args.verify_min_conf,
            )
    q_final = q_done or read_q(execute_rpc) or q_end
    result: Dict[str, Any] = {
        "ok": False,
        "label": label,
        "home_pose": args.home_pose,
        "target_name": target_name,
        "attempt": attempt_idx,
        "motion": "T102_one_shot",
        "spd": args.spd,
        "acc": args.acc,
        "move_elapsed_s": round(move_elapsed, 2),
        "lock_elapsed_s": round(lock_elapsed, 3),
        "verify_wait_elapsed_s": round(verify_wait_elapsed, 2),
        "cycle_elapsed_s": round(time.time() - cycle_t0, 2),
        "target_source": args.target,
        "joints_start": q_start.as_dict(),
        "joints_target": q_end.as_dict(),
        "joints_end": q_final.as_dict(),
        "joint_err_rad": round(joint_error(q_final, q_end), 4),
        "berry_before": {"px": px0, "py": py0, "depth_m": d0, "conf": conf0},
        "plan": plan_result,
        "updated_at": time.time(),
    }

    if m1:
        px1, py1, d1, conf1, _ = m1
        result["berry_after"] = {"px": px1, "py": py1, "depth_m": d1, "conf": conf1}
        expected = (plan_result or {}).get("target_image") or {}
        if label == "dom_final" and (load_or_seed_learned(learned_path, label=label).get("seeded_from")):
            expected = {}
        if not args.no_verify:
            ok, checks = verify_success(
                result["berry_before"],
                result["berry_after"],
                result["joints_start"],
                result["joints_target"],
                result["joints_end"],
                expected_px=expected.get("px"),
                expected_py=expected.get("py"),
                min_conf=args.verify_min_conf,
                depth_max_m=args.verify_depth_max,
            )
            result["ok"] = bool(ok)
            result["verify"] = checks
        else:
            result["ok"] = True
            result["verify"] = {"skipped": True, "depth_m": d1}
        print(
            f"[one] after px={px1:.0f} py={py1:.0f} d={d1:.3f} "
            f"({'OK' if result['ok'] else 'check'}) move={move_elapsed:.1f}s "
            f"cycle={result['cycle_elapsed_s']:.1f}s"
        )
        if result.get("verify", {}).get("reasons"):
            print(f"[one] verify reasons: {','.join(result['verify']['reasons'])}")
        _status(
            "BERRY_APPROACH",
            ok=int(bool(result.get("ok"))),
            depth=round(float(d1), 3),
            px=f"{px1:.0f}",
            py=f"{py1:.0f}",
            tgt_s=round(float(q_end.shoulder), 3),
            end_s=round(float(q_final.shoulder), 3),
        )
    else:
        result["berry_after"] = None
        if args.no_verify:
            result["ok"] = move_elapsed > 0.1
            result["verify"] = {"skipped": True}
            print(f"[one] no-verify move={move_elapsed:.1f}s cycle={result['cycle_elapsed_s']:.1f}s")
            _status("BERRY_APPROACH", ok=int(bool(result.get("ok"))), verify="skipped")
        else:
            print("[one] WARN berry lost after move")
            _status("STOP", reason="berry_lost_after_move")

    result["cycle_elapsed_s"] = round(time.time() - cycle_t0, 2)

    # Peduncle stage only after berry approach success + close standoff (no cut).
    use_peduncle_grasp = bool(getattr(args, "peduncle_grasp", False))
    use_peduncle_v3 = bool(
        getattr(args, "peduncle_v3", False)
        or getattr(args, "peduncle", False)
        or use_peduncle_grasp
    )
    use_legacy_obb = bool(getattr(args, "peduncle_legacy", False)) and not use_peduncle_grasp
    if use_peduncle_grasp:
        from pipelines.peduncle_v3.grasp_sequence import run_peduncle_grasp_sequence

        run_peduncle_grasp_sequence(
            args=args,
            result=result,
            provider=provider,
            tracker=tracker,
            execute_rpc=execute_rpc,
            plan_one_shot=plan_one_shot,
            move_t102_smooth=move_t102_smooth,
            q_from_dict=q_from_dict,
            read_q=read_q,
            load_or_seed_learned=load_or_seed_learned,
            learned_path=learned_path,
            label=label,
            attempt_idx=attempt_idx,
            q_final=q_final,
            calib_path=CALIB,
            gripper_open=GRIPPER_OPEN,
        )
    elif use_peduncle_v3 or use_legacy_obb:
        result["stage"] = StageState.BERRY_APPROACH.value if result.get("ok") else StageState.BERRY_LOCKED.value
        depth_for_peduncle = None
        px_p = py_p = None
        conf_p = None
        if result.get("berry_after"):
            depth_for_peduncle = result["berry_after"].get("depth_m")
            px_p = result["berry_after"].get("px")
            py_p = result["berry_after"].get("py")
            conf_p = result["berry_after"].get("conf")

        v3_cfg = load_peduncle_v3_config()
        dmin = float(v3_cfg.get("depth_min_m", 0.08))
        dmax = float(v3_cfg.get("depth_max_m", 0.17))
        require_ok = bool(v3_cfg.get("require_approach_ok", True))

        if require_ok and not result.get("ok"):
            result["peduncle_v3"] = {
                "ran": False,
                "reason": "approach_not_ok",
                "state": StageState.STOP.value,
                "depth_m": depth_for_peduncle,
            }
            print("[one] peduncle_v3 SKIP approach_not_ok → STOP")
        elif not depth_in_peduncle_gate(depth_for_peduncle, depth_min_m=dmin, depth_max_m=dmax):
            result["peduncle_v3"] = {
                "ran": False,
                "reason": "depth_out_of_gate",
                "state": StageState.STOP.value,
                "depth_m": depth_for_peduncle,
                "gate_m": [dmin, dmax],
            }
            print(
                f"[one] peduncle_v3 SKIP depth={depth_for_peduncle} "
                f"(need {dmin:.2f}–{dmax:.2f}m)"
            )
        else:
            bgr = None
            try:
                from scripts.roarm_strawberry_approach import _fetch_hub_bgr

                bgr, _ = _fetch_hub_bgr("http://127.0.0.1:8080")
            except Exception:
                bgr = None
            if bgr is None:
                try:
                    pair = provider.read(timeout_s=1.5)
                    bgr = pair.rgb_bgr if pair is not None else None
                except Exception:
                    bgr = None
            if bgr is None:
                result["peduncle_v3"] = {
                    "ran": False,
                    "reason": "no_frame",
                    "state": StageState.STOP.value,
                    "depth_m": depth_for_peduncle,
                }
                print("[one] peduncle_v3 SKIP no_frame")
            elif use_peduncle_v3:
                from pipelines.peduncle_v3.framing import run_peduncle_framing

                log_dir = REPO / str((v3_cfg.get("logging") or {}).get("dir") or "runs/peduncle_v3_runtime")
                # Avoid demo verify-anchor stuck on image edge (false lock).
                pref_x = float(px_p if px_p is not None else 320)
                pref_y = float(py_p if py_p is not None else 240)
                if pref_x > 560 or pref_x < 80 or pref_y < 40:
                    bb = result.get("berry_before") or {}
                    if bb.get("px") is not None:
                        pref_x, pref_y = float(bb["px"]), float(bb["py"])
                    else:
                        pref_x, pref_y = 320.0, 240.0
                framing = run_peduncle_framing(
                    provider=provider,
                    tracker=tracker,
                    execute_rpc=execute_rpc,
                    move_t102_smooth=move_t102_smooth,
                    read_q=read_q,
                    q_now=q_final,
                    preferred_px=pref_x,
                    preferred_py=pref_y,
                    depth_hint=float(depth_for_peduncle) if depth_for_peduncle is not None else None,
                    v3_cfg=v3_cfg,
                    log_dir=log_dir,
                    label=label,
                    attempt_idx=attempt_idx,
                    depth_max_m=dmax,
                    depth_min_m=dmin,
                    spd=float(args.spd),
                    acc=float(args.acc),
                )
                result["peduncle_framing"] = framing
                if not framing.get("ok"):
                    result["peduncle_v3"] = {
                        "ran": False,
                        "reason": f"framing_failed:{framing.get('reason')}",
                        "state": StageState.ABORT.value,
                        "depth_m": (framing.get("berry") or {}).get("depth_m", depth_for_peduncle),
                    }
                    result["stage"] = StageState.ABORT.value
                    print(f"[one] peduncle_v3 SKIP framing:{framing.get('reason')}")
                else:
                    berry_f = framing.get("berry") or {}
                    if berry_f.get("px") is not None:
                        px_p, py_p = float(berry_f["px"]), float(berry_f["py"])
                        conf_p = berry_f.get("conf", conf_p)
                    if berry_f.get("depth_m") is not None:
                        depth_for_peduncle = float(berry_f["depth_m"])
                    result["stage"] = StageState.PEDUNCLE_ACQUIRE.value
                    if framing.get("bbox") and len(framing["bbox"]) == 4:
                        bbox = tuple(float(v) for v in framing["bbox"])
                    else:
                        half = max(36.0, 0.08 * float(bgr.shape[1]))
                        if px_p is not None and py_p is not None:
                            bbox = (px_p - half, py_p - half, px_p + half, py_p + half)
                        else:
                            bbox = (0.0, 0.0, float(bgr.shape[1]), float(bgr.shape[0]))
                    # Fresh frame after framing
                    try:
                        from scripts.roarm_strawberry_approach import _fetch_hub_bgr

                        bgr2, _ = _fetch_hub_bgr("http://127.0.0.1:8080")
                        if bgr2 is not None:
                            bgr = bgr2
                    except Exception:
                        pass
                    pipe = getattr(args, "_peduncle_v3", None)
                    if pipe is None:
                        pipe = PeduncleV3Pipeline(cfg=v3_cfg)
                        args._peduncle_v3 = pipe
                    ped = pipe.process_focused_berry(
                        bgr, bbox, depth_m=depth_for_peduncle, draw_overlay=True
                    )
                    payload = ped.as_dict()
                    payload["ran"] = True
                    payload["framing"] = framing
                    payload["berry_lock"] = {
                        "px": px_p,
                        "py": py_p,
                        "conf": conf_p,
                        "depth_m": depth_for_peduncle,
                    }
                    result["stage"] = ped.state.value
                    result["peduncle_v3"] = payload
                    # Persist overlay + JSON attempt log
                    log_dir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    overlay_path = log_dir / f"{label}_attempt{attempt_idx}_{ts}_overlay.jpg"
                    json_path = log_dir / f"{label}_attempt{attempt_idx}_{ts}.json"
                    if ped.overlay_bgr is not None:
                        import cv2

                        cv2.imwrite(str(overlay_path), ped.overlay_bgr)
                        payload["overlay_path"] = str(overlay_path)
                    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    payload["log_path"] = str(json_path)
                    result["peduncle_v3"] = payload
                    decision = payload.get("decision")
                    print(
                        f"[one] peduncle_v3 {ped.state.value} decision={decision} "
                        f"reason={ped.reason} n={payload.get('n_peduncle')} "
                        f"t={payload.get('timings_ms', {}).get('total_ms')}ms"
                    )
                    _status(
                        StageState.PEDUNCLE_ACQUIRE.value,
                        state=ped.state.value,
                        decision=decision,
                        calyx=1 if ped.calyx_xy else 0,
                    )

                    # Optional safe stem standoff (no cut). Default OFF.
                    if (
                        getattr(args, "peduncle_standoff", False)
                        and ped.state == StageState.PEDUNCLE_CONFIRMED
                        and ped.association
                        and ped.association.selected is not None
                    ):
                        wp = ped.association.selected.work_point
                        from pipelines.roarm_perception import PerceptionConfig, sample_depth_median, pixel_to_camera_xyz

                        stem_depth = None
                        cam_xyz = None
                        try:
                            pair = provider.read(timeout_s=1.5)
                        except Exception:
                            pair = None
                        if pair is not None:
                            depth_img = pair.depth_m
                            if depth_img is not None:
                                stem_depth, _why = sample_depth_median(
                                    depth_img,
                                    int(round(wp[0])),
                                    int(round(wp[1])),
                                    radius=4,
                                    cfg=PerceptionConfig(),
                                )
                            K = None
                            try:
                                K = provider.get_intrinsics()
                            except Exception:
                                K = None
                            if stem_depth and K:
                                cam_xyz = pixel_to_camera_xyz(
                                    int(round(wp[0])),
                                    int(round(wp[1])),
                                    float(stem_depth),
                                    K,
                                )
                        standoff_info: Dict[str, Any] = {
                            "requested": True,
                            "executed": False,
                            "work_point_px": [round(wp[0], 1), round(wp[1], 1)],
                            "stem_depth_m": stem_depth,
                            "camera_xyz_m": [round(v, 4) for v in cam_xyz] if cam_xyz else None,
                        }
                        try:
                            tracker.reanchor(float(wp[0]), float(wp[1]))
                            q_now = read_q(execute_rpc) or q_final
                            learned_doc = load_or_seed_learned(learned_path, label=label)
                            plan2 = plan_one_shot(
                                q_now.as_dict() if hasattr(q_now, "as_dict") else q_now,
                                {
                                    "px": float(wp[0]),
                                    "py": float(wp[1]),
                                    "depth_m": float(stem_depth or depth_for_peduncle or 0.12),
                                    "conf": float(ped.association.selected.score),
                                },
                                learned_doc,
                                calib_path=CALIB,
                                standoff_m=float((v3_cfg.get("standoff") or {}).get("standoff_depth_m", 0.12)),
                            )
                            q_stem = q_from_dict(plan2["q_target"])
                            standoff_info["plan"] = {
                                "reason": plan2.get("reason"),
                                "prediction": plan2.get("prediction"),
                                "target_image": plan2.get("target_image"),
                                "delta_q": plan2.get("delta_q"),
                            }
                            if not args.dry_run:
                                print(
                                    f"[one] peduncle_standoff → stem px=({wp[0]:.0f},{wp[1]:.0f}) "
                                    f"d={stem_depth} (NO CUT)"
                                )
                                move_t102_smooth(
                                    execute_rpc,
                                    q_stem,
                                    spd=min(args.spd, 0.12),
                                    acc=min(args.acc, 8.0),
                                )
                                standoff_info["executed"] = True
                                result["stage"] = StageState.STOP.value
                            else:
                                standoff_info["reason"] = "dry_run"
                        except Exception as exc:
                            standoff_info["error"] = f"{type(exc).__name__}: {exc}"
                            result["stage"] = StageState.STOP.value
                            print(f"[one] peduncle_standoff STOP/err: {exc}")
                        result["peduncle_standoff"] = standoff_info
                    elif getattr(args, "peduncle_standoff", False):
                        result["peduncle_standoff"] = {
                            "requested": True,
                            "executed": False,
                            "reason": "NO_ASSOC_or_not_confirmed",
                        }
                        print("[one] peduncle_standoff skipped (no TARGET) → STOP")
            else:
                # Legacy OBB-only path (no association).
                ped_det = getattr(args, "_peduncle_detector", None) or PeduncleDetector()
                args._peduncle_detector = ped_det
                half = 40.0
                bbox = None
                if px_p is not None and py_p is not None:
                    bbox = (px_p - half, py_p - half, px_p + half, py_p + half)
                ped = ped_det.detect(bgr, depth_m=depth_for_peduncle, berry_bbox=bbox)
                result["peduncle"] = ped.as_dict()
                if ped.ran and ped.hits:
                    top = ped.hits[0]
                    print(
                        f"[one] peduncle-legacy OK n={len(ped.hits)} conf={top.conf:.2f} "
                        f"xy=({top.cx:.0f},{top.cy:.0f}) t={ped.elapsed_s:.2f}s"
                    )
                else:
                    print(f"[one] peduncle-legacy {ped.reason} t={ped.elapsed_s:.2f}s")

    ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
    episode_path = ATTEMPT_DIR / f"{label}_attempt{attempt_idx}_{int(result['updated_at'])}.json"
    episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["episode_path"] = str(episode_path)
    episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    append_episode_to_learned(learned_path, label, result)
    print(f"[one] saved → {episode_path}")

    if args.return_home:
        print(f"[one] return → {args.home_pose}")
        # Same full-travel + sticky-shoulder recovery as attempt start.
        ret_timeout = est_move_duration(
            q_final, q_home_move, spd=args.return_spd, acc=args.return_acc
        ) + 6.0
        ret_timeout = max(ret_timeout, 14.0)
        ret_timeout = min(ret_timeout, 28.0)
        q_ret, err = ensure_home_reached(
            execute_rpc,
            q_home_move,
            spd=args.return_spd,
            acc=args.return_acc,
            tol=float(args.return_tol),
            timeout_s=ret_timeout,
        )
        if err > float(args.return_tol):
            print(f"[one] WARN return incomplete err={err:.3f}; not marking at_home")
            result["return_home_ok"] = False
        else:
            result["return_home_ok"] = True
        time.sleep(args.settle_s)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        choices=("planner", "learned", "yaml", "delta", "last_run"),
        default="planner",
    )
    ap.add_argument("--spd", type=float, default=0.08, help="T:102 speed (lower = smoother)")
    ap.add_argument("--acc", type=float, default=3.5, help="T:102 accel (lower = smoother)")
    ap.add_argument("--return-spd", type=float, default=0.10, help="Return-to-home T:102 speed")
    ap.add_argument("--return-acc", type=float, default=4.5, help="Return-to-home T:102 accel")
    ap.add_argument("--home-pose", default="DOM_FINAL", help="Pose name from config/roarm_home_joints.yaml")
    ap.add_argument("--repeat", type=int, default=1, help="Number of attempts from the fixed home pose")
    ap.add_argument("--target-px", type=float, default=250.0, help="Preferred berry x when multiple berries are visible")
    ap.add_argument("--target-py", type=float, default=145.0, help="Preferred berry y when multiple berries are visible")
    ap.add_argument("--target-sequence", default="", help="Comma list name:px:py; runs all in one process")
    ap.add_argument("--target-max-dist", type=float, default=190.0, help="Reject initial lock farther than this from preferred target")
    ap.add_argument("--max-target-depth", type=float, default=0.80,
                    help="Only consider berries at or nearer than this distance")
    ap.add_argument("--preview-lock", dest="preview_lock", action="store_true", default=True,
                    help="Use warm fleet-agent detections and choose nearest valid depth")
    ap.add_argument("--no-preview-lock", dest="preview_lock", action="store_false",
                    help="Force local detector lock instead of warm preview cache")
    ap.add_argument("--lock-radius-px", type=float, default=85.0, help="After lock, ignore other berries outside this radius")
    ap.add_argument("--min-elbow-target", type=float, default=1.35, help="Clamp target elbow lower bound for DOM_FINAL standoff")
    ap.add_argument("--lock-min-conf", type=float, default=min_berry_conf(),
                    help="Minimum detector confidence to lock a berry")
    ap.add_argument("--verify-min-conf", type=float, default=min_berry_conf(),
                    help="Minimum detector confidence for close-range verify")
    ap.add_argument("--verify-depth-max", type=float, default=0.17, help="Maximum accepted standoff depth")
    ap.add_argument(
        "--verify-depth-samples",
        type=int,
        default=5,
        help="Fresh depth samples for median after approach (stable verify)",
    )
    ap.add_argument(
        "--micro-depth-slack-m",
        type=float,
        default=0.030,
        help="If median depth in (max, max+slack], allow one REACH micro-approach (gate unchanged)",
    )
    ap.add_argument(
        "--micro-aim-depth-m",
        type=float,
        default=0.14,
        help="Target depth for micro REACH (center of standoff band)",
    )
    ap.add_argument(
        "--micro-max-step-rad",
        type=float,
        default=0.08,
        help="Cap for one explicit_reach_delta micro step",
    )
    ap.add_argument("--reach-tol", type=float, default=0.045, help="Joint tolerance for approach completion")
    ap.add_argument("--return-tol", type=float, default=0.085, help="Joint tolerance for return-home completion")
    ap.add_argument("--settle-s", type=float, default=0.12, help="Short camera settle after each move")
    ap.add_argument("--verify-wait-s", type=float, default=0.9,
                    help="Wait for camera depth to settle near standoff")
    ap.add_argument("--verify-frames", type=int, default=1,
                    help="Frames per close-range verification sample (1 is fastest)")
    ap.add_argument("--return-wait-s", type=float, default=5.0, help="Max wait for return-home reach in sequence")
    ap.add_argument("--reset-target-each-attempt", dest="reset_target_each_attempt", action="store_true", default=True)
    ap.add_argument("--keep-target-between-attempts", dest="reset_target_each_attempt", action="store_false")
    ap.add_argument("--no-return-home", dest="return_home", action="store_false", help="Do not return home after each attempt")
    ap.set_defaults(return_home=True)
    ap.add_argument("--skip-home", action="store_true", help="Assume already at start pose (test4)")
    ap.add_argument("--from-current", action="store_true", help="Start joints = T:105 now (no HOME2)")
    ap.add_argument("--label", default="", help="e.g. test4 for artifacts")
    ap.add_argument("--recover-label", default="", help="T102 back to {label}_lock joints first")
    ap.add_argument("--lock-only", action="store_true",
                    help="Perception only: lock berry and exit (no arm motion)")
    ap.add_argument("--all-berries", action="store_true",
                    help="Auto-discover every visible berry (left→right) and approach each")
    ap.add_argument("--fast", action="store_true",
                    help="Speed preset: spd=0.22 acc=12 verify=0.25s settle=0.05")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip close-range verify wait (benchmark move only)")
    ap.add_argument("--verify-preview", action="store_true",
                    help="Fast verify via fleet-agent preview cache (no local YOLO)")
    ap.add_argument(
        "--peduncle",
        action="store_true",
        help="After successful berry approach: run peduncle v3 (calyx+OBB+assoc). No cut.",
    )
    ap.add_argument(
        "--peduncle-v3",
        action="store_true",
        help="Alias of --peduncle (explicit v3 pipeline).",
    )
    ap.add_argument(
        "--peduncle-legacy",
        action="store_true",
        help="Legacy OBB-only peduncle detect (no calyx/association).",
    )
    ap.add_argument(
        "--peduncle-standoff",
        action="store_true",
        help="After TARGET stem: one safe standoff move to stem work-point, then STOP (no cut).",
    )
    ap.add_argument(
        "--peduncle-grasp",
        action="store_true",
        help=(
            "Opt-in: berry approach (default DOM_FINAL learned)→peduncle v3→temporal→standoff→"
            "grasp approach→gentle gripper close→verify→STOP (NO cut / NO yank)."
        ),
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="Go to home pose, lock berry, run planner STATUS/dry-run gate, do not move to target.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.peduncle_grasp:
        # Keep calibrated DOM_FINAL start (has demos). Do NOT rewrite to HOME:
        # home_learned has 0 demos → zero prior → zero motion.
        if not getattr(args, "_return_home_explicit", False):
            if "--no-return-home" not in sys.argv and "--return-home" not in sys.argv:
                args.return_home = False
        args.peduncle = True
    if args.fast:
        args.spd = 0.22
        args.acc = 12.0
        args.verify_wait_s = 0.25
        args.settle_s = 0.05
        args.reach_tol = 0.09
        args.return_spd = 0.18
        args.return_acc = 10.0
        args.return_wait_s = 4.0

    _load_fleet_env()
    cfg = yaml.safe_load((REPO / "config/roarm_home_joints.yaml").read_text()) or {}
    if args.home_pose not in cfg:
        raise KeyError(f"home pose {args.home_pose!r} not found in config/roarm_home_joints.yaml")
    q_home = q_named_cfg(cfg, args.home_pose, force_open=bool(args.peduncle_grasp))
    label = args.label or args.home_pose.lower()
    learned_path = REPO / "runs/roarm_learn" / f"{label}_learned.json"

    from roarm_proxy import execute_rpc, reset_client

    reset_client()
    provider = Ros2RgbDepthProvider(
        rgb_topic=STEREO_RGB, depth_topic=STEREO_DEPTH, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=STEREO_INFO)
    tracker = StrawberryTargetTracker(
        preferred_px=args.target_px,
        preferred_py=args.target_py,
        preferred_max_dist_px=args.target_max_dist,
        strict_lock=True,
        lock_radius_px=args.lock_radius_px,
    )

    print(
        f"[one] repeat approach label={label} home={args.home_pose} repeat={args.repeat} "
        f"spd={args.spd} acc={args.acc} target={args.target}"
    )

    if args.dry_run:
        print(f"[one] dry-run learned={learned_path}")
        provider.close()
        return 0

    if args.lock_only:
        t_lock = time.time()
        m0 = lock_berry(
            provider,
            tracker,
            preview_first=args.preview_lock,
            max_depth_m=args.max_target_depth,
            min_conf=args.lock_min_conf,
        )
        lock_elapsed = time.time() - t_lock
        if not m0:
            print(f"[one] lock-only FAIL no_berry (min_conf={args.lock_min_conf:.2f}) t={lock_elapsed:.3f}s")
            provider.close()
            return 1
        px0, py0, d0, conf0, _ = m0
        print(
            f"[one] lock-only OK px={px0:.0f} py={py0:.0f} d={d0:.3f} "
            f"conf={conf0:.2f} t={lock_elapsed:.3f}s"
        )
        provider.close()
        return 0

    if args.recover_label:
        q_rec = load_recover_joints(args.recover_label)
        if q_rec:
            print(f"[one] recover → {args.recover_label} start pose (learned)")
            move_t102_smooth(execute_rpc, q_rec, spd=args.return_spd, acc=args.return_acc)
            time.sleep(1.2)
    results = []
    try:
        load_or_seed_learned(learned_path, label=label)
        # Berry monitoring always starts from home pose (DOM_FINAL / dashboard «дом»).
        # Discover AFTER arriving — current camera pose may be elsewhere.
        at_home = bool(args.skip_home)
        if args.all_berries and not args.skip_home:
            q_home_move = JointState(
                base=q_home.base,
                shoulder=q_home.shoulder,
                elbow=q_home.elbow,
                wrist=q_home.wrist,
                roll=q_home.roll,
                hand=GRIPPER_OPEN,
            )
            q_now = read_q(execute_rpc) or q_home_move
            print(f"[one] go monitoring pose → {args.home_pose} before berry discover")
            move_t102_smooth(execute_rpc, q_home_move, spd=args.return_spd, acc=args.return_acc)
            home_timeout = est_move_duration(
                q_now, q_home_move, spd=args.return_spd, acc=args.return_acc
            ) + 4.0
            home_timeout = max(home_timeout, float(args.return_wait_s), 12.0)
            home_timeout = min(home_timeout, 28.0)
            wait_reach(
                execute_rpc,
                q_home_move,
                timeout_s=home_timeout,
                tol=args.return_tol,
            )
            time.sleep(max(0.35, float(args.settle_s)))
            at_home = True
        if args.all_berries:
            targets = discover_berry_targets(
                max_depth_m=args.max_target_depth,
                min_conf=args.lock_min_conf,
            )
            if not targets:
                print("[one] FAIL no berries for sweep (after monitoring pose)")
                return 1
            print(f"[one] berry sweep: {len(targets)} targets (L→R)")
            for name, px, py in targets:
                print(f"  {name}: px={px:.0f} py={py:.0f}")
        else:
            targets = parse_target_sequence(args.target_sequence, args.target_px, args.target_py)
        attempt_idx = 0
        for target_name, target_px, target_py in targets:
            args.target_px = target_px
            args.target_py = target_py
            repeats = 1 if args.all_berries else max(1, args.repeat)
            for _ in range(repeats):
                attempt_idx += 1
                result = run_attempt(
                    attempt_idx=attempt_idx,
                    execute_rpc=execute_rpc,
                    provider=provider,
                    tracker=tracker,
                    q_home=q_home,
                    cfg=cfg,
                    args=args,
                    label=label,
                    learned_path=learned_path,
                    target_name=target_name,
                    at_home=at_home,
                )
                results.append(result)
                # Only skip next home go if return actually reached DOM_FINAL.
                # (Old bug: at_home=True after every return_home even when wait
                # timed out at ~5s mid-travel → plan from drifted shoulder.)
                at_home = bool(args.return_home) and bool(result.get("return_home_ok"))
                if not result.get("ok") and (repeats > 1 or len(targets) > 1):
                    print(f"[one] attempt {attempt_idx} ({target_name}) not OK; continuing sequence")
    finally:
        provider.close()

    ok_count = sum(1 for r in results if r.get("ok"))
    cycles = [float(r.get("cycle_elapsed_s", 0.0)) for r in results if r.get("cycle_elapsed_s")]
    summary = {
        "label": label,
        "home_pose": args.home_pose,
        "repeat": max(1, args.repeat),
        "all_berries": bool(args.all_berries),
        "fast": bool(args.fast),
        "ok_count": ok_count,
        "cycle_elapsed_s": {
            "min": min(cycles) if cycles else None,
            "max": max(cycles) if cycles else None,
            "mean": round(sum(cycles) / len(cycles), 2) if cycles else None,
        },
        "results": results,
        "updated_at": time.time(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[one] summary ok={ok_count}/{len(results)} → {SUMMARY}")
    return 0 if results and all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
