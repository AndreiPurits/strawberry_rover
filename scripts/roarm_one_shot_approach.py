#!/usr/bin/env python3
"""One smooth T:102 move: HOME2 → approach target (joint space).

Bench: target from last learned success joints.
Prod (later): target computed from strawberry lock at HOME2.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    verify_success,
    write_learned,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from scripts.roarm_jacobian_probe import (  # noqa: E402
    STEREO_DEPTH,
    STEREO_INFO,
    STEREO_RGB,
    _joints_params,
    clamp_joint,
    read_q,
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
GRIPPER_OPEN = 1.08

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
) -> Optional[JointState]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        q = read_q(execute_rpc)
        if q and joint_error(q, q_tgt) <= tol:
            return q
        time.sleep(0.12)
    return read_q(execute_rpc)


def lock_berry(provider, tracker) -> Optional[Tuple]:
    for _ in range(14):
        m = measure(provider, tracker, "http://127.0.0.1:8080", frames=2)
        if m:
            return m
        time.sleep(0.2)
    return None


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
) -> Dict[str, Any]:
    q_home_move = JointState(
        base=q_home.base,
        shoulder=q_home.shoulder,
        elbow=q_home.elbow,
        wrist=q_home.wrist,
        roll=q_home.roll,
        hand=GRIPPER_OPEN,
    )
    print(f"[one] attempt {attempt_idx}/{args.repeat} → {args.home_pose} (open grip)")
    move_t102_smooth(execute_rpc, q_home_move, spd=args.return_spd, acc=args.return_acc)
    q_start = wait_reach(execute_rpc, q_home_move, timeout_s=est_move_duration(q_home_move, q_home_move, spd=args.return_spd, acc=args.return_acc) + 4.0)
    q_start = q_start or read_q(execute_rpc) or q_home_move
    q_start.hand = GRIPPER_OPEN

    print(
        f"[one] START b={q_start.base:.3f} s={q_start.shoulder:.3f} e={q_start.elbow:.3f}"
    )
    m0 = lock_berry(provider, tracker)
    out_path = REPO / "runs/roarm_learn" / f"one_shot_{label}.json"
    lock_path = REPO / "runs/roarm_learn" / f"{label}_lock.json"
    if not m0:
        result = {
            "ok": False,
            "label": label,
            "home_pose": args.home_pose,
            "attempt": attempt_idx,
            "error": "no_berry",
            "joints_start": q_start.as_dict(),
            "updated_at": time.time(),
        }
        ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
        episode_path = ATTEMPT_DIR / f"{label}_attempt{attempt_idx}_{int(result['updated_at'])}.json"
        result["episode_path"] = str(episode_path)
        episode_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        append_episode_to_learned(learned_path, label, result)
        print("[one] FAIL no_berry")
        return result
    px0, py0, d0, conf0, _ = m0
    print(f"[one] lock px={px0:.0f} py={py0:.0f} d={d0:.3f} conf={conf0:.2f}")

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
            plan_result.setdefault("adjustments", []).append(
                f"elbow_clamped_to_{args.min_elbow_target:.3f}"
            )
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
    move_t102_smooth(execute_rpc, q_end, spd=args.spd, acc=args.acc)
    q_done = wait_reach(execute_rpc, q_end, timeout_s=est_s + 6.0)
    move_elapsed = time.time() - t_move
    time.sleep(0.35)

    m1 = lock_berry(provider, tracker)
    q_final = q_done or read_q(execute_rpc) or q_end
    result: Dict[str, Any] = {
        "ok": False,
        "label": label,
        "home_pose": args.home_pose,
        "attempt": attempt_idx,
        "motion": "T102_one_shot",
        "spd": args.spd,
        "acc": args.acc,
        "move_elapsed_s": round(move_elapsed, 2),
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
        ok, checks = verify_success(
            result["berry_before"],
            result["berry_after"],
            result["joints_start"],
            result["joints_target"],
            result["joints_end"],
            expected_px=expected.get("px"),
            expected_py=expected.get("py"),
            min_conf=args.verify_min_conf,
        )
        result["ok"] = bool(ok)
        result["verify"] = checks
        print(
            f"[one] after px={px1:.0f} py={py1:.0f} d={d1:.3f} "
            f"({'OK' if result['ok'] else 'check'}) t={move_elapsed:.1f}s"
        )
        if checks.get("reasons"):
            print(f"[one] verify reasons: {','.join(checks['reasons'])}")
    else:
        result["berry_after"] = None
        print("[one] WARN berry lost after move")

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
        move_t102_smooth(execute_rpc, q_home_move, spd=args.return_spd, acc=args.return_acc)
        wait_reach(execute_rpc, q_home_move, timeout_s=est_move_duration(q_final, q_home_move, spd=args.return_spd, acc=args.return_acc) + 6.0)
        time.sleep(0.25)
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
    ap.add_argument("--min-elbow-target", type=float, default=1.35, help="Clamp target elbow lower bound for DOM_FINAL standoff")
    ap.add_argument("--verify-min-conf", type=float, default=0.30, help="Minimum detector confidence for close-range verify")
    ap.add_argument("--no-return-home", dest="return_home", action="store_false", help="Do not return home after each attempt")
    ap.set_defaults(return_home=True)
    ap.add_argument("--skip-home", action="store_true", help="Assume already at start pose (test4)")
    ap.add_argument("--from-current", action="store_true", help="Start joints = T:105 now (no HOME2)")
    ap.add_argument("--label", default="", help="e.g. test4 for artifacts")
    ap.add_argument("--recover-label", default="", help="T102 back to {label}_lock joints first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _load_fleet_env()
    cfg = yaml.safe_load((REPO / "config/roarm_home_joints.yaml").read_text()) or {}
    if args.home_pose not in cfg:
        raise KeyError(f"home pose {args.home_pose!r} not found in config/roarm_home_joints.yaml")
    q_home = q_named_cfg(cfg, args.home_pose, force_open=False)
    label = args.label or args.home_pose.lower()
    learned_path = REPO / "runs/roarm_learn" / f"{label}_learned.json"

    from roarm_proxy import execute_rpc, reset_client

    reset_client()
    provider = Ros2RgbDepthProvider(
        rgb_topic=STEREO_RGB, depth_topic=STEREO_DEPTH, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=STEREO_INFO)
    tracker = StrawberryTargetTracker()

    print(
        f"[one] repeat approach label={label} home={args.home_pose} repeat={args.repeat} "
        f"spd={args.spd} acc={args.acc} target={args.target}"
    )

    if args.dry_run:
        print(f"[one] dry-run learned={learned_path}")
        provider.close()
        return 0

    if args.recover_label:
        q_rec = load_recover_joints(args.recover_label)
        if q_rec:
            print(f"[one] recover → {args.recover_label} start pose (learned)")
            move_t102_smooth(execute_rpc, q_rec, spd=args.return_spd, acc=args.return_acc)
            time.sleep(2.5)
    results = []
    try:
        load_or_seed_learned(learned_path, label=label)
        for attempt_idx in range(1, max(1, args.repeat) + 1):
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
            )
            results.append(result)
            if not result.get("ok") and args.repeat > 1:
                print(f"[one] attempt {attempt_idx} not OK; continuing repeat loop")
    finally:
        provider.close()

    ok_count = sum(1 for r in results if r.get("ok"))
    summary = {
        "label": label,
        "home_pose": args.home_pose,
        "repeat": max(1, args.repeat),
        "ok_count": ok_count,
        "results": results,
        "updated_at": time.time(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[one] summary ok={ok_count}/{len(results)} → {SUMMARY}")
    return 0 if results and all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
