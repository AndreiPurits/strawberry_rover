#!/usr/bin/env python3
"""RoArm visual approach to a detected strawberry from HOME2 (eye-in-hand).

Phases (stage-1 Jacobian from runs/roarm_kinematics/calibration.json):
  CENTER — berry bbox center → image center via Jc_inv (base + pitch).
  REACH  — reduce camera→berry depth toward standoff (shoulder+ / elbow−).

Detection uses the same hub JPEG letterbox as the website (640×480).
Depth from ROS stereo, letterboxed to match.

Arm only — never rover drive / Mega.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ops/axm-monitor/agent") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ops/axm-monitor/agent"))

import yaml  # noqa: E402

from pipelines.roarm_kinematics import JointState  # noqa: E402
from pipelines.roarm_motion import PoseRef  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from pipelines.roarm_strawberry_target import (  # noqa: E402
    StrawberryTargetTracker,
    detect_strawberry_in_frame,
    min_berry_conf,
    _fetch_hub_bgr,
    _filter_padding,
    _get_detector,
    _letterbox_array,
)

from scripts.roarm_jacobian_probe import (  # noqa: E402
    STEREO_RGB,
    STEREO_DEPTH,
    STEREO_INFO,
    _joints_params,
    _pose_params,
    clamp_joint,
    read_q,
)

GRIPPER_OPEN_RAD = 1.08


def _open_gripper(execute_rpc) -> None:
    execute_rpc("gripper_open", {})
    time.sleep(0.35)


def _joints_open(q: JointState) -> JointState:
    q.hand = GRIPPER_OPEN_RAD
    return q


def _load_fleet_env() -> None:
    env_file = Path(os.environ.get("AXM_FLEET_ENV", Path.home() / ".config/axm/fleet-agent.env"))
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def calibrate_extend_vertical(
    provider: Ros2RgbDepthProvider,
    tracker: StrawberryTargetTracker,
    local_web: str,
    execute_rpc,
    q: JointState,
    *,
    delta_rad: float = 0.02,
    settle_s: float = 0.7,
    frames: int = 2,
) -> tuple:
    """Probe shoulder+/elbow- and return (extend_increases_py, dpy_per_rad, ddepth_per_rad)."""
    m0 = measure(provider, tracker, local_web, frames=frames)
    if m0 is None:
        return True, 55.0, -0.42
    py0 = m0[1]
    depth0 = m0[2]
    q_probe = JointState(
        base=q.base,
        shoulder=clamp_joint("shoulder", q.shoulder + delta_rad),
        elbow=clamp_joint("elbow", q.elbow - delta_rad),
        wrist=q.wrist,
        roll=q.roll,
        hand=q.hand,
    )
    execute_rpc("joints_move", _joints_params(q_probe))
    time.sleep(settle_s)
    m1 = measure(provider, tracker, local_web, frames=frames)
    execute_rpc("joints_move", _joints_params(q))
    time.sleep(settle_s)
    if m1 is None:
        return True, 55.0, -0.42
    dpy = float(m1[1] - py0)
    ddepth = float(m1[2] - depth0)
    if abs(dpy) < 2.0:
        return True, 55.0, -0.42
    return dpy > 0.0, abs(dpy) / delta_rad, ddepth / delta_rad


def load_pose(home_raw: dict, name: str) -> Optional[PoseRef]:
    block = home_raw.get(name)
    if not isinstance(block, dict):
        return None
    return PoseRef.from_dict(block)


def measure(
    provider: Ros2RgbDepthProvider,
    tracker: StrawberryTargetTracker,
    local_web: str,
    frames: int,
) -> Optional[tuple]:
    pxs, pys, ds, confs = [], [], [], []
    wh = (640, 480)
    det_model = None
    from pipelines.roarm_perception import PerceptionConfig, sample_depth_median

    perc = PerceptionConfig()
    for _ in range(max(1, frames)):
        pair = None
        try:
            pair = provider.read(timeout_s=2.0)
        except RuntimeError:
            pair = None
        depth_native = pair.depth_m if pair is not None else None
        bgr, st = _fetch_hub_bgr(local_web)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        wh = (w, h)
        if det_model is None:
            det_model = _get_detector(REPO_ROOT)
        det = detect_strawberry_in_frame(bgr, det_model, tracker)
        if det is None and tracker and tracker.can_hold and depth_native is not None:
            if float(tracker.last_conf) < min_berry_conf():
                tracker.hold_streak += 1
                continue
            cx = int(round(tracker.last_px))
            cy = int(round(tracker.last_py))
            depth_hub = _letterbox_array(depth_native, w, h)
            depth_val, _ = sample_depth_median(depth_hub, cx, cy, perc.depth_median_radius, perc)
            if depth_val is not None:
                pxs.append(cx)
                pys.append(cy)
                ds.append(depth_val)
                confs.append(float(tracker.last_conf))
                tracker.hold_streak += 1
            continue
        if det is None:
            if tracker:
                tracker.lost_streak += 1
                tracker.hold_streak += 1
            continue
        cx = int(round((det["x1"] + det["x2"]) * 0.5))
        cy = int(round((det["y1"] + det["y2"]) * 0.5))
        depth_val = None
        if depth_native is not None:
            depth_hub = _letterbox_array(depth_native, w, h)
            for dx in (0, -45, -80, 45, -110):
                tcx = int(np.clip(cx + dx, 8, w - 9))
                depth_val, _ = sample_depth_median(
                    depth_hub, tcx, cy, perc.depth_median_radius, perc
                )
                if depth_val is not None:
                    break
        if depth_val is None:
            continue
        pxs.append(cx)
        pys.append(cy)
        ds.append(depth_val)
        confs.append(float(det["conf"]))
        if tracker:
            tracker.update(det)
    if not pxs:
        if tracker:
            tracker.lost_streak += 1
        return None
    return (
        float(np.median(pxs)),
        float(np.median(pys)),
        float(np.median(ds)),
        float(np.median(confs)),
        wh,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="RoArm strawberry approach from HOME2")
    ap.add_argument("--home-config", default=str(REPO_ROOT / "config/roarm_home_joints.yaml"))
    ap.add_argument("--calib", default=str(REPO_ROOT / "runs/roarm_kinematics/calibration.json"))
    ap.add_argument("--start-pose", default="HOME2")
    ap.add_argument("--local-web", default="http://127.0.0.1:8080")
    ap.add_argument("--standoff-m", type=float, default=0.12,
                    help="Target camera→berry distance (m); ~0.12 pregrasp")
    ap.add_argument("--center-only", action="store_true")
    ap.add_argument("--aim-mode", choices=("lock_y", "center", "offset"), default="lock_y",
                    help="lock_y: only center px (HOME2); center: image center; offset: center+py offset")
    ap.add_argument("--aim-py-offset", type=float, default=0.0,
                    help="Extra pixels added to aim_py when aim-mode=offset")
    ap.add_argument("--center-tol-px", type=float, default=14.0)
    ap.add_argument("--center-gain", type=float, default=0.45)
    ap.add_argument("--max-move-px", type=float, default=18.0)
    ap.add_argument("--max-step-rad", type=float, default=0.05)
    ap.add_argument("--reach-tol-m", type=float, default=0.025)
    ap.add_argument("--reach-step-rad", type=float, default=0.04)
    ap.add_argument("--reach-gain", type=float, default=0.55)
    ap.add_argument("--depth-floor-m", type=float, default=0.02)
    ap.add_argument("--max-iters", type=int, default=35)
    ap.add_argument("--settle-s", type=float, default=0.65)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--lock-attempts", type=int, default=12)
    ap.add_argument("--min-start-depth-m", type=float, default=0.18,
                    help="Abort if berry closer than this at start (mis-detect)")
    args = ap.parse_args()
    _load_fleet_env()

    home_raw = yaml.safe_load(Path(args.home_config).read_text()) or {}
    pose = load_pose(home_raw, args.start_pose)
    if pose is None:
        print(f"[berry] unknown pose {args.start_pose}")
        return 2

    calib = json.loads(Path(args.calib).read_text())
    stage1 = calib.get("stage1") or {}
    if not stage1.get("Jc_inv_rows"):
        print("[berry] missing stage1 Jc_inv — run roarm_fit_jacobian.py --write")
        return 2
    Jc_inv = np.array(stage1["Jc_inv_rows"], dtype=np.float64)
    dd_pitch = float(stage1["control_jacobian_px_per_rad"]["pitch"]["ddepth_m"])
    print(f"[berry] start={args.start_pose} standoff={args.standoff_m}m dd_pitch={dd_pitch:.3f}")

    provider = Ros2RgbDepthProvider(rgb_topic=STEREO_RGB, depth_topic=STEREO_DEPTH, sync_slop_s=0.15)
    provider.open(camera_info_topic=STEREO_INFO)

    from roarm_proxy import execute_rpc

    print(f"[berry] moving to {args.start_pose} …")
    _open_gripper(execute_rpc)
    pose_dict = _pose_params(pose)
    pose_dict["hand"] = GRIPPER_OPEN_RAD
    execute_rpc("joints_move", pose_dict)
    time.sleep(args.settle_s * 1.5)

    tracker = StrawberryTargetTracker()
    meas = None
    for attempt in range(max(1, args.lock_attempts)):
        meas = measure(provider, tracker, args.local_web, frames=max(3, args.frames))
        if meas is not None:
            break
        print(f"[berry] lock attempt {attempt + 1}/{args.lock_attempts} — no detection")
        time.sleep(0.4)
    if meas is None:
        print("[berry] no strawberry detected at start")
        return 2
    px, py, depth, conf, wh = meas
    w, h = wh
    cx, cy = w / 2.0, h / 2.0
    if depth < args.min_start_depth_m:
        print(f"[berry] depth {depth:.3f}m < min {args.min_start_depth_m} — abort")
        return 2
    print(f"[berry] lock conf={conf:.2f} px={px:.0f} py={py:.0f} depth={depth:.3f}m "
          f"center=({cx:.0f},{cy:.0f})")

    if args.aim_mode == "lock_y":
        aim_px, aim_py = cx, py
    elif args.aim_mode == "offset":
        aim_px, aim_py = cx, cy + args.aim_py_offset
    else:
        aim_px, aim_py = cx, cy
    print(f"[berry] aim=({aim_px:.0f},{aim_py:.0f}) mode={args.aim_mode}")

    q = read_q(execute_rpc)
    if q is None:
        print("[berry] no joint feedback")
        return 2

    print("[berry] calibrating extend depth slope …")
    _extend_up_py, _dpy_per_extend, dd_extend = calibrate_extend_vertical(
        provider,
        tracker,
        args.local_web,
        execute_rpc,
        q,
        delta_rad=0.02,
        settle_s=args.settle_s,
        frames=max(2, args.frames),
    )
    print(f"[berry] extend ddepth={dd_extend:.3f}m/rad")
    meas = measure(provider, tracker, args.local_web, frames=max(2, args.frames))
    if meas is not None:
        px, py, depth, conf, wh = meas
        w, h = wh
        cx, cy = w / 2.0, h / 2.0
        print(f"[berry] after calib px={px:.0f} py={py:.0f} depth={depth:.3f}m")

    phase = "CENTER"
    traj: List[dict] = []
    stop_floor = args.standoff_m + args.depth_floor_m

    for it in range(args.max_iters):
        err_x = aim_px - px
        err_y = aim_py - py
        cerr_x = abs(err_x)
        cerr_y = abs(err_y)
        cerr = float(np.hypot(err_x, err_y))
        derr = depth - args.standoff_m

        if args.aim_mode == "lock_y":
            centered = cerr_x <= args.center_tol_px or (
                cerr_x <= args.center_tol_px + 10 and it >= 6
            )
        else:
            centered = cerr <= args.center_tol_px
        if phase == "CENTER" and centered:
            if args.center_only or derr <= args.reach_tol_m:
                phase = "DONE"
            else:
                phase = "REACH"
        elif phase == "REACH" and not centered and cerr_x > args.center_tol_px * 1.8:
            phase = "CENTER"

        if phase == "DONE":
            print(f"[berry] DONE it={it}: cerr={cerr:.0f}px depth={depth:.3f}m")
            break

        db = ds = 0.0
        note = phase
        if phase == "CENTER" or (phase == "REACH" and cerr_x > args.center_tol_px):
            db = ds = 0.0
            if args.aim_mode != "lock_y" and abs(err_y) >= abs(err_x):
                want_dpy = float(np.clip(err_y * args.center_gain, -args.max_move_px, args.max_move_px))
                ds = want_dpy / max(_dpy_per_extend, 20.0)
                if not _extend_up_py:
                    ds = -ds
                ds = float(np.clip(ds, -args.max_step_rad, args.max_step_rad))
                q.shoulder = clamp_joint("shoulder", q.shoulder + ds)
                q.elbow = clamp_joint("elbow", q.elbow - ds)
            elif abs(err_x) > 2.0:
                want_dpx = float(np.clip(err_x * args.center_gain, -args.max_move_px, args.max_move_px))
                db = float(
                    np.clip(
                        want_dpx / max(float(stage1["control_jacobian_px_per_rad"]["base"]["dpx"]), 200.0),
                        -args.max_step_rad,
                        args.max_step_rad,
                    )
                )
                q.base = clamp_joint("base", q.base + db)
            note = "center"
        elif phase == "REACH":
            dd_reach = dd_extend if abs(dd_extend) > 0.05 else dd_pitch
            want_depth = depth - derr * args.reach_gain
            ds = (want_depth - depth) / dd_reach if abs(dd_reach) > 1e-3 else 0.0
            ds = float(np.clip(ds, -args.reach_step_rad, args.reach_step_rad))
            pred_depth = depth + ds * dd_reach
            if pred_depth < stop_floor - 1e-3:
                ds = (stop_floor - depth) / dd_reach
                ds = float(np.clip(ds, -args.reach_step_rad, args.reach_step_rad))
            if ds <= 1e-4:
                print("[berry] reach step ~0 — at standoff")
                break
            q.shoulder = clamp_joint("shoulder", q.shoulder + ds)
            q.elbow = clamp_joint("elbow", q.elbow - ds)
            note = "reach"

        print(f"[berry] it={it:2d} [{note:6s}] cerr={cerr:5.0f}px d={depth:.3f} "
              f"db={db:+.3f} ds={ds:+.3f}")
        _joints_open(q)
        execute_rpc("joints_move", _joints_params(q))
        time.sleep(args.settle_s)

        q_fb = read_q(execute_rpc)
        after = measure(provider, tracker, args.local_web, frames=args.frames)
        if after is None or tracker.is_lost:
            print(f"[berry] target lost during {note}")
            break
        npx, npy, ndepth, nconf, _ = after
        print(f"[berry]   -> px {px:.0f}->{npx:.0f} py {py:.0f}->{npy:.0f} "
              f"d {depth:.3f}->{ndepth:.3f} conf={nconf:.2f}")
        traj.append({
            "it": it, "phase": note, "cerr_px": round(cerr, 1),
            "depth_m": round(depth, 4), "px": round(npx, 1), "py": round(npy, 1),
        })
        if q_fb is not None:
            q = q_fb
        px, py, depth = npx, npy, ndepth
    else:
        print(f"[berry] max_iters={args.max_iters}")

    if args.aim_mode == "lock_y":
        final_cerr = abs(aim_px - px)
    else:
        final_cerr = float(np.hypot(aim_px - px, aim_py - py))
    print(f"\n[berry] final cerr={final_cerr:.0f}px depth={depth:.3f}m "
          f"(standoff {args.standoff_m}m)")

    out = REPO_ROOT / "runs/roarm_learn/logs/strawberry_approach.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "start_pose": args.start_pose,
                "standoff_m": args.standoff_m,
                "final_center_err_px": round(final_cerr, 1),
                "final_depth_m": round(depth, 4),
                "center_only": args.center_only,
                "trajectory": traj,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[berry] log → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
