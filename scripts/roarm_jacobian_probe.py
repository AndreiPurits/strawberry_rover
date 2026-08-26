#!/usr/bin/env python3
"""Controlled image-Jacobian probe for RoArm stage-1 calibration.

From a set of start poses, lock a chessboard corner (template) and perturb ONE
joint at a time by ±delta, measuring how the target moves in image (px, py) and
depth. This yields clean columns of the image Jacobian:

    [dpx, dpy, ddepth]^T ≈ J @ [d_shoulder, d_elbow, d_base]^T

Samples are saved to a probe JSON and appended to the knowledge base so
`roarm_fit_jacobian.py` can fit the model.

Arm motion only (RoArm), never touches rover drive / Mega.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ops/axm-monitor/agent") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ops/axm-monitor/agent"))

import yaml  # noqa: E402

from pipelines.roarm_kinematics import JointState  # noqa: E402
from pipelines.roarm_motion import PoseRef  # noqa: E402
from pipelines.roarm_perception import PerceptionConfig  # noqa: E402
from pipelines.roarm_template_track import TemplateCornerTracker, build_template_target  # noqa: E402
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider  # noqa: E402
from pipelines.roarm_knowledge import DEFAULT_KB_PATH, load_kb, save_kb  # noqa: E402
from pipelines.roarm_joint_limits import HARD_JOINT_LIMITS  # noqa: E402
from pipelines.roarm_live_approach import write_live_approach  # noqa: E402

STEREO_RGB = "/stereo_camera/color/image_rect_raw"
STEREO_DEPTH = "/stereo_camera/depth/image_rect_raw"
STEREO_INFO = "/stereo_camera/color/camera_info"

JOINT_LIMITS = dict(HARD_JOINT_LIMITS)


def build_perc_cfg(learn: dict) -> PerceptionConfig:
    geom = learn.get("geometry") or {}
    perc = learn.get("perception") or {}
    return PerceptionConfig(
        chessboard_cols=int(perc.get("chessboard_cols", 9)),
        chessboard_rows=int(perc.get("chessboard_rows", 6)),
        corner_strategy=str(perc.get("corner_strategy", "depth_center")),
        depth_median_radius=int(geom.get("depth_median_radius", 5)),
        depth_valid_min_m=float(geom.get("depth_valid_min_m", 0.05)),
        depth_valid_max_m=float(geom.get("depth_valid_max_m", 2.5)),
        depth_jump_max_m=float(geom.get("depth_jump_max_m", 0.4)),
        target_camera_distance_m=float(geom.get("target_camera_distance_m", 0.11)),
        uv_deadzone=float(geom.get("uv_deadzone", 0.06)),
        use_template_lock=True,
        template_patch_px=int(perc.get("template_patch_px", 64)),
        template_search_radius_px=int(perc.get("template_search_radius_px", 48)),
        template_reacquire_radius_px=int(perc.get("template_reacquire_radius_px", 140)),
        template_min_score=float(perc.get("template_min_score", 0.52)),
        template_spatial_penalty=float(perc.get("template_spatial_penalty", 0.45)),
        template_max_jump_px=float(perc.get("template_max_jump_px", 40.0)),
    )


def candidate_poses(home_raw: dict, names: List[str]) -> List[Tuple[str, PoseRef]]:
    home = PoseRef.from_dict(home_raw.get("HOME") or {})
    grid = PoseRef.from_dict(home_raw.get("GRID_VIEW") or {})
    presets: Dict[str, PoseRef] = {
        "HOME": home,
        "GRID_VIEW": grid,
        "MID": PoseRef(
            base=grid.base,
            shoulder=(home.shoulder + grid.shoulder) * 0.5,
            elbow=(home.elbow + grid.elbow) * 0.5,
            wrist=home.wrist,
            roll=home.roll,
            hand=home.hand,
        ),
        "MID_LOW": PoseRef(
            base=grid.base,
            shoulder=(home.shoulder + grid.shoulder) * 0.5 - 0.15,
            elbow=(home.elbow + grid.elbow) * 0.5 + 0.15,
            wrist=home.wrist,
            roll=home.roll,
            hand=home.hand,
        ),
        "GRID_LEFT": PoseRef(
            base=grid.base + 0.20,
            shoulder=grid.shoulder,
            elbow=grid.elbow,
            wrist=grid.wrist,
            roll=grid.roll,
            hand=grid.hand,
        ),
        "GRID_RIGHT": PoseRef(
            base=grid.base - 0.20,
            shoulder=grid.shoulder,
            elbow=grid.elbow,
            wrist=grid.wrist,
            roll=grid.roll,
            hand=grid.hand,
        ),
    }
    out = []
    for n in names:
        if n in presets:
            out.append((n, presets[n]))
    return out


def _pose_params(pose: PoseRef, acc: float = 8.0) -> Dict[str, Any]:
    return {
        "base": pose.base,
        "shoulder": pose.shoulder,
        "elbow": pose.elbow,
        "wrist": pose.wrist,
        "roll": pose.roll,
        "hand": pose.hand,
        "spd": 0.0,
        "acc": acc,
    }


def _joints_params(q: JointState, acc: float = 8.0, spd: float = 0.0) -> Dict[str, Any]:
    return {
        "base": q.base,
        "shoulder": q.shoulder,
        "elbow": q.elbow,
        "wrist": q.wrist,
        "roll": q.roll,
        "hand": q.hand,
        "spd": float(spd),
        "acc": acc,
    }


def move_joints_simultaneous(
    execute_rpc, q: JointState, *, acc: float = 16.0, spd: float = 0.0
) -> None:
    """T:102 — all joints in one command (not T:101 staged one-by-one)."""
    execute_rpc("joints_move", _joints_params(q, acc=acc, spd=spd))


def read_q(execute_rpc) -> Optional[JointState]:
    fb = execute_rpc("feedback", {})
    if fb.get("ok") and isinstance(fb.get("feedback"), dict):
        return JointState.from_feedback(fb["feedback"])
    return None


def measure_target(
    provider: Ros2RgbDepthProvider,
    perc_cfg: PerceptionConfig,
    tracker: TemplateCornerTracker,
    *,
    corner_idx: Optional[int],
    frames: int = 3,
    timeout_s: float = 4.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Median (px, py, depth, score) over a few frames using the template lock."""
    pxs, pys, ds, scores = [], [], [], []
    intr = provider.get_intrinsics()
    for _ in range(frames):
        pair = provider.read(timeout_s=timeout_s)
        t = build_template_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc_cfg,
            tracker,
            forced_corner_idx=corner_idx if not tracker.locked else None,
            intrinsics=intr,
        )
        if t.valid:
            pxs.append(float(t.px))
            pys.append(float(t.py))
            ds.append(float(t.depth_m))
            scores.append(float(tracker.last_score))
        time.sleep(0.03)
    if len(pxs) < max(1, frames // 2):
        return None
    import numpy as np

    return (
        float(np.median(pxs)),
        float(np.median(pys)),
        float(np.median(ds)),
        float(np.median(scores)) if scores else 0.0,
    )


def clamp_joint(name: str, value: float) -> float:
    lo, hi = JOINT_LIMITS.get(name, (-3.14, 3.14))
    return max(lo, min(hi, value))


def probe_pose(
    *,
    provider,
    perc_cfg,
    execute_rpc,
    pose_name: str,
    pose: PoseRef,
    corner_idx: Optional[int],
    delta: float,
    settle_s: float,
    frames: int,
    timeout_s: float,
) -> List[dict]:
    samples: List[dict] = []
    execute_rpc("home_joints_staged", _pose_params(pose))
    time.sleep(settle_s * 1.5)

    tracker = TemplateCornerTracker.from_config(perc_cfg)
    base_meas = measure_target(
        provider, perc_cfg, tracker, corner_idx=corner_idx, frames=frames, timeout_s=timeout_s
    )
    if base_meas is None:
        print(f"[probe] {pose_name}: no target lock, skip")
        return samples
    locked_corner = tracker.corner_idx
    print(
        f"[probe] {pose_name}: locked corner={locked_corner} "
        f"px={base_meas[0]:.0f} py={base_meas[1]:.0f} d={base_meas[2]:.3f} score={base_meas[3]:.2f}"
    )

    q0 = read_q(execute_rpc)
    if q0 is None:
        print(f"[probe] {pose_name}: no feedback, skip")
        return samples

    # Perturbation set chosen to break shoulder/elbow collinearity: pure single
    # joints + same-phase + anti-phase (extend / retract) combos + base.
    d = delta
    combos: List[Tuple[str, Dict[str, float]]] = [
        ("shoulder+", {"shoulder": +d}),
        ("shoulder-", {"shoulder": -d}),
        ("elbow+", {"elbow": +d}),
        ("elbow-", {"elbow": -d}),
        ("extend", {"shoulder": +d, "elbow": -d}),   # reach out
        ("retract", {"shoulder": -d, "elbow": +d}),  # pull in
        ("base+", {"base": +d}),
        ("base-", {"base": -d}),
    ]

    for label, deltas in combos:
        base = measure_target(
            provider, perc_cfg, tracker, corner_idx=None, frames=frames, timeout_s=timeout_s
        )
        if base is None:
            continue
        q_cmd = JointState(**q0.as_dict())
        moved_cmd = False
        for jn, dv in deltas.items():
            tv = clamp_joint(jn, getattr(q0, jn) + dv)
            setattr(q_cmd, jn, tv)
            if abs(tv - getattr(q0, jn)) > 0.01:
                moved_cmd = True
        if not moved_cmd:
            continue

        execute_rpc("home_joints_staged", _joints_params(q_cmd))
        time.sleep(settle_s)
        q1 = read_q(execute_rpc)
        after = measure_target(
            provider, perc_cfg, tracker, corner_idx=None, frames=frames, timeout_s=timeout_s
        )

        execute_rpc("home_joints_staged", _joints_params(q0))
        time.sleep(settle_s)

        if q1 is None or after is None:
            continue
        dq = {
            "base": round(q1.base - q0.base, 5),
            "shoulder": round(q1.shoulder - q0.shoulder, 5),
            "elbow": round(q1.elbow - q0.elbow, 5),
        }
        dpx = after[0] - base[0]
        dpy = after[1] - base[1]
        # Guard against template jump: implausibly large image move for a small
        # joint step, or lost lock -> re-lock and skip.
        if tracker.is_lost or after[3] < perc_cfg.template_min_score or abs(dpx) > 200 or abs(dpy) > 200:
            tracker = TemplateCornerTracker.from_config(perc_cfg)
            measure_target(
                provider, perc_cfg, tracker, corner_idx=locked_corner,
                frames=frames, timeout_s=timeout_s,
            )
            print(f"  {label:9s}: template jump/lost, re-locked, skip")
            continue

        sample = {
            "ts": time.time(),
            "pose": pose_name,
            "combo": label,
            "corner_idx": locked_corner,
            "q0": {k: round(getattr(q0, k), 5) for k in ("base", "shoulder", "elbow")},
            "dq": dq,
            "px0": round(base[0], 2),
            "py0": round(base[1], 2),
            "d0": round(base[2], 4),
            "dpx": round(dpx, 3),
            "dpy": round(dpy, 3),
            "ddepth_m": round(after[2] - base[2], 5),
            "score": round(after[3], 3),
            "kind": "probe",
        }
        samples.append(sample)
        print(
            f"  {label:9s}: dq_s={dq['shoulder']:+.3f} dq_e={dq['elbow']:+.3f} "
            f"dq_b={dq['base']:+.3f} -> dpx={sample['dpx']:+.1f} dpy={sample['dpy']:+.1f} "
            f"dd={sample['ddepth_m']:+.4f}"
        )
        write_live_approach(
            target={
                "px": int(after[0]), "py": int(after[1]), "valid": True,
                "u": 0.0, "v": 0.0, "depth_m": after[2], "cam_err_m": 0.0,
                "source": "probe", "corner_idx": locked_corner,
            },
            meta={"image_wh": [848, 480]},
            tracking={"anchor_px": tracker.anchor_px, "anchor_py": tracker.anchor_py,
                      "track_px": tracker.track_px, "track_py": tracker.track_py,
                      "template_score": after[3]},
            status="running",
            status_text=f"probe {pose_name} {label}",
        )

    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="RoArm image-Jacobian probe (stage 1)")
    ap.add_argument("--config", default=str(REPO_ROOT / "config/roarm_learn.yaml"))
    ap.add_argument("--home-config", default=str(REPO_ROOT / "config/roarm_home_joints.yaml"))
    ap.add_argument("--corner-idx", default="auto")
    ap.add_argument("--delta", type=float, default=0.07, help="Per-joint perturbation (rad)")
    ap.add_argument("--settle-s", type=float, default=0.7)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--timeout-s", type=float, default=4.0)
    ap.add_argument(
        "--poses",
        default="GRID_VIEW,MID,MID_LOW,GRID_LEFT,GRID_RIGHT",
        help="Comma list of start poses",
    )
    ap.add_argument("--kb-path", default=str(DEFAULT_KB_PATH))
    ap.add_argument("--save", default=str(REPO_ROOT / "runs/roarm_learn/logs/jacobian_probe.json"))
    args = ap.parse_args()

    learn = yaml.safe_load(Path(args.config).read_text()) or {}
    home_raw = yaml.safe_load(Path(args.home_config).read_text()) or {}
    perc_cfg = build_perc_cfg(learn)

    corner_raw = str(args.corner_idx).strip().lower()
    corner_idx: Optional[int] = None
    if corner_raw not in ("auto", "", "none"):
        try:
            corner_idx = int(corner_raw)
        except ValueError:
            corner_idx = None

    provider = Ros2RgbDepthProvider(
        rgb_topic=STEREO_RGB, depth_topic=STEREO_DEPTH, sync_slop_s=0.15
    )
    provider.open(camera_info_topic=STEREO_INFO)
    t0 = time.time()
    while provider.get_intrinsics() is None and (time.time() - t0) < 6.0:
        time.sleep(0.05)

    from roarm_proxy import execute_rpc

    poses = candidate_poses(home_raw, [p.strip() for p in args.poses.split(",") if p.strip()])
    print(f"[probe] poses: {[n for n, _ in poses]} delta={args.delta} corner={corner_raw}")

    all_samples: List[dict] = []
    for name, pose in poses:
        s = probe_pose(
            provider=provider,
            perc_cfg=perc_cfg,
            execute_rpc=execute_rpc,
            pose_name=name,
            pose=pose,
            corner_idx=corner_idx,
            delta=float(args.delta),
            settle_s=float(args.settle_s),
            frames=int(args.frames),
            timeout_s=float(args.timeout_s),
        )
        all_samples.extend(s)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps({"samples": all_samples}, indent=2, ensure_ascii=False))

    kb = load_kb(Path(args.kb_path))
    probe_samples = list(kb.get("probe_samples") or [])
    probe_samples.extend(all_samples)
    kb["probe_samples"] = probe_samples[-4000:]
    save_kb(kb, Path(args.kb_path))

    valid = [s for s in all_samples if abs(s["dpx"]) + abs(s["dpy"]) > 0.5]
    print(f"\n[probe] collected {len(all_samples)} samples ({len(valid)} with image motion)")
    print(f"[probe] saved → {save_path}")
    print(f"[probe] KB probe_samples total → {len(kb['probe_samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
