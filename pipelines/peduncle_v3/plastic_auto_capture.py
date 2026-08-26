"""Safe plastic_lab auto-capture: small joint perturbations around a working pose.

Uses existing HARD/SOFT joint limits from pipelines/roarm_joint_limits.py.
Viewpoint joints: base, shoulder, elbow (optional tiny wrist). Gripper frozen.
Does NOT use peduncle/calyx models for control. Does NOT change berry approach.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipelines.peduncle_v3.geometry import build_focus_roi
from pipelines.roarm_joint_limits import HARD_JOINT_LIMITS, SOFT_JOINT_LIMITS, clamp_joint
from pipelines.roarm_kinematics import JointState

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "peduncle_plastic_lab" / "raw"

# Camera viewpoint joints only (hand/gripper never commanded for diversity).
VIEW_JOINTS = ("base", "shoulder", "elbow")
OPTIONAL_WRIST = ("wrist",)  # tiny only; roll unused for viewpoint diversity

# Lab envelope: max |Δ| from baseline pose (additional to soft/hard clamps).
DEFAULT_ENVELOPE = {
    "base": 0.16,
    "shoulder": 0.12,
    "elbow": 0.14,
    "wrist": 0.06,
}

# Single-joint step sizes (rad). Signs: base+ left, shoulder+ forward, elbow+ fold.
DEFAULT_STEPS = {
    "base": (0.03, 0.06, 0.09, 0.12, 0.15, -0.03, -0.06, -0.09, -0.12, -0.15),
    "shoulder": (0.03, 0.06, 0.09, -0.03, -0.06, -0.09),
    "elbow": (0.04, 0.08, 0.12, -0.04, -0.08, -0.12),
    "wrist": (0.03, 0.05, -0.03, -0.05),
}

# Two-joint combo magnitude (smaller).
COMBO_STEP = 0.05
COMBO_STEP_B = 0.08


@dataclass
class PlannedPose:
    pose_id: str
    deltas: Dict[str, float]
    joints: Dict[str, float]
    n_changed: int


@dataclass
class CoverageState:
    frames: int = 0
    unique_poses: int = 0
    berry_lost: int = 0
    berry_visible: int = 0
    recoveries: int = 0
    bins: Dict[str, int] = field(
        default_factory=lambda: {
            "x_left": 0,
            "x_center": 0,
            "x_right": 0,
            "y_high": 0,
            "y_center": 0,
            "y_low": 0,
            "d_near": 0,
            "d_medium": 0,
            "d_far": 0,
        }
    )
    pose_ids: List[str] = field(default_factory=list)

    def viewpoint_bin(self, px: float, py: float, depth: Optional[float], wh: Tuple[int, int]) -> str:
        w, h = wh
        if px < w * 0.38:
            xb = "x_left"
        elif px > w * 0.62:
            xb = "x_right"
        else:
            xb = "x_center"
        if py < h * 0.38:
            yb = "y_high"
        elif py > h * 0.62:
            yb = "y_low"
        else:
            yb = "y_center"
        if depth is None or depth <= 0:
            db = "d_medium"
        elif depth < 0.20:
            db = "d_near"
        elif depth < 0.45:
            db = "d_medium"
        else:
            db = "d_far"
        return f"{xb}|{yb}|{db}"

    def add_frame(self, px: float, py: float, depth: Optional[float], wh: Tuple[int, int], pose_id: str) -> str:
        self.frames += 1
        self.berry_visible += 1
        vb = self.viewpoint_bin(px, py, depth, wh)
        for part in vb.split("|"):
            self.bins[part] = self.bins.get(part, 0) + 1
        if pose_id not in self.pose_ids:
            self.pose_ids.append(pose_id)
            self.unique_poses = len(self.pose_ids)
        return vb


def _copy_q(q: JointState) -> JointState:
    return JointState(**q.as_dict())


def apply_lab_clamp(
    name: str,
    value: float,
    *,
    baseline: float,
    envelope: Dict[str, float],
) -> float:
    """Clamp to HARD ∩ SOFT ∩ baseline±envelope."""
    v = clamp_joint(name, value, hard=True)
    if name in SOFT_JOINT_LIMITS:
        lo_s, hi_s = SOFT_JOINT_LIMITS[name]
        v = max(lo_s, min(hi_s, v))
    env = float(envelope.get(name, 0.1))
    v = max(baseline - env, min(baseline + env, v))
    # re-apply hard after envelope
    return clamp_joint(name, v, hard=True)


def build_target_q(
    baseline: JointState,
    deltas: Dict[str, float],
    *,
    envelope: Dict[str, float],
    freeze_hand: float,
) -> JointState:
    q = _copy_q(baseline)
    for name, d in deltas.items():
        if name == "hand":
            continue
        cur = getattr(q, name)
        setattr(q, name, apply_lab_clamp(name, cur + float(d), baseline=getattr(baseline, name), envelope=envelope))
    q.hand = float(freeze_hand)  # never perturb gripper
    q.roll = float(baseline.roll)  # keep roll fixed (not a viewpoint sweep axis)
    return q


def plan_perturbations(
    baseline: JointState,
    *,
    envelope: Optional[Dict[str, float]] = None,
    include_wrist: bool = True,
    max_poses: int = 48,
) -> List[PlannedPose]:
    """Deterministic safe poses: 1-joint then 2-joint combos. No random."""
    envelope = dict(DEFAULT_ENVELOPE if envelope is None else envelope)
    freeze_hand = float(baseline.hand)
    planned: List[PlannedPose] = []
    seen = set()

    def _add(deltas: Dict[str, float], tag: str) -> None:
        if len(planned) >= max_poses:
            return
        # drop zero / hand
        deltas = {k: float(v) for k, v in deltas.items() if abs(v) > 1e-6 and k != "hand"}
        if not deltas or len(deltas) > 2:
            return
        q = build_target_q(baseline, deltas, envelope=envelope, freeze_hand=freeze_hand)
        key = tuple(round(getattr(q, k), 4) for k in ("base", "shoulder", "elbow", "wrist"))
        if key in seen:
            return
        # skip if effectively identical to baseline on all viewpoint axes
        if all(abs(getattr(q, k) - getattr(baseline, k)) < 0.012 for k in ("base", "shoulder", "elbow", "wrist")):
            return
        seen.add(key)
        pose_id = f"p{len(planned):03d}_{tag}"
        planned.append(
            PlannedPose(
                pose_id=pose_id,
                deltas=deltas,
                joints=q.as_dict(),
                n_changed=len(deltas),
            )
        )

    # Always include baseline itself as pose 0
    planned.append(
        PlannedPose(
            pose_id="p000_baseline",
            deltas={},
            joints=baseline.as_dict(),
            n_changed=0,
        )
    )
    seen.add(tuple(round(getattr(baseline, k), 4) for k in ("base", "shoulder", "elbow", "wrist")))

    joints = list(VIEW_JOINTS)
    if include_wrist:
        joints.append("wrist")

    for j in joints:
        for step in DEFAULT_STEPS.get(j, ()):
            _add({j: step}, f"{j}{step:+.2f}")

    # 2-joint combos (small)
    pairs = [("base", "shoulder"), ("base", "elbow"), ("shoulder", "elbow")]
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for step in (COMBO_STEP, COMBO_STEP_B):
        for a, b in pairs:
            for sa, sb in signs:
                _add({a: sa * step, b: sb * step}, f"{a}{sa*step:+.2f}_{b}{sb*step:+.2f}")

    return planned[:max_poses]


def detect_berry_simple(
    provider,
    tracker,
    *,
    min_conf: float,
    depth_min_m: float = 0.08,
    depth_max_m: float = 1.20,
) -> Optional[Dict[str, Any]]:
    """Berry-only visibility (no peduncle/calyx). Works for DOM_FINAL and close-range."""
    from scripts.roarm_strawberry_approach import measure, _fetch_hub_bgr

    m = measure(provider, tracker, "http://127.0.0.1:8080", frames=1)
    if not m:
        return None
    # measure → (px, py, depth_m, conf, wh)
    px = float(m[0])
    py = float(m[1])
    depth = float(m[2]) if m[2] is not None else None
    conf = float(m[3]) if m[3] is not None else 0.0
    if conf < min_conf:
        return None
    if depth is not None and (depth < depth_min_m or depth > depth_max_m):
        return None

    bgr = None
    try:
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
        return None

    h, w = bgr.shape[:2]
    if len(m) > 4 and m[4] is not None:
        try:
            w, h = int(m[4][0]), int(m[4][1])
        except Exception:
            pass
    margin = 12
    if px < margin or py < margin or px > w - margin or py > h - margin:
        return None

    half = max(36.0, 0.08 * float(w))
    bbox = (px - half, py - half, px + half, py + half)
    try:
        lb = getattr(tracker, "last_bbox", None) if tracker is not None else None
        if isinstance(lb, dict) and all(k in lb for k in ("x1", "y1", "x2", "y2")):
            bbox = (float(lb["x1"]), float(lb["y1"]), float(lb["x2"]), float(lb["y2"]))
    except Exception:
        pass

    return {
        "px": px,
        "py": py,
        "depth_m": depth,
        "conf": conf,
        "bbox": list(bbox),
        "bgr": bgr,
        "wh": [w, h],
    }


def save_auto_shot(
    *,
    bgr: np.ndarray,
    berry: Dict[str, Any],
    joints: Dict[str, float],
    out_dir: Path,
    session_id: str,
    pose_id: str,
    sample_idx: int,
    viewpoint_bin: str,
    focus_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_cfg = focus_cfg or {}
    h, w = bgr.shape[:2]
    bbox = tuple(float(v) for v in berry["bbox"])
    roi = build_focus_roi(
        bbox,  # type: ignore[arg-type]
        (w, h),
        width_scale=float(focus_cfg.get("width_scale", 1.6)),
        height_above_scale=float(focus_cfg.get("height_above_scale", 0.85)),
        height_below_scale=float(focus_cfg.get("height_below_scale", 0.45)),
        min_size_px=int(focus_cfg.get("min_size_px", 160)),
        max_size_px=int(focus_cfg.get("max_size_px", 640)),
        pad_px=int(focus_cfg.get("pad_px", 8)),
    )
    rx1, ry1, rx2, ry2 = roi
    crop = bgr[ry1:ry2, rx1:rx2].copy()
    stem = f"plastic_lab__{session_id}__{pose_id}__s{sample_idx:03d}"
    full_p = out_dir / f"{stem}_full.jpg"
    crop_p = out_dir / f"{stem}_crop.jpg"
    meta_p = out_dir / f"{stem}.json"
    cv2.imwrite(str(full_p), bgr)
    cv2.imwrite(str(crop_p), crop)
    meta = {
        "group": f"plastic_lab__{session_id}",
        "stem": stem,
        "session_id": session_id,
        "pose_id": pose_id,
        "sample_idx": sample_idx,
        "ts": time.time(),
        "domain": "plastic_lab",
        "capture_mode": "auto_perturb",
        "full_path": str(full_p.relative_to(REPO)) if str(full_p).startswith(str(REPO)) else str(full_p),
        "crop_path": str(crop_p.relative_to(REPO)) if str(crop_p).startswith(str(REPO)) else str(crop_p),
        "image_wh": [w, h],
        "berry_bbox_xyxy": list(bbox),
        "berry_px": berry["px"],
        "berry_py": berry["py"],
        "berry_conf": berry["conf"],
        "depth_m": berry.get("depth_m"),
        "peduncle_roi_xyxy": [int(rx1), int(ry1), int(rx2), int(ry2)],
        "joints": joints,
        "viewpoint_bin": viewpoint_bin,
        "labeled": False,
    }
    try:
        meta["full_path"] = str(full_p.resolve().relative_to(REPO.resolve()))
        meta["crop_path"] = str(crop_p.resolve().relative_to(REPO.resolve()))
    except Exception:
        pass
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta


def status_line(cov: CoverageState, *, target: int, q: Optional[JointState], berry: Optional[Dict[str, Any]], vb: str) -> str:
    j = ""
    if q is not None:
        j = f"b={q.base:.3f} s={q.shoulder:.3f} e={q.elbow:.3f}"
    bp = ""
    if berry is not None:
        d = berry.get("depth_m")
        bp = f"px={berry['px']:.0f} py={berry['py']:.0f} d={d if d is not None else float('nan'):.3f}"
    bins = ",".join(f"{k}:{v}" for k, v in cov.bins.items() if k.startswith("x_") or k.startswith("y_") or k.startswith("d_"))
    return (
        f"[STATUS] frames={cov.frames}/{target} poses={cov.unique_poses} "
        f"lost={cov.berry_lost} recoveries={cov.recoveries} | {j} | {bp} | bin={vb} | {bins}"
    )


def run_auto_capture(
    *,
    execute_rpc: Callable,
    provider,
    tracker,
    move_t102_smooth: Callable,
    wait_reach: Callable,
    read_q: Callable,
    baseline: JointState,
    planned: Sequence[PlannedPose],
    out_dir: Path,
    target_shots: int = 150,
    max_frames_per_pose: int = 1,
    settle_s: float = 0.45,
    spd: float = 0.06,
    acc: float = 2.5,
    reach_tol: float = 0.05,
    min_conf: float = 0.70,
    dry_run: bool = False,
    smoke_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute planned poses. smoke_n limits pose count (e.g. 5)."""
    session_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_hand = float(baseline.hand)
    poses = list(planned)
    if smoke_n is not None:
        poses = poses[: max(1, int(smoke_n))]

    plan_doc = {
        "session_id": session_id,
        "mode": "dry_run" if dry_run else ("smoke" if smoke_n else "full"),
        "baseline": baseline.as_dict(),
        "freeze_hand": freeze_hand,
        "view_joints": list(VIEW_JOINTS) + (["wrist"] if any("wrist" in p.deltas for p in poses) else []),
        "envelope": DEFAULT_ENVELOPE,
        "hard_limits": {k: list(v) for k, v in HARD_JOINT_LIMITS.items()},
        "soft_limits": {k: list(v) for k, v in SOFT_JOINT_LIMITS.items()},
        "spd": spd,
        "acc": acc,
        "target_shots": target_shots,
        "n_planned_poses": len(poses),
        "poses": [asdict(p) for p in poses],
    }
    (out_dir / f"plan_{session_id}.json").write_text(json.dumps(plan_doc, indent=2), encoding="utf-8")

    if dry_run:
        print(f"[plastic-auto] DRY-RUN session={session_id} planned_poses={len(poses)}")
        for p in poses:
            print(f"  {p.pose_id} deltas={p.deltas} joints={{b:{p.joints['base']:.3f},s:{p.joints['shoulder']:.3f},e:{p.joints['elbow']:.3f}}}")
        print(f"[plastic-auto] plan → {out_dir / f'plan_{session_id}.json'}")
        return {
            "ok": True,
            "dry_run": True,
            "session_id": session_id,
            "n_planned": len(poses),
            "out_dir": str(out_dir),
            "plan": plan_doc,
        }

    cov = CoverageState()
    last_safe = _copy_q(baseline)
    sample_idx = 0
    focus_cfg: Dict[str, Any] = {}

    # Ensure at baseline first
    print(f"[plastic-auto] go baseline hand={freeze_hand:.3f} (gripper frozen)")
    q_base = _copy_q(baseline)
    q_base.hand = freeze_hand
    move_t102_smooth(execute_rpc, q_base, spd=spd, acc=acc)
    wait_reach(execute_rpc, q_base, timeout_s=25.0, tol=reach_tol, stable_s=0.35)
    time.sleep(settle_s)

    for p in poses:
        if cov.frames >= target_shots:
            break
        q_tgt = JointState(**p.joints)
        q_tgt.hand = freeze_hand
        print(f"[plastic-auto] → {p.pose_id} deltas={p.deltas}")
        move_t102_smooth(execute_rpc, q_tgt, spd=spd, acc=acc)
        q_fb = wait_reach(execute_rpc, q_tgt, timeout_s=25.0, tol=reach_tol, stable_s=0.35)
        time.sleep(settle_s)
        q_now = q_fb or read_q(execute_rpc) or q_tgt

        frames_this = 0
        while frames_this < max_frames_per_pose and cov.frames < target_shots:
            berry = detect_berry_simple(provider, tracker, min_conf=min_conf)
            if berry is None:
                cov.berry_lost += 1
                print(status_line(cov, target=target_shots, q=q_now, berry=None, vb="LOST"))
                print(f"[plastic-auto] berry LOST at {p.pose_id} → recover last_safe")
                cov.recoveries += 1
                move_t102_smooth(execute_rpc, last_safe, spd=spd, acc=acc)
                wait_reach(execute_rpc, last_safe, timeout_s=25.0, tol=reach_tol, stable_s=0.35)
                time.sleep(settle_s)
                # verify recovery
                berry_r = detect_berry_simple(provider, tracker, min_conf=min_conf)
                if berry_r is None:
                    print("[plastic-auto] recovery failed → baseline")
                    move_t102_smooth(execute_rpc, q_base, spd=spd, acc=acc)
                    wait_reach(execute_rpc, q_base, timeout_s=25.0, tol=reach_tol, stable_s=0.35)
                    time.sleep(settle_s)
                    last_safe = _copy_q(q_base)
                else:
                    last_safe = _copy_q(last_safe)
                break  # next pose variant

            wh = (berry["wh"][0], berry["wh"][1])
            vb = cov.add_frame(berry["px"], berry["py"], berry.get("depth_m"), wh, p.pose_id)
            meta = save_auto_shot(
                bgr=berry["bgr"],
                berry=berry,
                joints=(q_now.as_dict() if hasattr(q_now, "as_dict") else p.joints),
                out_dir=out_dir,
                session_id=session_id,
                pose_id=p.pose_id,
                sample_idx=sample_idx,
                viewpoint_bin=vb,
                focus_cfg=focus_cfg,
            )
            sample_idx += 1
            frames_this += 1
            last_safe = _copy_q(q_now) if hasattr(q_now, "base") else last_safe
            print(status_line(cov, target=target_shots, q=q_now, berry=berry, vb=vb))
            print(f"[plastic-auto] saved {meta['stem']}")
            if frames_this < max_frames_per_pose:
                time.sleep(0.25)  # tiny settle only; no extra joint motion

    # Return home/baseline
    move_t102_smooth(execute_rpc, q_base, spd=spd, acc=acc)
    wait_reach(execute_rpc, q_base, timeout_s=25.0, tol=reach_tol, stable_s=0.25)

    summary = {
        "ok": True,
        "session_id": session_id,
        "out_dir": str(out_dir),
        "frames": cov.frames,
        "unique_poses": cov.unique_poses,
        "berry_lost": cov.berry_lost,
        "recoveries": cov.recoveries,
        "bins": cov.bins,
        "pose_ids": cov.pose_ids,
        "view_joints_used": list(VIEW_JOINTS) + (["wrist"] if any("wrist" in p.deltas for p in poses) else []),
        "envelope": DEFAULT_ENVELOPE,
        "spd": spd,
        "acc": acc,
        "target_shots": target_shots,
        "mode": "smoke" if smoke_n else "full",
    }
    (out_dir / f"session_auto_{session_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[plastic-auto] DONE {json.dumps({k: summary[k] for k in ('frames','unique_poses','berry_lost','bins')})}")
    return summary
