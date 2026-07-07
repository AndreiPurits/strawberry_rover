"""Persistent RoArm approach knowledge (warm-start joints, corner stats, episodes)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_PATH = REPO_ROOT / "runs/roarm_learn/knowledge.json"
MAX_EPISODES = 80
MAX_JACOBIAN_SAMPLES = 2000


def _q_bin(q: dict, *, step: float = 0.12) -> str:
    s = float(q.get("shoulder", q.get("s", 0)))
    e = float(q.get("elbow", q.get("e", 0)))
    return f"s:{round(s / step) * step:.2f}_e:{round(e / step) * step:.2f}"


def append_jacobian_samples(kb: Dict[str, Any], approach: Dict[str, Any]) -> None:
    """Record (q, image error, Δq, Δerror) pairs from motion step logs."""
    steps = approach.get("steps") or []
    if len(steps) < 1:
        return
    samples: List[dict] = list(kb.get("jacobian_samples") or [])
    corner_idx = int((approach.get("final_target") or {}).get("corner_idx", -1))
    source = str((approach.get("final_target") or {}).get("source", ""))

    for i, step in enumerate(steps):
        dq = step.get("delta_q_cmd") or {}
        q_before = step.get("q_before") or {}
        if not q_before or not dq:
            continue
        entry = {
            "ts": step.get("ts", time.time()),
            "corner_idx": corner_idx,
            "source": source,
            "q_bin": _q_bin(q_before),
            "q": {k: round(float(q_before.get(k, 0)), 4) for k in ("base", "shoulder", "elbow")},
            "u": step.get("u"),
            "v": step.get("v"),
            "depth_m": step.get("depth_m"),
            "cam_err_m": step.get("cam_err_m"),
            "dq": {k: round(float(dq.get(k, 0)), 5) for k in ("base", "shoulder", "elbow")},
            "phase": step.get("phase"),
            "reason": step.get("reason"),
            "stall": step.get("stall"),
        }
        if i + 1 < len(steps):
            nxt = steps[i + 1]
            entry["du"] = round(float(nxt.get("u", 0)) - float(step.get("u", 0)), 5)
            entry["dv"] = round(float(nxt.get("v", 0)) - float(step.get("v", 0)), 5)
            entry["dcam_err_m"] = round(
                float(nxt.get("cam_err_m", 0)) - float(step.get("cam_err_m", 0)), 5
            )
        samples.append(entry)

    kb["jacobian_samples"] = samples[-MAX_JACOBIAN_SAMPLES:]
    bins: Dict[str, dict] = dict(kb.get("jacobian_bins") or {})
    for s in samples[-len(steps) :]:
        b = s.get("q_bin")
        if not b:
            continue
        agg = dict(bins.get(b) or {"n": 0})
        agg["n"] = int(agg.get("n", 0)) + 1
        bins[b] = agg
    kb["jacobian_bins"] = bins


def _empty_kb() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": 0.0,
        "session_id": uuid.uuid4().hex[:12],
        "last_q": None,
        "last_target": None,
        "corners": {},
        "episodes": [],
        "stats": {
            "total_runs": 0,
            "successes": 0,
            "failures": 0,
        },
    }


def load_kb(path: Path = DEFAULT_KB_PATH) -> Dict[str, Any]:
    if not path.is_file():
        return _empty_kb()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_kb()
        data.setdefault("version", 1)
        data.setdefault("corners", {})
        data.setdefault("episodes", [])
        data.setdefault("stats", {"total_runs": 0, "successes": 0, "failures": 0})
        return data
    except (OSError, json.JSONDecodeError):
        return _empty_kb()


def save_kb(kb: Dict[str, Any], path: Path = DEFAULT_KB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kb["updated_at"] = time.time()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _corner_key(corner_idx: int, source: str = "") -> str:
    if corner_idx < 0:
        return "auto"
    kind = source.split(":", 1)[-1] if ":" in source else ""
    return f"{corner_idx}:{kind}" if kind else str(corner_idx)


def pick_corner_auto(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    perc_cfg,
    *,
    kb: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Pick any grid corner with valid depth; prefer least-tried in KB."""
    from pipelines.roarm_perception import detect_grid_corners, sample_depth_median

    corners, _kind = detect_grid_corners(bgr, perc_cfg)
    if corners is None or depth_m is None:
        return None
    h, w = bgr.shape[:2]
    valid: List[Tuple[int, float]] = []
    for i, pt in enumerate(corners.reshape(-1, 2)):
        px, py = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= px < w and 0 <= py < h):
            continue
        z = sample_depth_median(depth_m, px, py, perc_cfg.depth_median_radius)
        if z is None:
            continue
        if not (perc_cfg.depth_valid_min_m <= z <= perc_cfg.depth_valid_max_m):
            continue
        valid.append((i, float(z)))

    if not valid:
        return None

    corners_kb = (kb or {}).get("corners") or {}
    scored: List[Tuple[float, int]] = []
    for idx, z in valid:
        key = str(idx)
        entry = corners_kb.get(key) or {}
        tries = int(entry.get("attempts", 0))
        successes = int(entry.get("successes", 0))
        # Lower score = higher priority: fewer tries, more successes, closer depth
        score = tries * 2.0 - successes * 5.0 + abs(z - 0.8) * 0.1
        scored.append((score, idx))
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def update_kb_from_run(
    kb: Dict[str, Any],
    *,
    report: Dict[str, Any],
    approach: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge one approach run into knowledge base."""
    perception = report.get("perception") or {}
    tracking = approach.get("tracking") or {}
    final_t = approach.get("final_target") or perception
    corner_idx = int(final_t.get("corner_idx", perception.get("corner_idx", -1)))
    source = str(final_t.get("source") or perception.get("source") or "")
    key = _corner_key(corner_idx, source)

    q_final = approach.get("q_final")
    if isinstance(q_final, dict):
        kb["last_q"] = dict(q_final)

    kb["last_target"] = {
        "corner_idx": corner_idx,
        "source": source,
        "anchor_px": tracking.get("anchor_px") or final_t.get("px"),
        "anchor_py": tracking.get("anchor_py") or final_t.get("py"),
        "depth_m": final_t.get("depth_m"),
        "cam_err_m": final_t.get("cam_err_m"),
        "plane": (report.get("meta") or {}).get("plane"),
        "updated_at": time.time(),
    }

    corners = kb.setdefault("corners", {})
    entry = dict(corners.get(key) or {})
    entry["corner_idx"] = corner_idx
    entry["source"] = source
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    if approach.get("success"):
        entry["successes"] = int(entry.get("successes", 0)) + 1
        if isinstance(q_final, dict):
            entry["best_q"] = dict(q_final)
            entry["best_cam_err_m"] = float(final_t.get("cam_err_m") or 0)
    else:
        entry["failures"] = int(entry.get("failures", 0)) + 1
    entry["last_cam_err_m"] = float(final_t.get("cam_err_m") or approach.get("initial_depth_m") or 0)
    entry["last_success_reason"] = str(approach.get("success_reason", ""))
    entry["updated_at"] = time.time()
    corners[key] = entry

    stats = kb.setdefault("stats", {"total_runs": 0, "successes": 0, "failures": 0})
    stats["total_runs"] = int(stats.get("total_runs", 0)) + 1
    if approach.get("success"):
        stats["successes"] = int(stats.get("successes", 0)) + 1
    else:
        stats["failures"] = int(stats.get("failures", 0)) + 1

    episode = {
        "ts": time.time(),
        "success": bool(approach.get("success")),
        "success_reason": approach.get("success_reason"),
        "corner_idx": corner_idx,
        "source": source,
        "steps": len(approach.get("steps") or []),
        "q_final": q_final,
        "final_cam_err_m": final_t.get("cam_err_m"),
        "attempt": approach.get("attempt", 1),
    }
    episodes: List[dict] = list(kb.get("episodes") or [])
    episodes.append(episode)
    kb["episodes"] = episodes[-MAX_EPISODES:]
    append_jacobian_samples(kb, approach)
    return kb


def warm_start_q(kb: Dict[str, Any], corner_idx: int, source: str = "") -> Optional[Dict[str, float]]:
    """Best known joints for a corner (for fast re-approach)."""
    key = _corner_key(corner_idx, source)
    corners = kb.get("corners") or {}
    entry = corners.get(key) or corners.get(str(corner_idx))
    if not entry:
        return None
    best = entry.get("best_q")
    if isinstance(best, dict):
        return dict(best)
    return None
