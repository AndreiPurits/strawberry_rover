"""Plastic-lab capture: full frame + peduncle focus crop after berry approach.

Uses existing framing CENTER/REACH only — no new joint poses, no calyx/association.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipelines.peduncle_v3.geometry import build_focus_roi
from pipelines.peduncle_v3.framing import run_peduncle_framing
from pipelines.peduncle_v3.pipeline import load_config as load_peduncle_v3_config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "peduncle_plastic_lab" / "raw"
BBox = Tuple[float, float, float, float]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(p)


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


def _bbox_from_point(px: float, py: float, w: int, h: int, half: float = 40.0) -> BBox:
    return (
        max(0.0, px - half),
        max(0.0, py - half),
        min(float(w), px + half),
        min(float(h), py + half),
    )


def save_plastic_shot(
    *,
    bgr: np.ndarray,
    bbox: BBox,
    out_dir: Path,
    session_id: str,
    shot_idx: int,
    focus_cfg: Dict[str, Any],
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write full frame + focus crop + sidecar JSON. Returns paths/meta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = bgr.shape[:2]
    roi = build_focus_roi(
        bbox,
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
    stem = f"plastic_lab__{session_id}__shot{shot_idx:03d}"
    full_path = out_dir / f"{stem}_full.jpg"
    crop_path = out_dir / f"{stem}_crop.jpg"
    meta_path = out_dir / f"{stem}.json"
    cv2.imwrite(str(full_path), bgr)
    cv2.imwrite(str(crop_path), crop)
    meta: Dict[str, Any] = {
        "group": f"plastic_lab__{session_id}",
        "stem": stem,
        "session_id": session_id,
        "shot_idx": shot_idx,
        "full_path": _rel(full_path),
        "crop_path": _rel(crop_path),
        "image_wh": [w, h],
        "berry_bbox_xyxy": [float(v) for v in bbox],
        "peduncle_roi_xyxy": [int(rx1), int(ry1), int(rx2), int(ry2)],
        "crop_wh": [int(crop.shape[1]), int(crop.shape[0])],
        "domain": "plastic_lab",
        "labeled": False,
        "ts": time.time(),
    }
    if meta_extra:
        meta.update(meta_extra)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # append manifest
    man = out_dir / "manifest.jsonl"
    with man.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta


def run_plastic_capture_session(
    *,
    provider,
    tracker,
    execute_rpc: Callable,
    move_t102_smooth: Callable,
    read_q: Callable,
    q_now: Any,
    preferred_px: float,
    preferred_py: float,
    depth_hint: Optional[float],
    v3_cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
    n_shots: int = 8,
    settle_between_s: float = 0.35,
    label: str = "plastic",
    attempt_idx: int = 1,
    spd: float = 0.12,
    acc: float = 8.0,
    depth_min_m: float = 0.08,
    depth_max_m: float = 0.17,
) -> Dict[str, Any]:
    """After successful berry approach: framing + N saves using only CENTER/REACH.

    Between shots, re-run framing so existing safe CENTER/REACH produce viewpoint
    diversity. Does not call calyx / OBB / association / grasp.
    """
    v3_cfg = v3_cfg or load_peduncle_v3_config()
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_cfg = dict(v3_cfg.get("focus_roi") or {})
    log_dir = REPO / str((v3_cfg.get("logging") or {}).get("dir") or "runs/peduncle_v3_runtime")
    log_dir.mkdir(parents=True, exist_ok=True)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    shots: List[Dict[str, Any]] = []
    pref_x, pref_y = float(preferred_px), float(preferred_py)
    depth = float(depth_hint) if depth_hint is not None else None
    q = q_now

    for i in range(max(1, int(n_shots))):
        framing = run_peduncle_framing(
            provider=provider,
            tracker=tracker,
            execute_rpc=execute_rpc,
            move_t102_smooth=move_t102_smooth,
            read_q=read_q,
            q_now=q,
            preferred_px=pref_x,
            preferred_py=pref_y,
            depth_hint=depth,
            v3_cfg=v3_cfg,
            log_dir=log_dir,
            label=f"{label}_plastic",
            attempt_idx=attempt_idx * 100 + i,
            depth_max_m=depth_max_m,
            depth_min_m=depth_min_m,
            spd=spd,
            acc=acc,
        )
        if not framing.get("ok"):
            return {
                "ok": False,
                "reason": f"framing_failed:{framing.get('reason')}",
                "session_id": session_id,
                "shots": shots,
                "n_saved": len(shots),
            }

        berry_f = framing.get("berry") or {}
        if berry_f.get("px") is not None:
            pref_x, pref_y = float(berry_f["px"]), float(berry_f["py"])
        if berry_f.get("depth_m") is not None:
            depth = float(berry_f["depth_m"])

        bgr = _fetch_bgr(provider)
        if bgr is None:
            return {
                "ok": False,
                "reason": "no_frame",
                "session_id": session_id,
                "shots": shots,
                "n_saved": len(shots),
            }

        if framing.get("bbox") and len(framing["bbox"]) == 4:
            bbox = tuple(float(v) for v in framing["bbox"])  # type: ignore
        else:
            bbox = _bbox_from_point(pref_x, pref_y, bgr.shape[1], bgr.shape[0])

        meta = save_plastic_shot(
            bgr=bgr,
            bbox=bbox,  # type: ignore[arg-type]
            out_dir=out_dir,
            session_id=session_id,
            shot_idx=i,
            focus_cfg=focus_cfg,
            meta_extra={
                "depth_m": depth,
                "framing_action": framing.get("action"),
                "framing_reason": framing.get("reason"),
                "berry_pxpy": [pref_x, pref_y],
            },
        )
        shots.append(meta)
        print(
            f"[plastic-capture] shot {i+1}/{n_shots} → {meta['stem']} "
            f"crop={meta['crop_wh']} action={framing.get('action')}"
        )

        try:
            q = read_q()
        except Exception:
            pass
        if i + 1 < n_shots and settle_between_s > 0:
            time.sleep(float(settle_between_s))

    summary = {
        "ok": True,
        "reason": "ok",
        "session_id": session_id,
        "group": f"plastic_lab__{session_id}",
        "out_dir": str(out_dir),
        "n_saved": len(shots),
        "shots": shots,
    }
    (out_dir / f"session_{session_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
