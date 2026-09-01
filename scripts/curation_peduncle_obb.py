#!/usr/bin/env python3
"""Peduncle OBB editor for approved crops (4-corner / YOLO-OBB).

Dataset: data/плодоножки апрувд/
  labels/           — berry AABB class 0 (editable in GUI)
  labels_peduncle/  — peduncle OBB class 1: `1 x1 y1 x2 y2 x3 y3 x4 y4` (normalized)

UI: ?dataset=peduncle_obb
Draw: drag along stem (calyx → tip), move to set width, click to confirm.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Active labeling root shown in UI. Approved set used for balance counters.
APPROVED_ROOT = REPO_ROOT / "data" / "плодоножки апрувд"
QUEUE_ROOT = REPO_ROOT / "data" / "peduncle_labeling_queue"
ROOT = APPROVED_ROOT
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EDITOR_MAX_SIDE = 1280
CLASS_PEDUNCLE = 1


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ensure_symlink(link: Path, target: Path) -> None:
    """Create/replace symlink link -> target (relative if possible)."""
    if not target.exists():
        return
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return  # real dir/file — do not clobber
    try:
        rel = target.resolve().relative_to(link.parent.resolve())
        link.symlink_to(rel)
    except Exception:
        link.symlink_to(target.resolve())


def ensure_layout() -> None:
    # Queue layout uses crops/ + labels_berry/; UI expects images/ + labels/
    if (ROOT / "crops").is_dir():
        _ensure_symlink(ROOT / "images", ROOT / "crops")
    if (ROOT / "labels_berry").is_dir():
        _ensure_symlink(ROOT / "labels", ROOT / "labels_berry")
    for d in ("images", "labels", "labels_peduncle", "labels_peduncle_meta", "meta", "bbox_vis", "reports", "calyx_meta", "deleted"):
        _safe_mkdir(ROOT / d)
    for d in ("images", "labels", "labels_peduncle", "labels_peduncle_meta", "meta", "bbox_vis", "calyx_meta"):
        _safe_mkdir(ROOT / "deleted" / d)


def delete_frame(filename: str) -> Tuple[bool, str]:
    """Move one crop out of the labeling queue into deleted/ (recoverable)."""
    p = image_path(filename)
    if not p.is_file():
        return False, "image not found"
    ensure_layout()
    stem = p.stem
    name = p.name
    mapping = [
        (ROOT / "images" / name, ROOT / "deleted" / "images" / name),
        (ROOT / "labels" / f"{stem}.txt", ROOT / "deleted" / "labels" / f"{stem}.txt"),
        (ROOT / "labels_peduncle" / f"{stem}.txt", ROOT / "deleted" / "labels_peduncle" / f"{stem}.txt"),
        (ROOT / "labels_peduncle_meta" / f"{stem}.json", ROOT / "deleted" / "labels_peduncle_meta" / f"{stem}.json"),
        (ROOT / "meta" / f"{stem}.json", ROOT / "deleted" / "meta" / f"{stem}.json"),
        (ROOT / "calyx_meta" / f"{stem}.json", ROOT / "deleted" / "calyx_meta" / f"{stem}.json"),
        (ROOT / "bbox_vis" / name, ROOT / "deleted" / "bbox_vis" / name),
    ]
    moved = 0
    for src, dst in mapping:
        if not src.is_file():
            continue
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        moved += 1
    if moved == 0:
        return False, "nothing to delete"
    return True, "ok"


def _file_stem(filename: str) -> str:
    p = Path(filename)
    if p.suffix.lower() in IMG_EXTS | {".txt", ".json"}:
        return p.stem
    return p.name


def calyx_meta_path(filename: str) -> Path:
    return ROOT / "calyx_meta" / f"{_file_stem(filename)}.json"


def peduncle_meta_path(filename: str) -> Path:
    return ROOT / "labels_peduncle_meta" / f"{_file_stem(filename)}.json"


def _norm_relation(v: Any) -> Optional[str]:
    if v in ("target", "other"):
        return str(v)
    return None


def _obbs_relations_complete(obbs: List[Dict[str, Any]]) -> bool:
    if not obbs:
        return True
    return all(_norm_relation(o.get("relation")) is not None for o in obbs)


def frame_is_reviewed(filename: str, obbs: Optional[List[Dict[str, Any]]] = None) -> bool:
    calyx = _load_calyx_meta(filename)
    if not calyx.get("reviewed"):
        return False
    if obbs is not None:
        if not _obbs_relations_complete(obbs):
            return False
        try:
            return bool(json.loads(peduncle_meta_path(filename).read_text(encoding="utf-8")).get("confirmed"))
        except Exception:
            return False
    lab = peduncle_label_path(filename)
    n = 0
    if lab.is_file():
        n = sum(1 for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
    if n == 0:
        # no peduncle OBB: reviewed if calyx done + Save confirmed (empty sidecar is OK)
        try:
            return bool(json.loads(peduncle_meta_path(filename).read_text(encoding="utf-8")).get("confirmed"))
        except Exception:
            return False
    side = _load_relation_sidecar(lab)
    if not side:
        return False
    if len(side) != n:
        return False
    if not all(_norm_relation(s.get("relation")) is not None for s in side):
        return False
    try:
        conf = json.loads(peduncle_meta_path(filename).read_text(encoding="utf-8")).get("confirmed")
    except Exception:
        conf = False
    return bool(conf)


def _normalize_calyx_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unset visibility → not visible. Point kept if present (inferred / hidden)."""
    out = dict(data) if data else {}
    vis = out.get("calyx_visible")
    if vis is None:
        out["calyx_visible"] = False
    else:
        out["calyx_visible"] = bool(vis)
    pt = out.get("calyx_point")
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        try:
            out["calyx_point"] = [float(pt[0]), float(pt[1])]
        except Exception:
            out["calyx_point"] = None
    else:
        out["calyx_point"] = None
    out["reviewed"] = bool(out.get("reviewed", False))
    return out


def _load_calyx_meta(filename: str) -> Dict[str, Any]:
    p = calyx_meta_path(filename)
    if p.is_file():
        try:
            return _normalize_calyx_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {"calyx_visible": False, "calyx_point": None, "reviewed": False}


def _save_calyx_meta(filename: str, calyx_visible: Any, calyx_point: Any) -> None:
    ensure_layout()
    # unset visibility → not visible (point may still exist as inferred)
    data = _normalize_calyx_dict(
        {
            "calyx_visible": False if calyx_visible is None else bool(calyx_visible),
            "calyx_point": calyx_point,
            "reviewed": True,
        }
    )
    calyx_meta_path(filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def migrate_calyx_null_visibility() -> Dict[str, int]:
    """Disk: calyx_visible=null → false. Keep point if any (invisible+point); else invis+no point."""
    ensure_layout()
    stats = {"scanned": 0, "updated": 0, "invis_point": 0, "invis_no_point": 0, "skipped": 0}
    d = ROOT / "calyx_meta"
    if not d.is_dir():
        return stats
    for p in d.glob("*.json"):
        stats["scanned"] += 1
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            stats["skipped"] += 1
            continue
        if raw.get("calyx_visible") is not None:
            stats["skipped"] += 1
            continue
        norm = _normalize_calyx_dict({**raw, "calyx_visible": False, "reviewed": True if raw.get("reviewed") else raw.get("reviewed", False)})
        # if file was already marked reviewed with null vis — keep reviewed; else leave as-is
        if raw.get("reviewed"):
            norm["reviewed"] = True
        p.write_text(json.dumps(norm, indent=2), encoding="utf-8")
        stats["updated"] += 1
        if norm.get("calyx_point"):
            stats["invis_point"] += 1
        else:
            stats["invis_no_point"] += 1
    return stats


def frame_has_unreviewed_relation(filename: str) -> bool:
    lab = peduncle_label_path(filename)
    n = 0
    if lab.is_file():
        n = sum(1 for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
    if n == 0:
        return False
    side = _load_relation_sidecar(lab)
    if len(side) != n:
        return True
    return any(_norm_relation(s.get("relation")) is None for s in side)


def calyx_review_stats() -> Dict[str, int]:
    st = dataset_stats()
    return {"total": st["frames_total"], "reviewed": st["frames_reviewed"], "unreviewed": st["frames_unreviewed"]}


def list_images() -> List[Path]:
    d = ROOT / "images"
    if not d.is_dir():
        return []
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    def _key(p: Path):
        # 1) OBB without target/other first, 2) then other unreviewed, 3) reviewed
        rel_gap = 0 if frame_has_unreviewed_relation(p.name) else 1
        reviewed = 1 if frame_is_reviewed(p.name) else 0
        return (rel_gap, reviewed, p.name)
    return sorted(imgs, key=_key)


def image_path(filename: str) -> Path:
    return ROOT / "images" / Path(filename).name


def berry_label_path(filename: str) -> Path:
    return ROOT / "labels" / f"{Path(filename).stem}.txt"


def peduncle_label_path(filename: str) -> Path:
    return ROOT / "labels_peduncle" / f"{Path(filename).stem}.txt"


def _editor_display_size(w: int, h: int, max_side: int = EDITOR_MAX_SIDE) -> Tuple[int, int, float]:
    side = max(w, h)
    if side <= max_side:
        return w, h, 1.0
    s = max_side / float(side)
    return int(round(w * s)), int(round(h * s)), s


def _parse_berry_aabb(label_file: Path, w: int, h: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if not label_file.is_file():
        return out
    for raw in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 5:
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, bw, bh = map(float, toks[1:5])
        except Exception:
            continue
        if cid != 0:
            continue
        x1 = max(0.0, (xc - bw / 2.0) * w)
        y1 = max(0.0, (yc - bh / 2.0) * h)
        x2 = min(float(w), (xc + bw / 2.0) * w)
        y2 = min(float(h), (yc + bh / 2.0) * h)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append({"cls": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def _berry_to_yolo_line(box: Dict[str, Any], w: int, h: int) -> Optional[str]:
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 2:
        return None
    xc = ((x1 + x2) / 2.0) / float(w)
    yc = ((y1 + y2) / 2.0) / float(h)
    return f"0 {xc:.6f} {yc:.6f} {bw / float(w):.6f} {bh / float(h):.6f}"


def _order_corners_quad(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Order 4 points clockwise starting from top-most then left-most."""
    if len(pts) != 4:
        return pts
    arr = np.array(pts, dtype=np.float32)
    # centroid
    c = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    order = np.argsort(angles)  # CCW from +x; reverse for CW visual consistency
    ordered = arr[order]
    # start from point with smallest y (then x)
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1]))[0])
    ordered = np.roll(ordered, -start, axis=0)
    return [(float(p[0]), float(p[1])) for p in ordered]


def _parse_peduncle_obb(label_file: Path, w: int, h: int) -> List[Dict[str, Any]]:
    """Parse YOLO-OBB (class + 8 normalized coords) or legacy AABB (class + 4)."""
    out: List[Dict[str, Any]] = []
    if not label_file.is_file():
        return out
    for raw in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 5:
            continue
        try:
            cid = int(float(toks[0]))
        except Exception:
            continue
        if cid != CLASS_PEDUNCLE and cid != 1:
            # allow class 1 only for peduncle file
            pass
        try:
            if len(toks) >= 9:
                vals = list(map(float, toks[1:9]))
                pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 8, 2)]
                pts = _order_corners_quad(pts)
            else:
                xc, yc, bw, bh = map(float, toks[1:5])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        except Exception:
            continue
        out.append(
            {
                "cls": CLASS_PEDUNCLE,
                "points": [{"x": float(x), "y": float(y)} for x, y in pts],
                "relation": None,
            }
        )
    _apply_relations(label_file, out, w, h)
    return out


def _obb_centroid(obb: Dict[str, Any]) -> Tuple[float, float]:
    pts = obb.get("points") or []
    if not pts:
        return (0.0, 0.0)
    xs = [float(p["x"]) for p in pts]
    ys = [float(p["y"]) for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _load_relation_sidecar(label_file: Path) -> List[Dict[str, Any]]:
    meta_p = ROOT / "labels_peduncle_meta" / f"{label_file.stem}.json"
    if not meta_p.is_file():
        return []
    try:
        data = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return []
    obbs = data.get("obbs")
    if not isinstance(obbs, list):
        return []
    return [x for x in obbs if isinstance(x, dict)]


def _apply_relations(label_file: Path, out: List[Dict[str, Any]], w: int, h: int) -> None:
    side = _load_relation_sidecar(label_file)
    if not out:
        return
    if side and len(side) == len(out):
        for dst, src in zip(out, side):
            dst["relation"] = _norm_relation(src.get("relation"))
        return
    if side:
        used = set()
        for dst in out:
            dcx, dcy = _obb_centroid(dst)
            best_i, best_d = -1, 1e18
            for i, src in enumerate(side):
                if i in used:
                    continue
                pts = src.get("points") or src.get("xyxyxyxy_norm")
                if isinstance(pts, list) and len(pts) == 4 and isinstance(pts[0], dict):
                    sx = sum(float(p["x"]) for p in pts) / 4.0
                    sy = sum(float(p["y"]) for p in pts) / 4.0
                elif isinstance(pts, list) and len(pts) >= 8:
                    sx = sum(float(pts[i]) * w for i in range(0, 8, 2)) / 4.0
                    sy = sum(float(pts[i]) * h for i in range(1, 8, 2)) / 4.0
                else:
                    continue
                d = (dcx - sx) ** 2 + (dcy - sy) ** 2
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                used.add(best_i)
                dst["relation"] = _norm_relation(side[best_i].get("relation"))
        return
    # no sidecar: safe default only for a single OBB on an already calyx-reviewed frame
    calyx = _load_calyx_meta(label_file.stem)
    if len(out) == 1 and calyx.get("reviewed"):
        out[0]["relation"] = "target"


def _save_relation_sidecar(
    filename: str,
    peduncle_obbs: List[Dict[str, Any]],
    w: int,
    h: int,
    *,
    confirmed: bool = True,
) -> None:
    ensure_layout()
    obbs = []
    for obb in peduncle_obbs or []:
        pts = obb.get("points")
        if not isinstance(pts, list) or len(pts) != 4:
            continue
        line = _obb_to_yolo_line(pts, w, h)
        if not line:
            continue
        coords = [float(x) for x in line.split()[1:9]]
        obbs.append(
            {
                "cls": CLASS_PEDUNCLE,
                "relation": _norm_relation(obb.get("relation")),
                "xyxyxyxy_norm": coords,
            }
        )
    if not obbs:
        confirmed = bool(confirmed)
    else:
        confirmed = bool(confirmed) and all(_norm_relation(o.get("relation")) is not None for o in obbs)
    peduncle_meta_path(filename).write_text(
        json.dumps(
            {
                "format": "peduncle_relation_v1",
                "yolo_class": 1,
                "yolo_class_name": "peduncle",
                "note": "relation is metadata only; do not use as a YOLO train class",
                "confirmed": confirmed,
                "obbs": obbs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def migrate_relations() -> Dict[str, int]:
    """Write missing sidecars. Single OBB + calyx-reviewed → target; else unreviewed."""
    ensure_layout()
    stats = {"wrote": 0, "safe_target": 0, "unreviewed_rel": 0, "empty": 0, "skipped": 0}
    img_dir = ROOT / "images"
    if not img_dir.is_dir():
        return stats
    for p in img_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        meta_p = peduncle_meta_path(p.name)
        if meta_p.is_file():
            try:
                existing = json.loads(meta_p.read_text(encoding="utf-8"))
                rels = [_norm_relation((x or {}).get("relation")) for x in (existing.get("obbs") or [])]
                if any(r is not None for r in rels):
                    stats["skipped"] += 1
                    continue
            except Exception:
                pass
            try:
                meta_p.unlink()
            except Exception:
                pass
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        obbs = _parse_peduncle_obb(peduncle_label_path(p.name), w, h)
        _save_relation_sidecar(p.name, obbs, w, h, confirmed=False)
        stats["wrote"] += 1
        rels = [_norm_relation(o.get("relation")) for o in obbs]
        if not obbs:
            stats["empty"] += 1
        elif rels == ["target"]:
            stats["safe_target"] += 1
        else:
            stats["unreviewed_rel"] += 1
    return stats


def _pct(n: int, d: int) -> float:
    return 0.0 if d <= 0 else round(100.0 * n / d, 1)


def balance_plan(
    *,
    with_target: int,
    only_other: int,
    no_stem: int,
    target_share: float = 0.70,
    other_share: float = 0.15,
    no_stem_share: float = 0.15,
) -> Dict[str, Any]:
    """Minimal adds of only-other / no-stem so each is ~15%, target fixed (no new target crops)."""
    t = max(0, int(with_target))
    o = max(0, int(only_other))
    ns = max(0, int(no_stem))
    if t <= 0:
        return {
            "mode": "fixed_target",
            "target_fixed": t,
            "only_other_now": o,
            "no_stem_now": ns,
            "frames_now": t + o + ns,
            "frames_final": t + o + ns,
            "only_other_final": o,
            "no_stem_final": ns,
            "need_only_other": 0,
            "need_no_stem": 0,
            "need_total": 0,
            "target_final_pct": 0.0,
            "only_other_final_pct": 0.0,
            "no_stem_final_pct": 0.0,
            "note": "нет target-кадров — план не считается",
        }
    # N >= T / target_share so target ≤ 70%; O and NS each ≥ 15%
    n_final = int(math.ceil(t / float(target_share)))
    o_final = int(math.ceil(other_share * n_final))
    ns_final = int(math.ceil(no_stem_share * n_final))
    # exact sum: keep target fixed, pad the larger of the two minorities if needed
    while t + o_final + ns_final < n_final:
        # prefer balancing the lagging share
        if o_final <= ns_final:
            o_final += 1
        else:
            ns_final += 1
    n_final = t + o_final + ns_final
    need_o = max(0, o_final - o)
    need_ns = max(0, ns_final - ns)
    return {
        "mode": "fixed_target",
        "source_hint": "очередь data/peduncle_labeling_queue (кропы; berry prefilled)",
        "target_fixed": t,
        "only_other_now": o,
        "no_stem_now": ns,
        "frames_now": t + o + ns,
        "frames_final": n_final,
        "only_other_final": o_final,
        "no_stem_final": ns_final,
        "need_only_other": need_o,
        "need_no_stem": need_ns,
        "need_total": need_o + need_ns,
        "target_final_pct": _pct(t, n_final),
        "only_other_final_pct": _pct(o_final, n_final),
        "no_stem_final_pct": _pct(ns_final, n_final),
        "shares": {
            "target_stem": f"{int(target_share * 100)}%",
            "only_other": f"{int(other_share * 100)}%",
            "no_stem": f"{int(no_stem_share * 100)}%",
        },
        "note": "target не добавляем; только only-other и no-stem до ~15% каждый",
    }


def _group_counts_for_root(root: Path) -> Dict[str, int]:
    """Frame-group counts for a dataset root (images + labels_peduncle + meta)."""
    img_dir = root / "images"
    if not img_dir.is_dir() and (root / "crops").is_dir():
        img_dir = root / "crops"
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS] if img_dir.is_dir() else []
    with_target = only_other = no_stem = 0
    for p in files:
        stem = _file_stem(p.name)
        lab = root / "labels_peduncle" / f"{stem}.txt"
        n = 0
        if lab.is_file():
            n = sum(1 for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
        meta_p = root / "labels_peduncle_meta" / f"{stem}.json"
        side: List[Dict[str, Any]] = []
        if meta_p.is_file():
            try:
                data = json.loads(meta_p.read_text(encoding="utf-8"))
                obbs = data.get("obbs")
                if isinstance(obbs, list):
                    side = [x for x in obbs if isinstance(x, dict)]
            except Exception:
                side = []
        rels: List[Optional[str]] = []
        if side and len(side) == n:
            rels = [_norm_relation(s.get("relation")) for s in side]
        else:
            rels = [_norm_relation(side[i].get("relation")) if i < len(side) else None for i in range(n)]
        n_t = sum(1 for r in rels if r == "target")
        n_o = sum(1 for r in rels if r == "other")
        if n_t > 0:
            with_target += 1
        elif n_o > 0:
            only_other += 1
        if n == 0:
            no_stem += 1
    return {
        "frames_total": len(files),
        "with_target": with_target,
        "only_other": only_other,
        "no_stem": no_stem,
    }


def dataset_stats() -> Dict[str, Any]:
    img_dir = ROOT / "images"
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS] if img_dir.is_dir() else []
    frames_total = len(files)
    frames_reviewed = 0
    with_target = 0
    only_other = 0
    no_stem = 0
    total_obb = 0
    target_obb = 0
    other_obb = 0
    rel_unreviewed_obb = 0
    calyx_vis_pt = 0
    calyx_hid_pt = 0
    calyx_hid_nopt = 0
    calyx_unrev = 0
    for p in files:
        lab = peduncle_label_path(p.name)
        n = 0
        if lab.is_file():
            n = sum(1 for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())
        side = _load_relation_sidecar(lab)
        rels: List[Optional[str]] = []
        if side and len(side) == n:
            rels = [_norm_relation(s.get("relation")) for s in side]
        elif n == 1 and _load_calyx_meta(p.name).get("reviewed") and not side:
            rels = ["target"]
        else:
            rels = [_norm_relation(s.get("relation")) if i < len(side) else None for i, s in enumerate((side or [])[:n])]
            if len(rels) < n:
                rels.extend([None] * (n - len(rels)))
        n_t = sum(1 for r in rels if r == "target")
        n_o = sum(1 for r in rels if r == "other")
        n_u = n - n_t - n_o
        total_obb += n
        target_obb += n_t
        other_obb += n_o
        rel_unreviewed_obb += n_u
        if n_t > 0:
            with_target += 1
        elif n_o > 0:
            only_other += 1
        if n == 0:
            no_stem += 1
        if frame_is_reviewed(p.name):
            frames_reviewed += 1
        calyx = _load_calyx_meta(p.name)
        vis = calyx.get("calyx_visible")
        pt = calyx.get("calyx_point")
        rev = bool(calyx.get("reviewed"))
        has_pt = isinstance(pt, (list, tuple)) and len(pt) == 2
        if not rev:
            calyx_unrev += 1
        elif vis is True and has_pt:
            calyx_vis_pt += 1
        elif vis is False and has_pt:
            calyx_hid_pt += 1
        elif vis is False and not has_pt:
            calyx_hid_nopt += 1
        else:
            calyx_unrev += 1
    frames_unreviewed = frames_total - frames_reviewed
    warnings: List[str] = []
    no_stem_pct = _pct(no_stem, frames_total)
    target_frame_pct = _pct(with_target, frames_total)
    if no_stem_pct > 25 and frames_reviewed > 0:
        warnings.append("negative кадров >25% — возможно слишком много фона")
    if target_frame_pct < 40 and frames_reviewed > frames_total * 0.5:
        warnings.append("target stem <40% — недостаточно положительных примеров")
    if other_obb == 0 and frames_total and frames_reviewed > 0:
        warnings.append("other stem = 0 — проверить, что сложные кусты не размечаются неправильно")
    # Balance plan always vs approved labeled set (not the active queue)
    approved = _group_counts_for_root(APPROVED_ROOT)
    plan = balance_plan(
        with_target=approved["with_target"],
        only_other=approved["only_other"],
        no_stem=approved["no_stem"],
    )
    plan["approved_root"] = str(APPROVED_ROOT)
    plan["queue_root"] = str(ROOT)
    plan["queue_frames"] = frames_total
    if plan["need_total"] > 0:
        warnings.append(
            f"до баланса 70/15/15 (по апрувд) осталось: only-other {plan['need_only_other']}, "
            f"no-stem {plan['need_no_stem']} (итого +{plan['need_total']} → {plan['frames_final']})"
        )
    return {
        "frames_total": frames_total,
        "frames_reviewed": frames_reviewed,
        "frames_unreviewed": frames_unreviewed,
        "frames_reviewed_pct": _pct(frames_reviewed, frames_total),
        "frames_unreviewed_pct": _pct(frames_unreviewed, frames_total),
        "frames_with_target_stem": with_target,
        "frames_with_target_stem_pct": target_frame_pct,
        "frames_with_only_other_stem": only_other,
        "frames_with_only_other_stem_pct": _pct(only_other, frames_total),
        "frames_with_no_stem": no_stem,
        "frames_with_no_stem_pct": no_stem_pct,
        "obb_total": total_obb,
        "obb_target": target_obb,
        "obb_other": other_obb,
        "obb_relation_unreviewed": rel_unreviewed_obb,
        "obb_target_pct": _pct(target_obb, total_obb),
        "obb_other_pct": _pct(other_obb, total_obb),
        "obb_relation_unreviewed_pct": _pct(rel_unreviewed_obb, total_obb),
        "calyx_visible_point": calyx_vis_pt,
        "calyx_invisible_point": calyx_hid_pt,
        "calyx_invisible_no_point": calyx_hid_nopt,
        "calyx_unreviewed": calyx_unrev,
        "calyx_visible_point_pct": _pct(calyx_vis_pt, frames_total),
        "calyx_invisible_point_pct": _pct(calyx_hid_pt, frames_total),
        "calyx_invisible_no_point_pct": _pct(calyx_hid_nopt, frames_total),
        "calyx_unreviewed_pct": _pct(calyx_unrev, frames_total),
        "targets": {
            "target_stem_pct": "70% (фиксируем апрувд target)",
            "only_other_pct": "15%",
            "no_stem_pct": "15%",
        },
        "active_root": str(ROOT),
        "approved_counts": approved,
        "balance_plan": plan,
        "warnings": warnings,
    }


def _obb_to_yolo_line(points: List[Dict[str, Any]], w: int, h: int) -> Optional[str]:
    if len(points) != 4:
        return None
    coords: List[float] = []
    for p in points:
        try:
            x = float(p["x"]) / float(w)
            y = float(p["y"]) / float(h)
        except Exception:
            return None
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        coords.extend([x, y])
    # reject degenerate
    arr = np.array([(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)], dtype=np.float32)
    area = float(abs(cv2.contourArea(arr.reshape(-1, 1, 2))))
    if area < 16:
        return None
    return "1 " + " ".join(f"{v:.6f}" for v in coords)


def get_annotation(filename: str) -> Optional[Dict[str, Any]]:
    p = image_path(filename)
    if not p.is_file():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    dw, dh, _ = _editor_display_size(w, h)
    calyx = _load_calyx_meta(filename)
    return {
        "filename": p.name,
        "width": w,
        "height": h,
        "display_width": dw,
        "display_height": dh,
        "berry_boxes": _parse_berry_aabb(berry_label_path(filename), w, h),
        "peduncle_obbs": _parse_peduncle_obb(peduncle_label_path(filename), w, h),
        "calyx_visible": calyx.get("calyx_visible"),
        "calyx_point": calyx.get("calyx_point"),
        "calyx_reviewed": calyx.get("reviewed", False),
        "frame_reviewed": frame_is_reviewed(filename),
        "format": "yolo_obb_4pts",
        "tip_cm": "Цель: ~2.5 см стебля от чашечки (допустимо 2–3.5). Косой бокс по оси стебля.",
    }


def save_annotation(filename: str, peduncle_obbs: List[Dict[str, Any]],
                    calyx_visible: Any = None, calyx_point: Any = None,
                    berry_boxes: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, str]:
    p = image_path(filename)
    if not p.is_file():
        return False, "image not found"
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, "cannot read image"
    h, w = bgr.shape[:2]
    ensure_layout()
    lines: List[str] = []
    for obb in peduncle_obbs or []:
        pts = obb.get("points")
        if not isinstance(pts, list):
            continue
        line = _obb_to_yolo_line(pts, w, h)
        if line:
            lines.append(line)
    peduncle_label_path(filename).write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    _save_relation_sidecar(filename, peduncle_obbs, w, h)
    if berry_boxes is not None:
        blines: List[str] = []
        for b in berry_boxes:
            line = _berry_to_yolo_line(b, w, h)
            if line:
                blines.append(line)
        berry_label_path(filename).write_text(("\n".join(blines) + ("\n" if blines else "")), encoding="utf-8")

    # preview overlay
    try:
        vis = bgr.copy()
        vis_berries = berry_boxes if berry_boxes is not None else _parse_berry_aabb(berry_label_path(filename), w, h)
        for b in vis_berries:
            cv2.rectangle(
                vis,
                (int(b["x1"]), int(b["y1"])),
                (int(b["x2"]), int(b["y2"])),
                (0, 220, 0),
                2,
            )
        for obb in peduncle_obbs or []:
            pts = obb.get("points") or []
            if len(pts) != 4:
                continue
            poly = np.array([[int(round(float(p["x"]))), int(round(float(p["y"])))] for p in pts], dtype=np.int32)
            cv2.polylines(vis, [poly], True, (0, 140, 255), 2)
        cv2.imwrite(str(ROOT / "bbox_vis" / p.name), vis)
    except Exception:
        pass
    _save_calyx_meta(filename, calyx_visible, calyx_point)
    return True, "ok"


def render_raw_jpeg(filename: str, max_side: int = EDITOR_MAX_SIDE) -> Optional[bytes]:
    p = image_path(filename)
    if not p.is_file():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    dw, dh, _ = _editor_display_size(w, h, max_side)
    if (dw, dh) != (w, h):
        bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return bytes(buf) if ok else None


def render_peduncle_obb_html(*, index: int = 0) -> str:
    ensure_layout()
    images = list_images()
    n = len(images)
    if n == 0:
        return f"""<!doctype html><html lang="ru"><body style="font-family:system-ui;margin:16px">
        <h2>Peduncle OBB</h2>
        <p>Пусто: <code>{ROOT}/images</code></p>
        </body></html>"""

    index = max(0, min(int(index), n - 1))
    names_json = json.dumps([p.name for p in images], ensure_ascii=False)
    stats = dataset_stats()
    stats_json = json.dumps(stats, ensure_ascii=False)
    rel_gap = sum(1 for p in images if frame_has_unreviewed_relation(p.name))
    # open first relation-gap frame when present (ignore ?i= resume for this queue)
    if rel_gap > 0:
        index = next((i for i, p in enumerate(images) if frame_has_unreviewed_relation(p.name)), 0)
    cur = images[index]
    prev_i = max(0, index - 1)
    next_i = min(n - 1, index + 1)

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Peduncle OBB — косой бокс</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 12px; background:#111; color:#eee; }}
    .card {{ max-width: 1200px; }}
    .row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0; }}
    button, a.btn {{ padding:8px 12px; border-radius:8px; border:1px solid #555; background:#222; color:#eee; cursor:pointer; text-decoration:none; }}
    button.primary {{ background:#1b5e20; border-color:#2e7d32; }}
    button.danger {{ background:#b71c1c; border-color:#e53935; }}
    .meta {{ color:#bbb; font-size:13px; }}
    code {{ background:#333; padding:2px 6px; border-radius:4px; }}
    kbd {{ background:#333; padding:2px 6px; border-radius:4px; font-size:12px; }}
    #wrap {{ position:relative; display:inline-block; max-width:100%; border:1px solid #444; background:#000; }}
    canvas {{ display:block; max-width:100%; cursor:crosshair; }}
    #status {{ min-height:1.2em; color:#8bc34a; }}
    #status.err {{ color:#ef9a9a; }}
    .legend span {{ display:inline-block; padding:2px 8px; border-radius:6px; margin-right:6px; font-size:12px; }}
    .berry {{ background:#1b5e20; }}
    .ped-other {{ background:#0277bd; }}
    .rel-box {{ display:inline-flex; gap:4px; align-items:center; padding:4px 8px; border:1px solid #555; border-radius:8px; }}
    .rel-box button {{ padding:4px 8px; }}
    .rel-box button.on-target {{ background:#e65100; border-color:#ff9800; }}
    .rel-box button.on-other {{ background:#0277bd; border-color:#4fc3f7; }}
    .stats {{ font-size:12px; color:#bbb; background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:8px 10px; margin:8px 0; max-width:900px; }}
    .stats b {{ color:#eee; }}
    .stats .warn {{ color:#ef9a9a; }}
    .stats .ok {{ color:#8bc34a; }}
    .stats .row2 {{ display:flex; flex-wrap:wrap; gap:12px 18px; margin-top:4px; }}
    .calyx-leg {{ background:#6a1b9a; }}
    .calyx-toggle {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:8px; background:#222; border:1px solid #555; font-size:13px; }}
    .calyx-toggle.active {{ border-color:#ab47bc; background:#2a1040; }}
    .calyx-toggle.unreviewed {{ border-color:#ff8f00; }}
    .review-badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:12px; }}
    .review-badge.yes {{ background:#1b5e20; }}
    .review-badge.no {{ background:#e65100; }}
    #reviewStats {{ font-size:13px; color:#aaa; }}
    .hints {{ font-size:11px; color:#999; background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:6px 10px; margin:6px 0; max-width:700px; }}
    .hints b {{ color:#ccc; }}
  </style>
</head>
<body>
  <div class="card">
    <h2 style="margin:0 0 8px">Плодоножка — косой бокс (4 точки) + Calyx</h2>
    <p class="meta">
      Кадр <b id="pos">{index + 1}</b> / <b>{n}</b> · <code id="fname">{cur.name}</code>
      · <span id="reviewBadge" class="review-badge no">calyx unreviewed</span>
      · <span id="reviewStats">reviewed {stats['frames_reviewed']}/{stats['frames_total']}</span>
      · <a class="btn" href="/?dataset=peduncle">approve UI</a>
    </p>
    <div class="row legend">
      <span class="berry">berry (target)</span>
      <span class="ped">target OBB</span>
      <span class="ped-other">other OBB</span>
      <span class="calyx-leg">calyx point</span>
    </div>
    <div class="row">
      <button type="button" id="btnToolBerry">Berry box (Y)</button>
      <button type="button" id="btnToolPed">Peduncle OBB (X)</button>
      <button type="button" id="btnSave" class="primary">Save + next (S)</button>
      <label id="calyxToggle" class="calyx-toggle" title="Чашечка видна?">
        <input type="checkbox" id="chkCalyx" />
        Calyx visible
      </label>
      <button type="button" id="btnCalyxPlace" style="border-color:#ab47bc">Place calyx (V)</button>
      <button type="button" id="btnCalyxClear" style="border-color:#666">Clear calyx pt</button>
      <span class="rel-box">
        relation:
        <button type="button" id="btnRelTarget">Target (T)</button>
        <button type="button" id="btnRelOther">Other (O)</button>
      </span>
      <button type="button" id="btnDel" class="danger">Delete selected (Del)</button>
      <button type="button" id="btnDelFrame" class="danger">Удалить кадр полностью</button>
      <button type="button" id="btnClear">Clear peduncle</button>
      <button type="button" id="btnCancel">Cancel draw (Esc)</button>
      <a class="btn" id="btnPrev" href="/?dataset=peduncle_obb&i={prev_i}">← prev</a>
      <a class="btn" id="btnNext" href="/?dataset=peduncle_obb&i={next_i}">next →</a>
    </div>
    <div class="hints">
      <b>OBB:</b> Тяни вдоль стебля → ширина → клик.
      Stem виден от calyx → OBB от выхода stem.
      Начало закрыто → OBB с видимой части (не дорисовывать до calyx).
      Короткое перекрытие — один OBB. Длинное — не угадывать.<br/>
      <b>Calyx:</b> 3 состояния:
      1) <b>visible+point</b> — calyx виден, ставим точку;
      2) <b>hidden+point</b> — точка крепления закрыта, но положение уверенно восстановимо;
      3) <b>hidden, no point</b> — нельзя определить.<br/>
      <kbd>Y</kbd> ягода · <kbd>X</kbd> плодоножка · <kbd>T</kbd>/<kbd>O</kbd> relation · <kbd>V</kbd> calyx · <kbd>C</kbd> visible · <kbd>S</kbd> save+next · <kbd>Del</kbd>/<kbd>Backspace</kbd> выбранный бокс · кнопка «Удалить кадр полностью»
    </div>
    <div id="statsBox" class="stats"></div>
    <div id="status"></div>
    <div id="wrap"><canvas id="cv"></canvas></div>
  </div>
<script>
(function() {{
  const NAMES = {names_json};
  let STATS = {stats_json};
  let index = {index};
  const REL_GAP = {rel_gap};
  let dirty = false;
  let img = new Image();
  let berryBoxes = [];
  let pedObbs = [];
  let fullW = 1, fullH = 1;
  let dispW = 1, dispH = 1;
  let imgScale = 1;
  let scale = 1;
  let selected = -1;
  let selectedKind = 'ped'; // 'ped' | 'berry'
  let drawTool = 'peduncle'; // 'peduncle' | 'berry'
  let berryA = null;

  let calyxVisible = null;
  let calyxPoint = null;
  let calyxReviewed = false;
  let calyxPlaceMode = false;

  let mode = 'idle';
  let axisA = null, axisB = null;
  let widthHalf = 8;
  let curPt = null;

  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  const statusEl = document.getElementById('status');
  const chkCalyx = document.getElementById('chkCalyx');
  const calyxToggle = document.getElementById('calyxToggle');

  function setStatus(msg, err) {{
    statusEl.textContent = msg || '';
    statusEl.className = err ? 'err' : '';
  }}

  function renderStats(st) {{
    if (!st) return;
    const w = (st.warnings || []).map(x => '<div class="warn">⚠ ' + x + '</div>').join('');
    const bp = st.balance_plan || null;
    let planHtml = '';
    if (bp) {{
      const doneO = bp.need_only_other <= 0;
      const doneN = bp.need_no_stem <= 0;
      planHtml =
        '<div class="row2" style="margin-top:8px;padding-top:6px;border-top:1px solid #333">' +
        '<b style="color:#ffcc80">План добора (target фиксируем, без новых target-stem)</b>' +
        '</div><div class="row2">' +
        '<span>сейчас: target ' + bp.target_fixed + ' · only-other ' + bp.only_other_now + ' · no-stem ' + bp.no_stem_now + ' · всего ' + bp.frames_now + '</span>' +
        '</div><div class="row2">' +
        '<span>цель: target ' + bp.target_fixed + ' (' + bp.target_final_pct + '%) · only-other ' + bp.only_other_final + ' (' + bp.only_other_final_pct + '%) · no-stem ' + bp.no_stem_final + ' (' + bp.no_stem_final_pct + '%) · итого ' + bp.frames_final + '</span>' +
        '</div><div class="row2">' +
        '<span style="color:' + (doneO ? '#8bc34a' : '#ef9a9a') + '"><b>осталось only-other: ' + bp.need_only_other + '</b> (чужие плодоножки, без target)</span>' +
        '<span style="color:' + (doneN ? '#8bc34a' : '#ef9a9a') + '"><b>осталось no-stem: ' + bp.need_no_stem + '</b> (вообще без плодоножки)</span>' +
        '<span>всего добрать: <b>' + bp.need_total + '</b></span>' +
        '</div><div class="row2"><span style="color:#999">баланс 70/15/15 · target фиксируем · добор only-other / no-stem</span></div>';
    }}
    const box = document.getElementById('statsBox');
    box.innerHTML =
      '<b>Датасет</b> · ориентир плана: target ~70% (фикс) · only-other 15% · no-stem 15%' +
      '<div class="row2">' +
      '<span>кадры: ' + st.frames_reviewed + '/' + st.frames_total + ' reviewed (' + st.frames_reviewed_pct + '%) · unreviewed ' + st.frames_unreviewed + ' (' + st.frames_unreviewed_pct + '%)</span>' +
      '<span>target stem: ' + st.frames_with_target_stem + ' (' + st.frames_with_target_stem_pct + '%)</span>' +
      '<span>only other: ' + st.frames_with_only_other_stem + ' (' + st.frames_with_only_other_stem_pct + '%)</span>' +
      '<span>no stem: ' + st.frames_with_no_stem + ' (' + st.frames_with_no_stem_pct + '%)</span>' +
      '</div><div class="row2">' +
      '<span>OBB: ' + st.obb_total + ' · target ' + st.obb_target + ' (' + st.obb_target_pct + '%) · other ' + st.obb_other + ' (' + st.obb_other_pct + '%) · rel unreviewed ' + st.obb_relation_unreviewed + '</span>' +
      '</div><div class="row2">' +
      '<span>calyx vis+pt ' + st.calyx_visible_point + ' (' + st.calyx_visible_point_pct + '%)</span>' +
      '<span>invis+pt ' + st.calyx_invisible_point + ' (' + st.calyx_invisible_point_pct + '%)</span>' +
      '<span>invis+no pt ' + st.calyx_invisible_no_point + ' (' + st.calyx_invisible_no_point_pct + '%)</span>' +
      '<span>calyx unreviewed ' + st.calyx_unreviewed + ' (' + st.calyx_unreviewed_pct + '%)</span>' +
      '</div>' + planHtml + w;
  }}

  function updateRelationUI() {{
    const t = document.getElementById('btnRelTarget');
    const o = document.getElementById('btnRelOther');
    const rel = (selectedKind === 'ped' && selected >= 0 && pedObbs[selected]) ? pedObbs[selected].relation : null;
    t.classList.toggle('on-target', rel === 'target');
    o.classList.toggle('on-other', rel === 'other');
  }}

  function setSelectedRelation(rel) {{
    if (selectedKind !== 'ped' || selected < 0 || !pedObbs[selected]) {{ setStatus('сначала выбери OBB плодоножки', true); return; }}
    pedObbs[selected].relation = rel;
    dirty = true;
    updateRelationUI();
    draw();
    setStatus('#' + selected + ' → ' + rel);
  }}

  function updateCalyxUI() {{
    chkCalyx.checked = !!calyxVisible;
    calyxToggle.classList.toggle('active', !!calyxVisible);
    calyxToggle.classList.toggle('unreviewed', calyxVisible === null && !calyxReviewed);
    const badge = document.getElementById('reviewBadge');
    if (calyxReviewed && dataRelationsComplete()) {{
      badge.textContent = 'reviewed';
      badge.className = 'review-badge yes';
    }} else if (calyxReviewed) {{
      badge.textContent = 'relation unreviewed';
      badge.className = 'review-badge no';
    }} else {{
      badge.textContent = 'calyx unreviewed';
      badge.className = 'review-badge no';
    }}
  }}

  function dataRelationsComplete() {{
    return pedObbs.every(o => o.relation === 'target' || o.relation === 'other');
  }}

  function setDrawTool(tool) {{
    drawTool = tool === 'berry' ? 'berry' : 'peduncle';
    calyxPlaceMode = false;
    const btn = document.getElementById('btnCalyxPlace');
    if (btn) btn.style.background = '#222';
    document.getElementById('btnToolBerry').style.background = drawTool === 'berry' ? '#1b5e20' : '#222';
    document.getElementById('btnToolPed').style.background = drawTool === 'peduncle' ? '#e65100' : '#222';
    mode = 'idle';
    berryA = axisA = axisB = curPt = null;
    draw();
    setStatus(drawTool === 'berry' ? 'режим Y: тяни AABB ягоды (одна target berry)' : 'режим X: косой OBB плодоножки');
  }}

  function pointInBerry(x, y, b) {{
    const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2);
    const y1 = Math.min(b.y1, b.y2), y2 = Math.max(b.y1, b.y2);
    return x >= x1 && x <= x2 && y >= y1 && y <= y2;
  }}

  chkCalyx.addEventListener('change', () => {{
    calyxVisible = chkCalyx.checked;
    dirty = true;
    updateCalyxUI();
    draw();
  }});

  function displaySize() {{
    const maxW = Math.min(window.innerWidth - 40, 1100);
    const maxH = Math.min(window.innerHeight - 240, 900);
    const sx = maxW / dispW;
    const sy = maxH / dispH;
    scale = Math.min(1, sx, sy);
    cv.width = Math.max(1, Math.round(dispW * scale));
    cv.height = Math.max(1, Math.round(dispH * scale));
  }}

  function toFull(x, y) {{
    const r = cv.getBoundingClientRect();
    const px = (x - r.left) * (cv.width / r.width);
    const py = (y - r.top) * (cv.height / r.height);
    return [px / scale / imgScale, py / scale / imgScale];
  }}

  function toCanvas(x, y) {{
    return [x * imgScale * scale, y * imgScale * scale];
  }}

  function obbFromAxis(ax, ay, bx, by, halfW) {{
    let dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len; dy /= len;
    const px = -dy * halfW, py = dx * halfW;
    return [
      {{x: ax + px, y: ay + py}},
      {{x: ax - px, y: ay - py}},
      {{x: bx - px, y: by - py}},
      {{x: bx + px, y: by + py}},
    ];
  }}

  function pointInPoly(x, y, pts) {{
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {{
      const xi = pts[i].x, yi = pts[i].y;
      const xj = pts[j].x, yj = pts[j].y;
      const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
      if (intersect) inside = !inside;
    }}
    return inside;
  }}

  function drawPoly(pts, stroke, fill, lineW) {{
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    const [x0, y0] = toCanvas(pts[0].x, pts[0].y);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < pts.length; i++) {{
      const [x, y] = toCanvas(pts[i].x, pts[i].y);
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    if (fill) {{ ctx.fillStyle = fill; ctx.fill(); }}
    ctx.lineWidth = lineW || 2;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }}

  function draw() {{
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    for (let i = 0; i < berryBoxes.length; i++) {{
      const b = berryBoxes[i];
      const [x1, y1] = toCanvas(b.x1, b.y1);
      const [x2, y2] = toCanvas(b.x2, b.y2);
      const sel = selectedKind === 'berry' && i === selected;
      ctx.lineWidth = sel ? 3 : 2;
      ctx.strokeStyle = sel ? 'rgba(180,255,80,1)' : 'rgba(0,220,0,0.95)';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (sel) {{
        ctx.fillStyle = 'rgba(0,220,0,0.12)';
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }}
      ctx.fillStyle = sel ? 'rgba(180,255,80,1)' : 'rgba(0,220,0,0.95)';
      ctx.font = '13px sans-serif';
      ctx.fillText('berry', x1 + 3, Math.max(14, y1 - 4));
      if (sel) {{
        for (const [hx, hy] of [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) {{
          ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
        }}
      }}
    }}
    if (mode === 'berryrect' && berryA && curPt) {{
      const [x1, y1] = toCanvas(Math.min(berryA[0], curPt[0]), Math.min(berryA[1], curPt[1]));
      const [x2, y2] = toCanvas(Math.max(berryA[0], curPt[0]), Math.max(berryA[1], curPt[1]));
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = 'rgba(0,255,120,0.95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }}
    for (let i = 0; i < pedObbs.length; i++) {{
      const sel = selectedKind === 'ped' && i === selected;
      const rel = pedObbs[i].relation;
      let stroke = 'rgba(255,200,0,0.95)';
      let fill = 'rgba(255,200,0,0.10)';
      if (rel === 'target') {{
        stroke = sel ? 'rgba(255,200,0,1)' : 'rgba(255,120,0,0.95)';
        fill = sel ? 'rgba(255,180,0,0.18)' : 'rgba(255,120,0,0.08)';
      }} else if (rel === 'other') {{
        stroke = sel ? 'rgba(128,216,255,1)' : 'rgba(2,136,209,0.95)';
        fill = sel ? 'rgba(2,136,209,0.20)' : 'rgba(2,136,209,0.08)';
      }}
      drawPoly(pedObbs[i].points, stroke, fill, sel ? 3 : 2);
      const p0 = pedObbs[i].points[0];
      const [lx, ly] = toCanvas(p0.x, p0.y);
      ctx.fillStyle = stroke;
      ctx.font = '13px sans-serif';
      const tag = rel === 'target' ? 'target' : (rel === 'other' ? 'other' : 'rel?');
      ctx.fillText('ped #' + i + ' ' + tag, lx + 3, Math.max(14, ly - 4));
      for (const pt of pedObbs[i].points) {{
        const [cx, cy] = toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(cx, cy, sel ? 4 : 3, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      }}
    }}
    // calyx point (shown for both visible=true and visible=false with point)
    if (calyxPoint) {{
      const [cx, cy] = toCanvas(calyxPoint[0], calyxPoint[1]);
      const isVis = !!calyxVisible;
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = isVis ? 'rgba(171,71,188,0.7)' : 'rgba(171,71,188,0.35)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = isVis ? '#e1bee7' : '#9e9e9e';
      ctx.stroke();
      if (!isVis) {{ ctx.setLineDash([4,3]); }}
      ctx.beginPath();
      ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
      ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
      ctx.strokeStyle = isVis ? 'rgba(206,147,216,0.9)' : 'rgba(158,158,158,0.7)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = isVis ? '#e1bee7' : '#9e9e9e';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(isVis ? 'calyx' : 'calyx (inferred)', cx + 10, cy - 10);
    }}
    // draw state
    if (mode === 'axis' && axisA && curPt) {{
      const [ax, ay] = toCanvas(axisA[0], axisA[1]);
      const [bx, by] = toCanvas(curPt[0], curPt[1]);
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(255,180,0,0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      ctx.setLineDash([]);
    }}
    if (mode === 'width' && axisA && axisB) {{
      const pts = obbFromAxis(axisA[0], axisA[1], axisB[0], axisB[1], widthHalf);
      drawPoly(pts, 'rgba(255,180,0,0.95)', 'rgba(255,160,0,0.15)', 2);
    }}
  }}

  function calyxStatusText() {{
    if (!calyxReviewed) return 'unreviewed';
    return calyxVisible ? 'visible' : 'not visible';
  }}

  function cancelDraw() {{
    mode = 'idle';
    calyxPlaceMode = false;
    const btn = document.getElementById('btnCalyxPlace');
    if (btn) btn.style.background = '#222';
    berryA = axisA = axisB = curPt = null;
    setStatus('OBB: ' + pedObbs.length + ' · calyx: ' + calyxStatusText());
  }}

  async function loadIndex(i, force) {{
    if (dirty && !force) {{
      if (!confirm('Есть несохранённые изменения. Уйти без Save?')) return false;
    }}
    index = Math.max(0, Math.min(i, NAMES.length - 1));
    const name = NAMES[index];
    document.getElementById('pos').textContent = String(index + 1);
    document.getElementById('fname').textContent = name;
    document.getElementById('btnPrev').href = '/?dataset=peduncle_obb&i=' + Math.max(0, index - 1);
    document.getElementById('btnNext').href = '/?dataset=peduncle_obb&i=' + Math.min(NAMES.length - 1, index + 1);
    history.replaceState(null, '', '/?dataset=peduncle_obb&i=' + index);
    try {{ localStorage.setItem('peduncle_obb_last', name); }} catch (e) {{}}
    setStatus('loading…');
    cancelDraw();
    const res = await fetch('/api/peduncle_obb/anno/' + encodeURIComponent(name));
    if (!res.ok) {{ setStatus('load failed', true); return false; }}
    const data = await res.json();
    fullW = data.width; fullH = data.height;
    berryBoxes = data.berry_boxes || [];
    pedObbs = data.peduncle_obbs || [];
    calyxVisible = data.calyx_visible;
    calyxPoint = data.calyx_point;
    calyxReviewed = data.calyx_reviewed || false;
    selected = -1;
    selectedKind = 'ped';
    dirty = false;
    updateCalyxUI();
    updateRelationUI();
    await new Promise((resolve, reject) => {{
      img.onload = () => {{
        dispW = img.naturalWidth;
        dispH = img.naturalHeight;
        imgScale = dispW / fullW;
        resolve();
      }};
      img.onerror = reject;
      img.src = '/img/peduncle_obb_raw/' + encodeURIComponent(name) + '?t=' + Date.now();
    }});
    displaySize();
    draw();
    setStatus('OBB: ' + pedObbs.length + ' · berry: ' + berryBoxes.length + ' · calyx: ' + (calyxReviewed ? (calyxVisible ? 'visible' : 'not visible') : 'unreviewed'));
    return true;
  }}

  async function save() {{
    const name = NAMES[index];
    setStatus('saving…');
    const body = {{
      peduncle_obbs: pedObbs,
      berry_boxes: berryBoxes,
      calyx_visible: calyxVisible,
      calyx_point: calyxPoint,
    }};
    const res = await fetch('/api/peduncle_obb/anno/' + encodeURIComponent(name), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    const data = await res.json().catch(() => ({{ok:false}}));
    if (!res.ok || data.ok === false) {{
      setStatus('save failed: ' + (data.error || res.status), true);
      return false;
    }}
    dirty = false;
    calyxReviewed = true;
    updateCalyxUI();
    setStatus('saved · OBB=' + pedObbs.length + ' · calyx ' + (calyxVisible ? 'visible' : 'not visible'));
    try {{
      const sres = await fetch('/api/peduncle_obb/stats');
      if (sres.ok) {{
        STATS = await sres.json();
        renderStats(STATS);
        document.getElementById('reviewStats').textContent = 'reviewed ' + STATS.frames_reviewed + '/' + STATS.frames_total;
      }}
    }} catch (e) {{}}
    return true;
  }}

  async function saveAndNext() {{
    const ok = await save();
    if (!ok) return;
    if (index >= NAMES.length - 1) {{
      setStatus('saved · это последний кадр');
      return;
    }}
    await loadIndex(index + 1, true);
  }}

  cv.addEventListener('mousedown', (e) => {{
    const [nx, ny] = toFull(e.clientX, e.clientY);
    // calyx point placement mode
    if (calyxPlaceMode && mode === 'idle') {{
      calyxPoint = [nx, ny];
      dirty = true;
      selected = -1;
      draw();
      setStatus('calyx point set (' + Math.round(nx) + ', ' + Math.round(ny) + ') — клик ещё раз чтобы перенести · Esc выйти из режима');
      return;
    }}
    if (mode === 'width') {{
      const pts = obbFromAxis(axisA[0], axisA[1], axisB[0], axisB[1], widthHalf);
      const hasTarget = pedObbs.some(o => o.relation === 'target');
      pedObbs.push({{ cls: 1, points: pts, relation: hasTarget ? 'other' : 'target' }});
      selected = pedObbs.length - 1;
      selectedKind = 'ped';
      dirty = true;
      mode = 'idle';
      axisA = axisB = curPt = null;
      draw();
      updateRelationUI();
      setStatus('добавлен косой бокс · relation=' + pedObbs[selected].relation + ' · Save');
      return;
    }}
    if (mode === 'idle') {{
      for (let i = berryBoxes.length - 1; i >= 0; i--) {{
        if (pointInBerry(nx, ny, berryBoxes[i])) {{
          selected = i;
          selectedKind = 'berry';
          draw();
          updateRelationUI();
          setStatus('выбрана ягода #' + i + ' · Del удалить · Y + drag перерисовать');
          return;
        }}
      }}
      for (let i = pedObbs.length - 1; i >= 0; i--) {{
        if (pointInPoly(nx, ny, pedObbs[i].points)) {{
          selected = i;
          selectedKind = 'ped';
          draw();
          updateRelationUI();
          setStatus('выбран ped #' + i + ' · ' + (pedObbs[i].relation || 'relation?'));
          return;
        }}
      }}
      if (drawTool === 'berry') {{
        mode = 'berryrect';
        berryA = [nx, ny];
        curPt = [nx, ny];
        selected = -1;
        selectedKind = 'berry';
        draw();
        return;
      }}
      mode = 'axis';
      axisA = [nx, ny];
      curPt = [nx, ny];
      selected = -1;
      selectedKind = 'ped';
      draw();
    }}
  }});

  cv.addEventListener('mousemove', (e) => {{
    const [nx, ny] = toFull(e.clientX, e.clientY);
    curPt = [Math.max(0, Math.min(fullW, nx)), Math.max(0, Math.min(fullH, ny))];
    if (mode === 'axis' && axisA) {{
      draw();
    }} else if (mode === 'berryrect' && berryA) {{
      draw();
    }} else if (mode === 'width' && axisA && axisB) {{
      const ax = axisA[0], ay = axisA[1], bx = axisB[0], by = axisB[1];
      const dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const dist = Math.abs(dx * (ay - curPt[1]) - dy * (ax - curPt[0])) / len;
      widthHalf = Math.max(3, dist);
      draw();
    }}
  }});

  cv.addEventListener('mouseup', (e) => {{
    const [nx, ny] = toFull(e.clientX, e.clientY);
    if (mode === 'berryrect' && berryA) {{
      const x1 = Math.max(0, Math.min(berryA[0], nx));
      const y1 = Math.max(0, Math.min(berryA[1], ny));
      const x2 = Math.min(fullW, Math.max(berryA[0], nx));
      const y2 = Math.min(fullH, Math.max(berryA[1], ny));
      berryA = null;
      mode = 'idle';
      if ((x2 - x1) < 8 || (y2 - y1) < 8) {{
        draw();
        setStatus('слишком маленький berry box', true);
        return;
      }}
      const box = {{ cls: 0, x1: x1, y1: y1, x2: x2, y2: y2 }};
      berryBoxes = [box];
      selected = 0;
      selectedKind = 'berry';
      dirty = true;
      draw();
      setStatus('berry box обновлён · Save');
      return;
    }}
    if (mode !== 'axis' || !axisA) return;
    axisB = [Math.max(0, Math.min(fullW, nx)), Math.max(0, Math.min(fullH, ny))];
    const len = Math.hypot(axisB[0] - axisA[0], axisB[1] - axisA[1]);
    if (len < 12) {{
      cancelDraw();
      setStatus('слишком короткий жест — тяни вдоль стебля', true);
      return;
    }}
    mode = 'width';
    widthHalf = Math.max(4, len * 0.08);
    draw();
    setStatus('двигай мышь для ширины, клик — OK');
  }});

  function toggleCalyxPlaceMode() {{
    const entering = !calyxPlaceMode;
    if (entering) {{
      // cancel OBB draw first, then enable calyx mode
      mode = 'idle';
      axisA = axisB = curPt = null;
      widthHalf = 8;
    }}
    calyxPlaceMode = entering;
    const btn = document.getElementById('btnCalyxPlace');
    btn.style.background = calyxPlaceMode ? '#4a148c' : '#222';
    cv.style.cursor = calyxPlaceMode ? 'cell' : 'crosshair';
    draw();
    setStatus(calyxPlaceMode ? 'Calyx place mode ON — клик на изображение ставит точку · Esc выйти' : 'Calyx place mode OFF');
  }}

  document.getElementById('btnToolBerry').onclick = () => setDrawTool('berry');
  document.getElementById('btnToolPed').onclick = () => setDrawTool('peduncle');
  document.getElementById('btnRelTarget').onclick = () => setSelectedRelation('target');
  document.getElementById('btnRelOther').onclick = () => setSelectedRelation('other');
  document.getElementById('btnSave').onclick = () => saveAndNext();
  document.getElementById('btnCancel').onclick = () => cancelDraw();
  document.getElementById('btnCalyxPlace').onclick = () => toggleCalyxPlaceMode();
  document.getElementById('btnCalyxClear').onclick = () => {{
    calyxPoint = null;
    dirty = true;
    draw();
    setStatus('calyx point cleared');
  }};
  document.getElementById('btnDel').onclick = () => {{
    if (selectedKind === 'berry') {{
      if (selected < 0 || !berryBoxes[selected]) {{ setStatus('сначала выбери бокс ягоды', true); return; }}
      berryBoxes.splice(selected, 1);
      selected = -1;
      dirty = true;
      draw();
      setStatus('ягода удалена · Save');
      return;
    }}
    if (selected < 0) {{ setStatus('сначала выбери бокс', true); return; }}
    pedObbs.splice(selected, 1);
    selected = -1;
    dirty = true;
    draw();
    updateRelationUI();
    setStatus('peduncle OBB удалён · Save');
  }};
  async function deleteFrame() {{
    const name = NAMES[index];
    if (!name) return;
    if (!confirm('Удалить кадр ' + name + ' из очереди?')) return;
    setStatus('deleting…');
    const res = await fetch('/api/peduncle_obb/frame/' + encodeURIComponent(name), {{ method: 'DELETE' }});
    const data = await res.json().catch(() => ({{ok:false}}));
    if (!res.ok || data.ok === false) {{
      setStatus('delete failed: ' + (data.error || res.status), true);
      return;
    }}
    NAMES.splice(index, 1);
    dirty = false;
    if (!NAMES.length) {{
      setStatus('очередь пуста');
      return;
    }}
    if (index >= NAMES.length) index = NAMES.length - 1;
    await loadIndex(index, true);
    setStatus('кадр удалён · осталось ' + NAMES.length);
  }}
  document.getElementById('btnDelFrame').onclick = () => deleteFrame();
  document.getElementById('btnClear').onclick = () => {{
    if (!pedObbs.length) return;
    if (!confirm('Удалить все peduncle OBB на кадре?')) return;
    pedObbs = [];
    selected = -1;
    dirty = true;
    draw();
  }};

  document.addEventListener('keydown', async (e) => {{
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key === 's' || e.key === 'S') {{ e.preventDefault(); await saveAndNext(); }}
    else if (e.key === 'y' || e.key === 'Y') {{ e.preventDefault(); setDrawTool('berry'); }}
    else if (e.key === 'x' || e.key === 'X') {{ e.preventDefault(); setDrawTool('peduncle'); }}
    else if (e.key === 't' || e.key === 'T') {{ e.preventDefault(); setSelectedRelation('target'); }}
    else if (e.key === 'o' || e.key === 'O') {{ e.preventDefault(); setSelectedRelation('other'); }}
    else if (e.key === 'v' || e.key === 'V') {{ e.preventDefault(); toggleCalyxPlaceMode(); }}
    else if (e.key === 'c' || e.key === 'C') {{
      e.preventDefault();
      chkCalyx.checked = !chkCalyx.checked;
      chkCalyx.dispatchEvent(new Event('change'));
    }}
    else if (e.key === 'Escape') {{ e.preventDefault(); cancelDraw(); }}
    else if (e.key === 'Delete' || e.key === 'Backspace') {{
      e.preventDefault();
      document.getElementById('btnDel').click();
    }}
    else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
      e.preventDefault();
      await loadIndex(index - 1, false);
    }}
    else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{
      e.preventDefault();
      await loadIndex(index + 1, false);
    }}
  }});

  window.addEventListener('resize', () => {{ displaySize(); draw(); }});
  window.addEventListener('beforeunload', (e) => {{
    if (dirty) {{ e.preventDefault(); e.returnValue = ''; }}
  }});

  renderStats(STATS);
  setDrawTool('peduncle');
  let start = index;
  // пока есть OBB без target/other — начинаем с них (не с localStorage)
  if (REL_GAP > 0) {{
    start = index;
    setStatus('очередь: ' + REL_GAP + ' кадров с OBB без relation (target/other) — выставь T/O и Save');
  }} else {{
    try {{
      const last = localStorage.getItem('peduncle_obb_last');
      if (last) {{
        const j = NAMES.indexOf(last);
        if (j >= 0) start = j;
      }}
    }} catch (e) {{}}
  }}
  loadIndex(start, true);
}})();
</script>
</body>
</html>"""


__all__ = [
    "ROOT",
    "calyx_review_stats",
    "ensure_layout",
    "get_annotation",
    "image_path",
    "list_images",
    "dataset_stats",
    "balance_plan",
    "delete_frame",
    "migrate_relations",
    "migrate_calyx_null_visibility",
    "render_peduncle_obb_html",
    "render_raw_jpeg",
    "save_annotation",
]
