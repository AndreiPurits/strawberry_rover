#!/usr/bin/env python3
"""Peduncle crop curation (approve / reject).

Queue:    data/плодоножки/images/  (+ labels, labels_peduncle, meta)
Approve → data/плодоножки апрувд/{images,labels,labels_peduncle,meta}/
Reject  → data/плодоножки/rejected/{images,labels,labels_peduncle,meta}/
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

QUEUE_ROOT = REPO_ROOT / "data" / "плодоножки"
APPROVED_ROOT = REPO_ROOT / "data" / "плодоножки апрувд"
SIDE_SUBDIRS = ("images", "labels", "labels_peduncle", "meta", "bbox_vis")

STATE_PATH = QUEUE_ROOT / "reports" / "peduncle_curation_state.json"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_layout() -> None:
    for root in (QUEUE_ROOT, APPROVED_ROOT, QUEUE_ROOT / "rejected"):
        for sub in SIDE_SUBDIRS:
            _safe_mkdir(root / sub)
    _safe_mkdir(QUEUE_ROOT / "reports")
    _safe_mkdir(APPROVED_ROOT / "reports")


def load_state() -> Dict[str, Any]:
    default: Dict[str, Any] = {"version": 1, "skipped": [], "events": []}
    if not STATE_PATH.is_file():
        return default
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default
        data.setdefault("skipped", [])
        data.setdefault("events", [])
        return data
    except Exception:
        return default


def save_state(st: Dict[str, Any]) -> None:
    _safe_mkdir(STATE_PATH.parent)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if _is_image(p))


def iter_queue_images() -> List[Path]:
    d = QUEUE_ROOT / "images"
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if _is_image(p)], key=lambda p: p.name)


def visible_queue(*, include_skipped: bool) -> List[Path]:
    items = iter_queue_images()
    if include_skipped:
        return items
    skipped = set(load_state().get("skipped") or [])
    primary = [p for p in items if p.name not in skipped]
    deferred = [p for p in items if p.name in skipped]
    return primary + deferred


def add_skipped(filename: str) -> None:
    st = load_state()
    skipped = list(st.get("skipped") or [])
    if filename not in skipped:
        skipped.append(filename)
    st["skipped"] = skipped
    save_state(st)


def clear_skipped() -> None:
    st = load_state()
    st["skipped"] = []
    save_state(st)


def peduncle_image_file(filename: str) -> Path:
    return QUEUE_ROOT / "images" / filename


def _stem(filename: str) -> str:
    return Path(filename).stem


def _collect_sidecar_paths(root: Path, stem: str) -> List[Path]:
    found: List[Path] = []
    for sub in SIDE_SUBDIRS:
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.stem == stem and p.is_file():
                found.append(p)
    return found


def _move_bundle(filename: str, *, src_root: Path, dst_root: Path) -> List[Tuple[str, str]]:
    """Move image + sidecars. Returns list of (from, to) for undo."""
    stem = _stem(filename)
    moves: List[Tuple[str, str]] = []
    for p in _collect_sidecar_paths(src_root, stem):
        rel_sub = p.parent.name
        dest_dir = dst_root / rel_sub
        _safe_mkdir(dest_dir)
        dst = dest_dir / p.name
        if dst.exists():
            raise FileExistsError(str(dst))
        shutil.move(str(p), str(dst))
        moves.append((str(dst), str(p)))  # undo: from dst back to src
    return moves


def apply_peduncle_action(
    filename: str,
    action: str,
    *,
    log_fn: Callable[[Dict[str, Any]], None],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    ensure_layout()
    action = (action or "").strip().lower()
    if action not in ("approve", "reject"):
        return False, "invalid action", None

    src = peduncle_image_file(filename)
    if not _is_image(src):
        return False, "source not found", None

    if action == "approve":
        dst_root = APPROVED_ROOT
    else:
        dst_root = QUEUE_ROOT / "rejected"

    try:
        moves = _move_bundle(filename, src_root=QUEUE_ROOT, dst_root=dst_root)
    except FileExistsError as e:
        return False, f"destination exists: {e}", None
    except Exception as e:
        return False, f"move failed: {e}", None

    if not moves:
        return False, "nothing moved", None

    st = load_state()
    skipped = [x for x in (st.get("skipped") or []) if x != filename]
    st["skipped"] = skipped
    ev = {"action": action, "filename": filename, "ts": time.time()}
    st.setdefault("events", []).append(ev)
    if len(st["events"]) > 5000:
        st["events"] = st["events"][-4000:]
    save_state(st)
    log_fn({"action": f"peduncle_{action}", "filename": filename})

    undo = {"type": "peduncle_move", "filename": filename, "moves": moves}
    return True, "ok", undo


def undo_peduncle_move(undo: Dict[str, Any]) -> bool:
    moves = undo.get("moves") or []
    ok = False
    for from_path_s, to_path_s in moves:
        from_path = Path(from_path_s)
        to_path = Path(to_path_s)
        if not from_path.is_file():
            continue
        _safe_mkdir(to_path.parent)
        if to_path.exists():
            continue
        shutil.move(str(from_path), str(to_path))
        ok = True
    return ok


def _parse_yolo_xyxy(label_path: Path, iw: int, ih: int) -> Optional[Tuple[int, int, int, int]]:
    if not label_path.is_file():
        return None
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 5:
            continue
        try:
            xc, yc, w, h = map(float, toks[1:5])
        except Exception:
            continue
        bw, bh = w * iw, h * ih
        x1 = int(max(0, xc * iw - bw / 2))
        y1 = int(max(0, yc * ih - bh / 2))
        x2 = int(min(iw, xc * iw + bw / 2))
        y2 = int(min(ih, yc * ih + bh / 2))
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return None


def render_preview_jpeg(filename: str) -> Optional[bytes]:
    """JPEG with berry bbox (green) and peduncle bbox (orange) if present."""
    img_path = peduncle_image_file(filename)
    if not _is_image(img_path):
        return None
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    stem = _stem(filename)
    berry = _parse_yolo_xyxy(QUEUE_ROOT / "labels" / f"{stem}.txt", w, h)
    ped = _parse_yolo_xyxy(QUEUE_ROOT / "labels_peduncle" / f"{stem}.txt", w, h)
    vis = bgr.copy()
    if berry is not None:
        x1, y1, x2, y2 = berry
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(vis, "berry", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    if ped is not None:
        x1, y1, x2, y2 = ped
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(vis, "peduncle", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return bytes(buf)


def counts() -> Dict[str, int]:
    return {
        "queue": count_images(QUEUE_ROOT / "images"),
        "approved": count_images(APPROVED_ROOT / "images"),
        "rejected": count_images(QUEUE_ROOT / "rejected" / "images"),
        "skipped": len(load_state().get("skipped") or []),
    }


def render_peduncle_html(*, include_skipped: bool, requested_file: Optional[str], include_skipped_val: int) -> str:
    ensure_layout()
    visible = visible_queue(include_skipped=include_skipped)
    cur: Optional[Path] = None
    if visible:
        if requested_file:
            for p in visible:
                if p.name == requested_file:
                    cur = p
                    break
        if cur is None:
            cur = visible[0]

    c = counts()
    chk = "checked" if include_skipped else ""

    if cur is None:
        fname = ""
        img_html = ""
        pos = "0"
        meta_line = (
            "<p class='meta'><b>Очередь пуста</b> — разобрано, либо положи кадры в "
            "<code>data/плодоножки/images/</code>.</p>"
            "<p class='meta'>Новые интернет-кадры смотри здесь: "
            "<a href='/?dataset=new_straw'><code>/?dataset=new_straw</code></a></p>"
        )
        hint = ""
    else:
        fname = cur.name
        pos = str(visible.index(cur) + 1)
        stem = cur.stem
        ped_lbl = QUEUE_ROOT / "labels_peduncle" / f"{stem}.txt"
        has_ped = ped_lbl.is_file() and bool(ped_lbl.read_text(encoding="utf-8", errors="ignore").strip())
        tip = "есть bbox плодоножки" if has_ped else "плодоножку ещё не размечали (только ягода)"
        img_html = (
            f'<img alt="crop" src="/img/peduncle/{fname}?v={int(time.time())}" '
            f'style="max-width:min(100%,720px);max-height:75vh;border-radius:8px;border:1px solid #ddd;" />'
        )
        meta_line = (
            f"<p class='meta'>Позиция: <b>{pos}</b> / <b>{len(visible)}</b> &nbsp;|&nbsp; "
            f"файл: <code>{fname}</code></p>"
            f"<p class='meta'>{tip}</p>"
        )
        hint = (
            "<p class='meta' style='background:#fff8e1;padding:8px 10px;border-radius:8px;'>"
            "Критерий: одна ягода + <b>её</b> плодоножка видна и однозначно к ней относится. "
            "Иначе → reject.</p>"
        )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Плодоножки — curation</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; margin: 16px; background:#fafafa; }}
    .card {{ border:1px solid #ddd; border-radius:10px; padding:14px; max-width:960px; background:#fff; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:10px; }}
    button {{ padding:12px 16px; border-radius:10px; border:1px solid #ccc; background:#fff; cursor:pointer; font-size:15px; }}
    .ok {{ border-color:#2e7d32; background:#e8f5e9; font-weight:600; }}
    .del {{ border-color:#d93025; background:#ffebee; }}
    .meta {{ color:#444; font-size:14px; }}
    code {{ background:#f5f5f5; padding:2px 6px; border-radius:4px; }}
    kbd {{ background:#eee; padding:2px 6px; border-radius:4px; font-size:12px; }}
    table.ct {{ border-collapse:collapse; margin-top:8px; font-size:14px; }}
    table.ct th, table.ct td {{ border:1px solid #ddd; padding:4px 10px; text-align:left; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>Плодоножки — approve / reject</h2>
    <p class="meta">Очередь: <code>data/плодоножки/images/</code> → approve: <code>data/плодоножки апрувд/</code> · reject: <code>data/плодоножки/rejected/</code></p>
    <p class="meta">Также: <a href="/?dataset=classifier">classifier</a></p>
    <form method="get" action="/" class="row">
      <input type="hidden" name="dataset" value="peduncle" />
      <label><input type="checkbox" name="include_skipped" value="1" {chk}/> показать пропущенные в конце</label>
      <button type="submit">Обновить</button>
    </form>
    {hint}
    {meta_line}
    <div style="margin-top:12px;">{img_html}</div>
    <table class="ct">
      <tr><th>queue</th><th>approved</th><th>rejected</th><th>skipped</th></tr>
      <tr><td>{c['queue']}</td><td>{c['approved']}</td><td>{c['rejected']}</td><td>{c['skipped']}</td></tr>
    </table>
    <p class="meta">Клавиши: <kbd>A</kbd> / <kbd>Enter</kbd> approve · <kbd>D</kbd> reject · <kbd>S</kbd> skip · <kbd>U</kbd> undo</p>
    <div class="row">
      <form id="fok" method="post" action="/peduncle/do">
        <input type="hidden" name="filename" value="{fname}"/>
        <input type="hidden" name="include_skipped" value="{include_skipped_val}"/>
        <input type="hidden" name="action" value="approve"/>
        <button type="submit" class="ok" {"disabled" if not fname else ""}>approve → плодоножки апрувд</button>
      </form>
      <form id="frej" method="post" action="/peduncle/do">
        <input type="hidden" name="filename" value="{fname}"/>
        <input type="hidden" name="include_skipped" value="{include_skipped_val}"/>
        <input type="hidden" name="action" value="reject"/>
        <button type="submit" class="del" {"disabled" if not fname else ""}>reject</button>
      </form>
      <form id="fskip" method="post" action="/peduncle/skip">
        <input type="hidden" name="filename" value="{fname}"/>
        <input type="hidden" name="include_skipped" value="{include_skipped_val}"/>
        <button type="submit" {"disabled" if not fname else ""}>пропустить (S)</button>
      </form>
    </div>
    <div class="row">
      <form method="post" action="/peduncle/clear_skipped">
        <input type="hidden" name="include_skipped" value="{include_skipped_val}"/>
        <button type="submit">сбросить пропуски</button>
      </form>
      <form id="fundo" method="post" action="/peduncle/undo">
        <input type="hidden" name="include_skipped" value="{include_skipped_val}"/>
        <button type="submit">Undo (U)</button>
      </form>
    </div>
  </div>
  <script>
  (function() {{
    document.addEventListener("keydown", function(e) {{
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) return;
      var k = e.key;
      if (k === "a" || k === "A" || k === "Enter") {{ e.preventDefault(); var f=document.getElementById("fok"); if(f) f.submit(); }}
      else if (k === "d" || k === "D") {{ e.preventDefault(); var f=document.getElementById("frej"); if(f) f.submit(); }}
      else if (k === "s" || k === "S") {{ e.preventDefault(); var f=document.getElementById("fskip"); if(f) f.submit(); }}
      else if (k === "u" || k === "U") {{ e.preventDefault(); var f=document.getElementById("fundo"); if(f) f.submit(); }}
    }});
  }})();
  </script>
</body>
</html>"""


__all__ = [
    "add_skipped",
    "apply_peduncle_action",
    "clear_skipped",
    "peduncle_image_file",
    "render_peduncle_html",
    "render_preview_jpeg",
    "undo_peduncle_move",
    "ensure_layout",
]
