#!/usr/bin/env python3
"""Peduncle OBB editor for approved crops (4-corner / YOLO-OBB).

Dataset: data/плодоножки апрувд/
  labels/           — berry AABB class 0 (read-only overlay)
  labels_peduncle/  — peduncle OBB class 1: `1 x1 y1 x2 y2 x3 y3 x4 y4` (normalized)

UI: ?dataset=peduncle_obb
Draw: drag along stem (calyx → tip), move to set width, click to confirm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "data" / "плодоножки апрувд"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EDITOR_MAX_SIDE = 1280
CLASS_PEDUNCLE = 1


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ensure_layout() -> None:
    for d in ("images", "labels", "labels_peduncle", "meta", "bbox_vis", "reports"):
        _safe_mkdir(ROOT / d)


def list_images() -> List[Path]:
    d = ROOT / "images"
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: p.name)


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
            }
        )
    return out


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
    return {
        "filename": p.name,
        "width": w,
        "height": h,
        "display_width": dw,
        "display_height": dh,
        "berry_boxes": _parse_berry_aabb(berry_label_path(filename), w, h),
        "peduncle_obbs": _parse_peduncle_obb(peduncle_label_path(filename), w, h),
        "format": "yolo_obb_4pts",
        "tip_cm": "Цель: ~2.5 см стебля от чашечки (допустимо 2–3.5). Косой бокс по оси стебля.",
    }


def save_annotation(filename: str, peduncle_obbs: List[Dict[str, Any]]) -> Tuple[bool, str]:
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

    # preview overlay
    try:
        vis = bgr.copy()
        for b in _parse_berry_aabb(berry_label_path(filename), w, h):
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
    cur = images[index]
    prev_i = max(0, index - 1)
    next_i = min(n - 1, index + 1)
    names_json = json.dumps([p.name for p in images], ensure_ascii=False)

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
    .ped {{ background:#e65100; }}
  </style>
</head>
<body>
  <div class="card">
    <h2 style="margin:0 0 8px">Плодоножка — косой бокс (4 точки)</h2>
    <p class="meta">
      Кадр <b id="pos">{index + 1}</b> / <b>{n}</b> · <code id="fname">{cur.name}</code>
      · апрувд · цель ~2.5 см от чашечки
      · <a class="btn" href="/?dataset=peduncle">approve UI</a>
    </p>
    <div class="row legend">
      <span class="berry">berry (готово)</span>
      <span class="ped">peduncle OBB</span>
    </div>
    <div class="row">
      <button type="button" id="btnSave" class="primary">Save + next (S)</button>
      <button type="button" id="btnDel" class="danger">Delete selected (Del)</button>
      <button type="button" id="btnClear">Clear peduncle</button>
      <button type="button" id="btnCancel">Cancel draw (Esc)</button>
      <a class="btn" id="btnPrev" href="/?dataset=peduncle_obb&i={prev_i}">← prev</a>
      <a class="btn" id="btnNext" href="/?dataset=peduncle_obb&i={next_i}">next →</a>
    </div>
    <p class="meta">
      1) Тяни вдоль стебля (от чашечки наружу) · 2) двигай мышь — ширина · 3) клик — зафиксировать.
      Клик по боксу = выбор · <kbd>S</kbd> save + next · <kbd>Del</kbd> · <kbd>←</kbd>/<kbd>→</kbd>
    </p>
    <div id="status"></div>
    <div id="wrap"><canvas id="cv"></canvas></div>
  </div>
<script>
(function() {{
  const NAMES = {names_json};
  let index = {index};
  let dirty = false;
  let img = new Image();
  let berryBoxes = [];
  let pedObbs = []; // {{cls, points:[{{x,y}}x4]}} full-res pixels
  let fullW = 1, fullH = 1;
  let dispW = 1, dispH = 1;
  let imgScale = 1;
  let scale = 1;
  let selected = -1;

  // draw state machine: idle -> axis (dragging) -> width (set) -> done
  let mode = 'idle'; // idle | axis | width
  let axisA = null, axisB = null;
  let widthHalf = 8;
  let curPt = null;

  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  const statusEl = document.getElementById('status');

  function setStatus(msg, err) {{
    statusEl.textContent = msg || '';
    statusEl.className = err ? 'err' : '';
  }}

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
    // corners: A-left, A-right, B-right, B-left (around axis A->B)
    return [
      {{x: ax + px, y: ay + py}},
      {{x: ax - px, y: ay - py}},
      {{x: bx - px, y: by - py}},
      {{x: bx + px, y: by + py}},
    ];
  }}

  function pointInPoly(x, y, pts) {{
    // ray cast
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
    for (const b of berryBoxes) {{
      const [x1, y1] = toCanvas(b.x1, b.y1);
      const [x2, y2] = toCanvas(b.x2, b.y2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(0,220,0,0.95)';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = 'rgba(0,220,0,0.95)';
      ctx.font = '13px sans-serif';
      ctx.fillText('berry', x1 + 3, Math.max(14, y1 - 4));
    }}
    for (let i = 0; i < pedObbs.length; i++) {{
      const sel = i === selected;
      drawPoly(
        pedObbs[i].points,
        sel ? 'rgba(255,200,0,1)' : 'rgba(255,120,0,0.95)',
        sel ? 'rgba(255,180,0,0.18)' : 'rgba(255,120,0,0.08)',
        sel ? 3 : 2
      );
      const p0 = pedObbs[i].points[0];
      const [lx, ly] = toCanvas(p0.x, p0.y);
      ctx.fillStyle = sel ? 'rgba(255,220,80,1)' : 'rgba(255,140,0,1)';
      ctx.font = '13px sans-serif';
      ctx.fillText('peduncle #' + i, lx + 3, Math.max(14, ly - 4));
      // corner dots
      for (const pt of pedObbs[i].points) {{
        const [cx, cy] = toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(cx, cy, sel ? 4 : 3, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
      }}
    }}
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

  function cancelDraw() {{
    mode = 'idle';
    axisA = axisB = curPt = null;
    widthHalf = 8;
    draw();
    setStatus('боксов peduncle: ' + pedObbs.length);
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
    setStatus('loading…');
    cancelDraw();
    const res = await fetch('/api/peduncle_obb/anno/' + encodeURIComponent(name));
    if (!res.ok) {{ setStatus('load failed', true); return false; }}
    const data = await res.json();
    fullW = data.width; fullH = data.height;
    berryBoxes = data.berry_boxes || [];
    pedObbs = data.peduncle_obbs || [];
    selected = -1;
    dirty = false;
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
    setStatus('peduncle OBB: ' + pedObbs.length + ' · berry boxes: ' + berryBoxes.length);
    return true;
  }}

  async function save() {{
    const name = NAMES[index];
    setStatus('saving…');
    const res = await fetch('/api/peduncle_obb/anno/' + encodeURIComponent(name), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ peduncle_obbs: pedObbs }}),
    }});
    const data = await res.json().catch(() => ({{ok:false}}));
    if (!res.ok || data.ok === false) {{
      setStatus('save failed: ' + (data.error || res.status), true);
      return false;
    }}
    dirty = false;
    setStatus('saved · peduncle OBB=' + pedObbs.length);
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
    if (mode === 'width') {{
      // confirm
      const pts = obbFromAxis(axisA[0], axisA[1], axisB[0], axisB[1], widthHalf);
      pedObbs.push({{ cls: 1, points: pts }});
      selected = pedObbs.length - 1;
      dirty = true;
      mode = 'idle';
      axisA = axisB = curPt = null;
      draw();
      setStatus('добавлен косой бокс · Save');
      return;
    }}
    if (mode === 'idle') {{
      // hit test existing
      for (let i = pedObbs.length - 1; i >= 0; i--) {{
        if (pointInPoly(nx, ny, pedObbs[i].points)) {{
          selected = i;
          draw();
          setStatus('выбран #' + i);
          return;
        }}
      }}
      // start axis drag
      mode = 'axis';
      axisA = [nx, ny];
      curPt = [nx, ny];
      selected = -1;
      draw();
    }}
  }});

  cv.addEventListener('mousemove', (e) => {{
    const [nx, ny] = toFull(e.clientX, e.clientY);
    curPt = [Math.max(0, Math.min(fullW, nx)), Math.max(0, Math.min(fullH, ny))];
    if (mode === 'axis' && axisA) {{
      draw();
    }} else if (mode === 'width' && axisA && axisB) {{
      // distance from point to axis line = half width
      const ax = axisA[0], ay = axisA[1], bx = axisB[0], by = axisB[1];
      const dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const dist = Math.abs(dx * (ay - curPt[1]) - dy * (ax - curPt[0])) / len;
      widthHalf = Math.max(3, dist);
      draw();
    }}
  }});

  cv.addEventListener('mouseup', (e) => {{
    if (mode !== 'axis' || !axisA) return;
    const [nx, ny] = toFull(e.clientX, e.clientY);
    axisB = [Math.max(0, Math.min(fullW, nx)), Math.max(0, Math.min(fullH, ny))];
    const len = Math.hypot(axisB[0] - axisA[0], axisB[1] - axisA[1]);
    if (len < 12) {{
      cancelDraw();
      setStatus('слишком короткий жест — тяни вдоль стебля', true);
      return;
    }}
    mode = 'width';
    // default width ~ 8% of axis length, min 4px
    widthHalf = Math.max(4, len * 0.08);
    draw();
    setStatus('двигай мышь для ширины, клик — OK');
  }});

  document.getElementById('btnSave').onclick = () => saveAndNext();
  document.getElementById('btnCancel').onclick = () => cancelDraw();
  document.getElementById('btnDel').onclick = () => {{
    if (selected < 0) {{ setStatus('сначала выбери бокс', true); return; }}
    pedObbs.splice(selected, 1);
    selected = -1;
    dirty = true;
    draw();
    setStatus('удалён · Save');
  }};
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

  loadIndex(index, true);
}})();
</script>
</body>
</html>"""


__all__ = [
    "ROOT",
    "ensure_layout",
    "get_annotation",
    "image_path",
    "list_images",
    "render_peduncle_obb_html",
    "render_raw_jpeg",
    "save_annotation",
]
