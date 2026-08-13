#!/usr/bin/env python3
"""Interactive primary-review editor for new_straw (berry + future_crop boxes).

- Draw new boxes (drag)
- Click box to select, Delete / Del removes it
- Class: 0=berry, 1=future_crop
- Save writes YOLO labels next to the image
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_NAMES = {0: "berry", 1: "future_crop"}
# Full-res labels; editor loads a downscaled JPEG so browsers don't freeze on 8K web photos.
EDITOR_MAX_SIDE = 1920

# Named packs → review roots. peduncle_new = only frames from data/плодоножки/new.
PACKS: Dict[str, Path] = {
    "new_straw": REPO_ROOT / "data" / "new_straw_primary_review",
    "peduncle_new": REPO_ROOT / "data" / "плодоножки" / "new_primary_review",
    "large_frames": REPO_ROOT / "data" / "плодоножки" / "large_frames_primary_review",
    "field_photos": REPO_ROOT / "data" / "field_photos_primary_review",
}
PACK_TITLES: Dict[str, str] = {
    "new_straw": "New straw — редактор боксов",
    "peduncle_new": "Плодоножки/new — primary review",
    "large_frames": "Крупные кадры — primary review (больше→меньше)",
    "field_photos": "Полевые фото — детект ягоды (редактор)",
}
# Backward-compat default root
ROOT = PACKS["new_straw"]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_pack(pack: Optional[str]) -> str:
    raw = (pack or "new_straw").strip().lower()
    if raw in ("peduncle_new", "peduncle-new", "плодоножки_new", "new_peduncle", "pn"):
        return "peduncle_new"
    if raw in ("large_frames", "large-frames", "large", "lf", "big_frames", "big"):
        return "large_frames"
    if raw in ("field_photos", "field-photos", "field", "phone", "field_edit"):
        return "field_photos"
    if raw in ("new_straw", "new-straw", "newstraw", "web"):
        return "new_straw"
    return "new_straw" if raw not in PACKS else raw


def pack_root(pack: Optional[str] = None) -> Path:
    return PACKS[normalize_pack(pack)]


def ensure_layout(pack: Optional[str] = None) -> None:
    root = pack_root(pack)
    for d in (root / "images", root / "labels", root / "preview", root / "reports"):
        _safe_mkdir(d)


def list_images(pack: Optional[str] = None) -> List[Path]:
    d = pack_root(pack) / "images"
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: p.name)


def image_path(filename: str, pack: Optional[str] = None) -> Path:
    # prevent path traversal
    name = Path(filename).name
    return pack_root(pack) / "images" / name


def label_path(filename: str, pack: Optional[str] = None) -> Path:
    return pack_root(pack) / "labels" / f"{Path(filename).stem}.txt"


def _parse_yolo(label_file: Path, w: int, h: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
        x1 = max(0.0, (xc - bw / 2.0) * w)
        y1 = max(0.0, (yc - bh / 2.0) * h)
        x2 = min(float(w), (xc + bw / 2.0) * w)
        y2 = min(float(h), (yc + bh / 2.0) * h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"cls": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def _boxes_to_yolo(boxes: List[Dict[str, Any]], w: int, h: int) -> str:
    lines: List[str] = []
    for b in boxes:
        try:
            cid = int(b["cls"])
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        except Exception:
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(1.0, min(float(w), x2))
        y2 = max(1.0, min(float(h), y2))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        if bw < 2 or bh < 2:
            continue
        xc = (x1 + x2) / 2.0 / float(w)
        yc = (y1 + y2) / 2.0 / float(h)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw / float(w):.6f} {bh / float(h):.6f}")
    return ("\n".join(lines) + ("\n" if lines else ""))


def _editor_display_size(w: int, h: int, max_side: int = EDITOR_MAX_SIDE) -> Tuple[int, int, float]:
    side = max(w, h)
    if side <= max_side:
        return w, h, 1.0
    s = max_side / float(side)
    return int(round(w * s)), int(round(h * s)), s


def get_annotation(filename: str, pack: Optional[str] = None) -> Optional[Dict[str, Any]]:
    p = image_path(filename, pack)
    if not p.is_file():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    boxes = _parse_yolo(label_path(filename, pack), w, h)
    dw, dh, _ = _editor_display_size(w, h)
    return {
        "filename": p.name,
        "width": w,
        "height": h,
        "display_width": dw,
        "display_height": dh,
        "boxes": boxes,
        "classes": CLASS_NAMES,
        "pack": normalize_pack(pack),
    }


def render_raw_jpeg(filename: str, max_side: int = EDITOR_MAX_SIDE, pack: Optional[str] = None) -> Optional[bytes]:
    """Downscaled JPEG for the in-browser editor (labels stay full-resolution)."""
    p = image_path(filename, pack)
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


def save_annotation(filename: str, boxes: List[Dict[str, Any]], pack: Optional[str] = None) -> Tuple[bool, str]:
    p = image_path(filename, pack)
    if not p.is_file():
        return False, "image not found"
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, "cannot read image"
    h, w = bgr.shape[:2]
    ensure_layout(pack)
    text = _boxes_to_yolo(boxes, w, h)
    label_path(filename, pack).write_text(text, encoding="utf-8")
    # refresh preview overlay jpeg for convenience
    try:
        vis = bgr.copy()
        for b in boxes:
            cid = int(b["cls"])
            x1, y1, x2, y2 = [int(round(float(b[k]))) for k in ("x1", "y1", "x2", "y2")]
            color = (0, 220, 0) if cid == 0 else (0, 140, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                CLASS_NAMES.get(cid, str(cid)),
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(pack_root(pack) / "preview" / p.name), vis)
    except Exception:
        pass
    return True, "ok"


def render_labeled_jpeg(filename: str, max_side: int = 1280, pack: Optional[str] = None) -> Optional[bytes]:
    """Kept for compatibility; editor uses raw image + canvas."""
    anno = get_annotation(filename, pack)
    p = image_path(filename, pack)
    if anno is None or not p.is_file():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    vis = bgr.copy()
    for b in anno["boxes"]:
        cid = int(b["cls"])
        x1, y1, x2, y2 = [int(round(float(b[k]))) for k in ("x1", "y1", "x2", "y2")]
        color = (0, 220, 0) if cid == 0 else (0, 140, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    side = max(h, w)
    if side > max_side:
        scale = max_side / float(side)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return bytes(buf) if ok else None


def render_new_straw_html(*, index: int = 0, pack: Optional[str] = None) -> str:
    pack_key = normalize_pack(pack)
    root = pack_root(pack_key)
    title = PACK_TITLES.get(pack_key, "Box editor")
    ensure_layout(pack_key)
    images = list_images(pack_key)
    n = len(images)
    if n == 0:
        return f"""<!doctype html><html lang="ru"><body style="font-family:system-ui;margin:16px">
        <h2>{title}</h2>
        <p>Пусто. Ожидаются файлы в <code>{root}/images</code>.</p>
        <p><a href="/?dataset=peduncle">← peduncle approve</a></p>
        </body></html>"""

    index = max(0, min(int(index), n - 1))
    cur = images[index]
    prev_i = max(0, index - 1)
    next_i = min(n - 1, index + 1)
    names_json = json.dumps([p.name for p in images], ensure_ascii=False)
    ds = pack_key  # dataset query param

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 12px; background:#111; color:#eee; }}
    .card {{ max-width: 1200px; }}
    .row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0; }}
    button, a.btn, select {{ padding:8px 12px; border-radius:8px; border:1px solid #555; background:#222; color:#eee; cursor:pointer; text-decoration:none; }}
    button.primary {{ background:#1b5e20; border-color:#2e7d32; }}
    button.danger {{ background:#b71c1c; border-color:#e53935; }}
    button.active {{ outline:2px solid #ffeb3b; }}
    .meta {{ color:#bbb; font-size:13px; }}
    code {{ background:#333; padding:2px 6px; border-radius:4px; }}
    kbd {{ background:#333; padding:2px 6px; border-radius:4px; font-size:12px; }}
    #wrap {{ position:relative; display:inline-block; max-width:100%; border:1px solid #444; background:#000; }}
    canvas {{ display:block; max-width:100%; cursor:crosshair; }}
    #status {{ min-height:1.2em; color:#8bc34a; }}
    #status.err {{ color:#ef9a9a; }}
    .legend span {{ display:inline-block; padding:2px 8px; border-radius:6px; margin-right:6px; font-size:12px; }}
    .berry {{ background:#1b5e20; }}
    .crop {{ background:#e65100; }}
  </style>
</head>
<body>
  <div class="card">
    <h2 style="margin:0 0 8px">{title}</h2>
    <p class="meta">
      Кадр <b id="pos">{index + 1}</b> / <b>{n}</b> · <code id="fname">{cur.name}</code>
      · только эти кадры · порядок: больше → меньше · после Save нарежем в апрувд ·
      <a class="btn" href="/?dataset=peduncle">peduncle</a>
    </p>
    <div class="row legend">
      <span class="berry">0 berry</span>
      <span class="crop">1 future_crop</span>
    </div>
    <div class="row">
      <label>Класс:&nbsp;
        <select id="cls">
          <option value="0" selected>berry</option>
          <option value="1">future_crop</option>
        </select>
      </label>
      <button type="button" id="btnSave" class="primary">Save (S)</button>
      <button type="button" id="btnDel" class="danger">Delete selected (Del)</button>
      <button type="button" id="btnClear">Clear all</button>
      <a class="btn" id="btnPrev" href="/?dataset={ds}&i={prev_i}">← prev</a>
      <a class="btn" id="btnNext" href="/?dataset={ds}&i={next_i}">next →</a>
    </div>
    <p class="meta">
      Тяни мышью новый бокс · клик по боксу = выбор · <kbd>1</kbd>/<kbd>2</kbd> класс ·
      <kbd>S</kbd> save · <kbd>Del</kbd>/<kbd>Backspace</kbd> удалить · <kbd>←</kbd>/<kbd>→</kbd> кадр
      (с несохранёнными спросит)
    </p>
    <div id="status"></div>
    <div id="wrap"><canvas id="cv"></canvas></div>
  </div>
<script>
(function() {{
  const PACK = {json.dumps(pack_key)};
  const DS = {json.dumps(ds)};
  const NAMES = {names_json};
  let index = {index};
  let dirty = false;
  let img = new Image();
  let boxes = []; // {{cls,x1,y1,x2,y2}} in full-res image pixels
  let fullW = 1, fullH = 1;
  let dispW = 1, dispH = 1; // loaded JPEG size (<= full, server downscales huge frames)
  let imgScale = 1; // disp / full
  let scale = 1; // canvas viewport fit
  let selected = -1;
  let drawing = false;
  let startX = 0, startY = 0;
  let curX = 0, curY = 0;

  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  const statusEl = document.getElementById('status');
  const clsEl = document.getElementById('cls');

  function setStatus(msg, err) {{
    statusEl.textContent = msg || '';
    statusEl.className = err ? 'err' : '';
  }}

  function displaySize() {{
    const maxW = Math.min(window.innerWidth - 40, 1100);
    const maxH = Math.min(window.innerHeight - 220, 900);
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
    const dx = px / scale;
    const dy = py / scale;
    return [dx / imgScale, dy / imgScale];
  }}

  function colorFor(cls, sel) {{
    if (cls === 0) return sel ? 'rgba(0,255,0,0.95)' : 'rgba(0,200,0,0.9)';
    return sel ? 'rgba(255,180,0,0.95)' : 'rgba(255,120,0,0.9)';
  }}

  function draw() {{
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    for (let i = 0; i < boxes.length; i++) {{
      const b = boxes[i];
      const sel = i === selected;
      ctx.lineWidth = sel ? 3 : 2;
      ctx.strokeStyle = colorFor(b.cls, sel);
      ctx.fillStyle = sel ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0)';
      const x = b.x1 * imgScale * scale, y = b.y1 * imgScale * scale;
      const w = (b.x2 - b.x1) * imgScale * scale, h = (b.y2 - b.y1) * imgScale * scale;
      ctx.strokeRect(x, y, w, h);
      if (sel) ctx.fillRect(x, y, w, h);
      ctx.fillStyle = colorFor(b.cls, true);
      ctx.font = '14px sans-serif';
      const label = (b.cls === 0 ? 'berry' : 'future_crop') + ' #' + i;
      ctx.fillText(label, x + 4, Math.max(14, y - 4));
    }}
    if (drawing) {{
      const cls = parseInt(clsEl.value, 10);
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = colorFor(cls, true);
      ctx.lineWidth = 2;
      const x = Math.min(startX, curX) * imgScale * scale;
      const y = Math.min(startY, curY) * imgScale * scale;
      const w = Math.abs(curX - startX) * imgScale * scale;
      const h = Math.abs(curY - startY) * imgScale * scale;
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }}
  }}

  function hitTest(nx, ny) {{
    // topmost (last drawn) first
    for (let i = boxes.length - 1; i >= 0; i--) {{
      const b = boxes[i];
      if (nx >= b.x1 && nx <= b.x2 && ny >= b.y1 && ny <= b.y2) return i;
    }}
    return -1;
  }}

  async function loadIndex(i, force) {{
    if (dirty && !force) {{
      if (!confirm('Есть несохранённые изменения. Уйти без Save?')) return false;
    }}
    index = Math.max(0, Math.min(i, NAMES.length - 1));
    const name = NAMES[index];
    document.getElementById('pos').textContent = String(index + 1);
    document.getElementById('fname').textContent = name;
    document.getElementById('btnPrev').href = '/?dataset=' + DS + '&i=' + Math.max(0, index - 1);
    document.getElementById('btnNext').href = '/?dataset=' + DS + '&i=' + Math.min(NAMES.length - 1, index + 1);
    history.replaceState(null, '', '/?dataset=' + DS + '&i=' + index);
    setStatus('loading…');
    const res = await fetch('/api/box_editor/' + PACK + '/anno/' + encodeURIComponent(name));
    if (!res.ok) {{ setStatus('load failed', true); return false; }}
    const data = await res.json();
    fullW = data.width; fullH = data.height;
    boxes = data.boxes || [];
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
      img.src = '/img/box_editor_raw/' + PACK + '/' + encodeURIComponent(name) + '?t=' + Date.now();
    }});
    displaySize();
    draw();
    const down = (fullW > dispW || fullH > dispH) ? (' · preview ' + dispW + '×' + dispH) : '';
    setStatus('боксов: ' + boxes.length + down);
    return true;
  }}

  async function save() {{
    const name = NAMES[index];
    setStatus('saving…');
    const res = await fetch('/api/box_editor/' + PACK + '/anno/' + encodeURIComponent(name), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ boxes }}),
    }});
    const data = await res.json().catch(() => ({{Ok:false}}));
    if (!res.ok || data.ok === false) {{
      setStatus('save failed: ' + (data.error || res.status), true);
      return;
    }}
    dirty = false;
    setStatus('saved · boxes=' + boxes.length);
  }}

  cv.addEventListener('mousedown', (e) => {{
    const [nx, ny] = toFull(e.clientX, e.clientY);
    const hit = hitTest(nx, ny);
    if (hit >= 0 && !e.shiftKey) {{
      selected = hit;
      drawing = false;
      draw();
      return;
    }}
    drawing = true;
    selected = -1;
    startX = curX = nx;
    startY = curY = ny;
    draw();
  }});
  cv.addEventListener('mousemove', (e) => {{
    if (!drawing) return;
    const [nx, ny] = toFull(e.clientX, e.clientY);
    curX = Math.max(0, Math.min(fullW, nx));
    curY = Math.max(0, Math.min(fullH, ny));
    draw();
  }});
  function endDraw() {{
    if (!drawing) return;
    drawing = false;
    const x1 = Math.min(startX, curX), y1 = Math.min(startY, curY);
    const x2 = Math.max(startX, curX), y2 = Math.max(startY, curY);
    if ((x2 - x1) > 8 && (y2 - y1) > 8) {{
      boxes.push({{ cls: parseInt(clsEl.value, 10), x1, y1, x2, y2 }});
      selected = boxes.length - 1;
      dirty = true;
      setStatus('добавлен бокс · не забывай Save');
    }}
    draw();
  }}
  cv.addEventListener('mouseup', endDraw);
  cv.addEventListener('mouseleave', endDraw);

  document.getElementById('btnSave').onclick = () => save();
  document.getElementById('btnDel').onclick = () => {{
    if (selected < 0) {{ setStatus('сначала выбери бокс кликом', true); return; }}
    boxes.splice(selected, 1);
    selected = -1;
    dirty = true;
    draw();
    setStatus('удалён · Save чтобы записать');
  }};
  document.getElementById('btnClear').onclick = () => {{
    if (!boxes.length) return;
    if (!confirm('Удалить все боксы на этом кадре?')) return;
    boxes = [];
    selected = -1;
    dirty = true;
    draw();
  }};

  document.addEventListener('keydown', async (e) => {{
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) return;
    if (e.key === 's' || e.key === 'S') {{ e.preventDefault(); save(); }}
    else if (e.key === 'Delete' || e.key === 'Backspace') {{
      e.preventDefault();
      document.getElementById('btnDel').click();
    }}
    else if (e.key === '1') {{ clsEl.value = '0'; }}
    else if (e.key === '2') {{ clsEl.value = '1'; }}
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
    "PACKS",
    "ensure_layout",
    "get_annotation",
    "image_path",
    "list_images",
    "normalize_pack",
    "pack_root",
    "render_labeled_jpeg",
    "render_raw_jpeg",
    "render_new_straw_html",
    "save_annotation",
]
