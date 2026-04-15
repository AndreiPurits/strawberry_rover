#!/usr/bin/env python3
"""Manual labeling UI helpers for ripeness classifier crops.

Pipeline:
  data/classifier_candidates/{all,review_small}/ -> data/classification_manual/{green,turning,ripe,rotten,rejected}/
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSIFIER_ROOT = REPO_ROOT / "data" / "classifier_candidates"
CLASSIFIER_ALL = CLASSIFIER_ROOT / "all"
CLASSIFIER_REVIEW = CLASSIFIER_ROOT / "review_small"
CLASSIFIER_REJECT_AUTO = CLASSIFIER_ROOT / "rejected_auto"
CLASSIFIER_REPORTS = CLASSIFIER_ROOT / "reports"
CROPS_INDEX_CSV = CLASSIFIER_REPORTS / "crops_index.csv"

PRIORITY_ROOT = REPO_ROOT / "data" / "classifier_priority_queue"
PRIORITY_RIPE = PRIORITY_ROOT / "ripe"
PRIORITY_UNRIPE = PRIORITY_ROOT / "unripe"
PRIORITY_ROTTEN = PRIORITY_ROOT / "rotten"

MANUAL_ROOT = REPO_ROOT / "data" / "classification_manual"
MANUAL_REPORTS = MANUAL_ROOT / "reports"
LABELING_STATE_PATH = MANUAL_REPORTS / "labeling_state.json"
CLASS_COUNTS_TXT = MANUAL_REPORTS / "class_counts.txt"

RIPENESS_CLASSES = ("green", "turning", "ripe", "rotten")
CLASS_TARGETS: Dict[str, int] = {"green": 800, "turning": 700, "ripe": 800, "rotten": 400}

Bucket = Literal["all", "review_small"]
QueueSource = Literal["candidates", "priority"]


@dataclasses.dataclass(frozen=True)
class QueueItem:
    source: QueueSource
    bucket: Bucket
    filename: str
    path: Path
    priority_label: str = ""  # "ripe"/"unripe" if source == priority

    @property
    def key(self) -> str:
        # Used for skip tracking; must be stable and unique across sources.
        return f"{self.source}:{self.bucket}:{self.filename}"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_manual_layout() -> None:
    for d in (
        MANUAL_ROOT / "green",
        MANUAL_ROOT / "turning",
        MANUAL_ROOT / "ripe",
        MANUAL_ROOT / "rotten",
        MANUAL_ROOT / "rejected",
        MANUAL_REPORTS,
    ):
        _safe_mkdir(d)


def _bucket_dir(bucket: Bucket) -> Path:
    return CLASSIFIER_ALL if bucket == "all" else CLASSIFIER_REVIEW


def iter_bucket_images(bucket: Bucket) -> List[QueueItem]:
    d = _bucket_dir(bucket)
    if not d.is_dir():
        return []
    items = [p for p in d.iterdir() if _is_image(p)]
    items_sorted = sorted(items, key=lambda p: p.name)
    return [
        QueueItem(source="candidates", bucket=bucket, filename=p.name, path=p)
        for p in items_sorted
    ]


def iter_priority_images(bucket: Bucket) -> List[QueueItem]:
    # Priority queue is independent of bucket; we surface it in any bucket view.
    items: List[QueueItem] = []
    for lbl, d in (("ripe", PRIORITY_RIPE), ("unripe", PRIORITY_UNRIPE), ("rotten", PRIORITY_ROTTEN)):
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if _is_image(p):
                items.append(
                    QueueItem(
                        source="priority",
                        bucket=bucket,
                        filename=p.name,
                        path=p,
                        priority_label=lbl,
                    )
                )
    prio_rank = {"rotten": 0, "unripe": 1, "ripe": 2}
    return sorted(items, key=lambda it: (prio_rank.get(it.priority_label, 99), it.filename))


def load_labeling_state() -> Dict[str, Any]:
    default: Dict[str, Any] = {"version": 1, "skipped": {"all": [], "review_small": []}, "events": []}
    if not LABELING_STATE_PATH.is_file():
        return default
    try:
        data = json.loads(LABELING_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    data.setdefault("version", 1)
    data.setdefault("skipped", {"all": [], "review_small": []})
    data.setdefault("events", [])
    return data


def save_labeling_state(data: Dict[str, Any]) -> None:
    ensure_manual_layout()
    LABELING_STATE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_skipped_set(bucket: Bucket) -> set:
    st = load_labeling_state()
    raw = (st.get("skipped") or {}).get(bucket) or []
    # Backward compatible: old entries were stored as filenames (candidates-only).
    out = set()
    for x in raw:
        s = str(x)
        out.add(s)
        if ":" not in s:
            out.add(f"candidates:{bucket}:{s}")
    return out


def add_skipped(bucket: Bucket, filename: str) -> None:
    st = load_labeling_state()
    st.setdefault("skipped", {"all": [], "review_small": []})
    bucket_sk = list(st["skipped"].setdefault(bucket, []))
    if filename not in bucket_sk:
        bucket_sk.append(filename)
    st["skipped"][bucket] = bucket_sk
    save_labeling_state(st)


def add_skipped_item(item: QueueItem) -> None:
    st = load_labeling_state()
    st.setdefault("skipped", {"all": [], "review_small": []})
    bucket_sk = list(st["skipped"].setdefault(item.bucket, []))
    key = item.key
    if key not in bucket_sk:
        bucket_sk.append(key)
    st["skipped"][item.bucket] = bucket_sk
    save_labeling_state(st)


def clear_skipped(bucket: Bucket) -> None:
    st = load_labeling_state()
    st.setdefault("skipped", {"all": [], "review_small": []})
    st["skipped"][bucket] = []
    save_labeling_state(st)


def visible_classifier_queue(bucket: Bucket, include_skipped: bool) -> List[QueueItem]:
    # Priority always goes first.
    items = iter_priority_images(bucket) + iter_bucket_images(bucket)
    sk = get_skipped_set(bucket)

    def _is_skipped(it: QueueItem) -> bool:
        # Accept both new key-based and old filename-based skip entries.
        return it.key in sk or it.filename in sk

    front = [it for it in items if not _is_skipped(it)]
    deferred = [it for it in items if _is_skipped(it)]
    return (front + deferred) if include_skipped else front


def count_dir_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if _is_image(p))


def count_candidate_buckets() -> Dict[str, int]:
    return {
        "all": count_dir_images(CLASSIFIER_ALL),
        "review_small": count_dir_images(CLASSIFIER_REVIEW),
        "rejected_auto": count_dir_images(CLASSIFIER_REJECT_AUTO),
    }


def count_manual_classes() -> Dict[str, int]:
    ensure_manual_layout()
    return {
        "green": count_dir_images(MANUAL_ROOT / "green"),
        "turning": count_dir_images(MANUAL_ROOT / "turning"),
        "ripe": count_dir_images(MANUAL_ROOT / "ripe"),
        "rotten": count_dir_images(MANUAL_ROOT / "rotten"),
        "rejected": count_dir_images(MANUAL_ROOT / "rejected"),
    }


def load_crop_index_row(filename: str) -> Optional[Dict[str, str]]:
    if not CROPS_INDEX_CSV.is_file():
        return None
    with CROPS_INDEX_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("crop_filename") == filename:
                return row
    return None


def update_crops_index_for_file(filename: str, *, class_label: str, status: str, manual_rel: str = "") -> None:
    if not CROPS_INDEX_CSV.is_file():
        return
    with CROPS_INDEX_CSV.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    if "manual_relative_path" not in fieldnames:
        fieldnames.append("manual_relative_path")
    updated = False
    for row in rows:
        if row.get("crop_filename") == filename:
            row["class_label"] = class_label
            row["status"] = status
            row["manual_relative_path"] = manual_rel
            updated = True
    if not updated:
        return
    _safe_mkdir(CROPS_INDEX_CSV.parent)
    with CROPS_INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_class_counts_report() -> None:
    ensure_manual_layout()
    manual = count_manual_classes()
    cand = count_candidate_buckets()
    unlabeled_total = cand["all"] + cand["review_small"]
    lines = [
        "Classification manual dataset counts",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "",
        "Targets (for UI):",
    ]
    for c in RIPENESS_CLASSES:
        lines.append(f"  {c}: current={manual[c]} target={CLASS_TARGETS[c]}")
    lines.append(f"  rejected (manual): {manual['rejected']}")
    lines.append("")
    lines.append("Candidates still in inbox:")
    lines.append(f"  all (unlabeled queue): {cand['all']}")
    lines.append(f"  review_small: {cand['review_small']}")
    lines.append(f"  rejected_auto: {cand['rejected_auto']}")
    lines.append("")
    lines.append(f"unlabeled_in_candidates (all+review_small): {unlabeled_total}")
    _safe_mkdir(CLASS_COUNTS_TXT.parent)
    CLASS_COUNTS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_label_event(event: Dict[str, Any]) -> None:
    st = load_labeling_state()
    st.setdefault("events", [])
    ev = dict(event)
    ev.setdefault("ts", time.time())
    st["events"].append(ev)
    if len(st["events"]) > 5000:
        st["events"] = st["events"][-4000:]
    save_labeling_state(st)


def classifier_image_file(bucket: Bucket, filename: str, *, source: QueueSource = "candidates", priority_label: str = "") -> Path:
    if source == "priority":
        if priority_label == "ripe":
            return PRIORITY_RIPE / filename
        if priority_label == "unripe":
            return PRIORITY_UNRIPE / filename
        if priority_label == "rotten":
            return PRIORITY_ROTTEN / filename
        # Fallback search (older links): try both.
        p = PRIORITY_RIPE / filename
        if p.exists():
            return p
        p2 = PRIORITY_UNRIPE / filename
        if p2.exists():
            return p2
        return PRIORITY_ROTTEN / filename
    return _bucket_dir(bucket) / filename


def destination_for_action(action: str) -> Path:
    if action == "reject":
        return MANUAL_ROOT / "rejected"
    if action in RIPENESS_CLASSES:
        return MANUAL_ROOT / action
    raise ValueError(action)


def apply_classifier_action(
    bucket: Bucket,
    filename: str,
    action: str,
    *,
    log_fn: Callable[[Dict[str, Any]], None],
    source: QueueSource = "candidates",
    priority_label: str = "",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    ensure_manual_layout()
    if action not in RIPENESS_CLASSES and action != "reject":
        return False, "invalid action", None
    src = classifier_image_file(bucket, filename, source=source, priority_label=priority_label)
    if not _is_image(src):
        return False, "source not found", None
    dest_dir = destination_for_action(action)
    _safe_mkdir(dest_dir)
    dst = dest_dir / filename
    if dst.exists():
        return False, "destination exists", None

    shutil.move(str(src), str(dst))

    cls_lbl = "" if action == "reject" else action
    stat = "rejected" if action == "reject" else "labeled"
    try:
        rel = str(dst.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(dst)
    update_crops_index_for_file(filename, class_label=cls_lbl, status=stat, manual_rel=rel)
    write_class_counts_report()

    undo = {"type": "classifier_move", "bucket": bucket, "filename": filename, "from_path": str(dst), "to_path": str(src)}
    ev = {"action": action, "bucket": bucket, "crop_filename": filename, "dest": rel, "source": source}
    if source == "priority":
        ev["priority_label"] = priority_label
    append_label_event(ev)
    log_fn({"action": "classifier_label", **ev})
    return True, "ok", undo


def undo_classifier_move(undo: Dict[str, Any]) -> bool:
    from_path = Path(undo.get("from_path") or "")
    to_path = Path(undo.get("to_path") or "")
    if not from_path.is_file():
        return False
    _safe_mkdir(to_path.parent)
    shutil.move(str(from_path), str(to_path))
    fn = undo.get("filename") or from_path.name
    update_crops_index_for_file(fn, class_label="", status="unlabeled", manual_rel="")
    write_class_counts_report()
    return True


def render_classifier_html(bucket: Bucket, *, include_skipped: bool, requested_file: Optional[str], include_skipped_val: int) -> str:
    ensure_manual_layout()
    visible = visible_classifier_queue(bucket, include_skipped=include_skipped)
    cur: Optional[QueueItem] = None
    if visible:
        if requested_file:
            for it in visible:
                if it.filename == requested_file:
                    cur = it
                    break
        if cur is None:
            cur = visible[0]

    manual = count_manual_classes()
    cand = count_candidate_buckets()
    unlabeled_total = cand["all"] + cand["review_small"]

    if cur is None:
        body = "<p><b>Готово</b>: в этой очереди нет crop (или запустите extract_classifier_crops.py).</p>"
        fname = ""
        img_html = ""
        wh = "-"
        src_img = "-"
        very_small_note = ""
        pos = "0"
        source = "candidates"
        source_badge = ""
        priority_lbl = ""
    else:
        fname = cur.filename
        row = load_crop_index_row(fname) or {}
        w = row.get("width", "-")
        h = row.get("height", "-")
        wh = f"{w} x {h}"
        src_img = row.get("source_image", "-")
        vs = row.get("very_small") == "1" or (str(w).isdigit() and str(h).isdigit() and (int(w) < 40 or int(h) < 40))
        very_small_note = '<div class="warn">Мелкий crop (рекомендуется внимательнее).</div>' if vs else ""
        source = cur.source
        priority_lbl = cur.priority_label
        if source == "priority":
            source_badge = f' <span style="background:#fff3e0;border:1px solid #ffb74d;border-radius:8px;padding:2px 8px;font-size:12px;">priority/{priority_lbl}</span>'
            img_url = f"/img/classifier_priority/{priority_lbl}/{fname}"
        else:
            source_badge = ""
            img_url = f"/img/classifier_crop/{bucket}/{fname}"
        border = "3px solid #e65100" if vs else "1px solid #eee"
        img_html = f'<img id="clfimg" alt="crop" src="{img_url}" style="max-width: min(100%, 520px); max-height: 70vh; border-radius:8px; border:{border};" />'
        pos = str(visible.index(cur) + 1) if cur in visible else "?"
        body = ""

    chk = "checked" if include_skipped else ""
    b_all = "selected" if bucket == "all" else ""
    b_rev = "selected" if bucket == "review_small" else ""

    rows_targets = ""
    for c in RIPENESS_CLASSES:
        rows_targets += f"<tr><td>{c}</td><td>{manual[c]}</td><td>{CLASS_TARGETS[c]}</td></tr>"

    counters = f"""
    <table class="ct">
      <tr><th>class</th><th>current</th><th>target</th></tr>
      {rows_targets}
    </table>
    <p class="meta">Unlabeled in candidates (all+review): <b>{unlabeled_total}</b>
    &nbsp;|&nbsp; review_small folder: <b>{cand['review_small']}</b>
    &nbsp;|&nbsp; rejected_auto: <b>{cand['rejected_auto']}</b>
    &nbsp;|&nbsp; manual rejected: <b>{manual['rejected']}</b></p>
    """

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Classifier curation</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; margin: 16px; }}
    .card {{ border:1px solid #ddd; border-radius:10px; padding:12px; max-width:900px; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:10px; }}
    button {{ padding:10px 14px; border-radius:10px; border:1px solid #ccc; background:#fff; cursor:pointer; }}
    .g {{ border-color:#2e7d32; background:#e8f5e9; }}
    .t {{ border-color:#f9a825; background:#fffde7; }}
    .r {{ border-color:#c62828; background:#ffebee; }}
    .rot {{ border-color:#5d4037; background:#efebe9; }}
    .del {{ border-color:#d93025; }}
    .meta {{ color:#444; font-size:14px; }}
    .warn {{ color:#e65100; font-weight:600; margin:8px 0; }}
    table.ct {{ border-collapse:collapse; margin-top:8px; font-size:14px; }}
    table.ct th, table.ct td {{ border:1px solid #ddd; padding:4px 10px; text-align:left; }}
    code {{ background:#f5f5f5; padding:2px 6px; border-radius:4px; }}
    kbd {{ background:#eee; padding:2px 6px; border-radius:4px; font-size:12px; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>Classifier mode (ripeness)</h2>
    <p class="meta" style="background:#e3f2fd;padding:8px 10px;border-radius:8px;max-width:820px;">
      Если вы видите «<b>DONE</b>: больше нет изображений…» — вы не в classifier режиме.
      Откройте <code>/?dataset=classifier</code>.
    </p>
    <p class="meta">Очередь: <code>data/classifier_candidates/{bucket}/</code> → <code>data/classification_manual/&lt;class&gt;/</code></p>
    <form method="get" action="/" class="row">
      <input type="hidden" name="dataset" value="classifier" />
      <label>Bucket: <select name="bucket" onchange="this.form.submit()">
        <option value="all" {b_all}>all</option>
        <option value="review_small" {b_rev}>review_small</option>
      </select></label>
      <label><input type="checkbox" name="include_skipped" value="1" {chk}/> показать пропущенные в конце очереди</label>
      <button type="submit">Обновить</button>
    </form>
    <p class="meta">Позиция: <b>{pos}</b> / <b>{len(visible)}</b> &nbsp;|&nbsp; файл: <code>{fname}</code>{source_badge}</p>
    <p class="meta">Размер: {wh} &nbsp;|&nbsp; источник: <code>{src_img}</code></p>
    {very_small_note}
    {body}
    <div style="margin-top:12px;">{img_html}</div>
    {counters}
    <p class="meta">Горячие клавиши: <kbd>1</kbd> green · <kbd>2</kbd> turning · <kbd>3</kbd> ripe · <kbd>4</kbd> rotten · <kbd>D</kbd> delete/reject · <kbd>S</kbd> пропустить позже</p>
    <div class="row">
      <form id="fgreen" method="post" action="/classifier/do"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="action" value="green"/><button type="submit" class="g">green</button></form>
      <form id="fturn" method="post" action="/classifier/do"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="action" value="turning"/><button type="submit" class="t">turning</button></form>
      <form id="fripe" method="post" action="/classifier/do"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="action" value="ripe"/><button type="submit" class="r">ripe</button></form>
      <form id="frot" method="post" action="/classifier/do"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="action" value="rotten"/><button type="submit" class="rot">rotten</button></form>
      <form id="frej" method="post" action="/classifier/do"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="action" value="reject"/><button type="submit" class="del">delete → rejected</button></form>
      <form method="post" action="/classifier/skip"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="filename" value="{fname}"/><input type="hidden" name="source" value="{source}"/><input type="hidden" name="priority_label" value="{priority_lbl}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><button type="submit">пропустить (S)</button></form>
    </div>
    <div class="row">
      <form method="post" action="/classifier/clear_skipped"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><button type="submit">сбросить пропуски в этом bucket</button></form>
      <form method="post" action="/nav_classif" style="display:inline"><input type="hidden" name="bucket" value="{bucket}"/><input type="hidden" name="include_skipped" value="{include_skipped_val}"/><input type="hidden" name="nav" value="undo"/><button type="submit">Undo classifier</button></form>
    </div>
  </div>
  <script>
  (function() {{
    document.addEventListener("keydown", function(e) {{
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) return;
      var k = e.key;
      if (k === "1") {{ e.preventDefault(); document.getElementById("fgreen").submit(); }}
      else if (k === "2") {{ e.preventDefault(); document.getElementById("fturn").submit(); }}
      else if (k === "3") {{ e.preventDefault(); document.getElementById("fripe").submit(); }}
      else if (k === "4") {{ e.preventDefault(); document.getElementById("frot").submit(); }}
      else if (k === "d" || k === "D") {{ e.preventDefault(); document.getElementById("frej").submit(); }}
      else if (k === "s" || k === "S") {{
        e.preventDefault();
        var f = document.querySelector('form[action="/classifier/skip"]');
        if (f) f.submit();
      }}
    }});
  }})();
  </script>
</body>
</html>"""


__all__ = [
    "Bucket",
    "add_skipped",
    "apply_classifier_action",
    "classifier_image_file",
    "clear_skipped",
    "render_classifier_html",
    "undo_classifier_move",
    "write_class_counts_report",
]

