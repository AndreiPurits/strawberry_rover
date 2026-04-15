#!/usr/bin/env python3
"""Local dataset curation UI (Flask).

Currently used for ripeness classifier labeling (dataset=classifier).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from flask import Flask, Response, abort, jsonify, redirect, request, send_file, session

from curation_classifier import (
    add_skipped,
    add_skipped_item,
    apply_classifier_action,
    classifier_image_file,
    clear_skipped,
    render_classifier_html,
    undo_classifier_move,
    write_class_counts_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DIAG_DIR = REPO_ROOT / "diagnostics"
LOG_PATH = DIAG_DIR / "curation_actions.jsonl"

app = Flask(__name__)
app.secret_key = "curation_ui_local_only"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _now() -> float:
    return time.time()


def _log(event: Dict) -> None:
    _safe_mkdir(LOG_PATH.parent)
    event = dict(event)
    event.setdefault("ts", _now())
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _normalize_dataset_param(s: Optional[str]) -> str:
    raw = (s or "detection").strip().lower()
    if raw in ("classier", "classif"):
        raw = "classifier"
    if raw not in ("classifier", "detection", "classification"):
        raw = "detection"
    return raw


def _classifier_bucket(s: Optional[str]) -> str:
    return s if s in ("all", "review_small") else "all"


@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True})


@app.get("/")
def index() -> Response:
    dataset = _normalize_dataset_param(request.args.get("dataset"))
    if dataset != "classifier":
        return Response(
            "<p>This UI is running. Open <code>/?dataset=classifier</code>.</p>",
            mimetype="text/html",
        )
    bucket = _classifier_bucket(request.args.get("bucket"))
    include_skipped = request.args.get("include_skipped", "0") == "1"
    raw_file = request.args.get("file")
    requested_file = (raw_file.strip() if isinstance(raw_file, str) and raw_file.strip() else None)
    write_class_counts_report()
    html = render_classifier_html(
        bucket,  # type: ignore[arg-type]
        include_skipped=include_skipped,
        requested_file=requested_file,
        include_skipped_val=1 if include_skipped else 0,
    )
    return Response(html, mimetype="text/html")


@app.post("/classifier/do")
def classifier_do() -> Response:
    bucket = _classifier_bucket(request.form.get("bucket"))
    filename = (request.form.get("filename") or "").strip()
    action = (request.form.get("action") or "").strip()
    source = (request.form.get("source") or "candidates").strip().lower()
    priority_label = (request.form.get("priority_label") or "").strip().lower()
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    if not filename:
        return redirect(f"/?dataset=classifier&bucket={bucket}&include_skipped={include_skipped}")
    if source not in ("candidates", "priority"):
        source = "candidates"
    ok, _msg, undo = apply_classifier_action(
        bucket,
        filename,
        action,
        log_fn=_log,
        source=source,  # type: ignore[arg-type]
        priority_label=priority_label,
    )
    if ok and undo:
        session.setdefault("classifier_undo_stack", []).append(undo)
        session.modified = True
    return redirect(f"/?dataset=classifier&bucket={bucket}&include_skipped={include_skipped}")


@app.post("/classifier/skip")
def classifier_skip() -> Response:
    bucket = _classifier_bucket(request.form.get("bucket"))
    filename = (request.form.get("filename") or "").strip()
    source = (request.form.get("source") or "candidates").strip().lower()
    priority_label = (request.form.get("priority_label") or "").strip().lower()
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    if filename:
        if source == "priority":
            from curation_classifier import QueueItem  # local import to avoid leaking to public API

            it = QueueItem(
                source="priority",  # type: ignore[arg-type]
                bucket=bucket,  # type: ignore[arg-type]
                filename=filename,
                path=classifier_image_file(bucket, filename, source="priority", priority_label=priority_label),
                priority_label=priority_label,
            )
            add_skipped_item(it)
        else:
            add_skipped(bucket, filename)  # type: ignore[arg-type]
    return redirect(f"/?dataset=classifier&bucket={bucket}&include_skipped={include_skipped}")


@app.post("/classifier/clear_skipped")
def classifier_clear_skipped() -> Response:
    bucket = _classifier_bucket(request.form.get("bucket"))
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    clear_skipped(bucket)  # type: ignore[arg-type]
    return redirect(f"/?dataset=classifier&bucket={bucket}&include_skipped={include_skipped}")


@app.post("/nav_classif")
def nav_classif() -> Response:
    bucket = _classifier_bucket(request.form.get("bucket"))
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    nav = (request.form.get("nav") or "").strip().lower()
    if nav == "undo":
        stack = list(session.get("classifier_undo_stack") or [])
        if stack:
            undo = stack.pop()
            session["classifier_undo_stack"] = stack
            session.modified = True
            undo_classifier_move(undo)
    return redirect(f"/?dataset=classifier&bucket={bucket}&include_skipped={include_skipped}")


@app.get("/img/classifier_crop/<bucket>/<path:filename>")
def img_classifier_crop(bucket: str, filename: str) -> Response:
    if bucket not in ("all", "review_small"):
        abort(404)
    p = classifier_image_file(bucket, filename)  # type: ignore[arg-type]
    if not _is_image(p):
        abort(404)
    return send_file(p)


@app.get("/img/classifier_priority/<label>/<path:filename>")
def img_classifier_priority(label: str, filename: str) -> Response:
    if label not in ("ripe", "unripe", "rotten"):
        abort(404)
    p = classifier_image_file("all", filename, source="priority", priority_label=label)  # type: ignore[arg-type]
    if not _is_image(p):
        abort(404)
    return send_file(p)


def main() -> int:
    default_port = int(os.environ.get("CURATION_UI_PORT", "7860"))
    ap = argparse.ArgumentParser(description="Local dataset curation UI (Flask).")
    ap.add_argument("--host", default=os.environ.get("CURATION_UI_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=default_port)
    args = ap.parse_args()

    print(
        f"Curation UI: http://127.0.0.1:{args.port}/  (classifier: ?dataset=classifier)\n"
        f"Bind: {args.host}:{args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

