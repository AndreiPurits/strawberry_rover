#!/usr/bin/env python3
"""Local dataset curation UI (Flask).

Modes:
  ?dataset=classifier  — ripeness labeling
  ?dataset=peduncle    — peduncle approve / reject → data/плодоножки апрувд/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from io import BytesIO
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
from curation_peduncle import (
    add_skipped as peduncle_add_skipped,
    apply_peduncle_action,
    clear_skipped as peduncle_clear_skipped,
    ensure_layout as peduncle_ensure_layout,
    peduncle_image_file,
    render_peduncle_html,
    render_preview_jpeg,
    undo_peduncle_move,
)
from curation_new_straw import (
    ensure_layout as new_straw_ensure_layout,
    get_annotation as new_straw_get_annotation,
    normalize_pack as box_editor_normalize_pack,
    render_labeled_jpeg as new_straw_render_jpeg,
    render_raw_jpeg as new_straw_render_raw_jpeg,
    render_new_straw_html,
    save_annotation as new_straw_save_annotation,
)
from curation_peduncle_obb import (
    dataset_stats as peduncle_obb_dataset_stats,
    delete_frame as peduncle_obb_delete_frame,
    ensure_layout as peduncle_obb_ensure_layout,
    get_annotation as peduncle_obb_get_annotation,
    migrate_relations as peduncle_obb_migrate_relations,
    render_peduncle_obb_html,
    render_raw_jpeg as peduncle_obb_render_raw_jpeg,
    save_annotation as peduncle_obb_save_annotation,
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
    raw = (s or "peduncle").strip().lower()
    if raw in ("classier", "classif"):
        raw = "classifier"
    if raw in ("ped", "плодоножки", "peduncles"):
        raw = "peduncle"
    if raw in ("new_straw", "new-straw", "newstraw", "web", "straw"):
        raw = "new_straw"
    if raw in ("peduncle_new", "peduncle-new", "new_peduncle", "pn", "плодоножки_new"):
        raw = "peduncle_new"
    if raw in ("large_frames", "large-frames", "large", "lf", "big_frames", "big"):
        raw = "large_frames"
    if raw in ("field_photos", "field-photos", "field", "phone", "field_edit"):
        raw = "field_photos"
    if raw in ("field_peduncle", "field_ba", "field_before_after", "field-peduncle"):
        raw = "field_peduncle"
    if raw in ("field_grasp_v1", "field-grasp", "grasp_v1"):
        raw = "field_grasp_v1"
    if raw in ("peduncle_obb", "peduncle-obb", "obb", "peduncle_label", "peduncle_edit"):
        raw = "peduncle_obb"
    if raw in ("peduncle_obb_pred", "peduncle_obb_predict", "obb_pred", "peduncle-obb-pred"):
        raw = "peduncle_obb_pred"
    if raw not in (
        "classifier",
        "detection",
        "classification",
        "peduncle",
        "new_straw",
        "peduncle_new",
        "large_frames",
        "field_photos",
        "field_peduncle",
        "field_grasp_v1",
        "peduncle_obb",
        "peduncle_obb_pred",
    ):
        raw = "peduncle"
    return raw


def _classifier_bucket(s: Optional[str]) -> str:
    return s if s in ("all", "review_small") else "all"


@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True})


@app.get("/")
def index() -> Response:
    dataset = _normalize_dataset_param(request.args.get("dataset"))
    if dataset == "classifier":
        bucket = _classifier_bucket(request.args.get("bucket"))
        include_skipped = request.args.get("include_skipped", "0") == "1"
        raw_file = request.args.get("file")
        requested_file = raw_file.strip() if isinstance(raw_file, str) and raw_file.strip() else None
        write_class_counts_report()
        html = render_classifier_html(
            bucket,  # type: ignore[arg-type]
            include_skipped=include_skipped,
            requested_file=requested_file,
            include_skipped_val=1 if include_skipped else 0,
        )
        return Response(html, mimetype="text/html")

    if dataset == "peduncle":
        include_skipped = request.args.get("include_skipped", "0") == "1"
        raw_file = request.args.get("file")
        requested_file = raw_file.strip() if isinstance(raw_file, str) and raw_file.strip() else None
        peduncle_ensure_layout()
        html = render_peduncle_html(
            include_skipped=include_skipped,
            requested_file=requested_file,
            include_skipped_val=1 if include_skipped else 0,
        )
        return Response(html, mimetype="text/html")

    if dataset in ("new_straw", "peduncle_new", "large_frames", "field_photos"):
        new_straw_ensure_layout(dataset)
        try:
            idx = int(request.args.get("i") or 0)
        except Exception:
            idx = 0
        html = render_new_straw_html(index=idx, pack=dataset)
        return Response(html, mimetype="text/html")

    if dataset == "peduncle_obb":
        peduncle_obb_ensure_layout()
        try:
            idx = int(request.args.get("i") or 0)
        except Exception:
            idx = 0
        html = render_peduncle_obb_html(index=idx)
        return Response(html, mimetype="text/html")

    if dataset in ("peduncle_obb_pred", "peduncle_obb_predict", "obb_pred"):
        pred_dir = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "peduncle_obb"
            / "yolov8n_peduncle_v2_predict_val"
        )
        imgs = sorted(
            [
                p.name
                for p in pred_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        ) if pred_dir.is_dir() else []
        cards = "".join(
            f'<figure style="margin:0"><img src="/img/peduncle_obb_pred/{n}" '
            f'style="max-width:100%;height:auto;border:1px solid #444"/>'
            f'<figcaption style="font-size:12px;color:#aaa;margin:4px 0 12px">{n}</figcaption></figure>'
            for n in imgs
        )
        html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Peduncle OBB v2 predict val</title>
        <style>body{{font-family:system-ui;margin:12px;background:#111;color:#eee}}
        .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}}</style></head>
        <body><h2>yolov8n-obb peduncle v2.1 mask-calyx — val ({len(imgs)})</h2>
        <p class="meta" style="color:#aaa">маска ягоды · жёлтый=calyx tip · оранжевый=peduncle · красный=cut</p>
        <p><a href="/?dataset=peduncle_obb" style="color:#8bc34a">← labeling</a></p>
        <div class="grid">{cards or "<p>нет картинок — прогони predict_v2</p>"}</div></body></html>"""
        return Response(html, mimetype="text/html")

    if dataset == "field_peduncle":
        root = Path(__file__).resolve().parents[1] / "runs" / "field_peduncle_before_after"
        side_dir = root / "side"
        imgs = (
            sorted(
                [
                    p.name
                    for p in side_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            if side_dir.is_dir()
            else []
        )
        cards = "".join(
            f'<figure style="margin:0 0 18px;width:100%">'
            f'<img src="/img/field_peduncle/side/{n}" style="width:100%;max-width:1100px;height:auto;border:1px solid #444"/>'
            f'<figcaption style="font-size:13px;color:#aaa;margin:6px 0">'
            f'{n} · <a href="/img/field_peduncle/before/{n}" style="color:#8bc34a">before</a> · '
            f'<a href="/img/field_peduncle/after/{n}" style="color:#ffb74d">after</a>'
            f"</figcaption></figure>"
            for n in imgs
        )
        html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Field peduncle — было / стало</title>
        <style>body{{font-family:system-ui;margin:12px;background:#111;color:#eee}} a{{color:#8bc34a}}</style></head>
        <body>
        <h2>Полевые кадры — было / стало ({len(imgs)})</h2>
        <p style="color:#aaa">слева BEFORE (твои berry) · справа AFTER (+peduncle OBB, красный=cut)</p>
        <p><a href="/?dataset=field_photos">← править berry</a></p>
        {cards or "<p>нет картинок</p>"}
        </body></html>"""
        return Response(html, mimetype="text/html")

    if dataset == "field_grasp_v1":
        root = Path(__file__).resolve().parents[1] / "runs" / "field_peduncle_grasp_v1"
        side_dir = root / "side"
        imgs = (
            sorted(
                [
                    p.name
                    for p in side_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            if side_dir.is_dir()
            else []
        )
        cards = "".join(
            f'<figure style="margin:0 0 18px;width:100%">'
            f'<img src="/img/field_grasp_v1/side/{n}" style="width:100%;max-width:1100px;height:auto;border:1px solid #444"/>'
            f'<figcaption style="font-size:13px;color:#aaa;margin:6px 0">{n} · '
            f'<a href="/img/field_grasp_v1/debug/{n}" style="color:#ffb74d">debug</a></figcaption></figure>'
            for n in imgs
        )
        html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Peduncle grasp v1 — close-up association</title>
        <style>body{{font-family:system-ui;margin:12px;background:#111;color:#eee}} a{{color:#8bc34a}}</style></head>
        <body>
        <h2>Grasp v1: calyx → close ROI → strict assoc ({len(imgs)})</h2>
        <p style="color:#aaa">слева BEFORE · справа AFTER (calyx/ROI/кандидаты/cut) · READY только с model-calyx</p>
        <p><a href="/?dataset=field_photos">berry editor</a> · <a href="/?dataset=field_peduncle">legacy was/became</a></p>
        {cards or "<p>нет картинок — scripts/run_peduncle_grasp_field_demo.py</p>"}
        </body></html>"""
        return Response(html, mimetype="text/html")

    return Response(
        "<p>Curation UI. Open "
        "<code><a href='/?dataset=field_grasp_v1'>/?dataset=field_grasp_v1</a></code> · "
        "<code><a href='/?dataset=field_photos'>/?dataset=field_photos</a></code> · "
        "<code><a href='/?dataset=field_peduncle'>/?dataset=field_peduncle</a></code> · "
        "<code><a href='/?dataset=peduncle_obb'>/?dataset=peduncle_obb</a></code> · "
        "<code><a href='/?dataset=classifier'>/?dataset=classifier</a></code>.</p>",
        mimetype="text/html",
    )


@app.post("/peduncle/do")
def peduncle_do() -> Response:
    filename = (request.form.get("filename") or "").strip()
    action = (request.form.get("action") or "").strip()
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    if not filename:
        return redirect(f"/?dataset=peduncle&include_skipped={include_skipped}")
    ok, _msg, undo = apply_peduncle_action(filename, action, log_fn=_log)
    if ok and undo:
        session.setdefault("peduncle_undo_stack", []).append(undo)
        session.modified = True
    return redirect(f"/?dataset=peduncle&include_skipped={include_skipped}")


@app.post("/peduncle/skip")
def peduncle_skip() -> Response:
    filename = (request.form.get("filename") or "").strip()
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    if filename:
        peduncle_add_skipped(filename)
    return redirect(f"/?dataset=peduncle&include_skipped={include_skipped}")


@app.post("/peduncle/clear_skipped")
def peduncle_clear_skipped_route() -> Response:
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    peduncle_clear_skipped()
    return redirect(f"/?dataset=peduncle&include_skipped={include_skipped}")


@app.post("/peduncle/undo")
def peduncle_undo() -> Response:
    include_skipped = 1 if request.form.get("include_skipped") == "1" else 0
    stack = list(session.get("peduncle_undo_stack") or [])
    if stack:
        undo = stack.pop()
        session["peduncle_undo_stack"] = stack
        session.modified = True
        undo_peduncle_move(undo)
    return redirect(f"/?dataset=peduncle&include_skipped={include_skipped}")


@app.get("/img/new_straw/<path:filename>")
def img_new_straw(filename: str) -> Response:
    data = new_straw_render_jpeg(filename, pack="new_straw")
    if data is None:
        abort(404)
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.get("/img/new_straw_raw/<path:filename>")
def img_new_straw_raw(filename: str) -> Response:
    data = new_straw_render_raw_jpeg(filename, pack="new_straw")
    if data is None:
        abort(404)
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.get("/api/new_straw/anno/<path:filename>")
def api_new_straw_get(filename: str) -> Response:
    anno = new_straw_get_annotation(filename, pack="new_straw")
    if anno is None:
        abort(404)
    return jsonify(anno)


@app.post("/api/new_straw/anno/<path:filename>")
def api_new_straw_save(filename: str) -> Response:
    payload = request.get_json(silent=True) or {}
    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        return jsonify({"ok": False, "error": "boxes must be a list"}), 400
    ok, msg = new_straw_save_annotation(filename, boxes, pack="new_straw")
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    _log({"action": "new_straw_save", "filename": Path(filename).name, "n_boxes": len(boxes)})
    return jsonify({"ok": True, "n_boxes": len(boxes)})


@app.get("/img/box_editor/<pack>/<path:filename>")
def img_box_editor(pack: str, filename: str) -> Response:
    pack_key = box_editor_normalize_pack(pack)
    if pack_key not in ("new_straw", "peduncle_new", "large_frames", "field_photos"):
        abort(404)
    data = new_straw_render_jpeg(filename, pack=pack_key)
    if data is None:
        abort(404)
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.get("/img/box_editor_raw/<pack>/<path:filename>")
def img_box_editor_raw(pack: str, filename: str) -> Response:
    pack_key = box_editor_normalize_pack(pack)
    if pack_key not in ("new_straw", "peduncle_new", "large_frames", "field_photos"):
        abort(404)
    data = new_straw_render_raw_jpeg(filename, pack=pack_key)
    if data is None:
        abort(404)
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.get("/api/box_editor/<pack>/anno/<path:filename>")
def api_box_editor_get(pack: str, filename: str) -> Response:
    pack_key = box_editor_normalize_pack(pack)
    if pack_key not in ("new_straw", "peduncle_new", "large_frames", "field_photos"):
        abort(404)
    anno = new_straw_get_annotation(filename, pack=pack_key)
    if anno is None:
        abort(404)
    return jsonify(anno)


@app.post("/api/box_editor/<pack>/anno/<path:filename>")
def api_box_editor_save(pack: str, filename: str) -> Response:
    pack_key = box_editor_normalize_pack(pack)
    if pack_key not in ("new_straw", "peduncle_new", "large_frames", "field_photos"):
        abort(404)
    payload = request.get_json(silent=True) or {}
    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        return jsonify({"ok": False, "error": "boxes must be a list"}), 400
    ok, msg = new_straw_save_annotation(filename, boxes, pack=pack_key)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    _log({"action": "box_editor_save", "pack": pack_key, "filename": Path(filename).name, "n_boxes": len(boxes)})
    return jsonify({"ok": True, "n_boxes": len(boxes)})


@app.get("/img/peduncle/<path:filename>")
def img_peduncle(filename: str) -> Response:
    data = render_preview_jpeg(filename)
    if data is not None:
        return send_file(BytesIO(data), mimetype="image/jpeg")
    p = peduncle_image_file(filename)
    if not _is_image(p):
        abort(404)
    return send_file(p)


@app.get("/img/peduncle_obb_raw/<path:filename>")
def img_peduncle_obb_raw(filename: str) -> Response:
    data = peduncle_obb_render_raw_jpeg(filename)
    if data is None:
        abort(404)
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.get("/img/peduncle_obb_pred/<path:filename>")
def img_peduncle_obb_pred(filename: str) -> Response:
    pred_dir = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "peduncle_obb"
        / "yolov8n_peduncle_v2_predict_val"
    )
    p = pred_dir / Path(filename).name
    if not p.is_file():
        abort(404)
    return send_file(p)


@app.get("/img/field_peduncle/<kind>/<path:filename>")
def img_field_peduncle(kind: str, filename: str) -> Response:
    if kind not in ("before", "after", "side"):
        abort(404)
    root = Path(__file__).resolve().parents[1] / "runs" / "field_peduncle_before_after" / kind
    p = root / Path(filename).name
    if not p.is_file():
        abort(404)
    return send_file(p)


@app.get("/img/field_grasp_v1/<kind>/<path:filename>")
def img_field_grasp_v1(kind: str, filename: str) -> Response:
    if kind not in ("side", "debug", "json"):
        abort(404)
    root = Path(__file__).resolve().parents[1] / "runs" / "field_peduncle_grasp_v1" / kind
    p = root / Path(filename).name
    if not p.is_file():
        abort(404)
    return send_file(p)


@app.get("/api/peduncle_obb/anno/<path:filename>")
def api_peduncle_obb_get(filename: str) -> Response:
    anno = peduncle_obb_get_annotation(filename)
    if anno is None:
        abort(404)
    return jsonify(anno)


@app.post("/api/peduncle_obb/anno/<path:filename>")
def api_peduncle_obb_save(filename: str) -> Response:
    payload = request.get_json(silent=True) or {}
    peduncle_obbs = payload.get("peduncle_obbs")
    if not isinstance(peduncle_obbs, list):
        return jsonify({"ok": False, "error": "peduncle_obbs must be a list"}), 400
    calyx_visible = payload.get("calyx_visible")
    calyx_point = payload.get("calyx_point")
    berry_boxes = payload.get("berry_boxes")
    ok, msg = peduncle_obb_save_annotation(
        filename,
        peduncle_obbs,
        calyx_visible=calyx_visible,
        calyx_point=calyx_point,
        berry_boxes=berry_boxes if isinstance(berry_boxes, list) else None,
    )
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    _log(
        {
            "action": "peduncle_obb_save",
            "filename": Path(filename).name,
            "n_obb": len(peduncle_obbs),
            "calyx_visible": calyx_visible,
        }
    )
    return jsonify({"ok": True, "n_obb": len(peduncle_obbs)})


@app.delete("/api/peduncle_obb/frame/<path:filename>")
def api_peduncle_obb_delete_frame(filename: str) -> Response:
    ok, msg = peduncle_obb_delete_frame(filename)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    _log({"action": "peduncle_obb_delete_frame", "filename": Path(filename).name})
    return jsonify({"ok": True})


@app.get("/api/peduncle_obb/stats")
def api_peduncle_obb_stats() -> Response:
    return jsonify(peduncle_obb_dataset_stats())


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
            from curation_classifier import QueueItem

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

    peduncle_ensure_layout()
    peduncle_obb_ensure_layout()
    new_straw_ensure_layout("new_straw")
    new_straw_ensure_layout("peduncle_new")
    new_straw_ensure_layout("large_frames")
    new_straw_ensure_layout("field_photos")
    print(
        f"Curation UI: http://127.0.0.1:{args.port}/?dataset=field_photos\n"
        f"  peduncle_obb: http://127.0.0.1:{args.port}/?dataset=peduncle_obb\n"
        f"  large_frames: http://127.0.0.1:{args.port}/?dataset=large_frames\n"
        f"  peduncle_new: http://127.0.0.1:{args.port}/?dataset=peduncle_new\n"
        f"  new_straw:  http://127.0.0.1:{args.port}/?dataset=new_straw\n"
        f"  peduncle:   http://127.0.0.1:{args.port}/?dataset=peduncle\n"
        f"  classifier: http://127.0.0.1:{args.port}/?dataset=classifier\n"
        f"Bind: {args.host}:{args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
