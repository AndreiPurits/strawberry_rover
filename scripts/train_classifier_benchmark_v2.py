#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["green", "turning", "ripe", "rotten"]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _weights_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return -1.0


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]


def _confusion_and_metrics(y_true: List[int], y_pred: List[int], n_classes: int) -> Metrics:
    conf = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf[t][p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    correct = sum(conf[i][i] for i in range(n_classes))
    total = sum(sum(r) for r in conf)
    acc = (correct / total) if total else 0.0

    for i in range(n_classes):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(n_classes) if r != i)
        fn = sum(conf[i][c] for c in range(n_classes) if c != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[CLASSES[i]] = {"precision": prec, "recall": rec, "f1": f1}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    pm = sum(precisions) / n_classes
    rm = sum(recalls) / n_classes
    f1m = sum(f1s) / n_classes
    return Metrics(
        accuracy=acc,
        precision_macro=pm,
        recall_macro=rm,
        f1_macro=f1m,
        per_class=per_class,
        confusion=conf,
    )


def _make_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(name)


def _build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def _epoch_train(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optim.step()
        bs = int(x.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def _epoch_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int], List[List[float]]]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        bs = int(x.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend([int(v) for v in y.cpu().tolist()])
        y_pred.extend([int(v) for v in pred.cpu().tolist()])
        y_prob.extend([[float(z) for z in row] for row in probs.cpu().tolist()])
    return total_loss / max(1, n), y_true, y_pred, y_prob


@torch.no_grad()
def _timing_ms_per_image(model: nn.Module, loader: DataLoader, device: torch.device, *, warmup_batches: int = 5) -> Tuple[float, float]:
    model.eval()
    batches = []
    for i, (x, _y) in enumerate(loader):
        batches.append(x)
        if i >= warmup_batches + 10:
            break

    # Warmup
    for x in batches[:warmup_batches]:
        _ = model(x.to(device, non_blocking=True))
    torch.cuda.synchronize(device)

    # Timed
    t0 = time.time()
    n_img = 0
    for x in batches[warmup_batches:]:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        n_img += int(x.shape[0])
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    ms = (dt / max(1, n_img)) * 1000.0
    fps = (n_img / dt) if dt > 0 else 0.0
    return ms, fps


def _draw_preview(
    src_img_path: Path,
    *,
    true_cls: str,
    pred_cls: str,
    conf: float,
    out_path: Path,
) -> None:
    img = Image.open(src_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    txt = f"true: {true_cls} | pred: {pred_cls} | conf: {conf:.3f}"
    # simple readable banner
    pad = 6
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        # Pillow>=10: use textbbox
        x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
        tw, th = (x1 - x0), (y1 - y0)
    except Exception:
        # Fallback: estimate
        tw, th = (len(txt) * 6), 12
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), txt, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def train_one_model(
    *,
    model_name: str,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
    img_size: int,
    batch_size: int,
    workers: int,
    epochs: int,
    patience: int,
    seed: int,
    preview_count: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "ok"
    err: str = ""

    best_path = out_dir / "best.pt"
    try:
        _seed_all(seed)
        train_tf, eval_tf = _build_transforms(img_size)
        # ImageFolder uses alphabetical class order; we remap to fixed CLASSES order.
        ds_train_raw = datasets.ImageFolder(str(data_root / "train"), transform=train_tf)
        ds_val_raw = datasets.ImageFolder(str(data_root / "val"), transform=eval_tf)
        ds_test_raw = datasets.ImageFolder(str(data_root / "test"), transform=eval_tf)

        def _make_remap(ds: datasets.ImageFolder) -> Dict[int, int]:
            name_of = {i: n for i, n in enumerate(ds.classes)}
            desired = {n: i for i, n in enumerate(CLASSES)}
            remap: Dict[int, int] = {}
            for src_i, nm in name_of.items():
                if nm not in desired:
                    raise RuntimeError(f"Unexpected class folder '{nm}' in {ds.root}")
                remap[src_i] = desired[nm]
            return remap

        remap_train = _make_remap(ds_train_raw)
        remap_val = _make_remap(ds_val_raw)
        remap_test = _make_remap(ds_test_raw)

        ds_train = datasets.ImageFolder(
            str(data_root / "train"),
            transform=train_tf,
            target_transform=lambda t, rm=remap_train: rm[int(t)],
        )
        ds_val = datasets.ImageFolder(
            str(data_root / "val"),
            transform=eval_tf,
            target_transform=lambda t, rm=remap_val: rm[int(t)],
        )
        ds_test = datasets.ImageFolder(
            str(data_root / "test"),
            transform=eval_tf,
            target_transform=lambda t, rm=remap_test: rm[int(t)],
        )

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        model = _make_model(model_name, num_classes=len(CLASSES)).to(device)

        # Fast path: reuse existing best.pt (skip retraining)
        if best_path.is_file():
            ckpt = torch.load(best_path, map_location="cpu")
            sd = ckpt.get("state_dict")
            if isinstance(sd, dict):
                model.load_state_dict(sd)
                model.to(device)
                model.eval()
                te_loss, y_t, y_p, y_prob = _epoch_eval(model, dl_test, device)
                m_test = _confusion_and_metrics(y_t, y_p, len(CLASSES))
                ms, fps = _timing_ms_per_image(model, dl_test, device)
                preview_dir = REPO_ROOT / "data" / "classification_preview" / model_name
                preview_dir.mkdir(parents=True, exist_ok=True)
                try:
                    n_prev = min(preview_count, len(ds_test))
                    for i in range(n_prev):
                        img_path, y = ds_test.samples[i]
                        probs = y_prob[i]
                        pred = int(max(range(len(probs)), key=lambda k: probs[k]))
                        conf = float(probs[pred])
                        _draw_preview(
                            Path(img_path),
                            true_cls=CLASSES[int(y)],
                            pred_cls=CLASSES[pred],
                            conf=conf,
                            out_path=preview_dir / Path(img_path).name,
                        )
                except Exception:
                    # Preview is non-critical
                    pass
                (out_dir / "confusion_matrix.json").write_text(json.dumps(m_test.confusion, indent=2) + "\n", encoding="utf-8")
                return {
                    "model_name": model_name,
                    "status": "ok",
                    "error": "",
                    "best_epoch": int(ckpt.get("epoch") or -1),
                    "best_weights_path": str(best_path),
                    "model_size_mb": _weights_size_mb(best_path),
                    "test_loss": te_loss,
                    "metrics": {
                        "accuracy": m_test.accuracy,
                        "precision_macro": m_test.precision_macro,
                        "recall_macro": m_test.recall_macro,
                        "f1_macro": m_test.f1_macro,
                        "per_class": m_test.per_class,
                    },
                    "confusion_matrix": m_test.confusion,
                    "timing": {"inference_time_ms_per_image": ms, "fps": fps},
                    "preview_dir": str(preview_dir),
                    "history": [],
                }

        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        best_f1 = -1.0
        best_epoch = -1
        history: List[Dict[str, Any]] = []
        bad = 0

        for ep in range(1, epochs + 1):
            t_ep = time.time()
            tr_loss = _epoch_train(model, dl_train, optim, device)
            va_loss, y_t, y_p, _ = _epoch_eval(model, dl_val, device)
            m = _confusion_and_metrics(y_t, y_p, len(CLASSES))

            history.append(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_accuracy": m.accuracy,
                    "val_f1_macro": m.f1_macro,
                    "val_precision_macro": m.precision_macro,
                    "val_recall_macro": m.recall_macro,
                    "seconds": time.time() - t_ep,
                }
            )

            if m.f1_macro > best_f1 + 1e-6:
                best_f1 = m.f1_macro
                best_epoch = ep
                bad = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "classes": CLASSES,
                        "img_size": img_size,
                        "state_dict": model.state_dict(),
                        "epoch": ep,
                        "val_metrics": {
                            "accuracy": m.accuracy,
                            "precision_macro": m.precision_macro,
                            "recall_macro": m.recall_macro,
                            "f1_macro": m.f1_macro,
                        },
                    },
                    best_path,
                )
            else:
                bad += 1

            if bad >= patience:
                break

        # Load best for test eval
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        te_loss, y_t, y_p, y_prob = _epoch_eval(model, dl_test, device)
        m_test = _confusion_and_metrics(y_t, y_p, len(CLASSES))

        # timing
        ms, fps = _timing_ms_per_image(model, dl_test, device)

        # previews: first N test samples (deterministic order)
        preview_dir = REPO_ROOT / "data" / "classification_preview" / model_name
        preview_dir.mkdir(parents=True, exist_ok=True)
        try:
            n_prev = min(preview_count, len(ds_test))
            for i in range(n_prev):
                img_path, y = ds_test.samples[i]
                probs = y_prob[i]
                pred = int(max(range(len(probs)), key=lambda k: probs[k]))
                conf = float(probs[pred])
                _draw_preview(
                    Path(img_path),
                    true_cls=CLASSES[int(y)],
                    pred_cls=CLASSES[pred],
                    conf=conf,
                    out_path=preview_dir / Path(img_path).name,
                )
        except Exception:
            pass

        # Confusion matrix as JSON + simple PNG heatmap (no matplotlib dependency)
        (out_dir / "confusion_matrix.json").write_text(json.dumps(m_test.confusion, indent=2) + "\n", encoding="utf-8")

        return {
            "model_name": model_name,
            "status": status,
            "error": err,
            "best_epoch": best_epoch,
            "best_weights_path": str(best_path),
            "model_size_mb": _weights_size_mb(best_path),
            "test_loss": te_loss,
            "metrics": {
                "accuracy": m_test.accuracy,
                "precision_macro": m_test.precision_macro,
                "recall_macro": m_test.recall_macro,
                "f1_macro": m_test.f1_macro,
                "per_class": m_test.per_class,
            },
            "confusion_matrix": m_test.confusion,
            "timing": {"inference_time_ms_per_image": ms, "fps": fps},
            "preview_dir": str(preview_dir),
            "history": history,
        }
    except Exception as e:
        status = "failed"
        err = f"{type(e).__name__}: {e}"
        return {
            "model_name": model_name,
            "status": status,
            "error": err,
            "best_epoch": -1,
            "best_weights_path": "",
            "model_size_mb": -1.0,
            "metrics": {},
            "confusion_matrix": [],
            "timing": {"inference_time_ms_per_image": -1.0, "fps": -1.0},
            "preview_dir": "",
            "history": [],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train 3 classifier backbones and write benchmark summary (dataset_v2).")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "classification_dataset_v2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64, help="Safe-ish default; lower if OOM on Orin.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-count", type=int, default=80)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "classification_benchmark_v2"))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]
    results: List[Dict[str, Any]] = []

    for name in models_to_run:
        r = train_one_model(
            model_name=name,
            data_root=data_root,
            out_dir=out_dir / name,
            device=device,
            img_size=int(args.img_size),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            epochs=int(args.epochs),
            patience=int(args.patience),
            seed=int(args.seed),
            preview_count=int(args.preview_count),
        )
        results.append(r)

    # Write benchmark summaries
    rep_dir = REPO_ROOT / "reports" / "classification_benchmark"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_json = rep_dir / "benchmark_summary.json"
    out_csv = rep_dir / "benchmark_summary.csv"
    out_md = rep_dir / "benchmark_summary.md"

    payload = {"ts": _now_utc(), "data_root": str(data_root), "results": results}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # CSV
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_name",
            "status",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "inference_time_ms_per_image",
            "fps",
            "model_size_mb",
            "best_weights_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            m = r.get("metrics") or {}
            w.writerow(
                {
                    "model_name": r.get("model_name", ""),
                    "status": r.get("status", ""),
                    "accuracy": (m.get("accuracy") if isinstance(m, dict) else ""),
                    "precision_macro": (m.get("precision_macro") if isinstance(m, dict) else ""),
                    "recall_macro": (m.get("recall_macro") if isinstance(m, dict) else ""),
                    "f1_macro": (m.get("f1_macro") if isinstance(m, dict) else ""),
                    "inference_time_ms_per_image": (r.get("timing") or {}).get("inference_time_ms_per_image", ""),
                    "fps": (r.get("timing") or {}).get("fps", ""),
                    "model_size_mb": r.get("model_size_mb", ""),
                    "best_weights_path": r.get("best_weights_path", ""),
                }
            )

    # Markdown
    def fnum(x: Any, nd: int = 4) -> str:
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "-"

    def fms(x: Any) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "-"

    md: List[str] = []
    md.append("## Classification benchmark summary (dataset_v2)")
    md.append("")
    md.append(f"- Generated (UTC): `{payload['ts']}`")
    md.append(f"- Dataset: `{data_root}`")
    md.append("")
    md.append("| model | status | acc | P_macro | R_macro | F1_macro | ms/img | FPS | size (MB) | best weights |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        m = r.get("metrics") or {}
        t = r.get("timing") or {}
        md.append(
            f"| {r.get('model_name','')} | {r.get('status','')} | {fnum(m.get('accuracy'))} | {fnum(m.get('precision_macro'))} | "
            f"{fnum(m.get('recall_macro'))} | {fnum(m.get('f1_macro'))} | {fms(t.get('inference_time_ms_per_image'))} | {fms(t.get('fps'))} | "
            f"{fms(r.get('model_size_mb'))} | `{r.get('best_weights_path','')}` |"
        )
    md.append("")
    md.append("Per-class metrics are stored in the JSON report.")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

