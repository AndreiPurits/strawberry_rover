#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASSES_DEFAULT = ["green", "turning", "ripe", "rotten"]


def _make_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if model_name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(model_name)


def _build_eval_tf(img_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def infer_one(
    img_path: Path,
    *,
    weights_path: Path,
    device: str,
    topk: int,
) -> Dict[str, Any]:
    ckpt = torch.load(weights_path, map_location="cpu")
    model_name = str(ckpt.get("model_name") or "")
    classes: List[str] = list(ckpt.get("classes") or CLASSES_DEFAULT)
    img_size = int(ckpt.get("img_size") or 224)
    state = ckpt.get("state_dict")
    if not model_name or not isinstance(state, dict):
        raise RuntimeError("Invalid checkpoint: expected keys model_name/classes/img_size/state_dict")

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = _make_model(model_name, num_classes=len(classes))
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    tf = _build_eval_tf(img_size)
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(dev)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top = order[: max(1, min(topk, len(order)))]
    pred_i = top[0]

    return {
        "image": str(img_path),
        "model_name": model_name,
        "weights": str(weights_path),
        "predicted_class": classes[pred_i],
        "confidence": float(probs[pred_i]),
        "topk": [{"class": classes[i], "prob": float(probs[i])} for i in top],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ripeness classifier inference on a single crop image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True, help="Path to .pt saved by train_classifier_benchmark_v2.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    out = infer_one(Path(args.image), weights_path=Path(args.weights), device=str(args.device), topk=int(args.topk))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

