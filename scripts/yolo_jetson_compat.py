"""Patch torchvision.ops.nms when Jetson torch and pip torchvision are ABI-incompatible."""

from __future__ import annotations

from typing import List

import torch


def apply_torchvision_nms_patch() -> None:
    try:
        import torchvision.ops as tv_ops
    except ImportError:
        return

    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            mask = iou <= iou_threshold
            if not bool(mask.any()):
                break
            order = rest[mask]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    tv_ops.nms = nms  # type: ignore[assignment]
