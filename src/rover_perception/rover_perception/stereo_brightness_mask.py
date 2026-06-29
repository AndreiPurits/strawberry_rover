"""Fixed mask for stereo auto-brightness (exclude gripper claws once)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

RectNorm = Tuple[float, float, float, float]  # x0, y0, x1, y1 in 0..1


def _norm_rects_to_pixels(
    rects: Sequence[RectNorm], width: int, height: int
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rects:
        xa = int(max(0.0, min(1.0, x0)) * width)
        ya = int(max(0.0, min(1.0, y0)) * height)
        xb = int(max(0.0, min(1.0, x1)) * width)
        yb = int(max(0.0, min(1.0, y1)) * height)
        if xb > xa and yb > ya:
            out.append((xa, ya, xb, yb))
    return out


def detect_claw_exclude_regions(
    gray: np.ndarray,
    *,
    dark_threshold: int = 55,
    bottom_frac: float = 0.58,
    min_area: float = 1200.0,
    morph_kernel: int = 7,
) -> List[RectNorm]:
    """One-shot: find claw blobs and return normalized exclude rectangles."""
    height, width = gray.shape[:2]
    y0 = int(height * max(0.0, min(1.0, bottom_frac)))
    if y0 >= height:
        return []

    roi = gray[y0:, :]
    dark = (roi < int(dark_threshold)).astype(np.uint8) * 255
    k = max(3, int(morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_h = height - y0
    regions: List[RectNorm] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if y + bh < roi_h * 0.25 or bh < roi_h * 0.12:
            continue
        ya = y0 + y
        yb = ya + bh
        xb = x + bw
        regions.append(
            (
                round(x / width, 4),
                round(ya / height, 4),
                round(xb / width, 4),
                round(yb / height, 4),
            )
        )
    return regions


def include_mask_from_regions(
    gray: np.ndarray, regions: Sequence[RectNorm]
) -> np.ndarray:
    """True = pixel participates in brightness mean."""
    height, width = gray.shape[:2]
    include = np.ones((height, width), dtype=bool)
    for xa, ya, xb, yb in _norm_rects_to_pixels(regions, width, height):
        include[ya:yb, xa:xb] = False
    return include


def masked_gray_mean(
    gray: np.ndarray,
    include_mask: np.ndarray,
    *,
    min_pixels: int = 500,
) -> float:
    sel = include_mask
    if int(sel.sum()) < min_pixels:
        return float(gray.mean())
    return float(gray[sel].mean())


def default_mask_path() -> str:
    return os.path.expanduser("~/.config/axm/stereo_brightness_exclude.json")


def save_brightness_mask(
    path: str,
    regions: Sequence[RectNorm],
    *,
    width: int,
    height: int,
    source: str = "claw_detect",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "version": 1,
        "source": source,
        "frame_width": int(width),
        "frame_height": int(height),
        "exclude_regions": [list(r) for r in regions],
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def load_brightness_mask(path: str) -> List[RectNorm]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("exclude_regions") or []
    regions: List[RectNorm] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        regions.append(
            (
                float(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
            )
        )
    return regions
