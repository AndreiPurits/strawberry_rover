#!/usr/bin/env python3
"""One-shot: detect claw exclude regions and save mask for auto-brightness."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, "src/rover_perception")
from rover_perception.stereo_brightness_mask import (  # noqa: E402
    default_mask_path,
    detect_claw_exclude_regions,
    save_brightness_mask,
)


def _frame_from_api(url: str) -> np.ndarray:
    raw = urllib.request.urlopen(url, timeout=8).read()
    data = json.loads(raw)
    if not data.get("ok"):
        raise RuntimeError(data)
    jpg = base64.b64decode(data["jpeg_b64"])
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("jpeg decode failed")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/api/perception/stereo_camera",
    )
    parser.add_argument("--out", default=default_mask_path())
    parser.add_argument("--dark-threshold", type=int, default=55)
    parser.add_argument("--bottom-frac", type=float, default=0.58)
    parser.add_argument("--min-area", type=float, default=1200.0)
    args = parser.parse_args()

    frame = _frame_from_api(args.url)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = detect_claw_exclude_regions(
        gray,
        dark_threshold=args.dark_threshold,
        bottom_frac=args.bottom_frac,
        min_area=args.min_area,
    )
    if not regions:
        print("No claw regions detected", file=sys.stderr)
        return 1
    save_brightness_mask(
        args.out,
        regions,
        width=int(frame.shape[1]),
        height=int(frame.shape[0]),
    )
    print(f"Saved {len(regions)} exclude regions -> {args.out}")
    for i, r in enumerate(regions):
        print(f"  [{i}] {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
