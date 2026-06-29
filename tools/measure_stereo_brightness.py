#!/usr/bin/env python3
"""Fetch /api/perception/stereo_camera JPEG and print brightness stats."""
from __future__ import annotations

import base64
import json
import sys
import urllib.request

import cv2
import numpy as np


def measure(url: str = "http://127.0.0.1:8080/api/perception/stereo_camera") -> dict:
    raw = urllib.request.urlopen(url, timeout=8).read()
    data = json.loads(raw)
    if not data.get("ok"):
        raise RuntimeError(data)
    jpg = base64.b64decode(data["jpeg_b64"])
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("jpeg decode failed")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "shape": list(frame.shape[:2]),
        "mean": float(gray.mean()),
        "mean_api": data.get("brightness_mean"),
        "excluded_pct": data.get("brightness_excluded_pct"),
        "p5": float(np.percentile(gray, 5)),
        "p95": float(np.percentile(gray, 95)),
        "sat_pct": float((gray > 250).mean() * 100.0),
        "dark_pct": float((gray < 20).mean() * 100.0),
    }


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "stereo"
    s = measure()
    print(
        f"[{label}] mean={s['mean']:.1f} p5={s['p5']:.1f} p95={s['p95']:.1f} "
        f"sat={s['sat_pct']:.1f}% dark={s['dark_pct']:.1f}% shape={s['shape']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
