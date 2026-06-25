#!/usr/bin/env python3
"""Detect Mega / LiDAR / GNSS / camera device paths on Orin."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, Optional

AGENT_DIR = os.path.join(os.path.dirname(__file__), "..", "ops", "axm-monitor", "agent")
if os.path.isdir(AGENT_DIR):
    sys.path.insert(0, AGENT_DIR)

from mega_client import port_busy, port_exists, probe_mega  # noqa: E402


def _udev_prop(dev: str, key: str) -> str:
    dev_path = f"/sys/class/tty/{os.path.basename(dev)}/device/../../../"
    # fallback: udevadm
    try:
        import subprocess

        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", dev],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode()
        for line in out.splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
    except Exception:
        pass
    return ""


def _video_name(video_path: str) -> str:
    base = os.path.basename(video_path)
    name_file = f"/sys/class/video4linux/{base}/name"
    try:
        with open(name_file, encoding="utf-8", errors="ignore") as fh:
            return fh.read().strip().lower()
    except OSError:
        return ""


def detect_mega_port() -> Optional[str]:
    """Return Mega serial port (CH340). Prefer PONG; else sole CH340 or port held by driver."""
    busy_mega: Optional[str] = None
    ch340_ports: list[str] = []
    for dev in sorted(glob.glob("/dev/ttyUSB*")):
        vid = _udev_prop(dev, "ID_VENDOR_ID")
        if vid == "10c4":
            continue
        if vid == "1a86":
            ch340_ports.append(dev)
        if port_busy(dev):
            if vid == "1a86":
                busy_mega = dev
            continue
        result = probe_mega(dev)
        if result.get("pong"):
            return dev
    if busy_mega:
        return busy_mega
    if len(ch340_ports) == 1:
        return ch340_ports[0]
    return None


def detect_lidar_port(mega_port: Optional[str]) -> Optional[str]:
    for dev in sorted(glob.glob("/dev/ttyUSB*")):
        if mega_port and dev == mega_port:
            continue
        if _udev_prop(dev, "ID_VENDOR_ID") == "10c4":
            return dev
    for dev in sorted(glob.glob("/dev/ttyUSB*")):
        if mega_port and dev == mega_port:
            continue
        return dev
    return None


def detect_gnss_port() -> Optional[str]:
    for pattern in (
        "/dev/serial/by-id/usb-u-blox_*",
        "/dev/serial/by-id/*u-blox*",
    ):
        for dev in sorted(glob.glob(pattern)):
            return dev
    for dev in sorted(glob.glob("/dev/ttyACM*")):
        vid = _udev_prop(dev, "ID_VENDOR_ID")
        if vid == "1546":
            return dev
    for dev in sorted(glob.glob("/dev/ttyACM*")):
        return dev
    return None


def list_video_devices() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(os.path.basename(path).replace("video", ""))
        except ValueError:
            continue
        out[idx] = _video_name(path)
    return out


def detect_stereo_camera(videos: Dict[int, str]) -> Optional[int]:
    realsense = sorted(
        idx for idx, name in videos.items() if "realsense" in name or "intel" in name
    )
    if not realsense:
        return None
    if 4 in realsense:
        return 4
    return realsense[len(realsense) // 2]


def detect_front_camera(videos: Dict[int, str], stereo_idx: Optional[int]) -> Optional[int]:
    usb_cams = []
    for idx, name in sorted(videos.items()):
        if stereo_idx is not None and idx == stereo_idx:
            continue
        if "realsense" in name or "intel" in name:
            continue
        if "web camera" in name or "usb" in name or "uvc" in name:
            usb_cams.append(idx)
    if usb_cams:
        return usb_cams[0]
    candidates = []
    for idx, name in sorted(videos.items()):
        if stereo_idx is not None and idx == stereo_idx:
            continue
        if "realsense" in name or "intel" in name:
            continue
        if "metadata" in name:
            continue
        candidates.append(idx)
    if not candidates:
        return None
    return candidates[-1][0] if isinstance(candidates[-1], tuple) else candidates[-1]


def detect_all() -> Dict[str, str]:
    mega = detect_mega_port()
    lidar = detect_lidar_port(mega)
    gnss = detect_gnss_port()
    videos = list_video_devices()
    stereo = detect_stereo_camera(videos)
    front = detect_front_camera(videos, stereo)

    out: Dict[str, str] = {}
    if mega:
        out["MEGA_PORT"] = mega
    if lidar:
        out["LIDAR_PORT"] = lidar
    if gnss:
        out["RTK_PORT"] = gnss
        try:
            from gnss_reader import probe_rtk_baud  # noqa: WPS433

            baud = probe_rtk_baud(gnss)
            if baud:
                out["RTK_BAUD"] = str(baud)
        except Exception:
            pass
    if front is not None:
        out["CAMERA_DEVICE"] = str(front)
    if stereo is not None:
        out["STEREO_CAMERA_DEVICE"] = str(stereo)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect AXM rover USB devices")
    parser.add_argument("--export", action="store_true", help="Print shell assignments")
    parser.add_argument("--write-env", metavar="PATH", help="Write KEY=VAL lines to file")
    args = parser.parse_args()

    found = detect_all()
    if args.write_env:
        os.makedirs(os.path.dirname(args.write_env) or ".", exist_ok=True)
        with open(args.write_env, "w", encoding="utf-8") as fh:
            fh.write("# Auto-detected by axm_detect_devices.py\n")
            for key, val in sorted(found.items()):
                fh.write(f"{key}={val}\n")

    if args.export:
        for key, val in sorted(found.items()):
            print(f'{key}="{val}"')
    else:
        for key, val in sorted(found.items()):
            print(f"{key}={val}")

    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
