#!/usr/bin/env python3
from __future__ import annotations


import argparse
import sys
import time
from dataclasses import dataclass
import os
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple, Dict, Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class FramePair:
    rgb_bgr: np.ndarray
    depth_m: Optional[np.ndarray]  # aligned to rgb, HxW; float32 meters
    stamp_s: float
    frame_id: str
    stamp_ros: Optional[Tuple[int, int]]  # (sec, nanosec) if available


_WARNED_RGB_ENCODINGS = set()


def _ros_to_bgr8(msg) -> Optional[np.ndarray]:
    enc = str(msg.encoding).lower()
    h = int(msg.height)
    w = int(msg.width)
    if h <= 0 or w <= 0:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "bgr8":
        if buf.size != h * w * 3:
            return None
        return buf.reshape((h, w, 3))
    if enc == "rgb8":
        if buf.size != h * w * 3:
            return None
        rgb = buf.reshape((h, w, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        if buf.size != h * w * 4:
            return None
        bgra = buf.reshape((h, w, 4))
        return bgra[:, :, :3].copy()
    if enc == "rgba8":
        if buf.size != h * w * 4:
            return None
        rgba = buf.reshape((h, w, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc not in _WARNED_RGB_ENCODINGS:
        _WARNED_RGB_ENCODINGS.add(enc)
        print(f"[strawberry_ensemble] WARNING: unsupported RGB encoding: {msg.encoding}")
    return None


def _ros_to_depth_meters(msg) -> Optional[np.ndarray]:
    h = int(msg.height)
    w = int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = str(msg.encoding).lower()
    if enc in ("16uc1",):
        buf = np.frombuffer(msg.data, dtype=np.uint16)
        if buf.size != h * w:
            return None
        depth_mm = buf.reshape((h, w))
        return (depth_mm.astype(np.float32) * 0.001).astype(np.float32)
    if enc in ("32fc1",):
        buf = np.frombuffer(msg.data, dtype=np.float32)
        if buf.size != h * w:
            return None
        return buf.reshape((h, w)).astype(np.float32)
    return None


class Ros2RgbDepthProvider:
    def __init__(
        self,
        *,
        rgb_topic: str,
        depth_topic: str,
        sync_slop_s: float,
        sync_enabled: bool = True,
        queue_size: int = 20,
    ) -> None:
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic)
        self.sync_slop_s = float(sync_slop_s)
        self.sync_enabled = bool(sync_enabled)
        self.queue_size = int(queue_size)

        self._node = None
        self._rgb_msg = None
        self._depth_msg = None
        self._color_info_msg = None
        self._color_intrinsics = None  # (fx, fy, cx, cy)
        self._points_pub = None
        self._points_topic = None
        self._pc2 = None
        self._pc2_import_error_logged = False
        self._rgb_buf: Deque[Tuple[float, object]] = deque(maxlen=self.queue_size)
        self._depth_buf: Deque[Tuple[float, object]] = deque(maxlen=self.queue_size)
        self._rgb_count = 0
        self._depth_count = 0
        self._last_rgb_rate_log_s = 0.0
        self._last_depth_rate_log_s = 0.0
        # Two separate notions of rate:
        # - callback rate: wall-time based, measures what this process receives (reliable for bringup)
        # - stamp rate: based on msg.header.stamp (can be misleading with non-monotonic or non-system timestamps)
        self._rgb_cb_window: Deque[float] = deque(maxlen=240)
        self._depth_cb_window: Deque[float] = deque(maxlen=240)
        self._rgb_stamp_window: Deque[float] = deque(maxlen=240)
        self._depth_stamp_window: Deque[float] = deque(maxlen=240)
        self._logger = None
        self._rgb_encoding_logged = False
        self._depth_encoding_logged = False
        self._last_align_warn_s = 0.0
        self._spin_thread = None
        self._spin_stop = None
        self._executor = None

    @staticmethod
    def _estimate_fps(stamps_s: Deque[float]) -> float:
        if stamps_s is None or len(stamps_s) < 2:
            return 0.0
        dt = float(stamps_s[-1] - stamps_s[0])
        if dt <= 1e-6:
            return 0.0
        return float(len(stamps_s) - 1) / dt

    def open(self, *, camera_info_topic: str = "/camera/color/camera_info") -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import Image
        from sensor_msgs.msg import CameraInfo
        import threading

        rclpy.init(args=None)
        # PointCloud2 publishing intentionally disabled (benchmark mode).
        self._pc2 = None
        self._points_pub = None
        self._points_topic = None

        class _Node(Node):
            def __init__(self, outer: "Ros2RgbDepthProvider") -> None:
                super().__init__("strawberry_ensemble_camera")
                self.outer = outer
                outer._logger = self.get_logger()
                qos = QoSProfile(depth=5)
                qos.history = HistoryPolicy.KEEP_LAST
                qos.reliability = ReliabilityPolicy.RELIABLE
                qos.durability = DurabilityPolicy.VOLATILE
                self.create_subscription(Image, outer.rgb_topic, self.on_rgb, qos)
                self.create_subscription(Image, outer.depth_topic, self.on_depth, qos)
                self.create_subscription(CameraInfo, str(camera_info_topic), self.on_color_info, qos)

            def on_rgb(self, msg: Image) -> None:
                self.outer._rgb_msg = msg
                now = time.time()
                ts = self.outer._stamp_s(msg)
                self.outer._rgb_buf.append((ts, msg))
                self.outer._rgb_count += 1
                self.outer._rgb_cb_window.append(float(now))
                self.outer._rgb_stamp_window.append(float(ts))
                if self.outer._logger is not None and (now - float(self.outer._last_rgb_rate_log_s)) >= 2.0:
                    self.outer._last_rgb_rate_log_s = now
                    cb_fps = self.outer._estimate_fps(self.outer._rgb_cb_window)
                    stamp_fps = self.outer._estimate_fps(self.outer._rgb_stamp_window)
                    self.outer._logger.info(
                        f"RGB callback fps={cb_fps:.1f} stamp_fps={stamp_fps:.1f} topic={self.outer.rgb_topic}"
                    )
                if not self.outer._rgb_encoding_logged:
                    self.outer._rgb_encoding_logged = True
                    if self.outer._logger is not None:
                        self.outer._logger.info(f"RGB encoding detected: {str(msg.encoding)}")

            def on_depth(self, msg: Image) -> None:
                self.outer._depth_msg = msg
                now = time.time()
                ts = self.outer._stamp_s(msg)
                self.outer._depth_buf.append((ts, msg))
                self.outer._depth_count += 1
                self.outer._depth_cb_window.append(float(now))
                self.outer._depth_stamp_window.append(float(ts))
                if self.outer._logger is not None and (now - float(self.outer._last_depth_rate_log_s)) >= 2.0:
                    self.outer._last_depth_rate_log_s = now
                    cb_fps = self.outer._estimate_fps(self.outer._depth_cb_window)
                    stamp_fps = self.outer._estimate_fps(self.outer._depth_stamp_window)
                    self.outer._logger.info(
                        f"Depth callback fps={cb_fps:.1f} stamp_fps={stamp_fps:.1f} topic={self.outer.depth_topic}"
                    )
                if not self.outer._depth_encoding_logged:
                    self.outer._depth_encoding_logged = True
                    if self.outer._logger is not None:
                        self.outer._logger.info(f"Depth encoding detected: {str(msg.encoding)}")

            def on_color_info(self, msg: CameraInfo) -> None:
                self.outer._color_info_msg = msg
                try:
                    k = list(msg.k)
                    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
                    if fx > 0 and fy > 0:
                        self.outer._color_intrinsics = (fx, fy, cx, cy)
                except Exception:
                    pass

        self._node = _Node(self)
        if self._logger is not None:
            self._logger.info(f"Subscribing RGB: {self.rgb_topic}")
            self._logger.info(f"Subscribing Depth: {self.depth_topic}")
            self._logger.info(f"Subscribing CameraInfo: {str(camera_info_topic)}")
            self._logger.info("PointCloud2 publishing disabled (no RViz/cloud).")

        # Spin in a background thread so callbacks keep up at full camera FPS even if
        # the main loop is busy doing inference / rendering.
        self._spin_stop = threading.Event()
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)

        def _spin() -> None:
            try:
                while (not self._spin_stop.is_set()) and rclpy.ok():
                    self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                pass

        self._spin_thread = threading.Thread(target=_spin, name="ros2_spin", daemon=True)
        self._spin_thread.start()

    def _stamp_s(self, msg) -> float:
        try:
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception:
            return time.time()

    def get_callback_fps(self) -> Tuple[float, float]:
        """Return (rgb_callback_fps, depth_callback_fps) based on wall time in callbacks."""
        return (self._estimate_fps(self._rgb_cb_window), self._estimate_fps(self._depth_cb_window))

    def read(self, timeout_s: float = 1.0) -> FramePair:
        if self._node is None:
            raise RuntimeError("ROS2 provider not opened")

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not self._rgb_buf:
                continue

            # No-sync mode: drive by latest RGB, attach latest depth if any.
            if not self.sync_enabled:
                rgb_msg = self._rgb_msg
                if rgb_msg is None:
                    continue
                rgb = _ros_to_bgr8(rgb_msg)
                if rgb is None:
                    continue
                depth_msg = self._depth_msg
                depth_m = _ros_to_depth_meters(depth_msg) if depth_msg is not None else None
                if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
                    # For visualization / ML crops we tolerate mismatch by disabling depth-distance usage.
                    depth_m = None
                fid = ""
                stamp_ros = None
                try:
                    fid = str(rgb_msg.header.frame_id)
                    stamp_ros = (int(rgb_msg.header.stamp.sec), int(rgb_msg.header.stamp.nanosec))
                except Exception:
                    fid = ""
                    stamp_ros = None
                return FramePair(rgb_bgr=rgb, depth_m=depth_m, stamp_s=float(self._stamp_s(rgb_msg)), frame_id=fid, stamp_ros=stamp_ros)

            if not self._depth_buf:
                continue

            # Pick the closest RGB/Depth pair by minimal dt across buffers.
            best_pair = None  # (dt_s, rgb_ts, rgb_msg, depth_ts, depth_msg)
            for rgb_ts, rgb_msg in self._rgb_buf:
                depth_ts, depth_msg = min(self._depth_buf, key=lambda p: abs(p[0] - rgb_ts))
                dt = abs(rgb_ts - depth_ts)
                if best_pair is None or dt < best_pair[0]:
                    best_pair = (dt, rgb_ts, rgb_msg, depth_ts, depth_msg)

            if best_pair is None:
                continue

            dt_s, rgb_ts, rgb_msg, depth_ts, depth_msg = best_pair
            if dt_s > self.sync_slop_s:
                continue

            rgb = _ros_to_bgr8(rgb_msg)
            if rgb is None:
                continue
            depth_m = _ros_to_depth_meters(depth_msg)

            if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
                now = time.time()
                if self._logger is not None and (now - self._last_align_warn_s) >= 1.0:
                    self._last_align_warn_s = now
                    self._logger.warn("depth not aligned, skipping distance")
                depth_m = None

            fid = ""
            stamp_ros = None
            try:
                fid = str(rgb_msg.header.frame_id)
                stamp_ros = (int(rgb_msg.header.stamp.sec), int(rgb_msg.header.stamp.nanosec))
            except Exception:
                fid = ""
                stamp_ros = None
            return FramePair(rgb_bgr=rgb, depth_m=depth_m, stamp_s=float(rgb_ts), frame_id=fid, stamp_ros=stamp_ros)
        last_rgb_s = self._stamp_s(self._rgb_msg) if self._rgb_msg is not None else None
        last_depth_s = self._stamp_s(self._depth_msg) if self._depth_msg is not None else None
        last_dt_s = (
            abs(float(last_rgb_s) - float(last_depth_s))
            if (last_rgb_s is not None and last_depth_s is not None)
            else None
        )
        best_dt_s = None
        best_depth_msg = None
        if self._rgb_buf and self._depth_buf:
            best_pair = None
            for rgb_ts, _rgb_msg in self._rgb_buf:
                depth_ts, _depth_msg = min(self._depth_buf, key=lambda p: abs(p[0] - rgb_ts))
                dt = abs(rgb_ts - depth_ts)
                if best_pair is None or dt < best_pair[0]:
                    best_pair = (dt, rgb_ts, _rgb_msg, depth_ts, _depth_msg)
            if best_pair is not None:
                best_dt_s = float(best_pair[0])
                last_rgb_s = float(best_pair[1])
                last_depth_s = float(best_pair[3])
                best_depth_msg = best_pair[4]
        if self._logger is not None:
            self._logger.error(
                "Timed out waiting for synced rgb+depth. "
                f"last_rgb_s={last_rgb_s} last_depth_s={last_depth_s} last_dt_s={last_dt_s} "
                f"rgb_count={self._rgb_count} depth_count={self._depth_count} "
                f"best_dt_s={best_dt_s}"
            )
        # Temporary fallback: return latest RGB + nearest depth (by timestamp) instead of failing.
        if self._rgb_msg is not None and best_depth_msg is not None:
            rgb = _ros_to_bgr8(self._rgb_msg)
            if rgb is not None:
                depth_m = _ros_to_depth_meters(best_depth_msg)
                if depth_m is not None and depth_m.shape[:2] != rgb.shape[:2]:
                    if self._logger is not None:
                        self._logger.warn("depth not aligned, skipping distance")
                    depth_m = None
                return FramePair(rgb_bgr=rgb, depth_m=depth_m, stamp_s=float(last_rgb_s))
        raise RuntimeError(
            f"Timed out waiting for synced rgb+depth (topics: rgb={self.rgb_topic}, depth={self.depth_topic}, slop={self.sync_slop_s}s)"
        )

    def close(self) -> None:
        try:
            if self._spin_stop is not None:
                self._spin_stop.set()
        except Exception:
            pass
        try:
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=1.0)
        except Exception:
            pass
        self._spin_thread = None
        self._spin_stop = None
        try:
            if self._executor is not None and self._node is not None:
                self._executor.remove_node(self._node)
        except Exception:
            pass
        self._executor = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def _mask_to_pointcloud_xyz32(
    *,
    depth_m: np.ndarray,
    mask_u8: np.ndarray,
    intr: Tuple[float, float, float, float],
    max_points: int,
) -> np.ndarray:
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Filter valid depth first.
    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.05) & (z < 20.0)
    if not bool(np.any(valid)):
        return np.zeros((0, 3), dtype=np.float32)
    ys = ys[valid]
    xs = xs[valid]
    z = z[valid]

    # Downsample if needed.
    n = int(z.size)
    if max_points > 0 and n > max_points:
        stride = int(np.ceil(n / max_points))
        ys = ys[::stride]
        xs = xs[::stride]
        z = z[::stride]

    x = (xs.astype(np.float32) - float(cx)) / float(fx) * z
    y = (ys.astype(np.float32) - float(cy)) / float(fy) * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def _estimate_dims_m_from_points(pts_xyz: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if pts_xyz.size == 0 or pts_xyz.shape[0] < 50:
        return None
    xs = pts_xyz[:, 0]
    ys = pts_xyz[:, 1]
    zs = pts_xyz[:, 2]
    # Robust extents via percentiles to suppress outliers.
    x0, x1 = np.percentile(xs, [5, 95])
    y0, y1 = np.percentile(ys, [5, 95])
    z0, z1 = np.percentile(zs, [5, 95])
    dx = float(max(0.0, x1 - x0))
    dy = float(max(0.0, y1 - y0))
    dz = float(max(0.0, z1 - z0))
    if dx <= 1e-6 or dy <= 1e-6 or dz <= 1e-6:
        return None
    return dx, dy, dz


def _estimate_mass_g_from_dims(
    *,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    density_g_per_cm3: float,
) -> float:
    # Ellipsoid approximation: V = 4/3*pi*a*b*c
    a = 0.5 * float(dx_m)
    b = 0.5 * float(dy_m)
    c = 0.5 * float(dz_m)
    vol_m3 = (4.0 / 3.0) * float(np.pi) * a * b * c
    vol_cm3 = vol_m3 * 1e6
    return float(density_g_per_cm3) * float(vol_cm3)


def _clean_mask_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Minimal cleanup: small open/close + keep largest connected component."""
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    out = (labels == best).astype(np.uint8) * 255
    # Slight dilation for nicer-looking cloud silhouette (video/demo).
    out = cv2.dilate(out, k, iterations=1)
    return out


def _median_depth_m_in_mask(depth_m: np.ndarray, mask_u8: np.ndarray) -> Optional[float]:
    if depth_m is None or mask_u8 is None:
        return None
    sel = mask_u8 > 0
    if not bool(np.any(sel)):
        return None
    z = depth_m[sel].astype(np.float32)
    z = z[np.isfinite(z)]
    z = z[(z > 0.05) & (z < 20.0)]
    if z.size == 0:
        return None
    return float(np.median(z))


def _estimate_mass_g_from_area_and_distance(
    *,
    area_px: int,
    distance_m: float,
    k_area_over_d2: float,
) -> Optional[float]:
    if area_px <= 0 or not np.isfinite(distance_m) or distance_m <= 1e-3:
        return None
    return float(k_area_over_d2) * float(area_px) / float(distance_m * distance_m)


def _cloud_visual_scale_about_centroid(pts_xyz: np.ndarray, scale: float) -> np.ndarray:
    s = float(scale)
    if not np.isfinite(s) or abs(s - 1.0) < 1e-6:
        return pts_xyz
    if pts_xyz is None or pts_xyz.size == 0:
        return pts_xyz
    c = np.mean(pts_xyz, axis=0).astype(np.float32)
    return (c + (pts_xyz - c) * np.float32(s)).astype(np.float32)

def _publish_points_xyz32(
    *,
    node: Any,
    pub: Any,
    topic: str,
    frame_id: str,
    stamp_ros: Optional[Tuple[int, int]],
    pts_xyz: np.ndarray,
    pc2: Any,
) -> None:
    # PointCloud2 publishing removed (benchmark mode).
    return


def _publish_points_xyzrgb32_constant(
    *,
    node: Any,
    pub: Any,
    frame_id: str,
    stamp_ros: Optional[Tuple[int, int]],
    pts_xyz: np.ndarray,
    pc2: Any,
    rgb_255: Tuple[int, int, int] = (255, 60, 60),
) -> None:
    # PointCloud2 publishing removed (benchmark mode).
    return




def _draw_label(img: np.ndarray, text: str, origin_xy: tuple) -> None:
    x, y = int(origin_xy[0]), int(origin_xy[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _depth_to_view(depth_m: Optional[np.ndarray], *, target_h: int, target_w: int) -> np.ndarray:
    if depth_m is None:
        view = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        view[:, :] = (20, 20, 20)
        _draw_label(view, "DEPTH (missing)", (10, 28))
        return view

    d = depth_m.astype(np.float32)
    valid = np.isfinite(d) & (d > 0.05) & (d < 20.0)
    if not bool(np.any(valid)):
        view = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        view[:, :] = (20, 20, 20)
        _draw_label(view, "DEPTH (invalid)", (10, 28))
        if view.shape[0] != target_h or view.shape[1] != target_w:
            view = cv2.resize(view, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return view

    # Robust normalization for visualization.
    v = d[valid]
    vmin = float(np.percentile(v, 5))
    vmax = float(np.percentile(v, 95))
    vmax = max(vmin + 1e-3, vmax)
    dn = (d - vmin) / (vmax - vmin)
    dn = np.clip(dn, 0.0, 1.0)
    dn[~valid] = 0.0
    gray = (dn * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if color.shape[0] != target_h or color.shape[1] != target_w:
        color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return color


def _stack_rgb_and_depth(rgb_bgr: np.ndarray, depth_view_bgr: np.ndarray) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    if depth_view_bgr.shape[:2] != (h, w):
        depth_view_bgr = cv2.resize(depth_view_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.concatenate([rgb_bgr, depth_view_bgr], axis=1)
    _draw_label(canvas, "RGB", (10, 28))
    _draw_label(canvas, "DEPTH", (rgb_bgr.shape[1] + 10, 28))
    return canvas


def _get_screen_size_px() -> Tuple[int, int]:
    # Best-effort: works on X11/Wayland when a GUI session exists.
    global _SCREEN_SIZE_CACHE  # type: ignore[global-variable-not-assigned]
    try:
        cached = _SCREEN_SIZE_CACHE  # type: ignore[name-defined]
        if cached is not None:
            return cached
    except Exception:
        pass
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            _SCREEN_SIZE_CACHE = (w, h)  # type: ignore[name-defined]
            return w, h
    except Exception:
        pass
    _SCREEN_SIZE_CACHE = (1920, 1080)  # type: ignore[name-defined]
    return 1920, 1080


def _fit_to_screen(img_bgr: np.ndarray, *, margin_px: int = 80) -> np.ndarray:
    sw, sh = _get_screen_size_px()
    max_w = max(1, int(sw) - int(margin_px))
    max_h = max(1, int(sh) - int(margin_px))
    h, w = img_bgr.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return img_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _stretch_to_screen(img_bgr: np.ndarray) -> np.ndarray:
    sw, sh = _get_screen_size_px()
    h, w = img_bgr.shape[:2]
    if (w, h) == (sw, sh):
        return img_bgr
    return cv2.resize(img_bgr, (int(sw), int(sh)), interpolation=cv2.INTER_AREA)


def _scale_and_center_crop(img_bgr: np.ndarray, *, out_w: int, out_h: int, scale: float) -> np.ndarray:
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    s = max(1e-3, float(scale))
    scaled_w = max(1, int(round(w * s)))
    scaled_h = max(1, int(round(h * s)))
    resized = cv2.resize(img_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # Center-crop (or pad via resize if somehow smaller).
    rh, rw = resized.shape[:2]
    if rw < out_w or rh < out_h:
        return cv2.resize(resized, (out_w, out_h), interpolation=cv2.INTER_AREA)

    x0 = int((rw - out_w) // 2)
    y0 = int((rh - out_h) // 2)
    return resized[y0 : y0 + out_h, x0 : x0 + out_w]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run strawberry detector+classifier+segmenter ensemble on live RGB+depth.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--sync-slop", type=float, default=0.10)
    ap.add_argument("--no-sync", action="store_true", help="Do not sync RGB/Depth by timestamp (lower latency).")
    ap.add_argument("--headless", action="store_true", help="Disable all GUI (imshow/waitKey); safe over SSH.")
    ap.add_argument("--view-only", action="store_true", help="Viewer only: show RGB+Depth without running ML models.")
    ap.add_argument(
        "--detector-only",
        action="store_true",
        help="Run only detector (disable classifier + segmenter).",
    )
    ap.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    ap.add_argument("--mass-k", type=float, default=0.0006, help="Mass model: g = k * area_px / distance_m^2")
    ap.add_argument("--mass-ema-alpha", type=float, default=0.25, help="EMA smoothing alpha for mass (0-1).")
    ap.add_argument("--fps-cap", type=float, default=20.0, help="Cap processing loop FPS (0 = uncapped).")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run forever). Useful for SSH benchmarks.",
    )
    ap.add_argument(
        "--log-every-s",
        type=float,
        default=1.0,
        help="Print timings at most once per this many seconds.",
    )

    ap.add_argument("--detector", default="")
    ap.add_argument("--classifier", default="")
    ap.add_argument("--segmenter", default="")
    ap.add_argument("--det-conf", type=float, default=0.35)
    ap.add_argument("--imgsz-det", type=int, default=640)
    ap.add_argument("--imgsz-seg", type=int, default=384)
    ap.add_argument("--seg-every", type=int, default=2, help="Run segmentation once per N frames (1 = every frame).")
    ap.add_argument("--seg-max-rois", type=int, default=8, help="Run segmentation only for top-K detections.")
    ap.add_argument(
        "--seg-min-det-conf",
        type=float,
        default=0.0,
        help="Skip segmentation unless detector_conf >= threshold.",
    )
    args = ap.parse_args()

    if bool(args.detector_only):
        # Keep input/output steady for camera bringup tests.
        args.classifier = ""
        args.segmenter = ""
        # Make detector-only runs stable by default.
        if float(args.fps_cap) <= 0:
            args.fps_cap = 30.0

    # CUDA diagnostics (must be visible in logs to prove GPU usage).
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        dev_req = str(args.device)
        msg = f"torch.cuda.is_available={cuda_ok} requested_device={dev_req}"
        if cuda_ok:
            try:
                msg += f" cuda_device_count={int(torch.cuda.device_count())}"
                msg += f" current_cuda_device={int(torch.cuda.current_device())}"
                msg += f" cuda_device_name={str(torch.cuda.get_device_name(torch.cuda.current_device()))}"
            except Exception:
                pass
        print(msg, flush=True)
        if "cuda" in dev_req and not cuda_ok:
            print("WARNING: CUDA requested but torch.cuda.is_available=False. Models may fall back to CPU.", flush=True)
    except Exception:
        pass

    if (
        not args.headless
        and os.name == "posix"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SystemExit(
            "No GUI display detected (DISPLAY/WAYLAND_DISPLAY not set). "
            "Run from Ubuntu Desktop session or enable X11 forwarding."
        )

    pipeline = None
    if not args.view_only:
        from pipelines.strawberry_ensemble import EnsembleConfig, StrawberryEnsemblePipeline, default_production_config

        cfg0 = default_production_config(REPO_ROOT)
        cfg = EnsembleConfig(
            detector_weights=args.detector or cfg0.detector_weights,
            classifier_weights=(args.classifier or cfg0.classifier_weights) if (not args.detector_only) else "",
            segmenter_weights=(args.segmenter or cfg0.segmenter_weights) if (not args.detector_only) else "",
            device=str(args.device),
            detector_conf=float(args.det_conf),
            detector_imgsz=int(args.imgsz_det),
            segmenter_imgsz=int(args.imgsz_seg),
            segment_every_n=int(args.seg_every),
            segment_max_rois=int(args.seg_max_rois),
            segment_min_det_conf=float(args.seg_min_det_conf),
            depth_sync_slop_s=float(args.sync_slop),
        )
        print(
            "[strawberry_ensemble] config "
            f"detector_weights={cfg.detector_weights} "
            f"classifier_weights={cfg.classifier_weights or '(disabled)'} "
            f"segmenter_weights={cfg.segmenter_weights or '(disabled)'} "
            f"detector_only={int(bool(args.detector_only))} "
            f"seg_every={int(cfg.segment_every_n)} "
            f"cls_every=8",
            flush=True,
        )
        pipeline = StrawberryEnsemblePipeline(cfg)

    provider = Ros2RgbDepthProvider(
        rgb_topic=str(args.rgb_topic),
        depth_topic=str(args.depth_topic),
        sync_slop_s=float(args.sync_slop),
        sync_enabled=(not args.no_sync),
    )

    provider.open(camera_info_topic=str(args.camera_info_topic))
    win = "strawberry_ensemble"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Place GUI in the top half of the screen (full width).
        try:
            sw, sh = _get_screen_size_px()
            cv2.resizeWindow(win, int(sw), int(sh // 2))
            cv2.moveWindow(win, 0, 0)
        except Exception:
            pass

    try:
        last_log_s = 0.0
        frames = 0
        last_frame_t = None
        viewer_fps = 0.0
        last_mass_g = None
        last_proc_log_s = 0.0
        last_loop_t = None
        while True:
            loop_t0 = time.time()
            pair = provider.read()
            frames += 1

            if args.view_only:
                now_t = time.time()
                if last_frame_t is not None:
                    dt = max(1e-6, now_t - last_frame_t)
                    viewer_fps = 1.0 / dt
                last_frame_t = now_t
                rgb_vis = pair.rgb_bgr
            else:
                assert pipeline is not None
                inst = pipeline.infer(pair.rgb_bgr, pair.depth_m)
                # Compute mass per instance (top-1 by default for speed, but keep list intact).
                inst_out = list(inst)
                if pair.depth_m is not None and provider._color_intrinsics is not None:
                    new_list = []
                    for r in inst_out:
                        mass_g = None
                        if r.mask_fullres is not None:
                            mclean = _clean_mask_u8(r.mask_fullres)
                            d = _median_depth_m_in_mask(pair.depth_m, mclean)
                            area_px = int(np.count_nonzero(mclean > 0))
                            if d is not None:
                                mass_g = _estimate_mass_g_from_area_and_distance(
                                    area_px=area_px,
                                    distance_m=float(d),
                                    k_area_over_d2=float(args.mass_k),
                                )
                                if mass_g is not None:
                                    if last_mass_g is None:
                                        last_mass_g = float(mass_g)
                                    else:
                                        a = float(args.mass_ema_alpha)
                                        last_mass_g = a * float(mass_g) + (1.0 - a) * float(last_mass_g)
                                    mass_g = float(last_mass_g)
                        new_list.append(
                            type(r)(
                                bbox_xyxy=r.bbox_xyxy,
                                detector_conf=r.detector_conf,
                                ripeness_class=r.ripeness_class,
                                classifier_conf=r.classifier_conf,
                                mask_fullres=r.mask_fullres,
                                contour_fullres=r.contour_fullres,
                                distance_m=r.distance_m,
                                center_xy=r.center_xy,
                                mass_g=mass_g,
                            )
                        )
                    inst_out = new_list

                rgb_vis = pipeline.render_overlay(pair.rgb_bgr, inst_out) if not args.headless else pair.rgb_bgr
                try:
                    rgb_cb_fps, depth_cb_fps = provider.get_callback_fps()
                    pipeline.set_input_fps(rgb_fps=rgb_cb_fps, depth_fps=depth_cb_fps)
                except Exception:
                    pass
            # Processing FPS + FPS cap (benchmark-friendly).
            now = time.time()
            if last_loop_t is not None:
                proc_fps = 1.0 / max(1e-6, now - float(last_loop_t))
            else:
                proc_fps = 0.0
            last_loop_t = now
            if provider._logger is not None and (now - float(last_proc_log_s)) >= 2.0:
                last_proc_log_s = now
                provider._logger.info(f"Processing fps={proc_fps:.1f} cap={float(args.fps_cap):.1f} view_only={int(bool(args.view_only))}")

            cap = float(args.fps_cap)
            if cap > 0:
                target_dt = 1.0 / cap
                elapsed = time.time() - loop_t0
                sleep_s = target_dt - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            if not args.headless:
                depth_vis = _depth_to_view(pair.depth_m, target_h=rgb_vis.shape[0], target_w=rgb_vis.shape[1])
                canvas = _stack_rgb_and_depth(rgb_vis, depth_vis)
                sw, sh = _get_screen_size_px()
                canvas = _scale_and_center_crop(canvas, out_w=int(sw), out_h=int(sh // 2), scale=1.2)
                cv2.imshow(win, canvas)
            now = time.time()
            if now - last_log_s >= float(args.log_every_s):
                last_log_s = now
                if args.view_only:
                    print(f"fps={viewer_fps:.1f} view_only=1")
                    continue
                assert pipeline is not None
                t = pipeline.timings
                if args.headless:
                    s = pipeline.stats
                    print(
                        f"fps={t.fps:.1f} det={t.detector_ms:.1f}ms cls={t.classifier_ms:.1f}ms "
                        f"seg={t.segmentation_ms:.1f}ms depth={t.depth_ms:.1f}ms total={t.total_ms:.1f}ms | "
                        f"det_frames={s.det_frames_total} det_pos={s.det_positive_frames} "
                        f"cls_calls={s.cls_calls} seg_calls={s.seg_calls}"
                    )
                else:
                    print(
                        f"fps={t.fps:.1f} det={t.detector_ms:.1f}ms cls={t.classifier_ms:.1f}ms "
                        f"seg={t.segmentation_ms:.1f}ms depth={t.depth_ms:.1f}ms total={t.total_ms:.1f}ms"
                    )
            if int(args.max_frames) > 0 and frames >= int(args.max_frames):
                break
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        provider.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if args.headless and (not args.view_only) and pipeline is not None:
            s = pipeline.stats
            print(
                "final_stats "
                f"det_frames={s.det_frames_total} det_pos={s.det_positive_frames} "
                f"cls_calls={s.cls_calls} seg_calls={s.seg_calls}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

