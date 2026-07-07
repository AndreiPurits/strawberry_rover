"""ROS2 RGB + depth frame provider for strawberry ML pipelines."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

_WARNED_RGB_ENCODINGS: set[str] = set()


@dataclass
class FramePair:
    rgb_bgr: np.ndarray
    depth_m: Optional[np.ndarray]  # aligned to rgb, HxW float32 meters
    stamp_s: float
    frame_id: str
    stamp_ros: Optional[Tuple[int, int]]  # (sec, nanosec) if available


def ros_to_bgr8(msg) -> Optional[np.ndarray]:
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
        print(f"[ros_rgb_depth] WARNING: unsupported RGB encoding: {msg.encoding}")
    return None


def ros_to_depth_meters(msg) -> Optional[np.ndarray]:
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


def align_depth_to_rgb(depth_m: Optional[np.ndarray], rgb_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Resize depth to RGB resolution (RealSense color/depth size mismatch)."""
    if depth_m is None:
        return None
    rh, rw = rgb_shape[:2]
    dh, dw = depth_m.shape[:2]
    if (dh, dw) == (rh, rw):
        return depth_m
    if dh <= 0 or dw <= 0 or rh <= 0 or rw <= 0:
        return None
    return cv2.resize(depth_m, (rw, rh), interpolation=cv2.INTER_NEAREST)


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
        self._color_intrinsics = None
        self._rgb_buf: Deque[Tuple[float, object]] = deque(maxlen=self.queue_size)
        self._depth_buf: Deque[Tuple[float, object]] = deque(maxlen=self.queue_size)
        self._rgb_count = 0
        self._depth_count = 0
        self._last_rgb_rate_log_s = 0.0
        self._last_depth_rate_log_s = 0.0
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

    def open(self, *, camera_info_topic: str = "/stereo_camera/color/camera_info") -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        rclpy.init(args=None)

        class _Node(Node):
            def __init__(self, outer: "Ros2RgbDepthProvider") -> None:
                super().__init__("ros_rgb_depth_provider")
                self.outer = outer
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
                    self.outer._logger.info(f"RGB fps={cb_fps:.1f} topic={self.outer.rgb_topic}")

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
                    self.outer._logger.info(f"Depth fps={cb_fps:.1f} topic={self.outer.depth_topic}")

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
        outer_logger = self._node.get_logger()
        self._logger = outer_logger
        outer_logger.info(f"RGB: {self.rgb_topic}")
        outer_logger.info(f"Depth: {self.depth_topic}")
        outer_logger.info(f"CameraInfo: {camera_info_topic}")

        self._spin_stop = threading.Event()
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)

        def _spin() -> None:
            try:
                while (not self._spin_stop.is_set()) and rclpy.ok():
                    self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                pass

        self._spin_thread = threading.Thread(target=_spin, name="ros_rgb_depth_spin", daemon=True)
        self._spin_thread.start()

    def _stamp_s(self, msg) -> float:
        try:
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception:
            return time.time()

    def get_callback_fps(self) -> Tuple[float, float]:
        return (self._estimate_fps(self._rgb_cb_window), self._estimate_fps(self._depth_cb_window))

    def _pair_from_msgs(self, rgb_msg, depth_msg) -> Optional[FramePair]:
        rgb = ros_to_bgr8(rgb_msg)
        if rgb is None:
            return None
        depth_m = ros_to_depth_meters(depth_msg) if depth_msg is not None else None
        depth_m = align_depth_to_rgb(depth_m, rgb.shape)
        fid = ""
        stamp_ros = None
        try:
            fid = str(rgb_msg.header.frame_id)
            stamp_ros = (int(rgb_msg.header.stamp.sec), int(rgb_msg.header.stamp.nanosec))
        except Exception:
            pass
        return FramePair(
            rgb_bgr=rgb,
            depth_m=depth_m,
            stamp_s=float(self._stamp_s(rgb_msg)),
            frame_id=fid,
            stamp_ros=stamp_ros,
        )

    def get_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        return self._color_intrinsics

    def read(self, timeout_s: float = 1.0) -> FramePair:
        if self._node is None:
            raise RuntimeError("ROS2 provider not opened")

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            time.sleep(0.01)
            if not self._rgb_buf:
                continue

            if not self.sync_enabled:
                if self._rgb_msg is None:
                    continue
                pair = self._pair_from_msgs(self._rgb_msg, self._depth_msg)
                if pair is not None:
                    return pair
                continue

            if not self._depth_buf:
                continue

            best_pair = None
            for rgb_ts, rgb_msg in self._rgb_buf:
                depth_ts, depth_msg = min(self._depth_buf, key=lambda p: abs(p[0] - rgb_ts))
                dt = abs(rgb_ts - depth_ts)
                if best_pair is None or dt < best_pair[0]:
                    best_pair = (dt, rgb_msg, depth_msg)

            if best_pair is None:
                continue
            dt_s, rgb_msg, depth_msg = best_pair
            if dt_s > self.sync_slop_s:
                continue
            pair = self._pair_from_msgs(rgb_msg, depth_msg)
            if pair is not None:
                return pair

        if self._rgb_msg is not None and self._depth_msg is not None:
            pair = self._pair_from_msgs(self._rgb_msg, self._depth_msg)
            if pair is not None:
                return pair
        raise RuntimeError(
            f"Timed out waiting for rgb+depth (rgb={self.rgb_topic}, depth={self.depth_topic})"
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
