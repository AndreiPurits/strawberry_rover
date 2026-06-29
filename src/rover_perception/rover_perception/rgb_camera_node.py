import json
import threading
import time
from typing import Any, Dict, Optional

import cv2
import rclpy
from rclpy.node import Node
from rover_perception.stereo_auto_brightness import StereoAutoBrightness
from rover_perception.v4l2_controls import apply_v4l2_controls
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RgbCameraNode(Node):
    """Publishes real RGB camera frames and basic camera metadata."""

    def __init__(self) -> None:
        super().__init__("rgb_camera_node")

        self.declare_parameter("device_index", 0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("use_v4l2_backend", True)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("debug_image_topic", "/camera/debug/image_annotated")
        self.declare_parameter("publish_camera_info", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("camera_name", "rover_rgb_camera")
        self.declare_parameter("auto_reconnect", True)
        self.declare_parameter("flip_horizontal", False)
        self.declare_parameter("prefer_max_fov", False)
        self.declare_parameter("auto_exposure", True)
        self.declare_parameter("exposure", -1)
        self.declare_parameter("gain", -1)
        self.declare_parameter("brightness", -1)
        self.declare_parameter("gamma", -1)
        self.declare_parameter("auto_brightness_enable", False)
        self.declare_parameter("auto_brightness_min", 110.0)
        self.declare_parameter("auto_brightness_max", 180.0)
        self.declare_parameter("auto_brightness_trial_interval_sec", 2.0)
        self.declare_parameter("tuning_topic", "")

        self._device_index = int(self.get_parameter("device_index").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._publish_rate = float(self.get_parameter("fps").value)
        self._image_width = int(self.get_parameter("width").value)
        self._image_height = int(self.get_parameter("height").value)
        self._use_v4l2_backend = bool(self.get_parameter("use_v4l2_backend").value)
        self._publish_camera_info = bool(self.get_parameter("publish_camera_info").value)
        self._publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self._camera_name = str(self.get_parameter("camera_name").value)
        self._auto_reconnect = bool(self.get_parameter("auto_reconnect").value)
        self._flip_horizontal = bool(self.get_parameter("flip_horizontal").value)
        self._prefer_max_fov = bool(self.get_parameter("prefer_max_fov").value)
        self._auto_exposure = bool(self.get_parameter("auto_exposure").value)
        self._exposure = int(self.get_parameter("exposure").value)
        self._gain = int(self.get_parameter("gain").value)
        self._brightness = int(self.get_parameter("brightness").value)
        self._gamma = int(self.get_parameter("gamma").value)
        self._auto_brightness_enable = bool(self.get_parameter("auto_brightness_enable").value)
        self._auto_brightness_min = float(self.get_parameter("auto_brightness_min").value)
        self._auto_brightness_max = float(self.get_parameter("auto_brightness_max").value)
        self._auto_brightness_trial_interval_sec = float(
            self.get_parameter("auto_brightness_trial_interval_sec").value
        )
        self._tuning_topic = str(self.get_parameter("tuning_topic").value).strip()

        image_topic = str(self.get_parameter("image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)

        if self._publish_rate <= 0.0:
            self.get_logger().warn("publish_rate must be > 0.0, defaulting to 15 Hz.")
            self._publish_rate = 15.0

        self._image_pub = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self._camera_info_pub = self.create_publisher(
            CameraInfo, camera_info_topic, qos_profile_sensor_data
        )
        self._debug_pub = self.create_publisher(
            Image, debug_image_topic, qos_profile_sensor_data
        )

        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_lock = threading.Lock()
        self._trial_thread: Optional[threading.Thread] = None
        self._last_reconnect_try = 0.0
        self._auto_tuner: Optional[StereoAutoBrightness] = None
        self._tuning_pub = None
        if self._auto_brightness_enable:
            init_brightness = self._brightness if self._brightness != -1 else 5
            init_gamma = self._gamma if self._gamma != -1 else 300
            self._auto_tuner = StereoAutoBrightness(
                device_index=self._device_index,
                target_min=self._auto_brightness_min,
                target_max=self._auto_brightness_max,
                brightness=init_brightness,
                gamma=init_gamma,
                trial_interval_sec=self._auto_brightness_trial_interval_sec,
                logger=self.get_logger(),
            )
            if self._tuning_topic:
                self._tuning_pub = self.create_publisher(String, self._tuning_topic, 10)
        self._open_camera()

        self._timer = self.create_timer(1.0 / self._publish_rate, self._on_timer)
        self.get_logger().info(
            "rgb_camera_node started. Publishing "
            f"{image_topic}, {camera_info_topic}, {debug_image_topic} at {self._publish_rate:.2f} Hz"
        )

    def _open_camera(self) -> None:
        """Open VideoCapture with configured size."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        backend_name = "CAP_V4L2" if self._use_v4l2_backend else "DEFAULT"
        if self._use_v4l2_backend:
            capture = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(self._device_index)

        if not capture.isOpened():
            capture.release()
            self.get_logger().error(
                f"Failed to open camera device {self._device_index} using backend {backend_name}. "
                "Will keep retrying if auto_reconnect is enabled."
            )
            return

        target_w, target_h = self._image_width, self._image_height
        if self._prefer_max_fov:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            best_area: Optional[int] = None
            best_mode: Optional[tuple[int, int]] = None
            for w, h in (
                (160, 120),
                (176, 144),
                (320, 240),
                (352, 288),
                (640, 480),
                (800, 600),
            ):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                fh, fw = frame.shape[:2]
                if fw <= 0 or fh <= 0:
                    continue
                area = fw * fh
                if best_area is None or area < best_area:
                    best_area = area
                    best_mode = (w, h)
                    target_w, target_h = fw, fh
            if best_mode is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(best_mode[0]))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(best_mode[1]))
        else:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._image_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._image_height))

        capture.set(cv2.CAP_PROP_FPS, float(self._publish_rate))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._apply_exposure_controls(capture, device_index=self._device_index)

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))

        self._image_width = actual_width
        self._image_height = actual_height
        self._capture = capture
        self.get_logger().info(
            "Connected RGB camera: "
            f"backend={backend_name}, device_index={self._device_index}, "
            f"resolution={actual_width}x{actual_height}, fps={actual_fps:.2f}, "
            f"max_fov={self._prefer_max_fov}"
        )

    def _apply_exposure_controls(
        self, capture: cv2.VideoCapture, *, device_index: int
    ) -> None:
        """OpenCV props + direct V4L2 ioctls (RealSense D405: brightness/gamma via UVC)."""
        brightness = self._brightness
        gamma = self._gamma
        if self._auto_tuner is not None:
            brightness = self._auto_tuner.active_brightness
            gamma = self._auto_tuner.active_gamma

        mode = "auto" if self._auto_exposure else "manual"
        if self._auto_exposure:
            for val in (0.75, 1.0, 3.0):
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
        else:
            for val in (0.25, 0.0, 1.0):
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            if self._exposure >= 0:
                capture.set(cv2.CAP_PROP_EXPOSURE, float(self._exposure))
            if self._gain >= 0:
                capture.set(cv2.CAP_PROP_GAIN, float(self._gain))
            if brightness >= 0:
                capture.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness))

        v4l2 = apply_v4l2_controls(
            device_index,
            auto_exposure=self._auto_exposure,
            exposure=self._exposure,
            gain=self._gain,
            brightness=brightness,
            gamma=gamma,
        )

        ae = capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp = capture.get(cv2.CAP_PROP_EXPOSURE)
        gain = capture.get(cv2.CAP_PROP_GAIN)
        bright = capture.get(cv2.CAP_PROP_BRIGHTNESS)
        self.get_logger().info(
            f"Exposure {mode}: opencv ae={ae:.3f} exp={exp:.1f} gain={gain:.1f} "
            f"bright={bright:.1f} | v4l2 set={v4l2.get('set')} readback={v4l2.get('readback')} "
            f"(requested auto={self._auto_exposure} exp={self._exposure} "
            f"gain={self._gain} bright={brightness} gamma={gamma})"
        )

    def _run_brightness_trial(self) -> None:
        if self._auto_tuner is None:
            return
        with self._capture_lock:
            if self._capture is None:
                return
            capture = self._capture
            candidate = self._auto_tuner.next_candidate()
            if candidate is None:
                return
            trial_b, trial_g = candidate
            self._publish_tuning_stats()
            self._auto_tuner.apply_values(trial_b, trial_g)
            time.sleep(0.08)
            for _ in range(4):
                capture.read()
            means = []
            for _ in range(3):
                ok, frame = capture.read()
                if ok and frame is not None:
                    means.append(StereoAutoBrightness.frame_mean(frame))
            trial_mean = sum(means) / len(means) if means else None
        self._auto_tuner.finish_trial(trial_mean)
        self._publish_tuning_stats()

    def _maybe_start_trial(self, now: float) -> None:
        if self._auto_tuner is None:
            return
        if not self._auto_tuner.should_trial(now):
            return
        if self._trial_thread is not None and self._trial_thread.is_alive():
            return
        self._trial_thread = threading.Thread(
            target=self._run_brightness_trial,
            name="stereo-brightness-trial",
            daemon=True,
        )
        self._trial_thread.start()

    def _publish_tuning_stats(self) -> None:
        if self._auto_tuner is None or self._tuning_pub is None:
            return
        msg = String()
        msg.data = json.dumps(self._auto_tuner.stats(), separators=(",", ":"))
        self._tuning_pub.publish(msg)

    def _on_timer(self) -> None:
        now = time.monotonic()
        self._maybe_start_trial(now)

        with self._capture_lock:
            if self._capture is None or not self._capture.isOpened():
                if self._auto_reconnect:
                    if now - self._last_reconnect_try >= 1.0:
                        self._last_reconnect_try = now
                        self._open_camera()
                return

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.get_logger().warn("Camera frame read failed.")
                if self._auto_reconnect:
                    self._open_camera()
                return

            for _ in range(1):
                grabbed = self._capture.grab()
                if not grabbed:
                    break
                ok2, newer = self._capture.retrieve()
                if ok2 and newer is not None:
                    frame = newer

        if self._flip_horizontal:
            frame = cv2.flip(frame, 1)

        if self._auto_tuner is not None:
            self._auto_tuner.update_from_frame(frame)
            self._publish_tuning_stats()

        stamp = self.get_clock().now().to_msg()
        image_msg = self._to_image_msg(frame, stamp)
        self._image_pub.publish(image_msg)

        if self._publish_camera_info:
            self._camera_info_pub.publish(self._build_camera_info(stamp, frame.shape[1], frame.shape[0]))

        if self._publish_debug_image:
            debug_frame = self._make_debug_frame(frame)
            self._debug_pub.publish(self._to_image_msg(debug_frame, stamp))

    def _to_image_msg(self, frame, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(frame.shape[1] * frame.shape[2])
        msg.data = frame.tobytes()
        return msg

    def _build_camera_info(self, stamp, width: int, height: int) -> CameraInfo:
        # Use simple pinhole defaults until calibrated intrinsics are available.
        fx = float(width)
        fy = float(height)
        cx = float(width) / 2.0
        cy = float(height) / 2.0

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _make_debug_frame(self, frame):
        debug = frame.copy()
        height, width = debug.shape[:2]
        cx = width // 2
        cy = height // 2
        cv2.line(debug, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 2)
        cv2.line(debug, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 2)
        cv2.putText(
            debug,
            self._camera_name,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    def destroy_node(self) -> bool:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
