import time
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


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
        self._last_reconnect_try = 0.0
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

        # Explicitly request known-safe camera settings.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._image_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._image_height))
        capture.set(cv2.CAP_PROP_FPS, float(self._publish_rate))

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))

        self._capture = capture
        self.get_logger().info(
            "Connected RGB camera: "
            f"backend={backend_name}, device_index={self._device_index}, "
            f"resolution={actual_width}x{actual_height}, fps={actual_fps:.2f}"
        )

    def _on_timer(self) -> None:
        if self._capture is None or not self._capture.isOpened():
            if self._auto_reconnect:
                now = time.monotonic()
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

        if self._flip_horizontal:
            frame = cv2.flip(frame, 1)

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
