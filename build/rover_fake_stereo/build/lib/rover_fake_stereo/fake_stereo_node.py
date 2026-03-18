import math
from typing import Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image


class FakeStereoNode(Node):
    """Publish synthetic bottom-view stereo image pair."""

    def __init__(self) -> None:
        super().__init__("fake_stereo_node")

        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("left_topic", "/sim/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/sim/stereo/right/image_raw")
        self.declare_parameter("debug_combined_topic", "/sim/stereo/debug/combined")
        self.declare_parameter("rover_pose_topic", "/sim/rover_pose")
        self.declare_parameter("frame_id", "stereo_link")
        self.declare_parameter("baseline_px", 10)
        self.declare_parameter("baseline_m", 0.10)
        self.declare_parameter("row_count", 5)
        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("bed_width", 0.8)
        self.declare_parameter("field_length", 22.0)
        self.declare_parameter("post_x_step", 4.0)
        self.declare_parameter("image_qos_reliability", "reliable")

        self._rate = max(1.0, float(self.get_parameter("publish_rate").value))
        self._width = max(160, int(self.get_parameter("image_width").value))
        self._height = max(120, int(self.get_parameter("image_height").value))
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._baseline_px = max(2, int(self.get_parameter("baseline_px").value))
        self._baseline_m = max(0.02, float(self.get_parameter("baseline_m").value))
        self._row_count = max(1, int(self.get_parameter("row_count").value))
        self._row_spacing = max(0.6, float(self.get_parameter("row_spacing").value))
        self._bed_width = max(0.2, float(self.get_parameter("bed_width").value))
        self._field_length = max(4.0, float(self.get_parameter("field_length").value))
        self._post_step = max(1.0, float(self.get_parameter("post_x_step").value))
        self._frame_counter = 0

        left_topic = str(self.get_parameter("left_topic").value)
        right_topic = str(self.get_parameter("right_topic").value)
        combined_topic = str(self.get_parameter("debug_combined_topic").value)
        pose_topic = str(self.get_parameter("rover_pose_topic").value)
        image_qos = self._build_image_qos(
            str(self.get_parameter("image_qos_reliability").value)
        )

        self._left_pub = self.create_publisher(Image, left_topic, image_qos)
        self._right_pub = self.create_publisher(Image, right_topic, image_qos)
        self._combined_pub = self.create_publisher(Image, combined_topic, image_qos)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self.create_timer(1.0 / self._rate, self._on_timer)

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self.get_logger().info(
            f"fake_stereo_node publishing {self._width}x{self._height} bgr8 images: "
            f"{left_topic}, {right_topic}, combined={combined_topic}, "
            f"qos={image_qos.reliability.name.lower()}"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._x = float(msg.pose.position.x)
        self._y = float(msg.pose.position.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self._yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

    def _on_timer(self) -> None:
        left = self._render_view(-self._baseline_m * 0.5)
        right = self._render_view(self._baseline_m * 0.5)
        combined = self._build_combined_debug(left, right)
        stamp = self.get_clock().now().to_msg()
        self._left_pub.publish(self._to_image(left, stamp))
        self._right_pub.publish(self._to_image(right, stamp))
        self._combined_pub.publish(self._to_image(combined, stamp))
        self._frame_counter += 1
        if self._frame_counter % int(max(1.0, self._rate) * 5) == 0:
            self.get_logger().info(
                f"published stereo frames #{self._frame_counter} ({self._width}x{self._height})"
            )

    def _render_view(self, camera_lateral_offset: float) -> np.ndarray:
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        img[:, :] = (63, 53, 41)  # soil

        active_row = round(self._y / self._row_spacing)
        row_center_y = active_row * self._row_spacing
        lateral_offset = (self._y - row_center_y) + camera_lateral_offset
        row_center_x = int((self._width // 2) - lateral_offset * 150.0)
        bed_half = int((self._bed_width / max(0.01, self._row_spacing)) * self._width * 0.45)
        left = max(0, row_center_x - bed_half)
        right = min(self._width, row_center_x + bed_half)
        img[:, left:right] = (45, 132, 44)

        phase = (
            self._x * 0.9
            + self._y * 1.2
            + self._yaw * 0.8
            + camera_lateral_offset * 9.0
        )
        for i in range(5):
            cx = int(row_center_x + math.sin(phase + i * 0.9) * (bed_half * 0.6))
            cy = int((i + 1) * self._height / 6.0)
            self._draw_circle(img, cx, cy, 22, (36, 145, 40))
            self._draw_circle(img, cx + 8, cy - 6, 14, (30, 120, 35))

        for i in range(3):
            bx = int(row_center_x + math.sin(phase + 0.6 + i * 1.3) * (bed_half * 0.48))
            by = int((i + 2) * self._height / 5.0)
            self._draw_circle(img, bx, by, 8, (22, 22, 220))
            self._draw_circle(img, bx - 2, by - 2, 3, (100, 100, 245))

        line_shift = int((self._x * 24.0) % 26)
        for yy in range(line_shift, self._height, 26):
            img[yy:yy + 1, left:right] = (58, 124, 50)

        # Draw projected poles to create logical parallax in left/right cameras.
        cam_x = self._x - (math.sin(self._yaw) * camera_lateral_offset)
        cam_y = self._y + (math.cos(self._yaw) * camera_lateral_offset)
        for px, py in self._iter_poles():
            dx = px - cam_x
            dy = py - cam_y
            fx = (math.cos(self._yaw) * dx) + (math.sin(self._yaw) * dy)
            fy = (-math.sin(self._yaw) * dx) + (math.cos(self._yaw) * dy)
            if fx <= 0.6 or fx > 8.0:
                continue
            xx = int((self._width // 2) + (fy / fx) * 360.0)
            if 0 <= xx < self._width:
                hh = int(max(3, min(34, 40.0 / fx)))
                yy0 = max(0, int(self._height * 0.2))
                img[yy0:yy0 + hh, max(0, xx - 1):min(self._width, xx + 1)] = (205, 205, 205)

        yaw_level = int(((self._yaw + math.pi) / (2.0 * math.pi)) * self._height)
        yaw_level = max(0, min(self._height - 1, yaw_level))
        img[yaw_level:yaw_level + 2, :] = (225, 195, 25)
        return img

    def _to_image(self, frame: np.ndarray, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(frame.shape[1] * 3)
        msg.data = frame.tobytes()
        return msg

    def _build_combined_debug(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        gap = np.zeros((self._height, 8, 3), dtype=np.uint8)
        gap[:, :, :] = (20, 20, 20)
        combined = np.concatenate([left, gap, right], axis=1)
        combined[0:14, 0:120] = (40, 40, 40)
        combined[0:14, -120:] = (40, 40, 40)
        return combined

    def _iter_poles(self):
        x = -self._field_length / 2.0
        while x <= (self._field_length / 2.0) + 1e-6:
            for pair_idx in range(max(0, self._row_count - 1)):
                yield x, (pair_idx + 0.5) * self._row_spacing
            x += self._post_step

    @staticmethod
    def _draw_circle(
        img: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]
    ) -> None:
        h, w = img.shape[:2]
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        rr2 = radius * radius
        for y in range(y0, y1):
            dy = y - cy
            for x in range(x0, x1):
                dx = x - cx
                if (dx * dx) + (dy * dy) <= rr2:
                    img[y, x] = color

    def _build_image_qos(self, reliability: str) -> QoSProfile:
        rel = reliability.strip().lower()
        policy = (
            ReliabilityPolicy.BEST_EFFORT
            if rel in ("best_effort", "besteffort", "sensor_data")
            else ReliabilityPolicy.RELIABLE
        )
        return QoSProfile(
            reliability=policy,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeStereoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
