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


class FakeCameraNode(Node):
    """Publish synthetic front and bottom RGB camera images."""

    def __init__(self) -> None:
        super().__init__("fake_camera_node")

        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("front_topic", "/sim/camera/front/image_raw")
        self.declare_parameter("bottom_rgb_topic", "/sim/camera/bottom_rgb/image_raw")
        self.declare_parameter("rover_pose_topic", "/sim/rover_pose")
        self.declare_parameter("front_frame_id", "camera_front_link")
        self.declare_parameter("bottom_rgb_frame_id", "bottom_rgb_camera_link")
        self.declare_parameter("row_count", 5)
        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("bed_width", 0.8)
        self.declare_parameter("field_length", 20.0)
        self.declare_parameter("start_side_extension", 0.0)
        self.declare_parameter("front_camera_pitch_deg", -15.0)
        self.declare_parameter("image_qos_reliability", "reliable")

        self._rate = max(1.0, float(self.get_parameter("publish_rate").value))
        self._width = max(160, int(self.get_parameter("image_width").value))
        self._height = max(120, int(self.get_parameter("image_height").value))
        self._row_count = max(1, int(self.get_parameter("row_count").value))
        self._row_spacing = max(0.6, float(self.get_parameter("row_spacing").value))
        self._bed_width = max(0.2, float(self.get_parameter("bed_width").value))
        self._field_length = max(4.0, float(self.get_parameter("field_length").value))
        self._start_side_extension = max(
            0.0, float(self.get_parameter("start_side_extension").value)
        )
        self._front_pitch_rad = math.radians(
            float(self.get_parameter("front_camera_pitch_deg").value)
        )
        self._field_x_min = -(self._field_length * 0.5) - self._start_side_extension
        self._field_x_max = self._field_length * 0.5
        self._front_frame_id = str(self.get_parameter("front_frame_id").value)
        self._bottom_rgb_frame_id = str(
            self.get_parameter("bottom_rgb_frame_id").value
        )
        self._frame_counter = 0

        front_topic = str(self.get_parameter("front_topic").value)
        bottom_rgb_topic = str(self.get_parameter("bottom_rgb_topic").value)
        pose_topic = str(self.get_parameter("rover_pose_topic").value)
        image_qos = self._build_image_qos(
            str(self.get_parameter("image_qos_reliability").value)
        )

        self._front_pub = self.create_publisher(Image, front_topic, image_qos)
        self._bottom_rgb_pub = self.create_publisher(Image, bottom_rgb_topic, image_qos)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self.create_timer(1.0 / self._rate, self._on_timer)

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self.get_logger().info(
            f"fake_camera_node publishing {self._width}x{self._height} bgr8 images: "
            f"{front_topic} ({self._front_frame_id}), "
            f"{bottom_rgb_topic} ({self._bottom_rgb_frame_id}), "
            f"qos={image_qos.reliability.name.lower()}"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._x = float(msg.pose.position.x)
        self._y = float(msg.pose.position.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self._yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        front = self._render_front_view()
        bottom_rgb = self._render_bottom_rgb_view()
        self._front_pub.publish(self._to_image(front, stamp, self._front_frame_id))
        self._bottom_rgb_pub.publish(
            self._to_image(bottom_rgb, stamp, self._bottom_rgb_frame_id)
        )
        self._frame_counter += 1
        if self._frame_counter % int(max(1.0, self._rate) * 5) == 0:
            self.get_logger().info(
                f"published camera frames #{self._frame_counter} ({self._width}x{self._height})"
            )

    def _render_front_view(self) -> np.ndarray:
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        horizon_ratio = max(
            0.15, min(0.55, 0.35 + (self._front_pitch_rad / math.pi))
        )
        horizon = int(self._height * horizon_ratio)
        img[:horizon, :] = (170, 195, 220)
        img[horizon:, :] = (58, 46, 34)  # soil

        # Ground-plane projection into image: this makes turns and row offset visible.
        gh = self._height - horizon
        nx = np.linspace(-1.0, 1.0, self._width, dtype=np.float32)[None, :]
        ny = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
        forward = 14.0 - (ny * 13.2)  # horizon->far, bottom->near
        half_lateral = 0.35 + (ny * 3.1)
        lateral = nx * half_lateral
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        world_x = self._x + (cy * forward) - (sy * lateral)
        world_y = self._y + (sy * forward) + (cy * lateral)

        bed_mask, edge_mask = self._compute_bed_masks(
            world_x=world_x,
            world_y=world_y,
            edge_tolerance=0.03,
        )
        img_ground = img[horizon:, :]
        img_ground[bed_mask] = (44, 142, 44)
        img_ground[edge_mask] = (76, 118, 70)

        return img

    def _render_bottom_rgb_view(self) -> np.ndarray:
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        img[:, :] = (64, 54, 42)  # soil

        # Bird's-eye projection under the rover arch.
        nx = np.linspace(-1.0, 1.0, self._width, dtype=np.float32)[None, :]
        ny = np.linspace(0.0, 1.0, self._height, dtype=np.float32)[:, None]
        local_forward = ((1.0 - ny) * 0.95) - 0.15
        local_lateral = nx * 0.9
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        world_x = self._x + (cy * local_forward) - (sy * local_lateral)
        world_y = self._y + (sy * local_forward) + (cy * local_lateral)

        bed_mask, edge_mask = self._compute_bed_masks(
            world_x=world_x,
            world_y=world_y,
            edge_tolerance=0.02,
        )

        img[bed_mask] = (48, 135, 46)
        img[edge_mask] = (72, 110, 60)

        return img

    def _to_image(self, frame: np.ndarray, stamp, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(frame.shape[1] * 3)
        msg.data = frame.tobytes()
        return msg

    def _compute_bed_masks(
        self, *, world_x: np.ndarray, world_y: np.ndarray, edge_tolerance: float
    ):
        row_index = np.rint(world_y / self._row_spacing)
        valid_row = (row_index >= 0.0) & (row_index <= float(self._row_count - 1))
        row_center = row_index * self._row_spacing
        within_field = (world_x >= self._field_x_min) & (world_x <= self._field_x_max)
        lateral_dist = np.abs(world_y - row_center)
        bed_mask = valid_row & within_field & (lateral_dist <= (self._bed_width * 0.5))
        edge_mask = valid_row & within_field & (
            np.abs(lateral_dist - (self._bed_width * 0.5)) <= edge_tolerance
        )
        return bed_mask, edge_mask

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
    node = FakeCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
