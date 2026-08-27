#!/usr/bin/env python3
"""Relay RealSense color → /stereo_camera/image_raw for chassis API / hub overlay."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class StereoRgbRelay(Node):
    def __init__(self) -> None:
        super().__init__("stereo_rgb_relay")
        self._pub = self.create_publisher(Image, "/stereo_camera/image_raw", qos_profile_sensor_data)
        self.create_subscription(
            Image,
            "/stereo_camera/color/image_rect_raw",
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info("Relay /stereo_camera/color/image_rect_raw → /stereo_camera/image_raw")

    def _on_image(self, msg: Image) -> None:
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = StereoRgbRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
