"""Demo pick sequence node for RoArm MVP over HTTP.

Sequence:
  home -> pregrasp -> grasp -> close_gripper -> retreat -> home
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node

from .roarm_client import RoArmClient, RoArmClientError, RoArmHttpConfig
from .safety import WorkspaceLimits, sanitize_target


class DemoPickSequenceNode(Node):
    """Executes a simple hard-coded pick sequence."""

    def __init__(self) -> None:
        super().__init__("demo_pick_sequence")

        self.declare_parameter("roarm_ip", "192.168.4.1")
        self.declare_parameter("api_path", "/js")
        self.declare_parameter("http_timeout_sec", 8.0)
        self.declare_parameter("step_sleep_sec", 1.5)

        self._step_sleep_sec = float(self.get_parameter("step_sleep_sec").value)
        config = RoArmHttpConfig(
            roarm_ip=self.get_parameter("roarm_ip").value,
            api_path=self.get_parameter("api_path").value,
            timeout_sec=float(self.get_parameter("http_timeout_sec").value),
        )
        self._client = RoArmClient(config)
        self._limits = WorkspaceLimits()

        self._poses: Dict[str, Tuple[float, float, float]] = {
            "pregrasp": (0.18, 0.02, 0.16),
            "grasp": (0.18, 0.02, 0.11),
            "retreat": (0.18, 0.02, 0.20),
        }

        self.get_logger().info("Starting demo pick sequence...")
        self._run_once()

    def _sleep_step(self) -> None:
        time.sleep(self._step_sleep_sec)

    def _safe_move(self, name: str) -> None:
        x, y, z = self._poses[name]
        safe_target, reason = sanitize_target(
            x=x,
            y=y,
            z=z,
            limits=self._limits,
            reject_if_unsafe=True,
        )
        if safe_target is None:
            self.get_logger().warning(f"Skip {name}: unsafe target {reason}")
            return

        sx, sy, sz = safe_target
        self.get_logger().info(f"Step: {name} -> move_xyz({sx:.3f}, {sy:.3f}, {sz:.3f})")
        response = self._client.move_xyz(sx, sy, sz)
        self.get_logger().info(f"{name} response: {response}")

    def _run_once(self) -> None:
        try:
            self.get_logger().info("Step: home")
            self.get_logger().info(f"home response: {self._client.go_home()}")
            self._sleep_step()

            self._safe_move("pregrasp")
            self._sleep_step()

            self._safe_move("grasp")
            self._sleep_step()

            self.get_logger().info("Step: close_gripper")
            self.get_logger().info(f"close_gripper response: {self._client.close_gripper()}")
            self._sleep_step()

            self._safe_move("retreat")
            self._sleep_step()

            self.get_logger().info("Step: home")
            self.get_logger().info(f"home response: {self._client.go_home()}")
            self.get_logger().info("Demo pick sequence completed.")
        except RoArmClientError as exc:
            self.get_logger().error(f"Demo pick sequence failed: {exc}")
        finally:
            # One-shot node: exit after sequence.
            self.get_logger().info("Shutting down demo_pick_sequence node.")
            rclpy.shutdown()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DemoPickSequenceNode()
    try:
        if rclpy.ok():
            rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

