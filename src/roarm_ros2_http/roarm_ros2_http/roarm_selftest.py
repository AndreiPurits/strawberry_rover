"""Minimal RoArm self-test node for transport and axis checks."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Tuple

import rclpy
from rclpy.node import Node

from .roarm_client import RoArmClient, RoArmClientError, RoArmHttpConfig


class RoArmSelftestNode(Node):
    """Runs one-shot self tests against RoArm HTTP API."""

    def __init__(self) -> None:
        super().__init__("roarm_selftest")
        self.declare_parameter("roarm_ip", "192.168.4.1")
        self.declare_parameter("api_path", "/js")
        self.declare_parameter("http_timeout_sec", 8.0)
        self.declare_parameter("mode", "transport")
        self.declare_parameter("step_sleep_sec", 1.5)
        self.declare_parameter("axis_delta_m", 0.02)
        self.declare_parameter("base_x_m", 0.16)
        self.declare_parameter("base_y_m", 0.0)
        self.declare_parameter("base_z_m", 0.12)

        config = RoArmHttpConfig(
            roarm_ip=str(self.get_parameter("roarm_ip").value),
            api_path=str(self.get_parameter("api_path").value),
            timeout_sec=float(self.get_parameter("http_timeout_sec").value),
        )
        self._client = RoArmClient(config)
        self._mode = str(self.get_parameter("mode").value).strip().lower()
        self._sleep_sec = float(self.get_parameter("step_sleep_sec").value)
        self._axis_delta_m = float(self.get_parameter("axis_delta_m").value)
        self._base_pose = (
            float(self.get_parameter("base_x_m").value),
            float(self.get_parameter("base_y_m").value),
            float(self.get_parameter("base_z_m").value),
        )

        self.get_logger().info(f"RoArm selftest start. mode={self._mode}")
        self._run_once()

    def _sleep_step(self) -> None:
        time.sleep(self._sleep_sec)

    def _log_status(self, prefix: str) -> None:
        status = self._client.get_status()
        self.get_logger().info(f"{prefix} status={json.dumps(status, ensure_ascii=True)}")
        self._log_position_hint(status)
        self._log_brief_status(status, prefix)

    def _log_position_hint(self, status: Dict[str, Any]) -> None:
        # Many firmware variants return different keys; log best-effort position.
        xyz_lower = {k: status[k] for k in ("x", "y", "z") if k in status}
        xyz_upper = {k.lower(): status[k] for k in ("X", "Y", "Z") if k in status}
        xyz = xyz_lower if xyz_lower else xyz_upper
        if xyz:
            self.get_logger().info(f"reported_position={xyz}")

    def _pick(self, status: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key in status:
                return status[key]
        return None

    def _log_brief_status(self, status: Dict[str, Any], prefix: str) -> None:
        x = self._pick(status, ("x", "X"))
        y = self._pick(status, ("y", "Y"))
        z = self._pick(status, ("z", "Z"))
        v = self._pick(status, ("v", "V"))
        self.get_logger().info(f"{prefix} brief: x={x} y={y} z={z} v={v}")

    def _run_status(self) -> None:
        self.get_logger().info("Status test: 3x get_status without motion")
        for idx in range(3):
            status = self._client.get_status()
            self.get_logger().info(f"status[{idx + 1}] source=curl")
            self._log_brief_status(status, f"status[{idx + 1}]")
            time.sleep(1.0)

    def _run_transport(self) -> None:
        self.get_logger().info("Transport test: initial get_status")
        try:
            self._log_status("transport/initial")
        except RoArmClientError as exc:
            self.get_logger().error(
                f"Initial get_status failed, stop transport test without motion: {exc}"
            )
            return
        self._sleep_step()

        self.get_logger().info("Transport test: go_home")
        self.get_logger().info(f"go_home response={self._client.go_home()}")
        self._sleep_step()

        self.get_logger().info("Transport test: post-home get_status")
        self._log_status("transport/post_home")
        self._sleep_step()

        self.get_logger().info("Transport test: open_gripper")
        self.get_logger().info(f"open_gripper response={self._client.open_gripper()}")
        self._sleep_step()

        self.get_logger().info("Transport test: close_gripper")
        self.get_logger().info(f"close_gripper response={self._client.close_gripper()}")
        self._sleep_step()

    def _run_axes(self) -> None:
        self.get_logger().info("Axes test: go_home")
        self.get_logger().info(f"go_home response={self._client.go_home()}")
        self._sleep_step()
        self._log_status("axes/home")

        bx, by, bz = self._base_pose
        d = self._axis_delta_m
        steps: Iterable[Tuple[str, Tuple[float, float, float]]] = (
            ("+X", (bx + d, by, bz)),
            ("-X", (bx - d, by, bz)),
            ("+Y", (bx, by + d, bz)),
            ("-Y", (bx, by - d, bz)),
            ("+Z", (bx, by, bz + d)),
            ("-Z", (bx, by, bz - d)),
        )

        for label, (x, y, z) in steps:
            self.get_logger().info(
                f"Axes test move {label}: target_m=({x:.3f}, {y:.3f}, {z:.3f})"
            )
            self.get_logger().info(f"move response={self._client.move_xyz(x, y, z)}")
            self._sleep_step()
            self._log_status(f"axes/{label}")

        self.get_logger().info("Axes test: go_home")
        self.get_logger().info(f"go_home response={self._client.go_home()}")
        self._sleep_step()
        self._log_status("axes/final_home")

    def _run_once(self) -> None:
        try:
            if self._mode == "status":
                self._run_status()
            elif self._mode == "transport":
                self._run_transport()
            elif self._mode == "axes":
                self._run_axes()
            else:
                self.get_logger().error("Unknown mode. Use mode:=status, mode:=transport, or mode:=axes")
        except RoArmClientError as exc:
            self.get_logger().error(f"Selftest failed: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RoArmSelftestNode()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

