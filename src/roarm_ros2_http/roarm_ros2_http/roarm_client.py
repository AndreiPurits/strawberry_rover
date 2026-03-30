"""HTTP client wrapper for Waveshare RoArm-M3 JSON API.

Build:
  colcon build --packages-select roarm_ros2_http
Run:
  ros2 launch roarm_ros2_http demo_pick.launch.py roarm_ip:=192.168.4.1
Manual target test:
  ros2 topic pub --once /roarm/target_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'roarm_base'}, pose: {position: {x: 0.16, y: 0.00, z: 0.12}, orientation: {w: 1.0}}}"
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import subprocess
from typing import Any, Dict


class RoArmClientError(RuntimeError):
    """Raised when RoArm HTTP command fails."""


@dataclass(frozen=True)
class RoArmHttpConfig:
    """HTTP transport configuration for RoArm JSON commands."""

    roarm_ip: str = "192.168.4.1"
    api_path: str = "/js"
    timeout_sec: float = 8.0

    @property
    def endpoint(self) -> str:
        return f"http://{self.roarm_ip}{self.api_path}"


class RoArmClient:
    """Thin client over RoArm-M3 HTTP JSON API."""

    def __init__(self, config: RoArmHttpConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    def _run_curl(self, cmd: list[str], url: str) -> str:
        self._logger.info(f"RoArm CURL URL: {url}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_sec,
            )
            if result.returncode != 0:
                raise RoArmClientError(
                    f"curl failed for {url}; stderr={result.stderr.strip()!r}; stdout={result.stdout.strip()!r}"
                )
            return result.stdout
        except subprocess.TimeoutExpired as exc:
            raise RoArmClientError(f"curl timed out for {url}") from exc

    def send_raw_json(self, cmd: Dict[str, Any]) -> str:
        """Send raw JSON command and return response text."""
        # Transport seam:
        # If you later switch to UART/serial, replace this method body with
        # pyserial write/read logic and keep public API methods unchanged.
        json_payload = json.dumps(cmd, separators=(",", ":"))
        url = f"{self._config.endpoint}?json={json_payload}"
        return self._run_curl(["curl", "-g", url], url)

    def get_status(self) -> Dict[str, Any]:
        """Fetch current arm status as JSON dictionary."""
        url = self._config.endpoint
        output = self._run_curl(["curl", url], url)
        try:
            data = json.loads(output)
        except ValueError as exc:
            raise RoArmClientError(
                f"Failed to decode curl status JSON: {exc}; payload={output!r}"
            ) from exc
        if not isinstance(data, dict):
            raise RoArmClientError(
                f"Unexpected curl status payload type: {type(data).__name__}"
            )
        return data

    def move_xyz(
        self,
        x: float,
        y: float,
        z: float,
        t: float = 0.0,
        r: float = 0.0,
        g: float = 3.14,
        spd: float = 0.25,
        acc: int = 0,
    ) -> str:
        """Send Cartesian move command."""
        x_mm = round(float(x) * 1000.0, 3)
        y_mm = round(float(y) * 1000.0, 3)
        z_mm = round(float(z) * 1000.0, 3)
        cmd = {
            "T": 104,
            "x": x_mm,
            "y": y_mm,
            "z": z_mm,
            "t": float(t),
            "r": float(r),
            "g": float(g),
            "spd": float(spd),
        }
        return self.send_raw_json(cmd)

    def open_gripper(self) -> str:
        """Open gripper."""
        return self.send_raw_json({"T": 106, "cmd": 1})

    def close_gripper(self) -> str:
        """Close gripper."""
        return self.send_raw_json({"T": 106, "cmd": 2})

    def go_home(self) -> str:
        """Move arm to home pose."""
        return self.send_raw_json({"T": 100})

