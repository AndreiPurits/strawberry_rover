"""Local curl-based client for Waveshare RoArm-M3 (no ROS dependencies)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict


class RoArmClientError(RuntimeError):
    """Raised when a curl call or JSON parsing fails."""


@dataclass
class RoArmClient:
    """Thin RoArm-M3 client using raw HTTP query + curl."""

    ip: str = "192.168.1.87"
    timeout_sec: float = 5.0

    def set_ip(self, ip: str) -> None:
        self.ip = ip.strip()

    def _status_url(self) -> str:
        return f"http://{self.ip}/js"

    def _command_url(self, cmd: Dict[str, Any]) -> str:
        payload = json.dumps(cmd, separators=(",", ":"))
        return f"http://{self.ip}/js?json={payload}"

    def _run_curl(self, url: str, use_globoff: bool, timeout_sec: float | None = None) -> str:
        cmd = ["curl"]
        if use_globoff:
            cmd.append("-g")
        cmd.append(url)
        effective_timeout = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RoArmClientError(f"curl timeout for URL: {url}") from exc

        if result.returncode != 0:
            raise RoArmClientError(
                f"curl failed (code={result.returncode}) for URL: {url}\n"
                f"stderr={result.stderr.strip()!r}\n"
                f"stdout={result.stdout.strip()!r}"
            )
        return result.stdout

    def send_raw_json(self, cmd: Dict[str, Any], timeout_sec: float | None = None) -> tuple[str, str]:
        """Send one JSON command via raw URL query."""
        url = self._command_url(cmd)
        response = self._run_curl(url=url, use_globoff=True, timeout_sec=timeout_sec)
        return url, response

    def get_status(self, timeout_sec: float | None = None) -> tuple[str, Dict[str, Any]]:
        """Read status via plain GET /js and parse JSON response."""
        url = self._status_url()
        response = self._run_curl(url=url, use_globoff=False, timeout_sec=timeout_sec)
        try:
            parsed = json.loads(response)
        except ValueError as exc:
            raise RoArmClientError(
                f"Failed to parse status JSON from {url}: {exc}\nRaw={response!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RoArmClientError(
                f"Unexpected status payload type {type(parsed).__name__} from {url}"
            )
        return url, parsed

    def home(self, timeout_sec: float | None = None) -> tuple[str, str]:
        return self.send_raw_json({"T": 100}, timeout_sec=timeout_sec)

    def joint_control(
        self,
        joint: int,
        rad: float,
        spd: float,
        acc: float,
        timeout_sec: float | None = None,
    ) -> tuple[str, str]:
        return self.send_raw_json(
            {"T": 101, "joint": int(joint), "rad": float(rad), "spd": float(spd), "acc": float(acc)},
            timeout_sec=timeout_sec,
        )

    def axis_control(self, axis: int, pos: float, spd: float, timeout_sec: float | None = None) -> tuple[str, str]:
        return self.send_raw_json(
            {"T": 103, "axis": int(axis), "pos": float(pos), "spd": float(spd)},
            timeout_sec=timeout_sec,
        )

    def move_xyz(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        r: float,
        g: float,
        spd: float,
        timeout_sec: float | None = None,
    ) -> tuple[str, str]:
        return self.send_raw_json(
            {
                "T": 104,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "t": float(t),
                "r": float(r),
                "g": float(g),
                "spd": float(spd),
            },
            timeout_sec=timeout_sec,
        )

    def move_xyz_direct(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        r: float,
        g: float,
        timeout_sec: float | None = None,
    ) -> tuple[str, str]:
        return self.send_raw_json(
            {
                "T": 1041,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "t": float(t),
                "r": float(r),
                "g": float(g),
            },
            timeout_sec=timeout_sec,
        )

    def gripper_open(self, timeout_sec: float | None = None) -> tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0}, timeout_sec=timeout_sec)

    def gripper_close(self, timeout_sec: float | None = None) -> tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0}, timeout_sec=timeout_sec)

    def torque(self, enabled: bool, timeout_sec: float | None = None) -> tuple[str, str]:
        return self.send_raw_json({"T": 210, "cmd": 1 if enabled else 0}, timeout_sec=timeout_sec)

