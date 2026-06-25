"""RoArm-M3 HTTP client for fleet agent (curl → ESP32 on local LAN)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Tuple


class RoArmClientError(RuntimeError):
    pass


@dataclass
class RoArmClient:
    ip: str = "192.168.1.87"
    timeout_sec: float = 5.0

    def _status_url(self) -> str:
        return f"http://{self.ip}/js"

    def _command_url(self, cmd: Dict[str, Any]) -> str:
        payload = json.dumps(cmd, separators=(",", ":"))
        return f"http://{self.ip}/js?json={payload}"

    def _run_curl(self, url: str, use_globoff: bool) -> str:
        cmd = ["curl", "-sS", "--max-time", str(self.timeout_sec)]
        if use_globoff:
            cmd.append("-g")
        cmd.append(url)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec + 1)
        except subprocess.TimeoutExpired as exc:
            raise RoArmClientError(f"curl timeout: {url}") from exc
        if result.returncode != 0:
            raise RoArmClientError(
                f"curl failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    def get_status(self) -> Tuple[str, Dict[str, Any]]:
        url = self._status_url()
        raw = self._run_curl(url, use_globoff=False)
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise RoArmClientError(f"invalid JSON from {url}: {raw[:120]!r}") from exc
        if not isinstance(parsed, dict):
            raise RoArmClientError(f"unexpected status type from {url}")
        return url, parsed

    def send_raw_json(self, cmd: Dict[str, Any]) -> Tuple[str, str]:
        url = self._command_url(cmd)
        return url, self._run_curl(url, use_globoff=True)

    def home(self) -> Tuple[str, str]:
        return self.send_raw_json({"T": 100})

    def gripper_open(self) -> Tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})

    def gripper_close(self) -> Tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})

    def move_xyz(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        r: float,
        g: float,
        spd: float,
    ) -> Tuple[str, str]:
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
            }
        )

    def torque(self, enabled: bool) -> Tuple[str, str]:
        return self.send_raw_json({"T": 210, "cmd": 1 if enabled else 0})
