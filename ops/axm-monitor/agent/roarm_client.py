"""RoArm-M3 HTTP client for fleet agent (urllib → ESP32 on local LAN)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
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

    def _http_get(self, url: str, timeout_sec: float | None = None) -> str:
        effective = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=effective) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RoArmClientError(f"HTTP {exc.code} for {url}: {body[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RoArmClientError(f"request failed for {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RoArmClientError(f"timeout for {url}") from exc

    def get_status(self, timeout_sec: float | None = None) -> Tuple[str, Dict[str, Any]]:
        url = self._status_url()
        raw = self._http_get(url, timeout_sec=timeout_sec)
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise RoArmClientError(f"invalid JSON from {url}: {raw[:120]!r}") from exc
        if not isinstance(parsed, dict):
            raise RoArmClientError(f"unexpected status type from {url}")
        return url, parsed

    def send_raw_json(self, cmd: Dict[str, Any], timeout_sec: float | None = None) -> Tuple[str, str]:
        url = self._command_url(cmd)
        return url, self._http_get(url, timeout_sec=timeout_sec)

    def home(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 100}, timeout_sec=timeout_sec)

    def gripper_open(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0}, timeout_sec=timeout_sec)

    def gripper_close(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0}, timeout_sec=timeout_sec)

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
    ) -> Tuple[str, str]:
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

    def torque(self, enabled: bool, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 210, "cmd": 1 if enabled else 0}, timeout_sec=timeout_sec)
