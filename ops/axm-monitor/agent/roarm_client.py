"""RoArm-M3 HTTP client for fleet agent (socket HTTP → ESP32 on LAN)."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from urllib.parse import quote

_HTTP_LOCK = threading.Lock()


class RoArmClientError(RuntimeError):
    pass


@dataclass
class RoArmClient:
    ip: str = "192.168.1.87"
    timeout_sec: float = 5.0
    _lock: threading.Lock = field(default_factory=lambda: _HTTP_LOCK, repr=False)

    def _command_url(self, cmd: Dict[str, Any]) -> str:
        payload = json.dumps(cmd, separators=(",", ":"))
        return f"http://{self.ip}/js?json={quote(payload, safe='')}"

    def tcp_open(self, timeout_sec: float | None = None) -> bool:
        effective = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        try:
            with socket.create_connection((self.ip, 80), timeout=effective):
                return True
        except OSError:
            return False

    def _http_get(self, url: str, timeout_sec: float | None = None) -> str:
        effective = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        if url.startswith("http://"):
            rest = url[7:]
        else:
            rest = url
        host, _, path = rest.partition("/")
        if not path:
            path = "/"
        else:
            path = "/" + path

        connect_timeout = min(3.0, effective)
        read_timeout = max(1.0, effective - connect_timeout)
        try:
            with self._lock:
                with socket.create_connection((host, 80), timeout=connect_timeout) as sock:
                    sock.settimeout(read_timeout)
                    req = (
                        f"GET {path} HTTP/1.1\r\n"
                        f"Host: {host}\r\n"
                        "Connection: close\r\n"
                        "User-Agent: axm-roarm/1\r\n"
                        "Accept: application/json,*/*\r\n"
                        "\r\n"
                    )
                    sock.sendall(req.encode("ascii"))
                    chunks: list[bytes] = []
                    while True:
                        try:
                            block = sock.recv(4096)
                        except socket.timeout:
                            break
                        if not block:
                            break
                        chunks.append(block)
        except socket.timeout as exc:
            raise RoArmClientError(
                f"HTTP read timeout for {url} — RoArm не ответил за {read_timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise RoArmClientError(f"request failed for {url}: {exc}") from exc

        raw = b"".join(chunks).decode("utf-8", errors="replace")
        if not raw.strip():
            raise RoArmClientError(
                f"empty HTTP response from {url} — перезагрузите RoArm или проверьте Wi‑Fi {self.ip}"
            )
        _, _, body = raw.partition("\r\n\r\n")
        if not body and "\n\n" in raw:
            _, _, body = raw.partition("\n\n")
        return body.strip() or raw.strip()

    def get_status(self, timeout_sec: float | None = None) -> Tuple[str, Dict[str, Any]]:
        """RoArm firmware does not answer bare GET /js — use T:105 feedback."""
        return self.servo_feedback(timeout_sec=timeout_sec)

    def send_raw_json(self, cmd: Dict[str, Any], timeout_sec: float | None = None) -> Tuple[str, str]:
        url = self._command_url(cmd)
        return url, self._http_get(url, timeout_sec=timeout_sec)

    def home(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 100}, timeout_sec=timeout_sec)

    def joint_control(
        self,
        joint: int,
        rad: float,
        spd: float = 0.0,
        acc: float = 10.0,
        timeout_sec: float | None = None,
    ) -> Tuple[str, str]:
        return self.send_raw_json(
            {"T": 101, "joint": int(joint), "rad": float(rad), "spd": float(spd), "acc": float(acc)},
            timeout_sec=timeout_sec,
        )

    def joints_rad_ctrl(
        self,
        *,
        base: float,
        shoulder: float,
        elbow: float,
        wrist: float,
        roll: float,
        hand: float,
        spd: float = 0.0,
        acc: float = 10.0,
        timeout_sec: float | None = None,
    ) -> Tuple[str, str]:
        return self.send_raw_json(
            {
                "T": 102,
                "base": float(base),
                "shoulder": float(shoulder),
                "elbow": float(elbow),
                "wrist": float(wrist),
                "roll": float(roll),
                "hand": float(hand),
                "spd": float(spd),
                "acc": float(acc),
            },
            timeout_sec=timeout_sec,
        )

    def servo_feedback(self, timeout_sec: float | None = None) -> Tuple[str, Dict[str, Any]]:
        url, raw = self.send_raw_json({"T": 105}, timeout_sec=timeout_sec)
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise RoArmClientError(f"invalid feedback JSON: {raw[:120]!r}") from exc
        if not isinstance(parsed, dict):
            raise RoArmClientError(f"unexpected feedback type from {url}")
        return url, parsed

    def set_servo_middle(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        """Save current physical pose as servo middle (affects T:100 / power-on init)."""
        return self.send_raw_json({"T": 502}, timeout_sec=timeout_sec)

    def gripper_open(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        return self.send_raw_json({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0}, timeout_sec=timeout_sec)

    def gripper_ctrl(
        self,
        cmd: float,
        spd: float = 0.0,
        acc: float = 0.0,
        timeout_sec: float | None = None,
    ) -> Tuple[str, str]:
        return self.send_raw_json(
            {"T": 106, "cmd": float(cmd), "spd": float(spd), "acc": float(acc)},
            timeout_sec=timeout_sec,
        )

    def dynamic_torque_limits(
        self,
        *,
        mode: int = 0,
        b: int = 1000,
        s: int = 1000,
        e: int = 1000,
        t: int = 1000,
        r: int = 1000,
        g: int = 1000,
        timeout_sec: float | None = None,
    ) -> Tuple[str, str]:
        """T:112 — per-joint torque caps; mode 0 sets limits, mode 1 enables load adaptation."""
        return self.send_raw_json(
            {"T": 112, "mode": int(mode), "b": b, "s": s, "e": e, "t": t, "r": r, "g": g},
            timeout_sec=timeout_sec,
        )

    def gripper_close_force(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        """Close with max grip torque + repeated T:106 (pads/blades need squeeze, not just angle)."""
        close_rad = float(os.environ.get("ROARM_GRIP_CLOSE_RAD", "3.14"))
        torque_g = int(os.environ.get("ROARM_GRIP_TORQUE_G", "1000"))
        pulses = max(1, int(os.environ.get("ROARM_GRIP_CLOSE_PULSES", "1")))
        pulse_delay = float(os.environ.get("ROARM_GRIP_PULSE_DELAY_S", "0.2"))
        close_acc = float(os.environ.get("ROARM_GRIP_CLOSE_ACC", "50"))

        self.dynamic_torque_limits(mode=0, g=torque_g, timeout_sec=timeout_sec)
        time.sleep(0.05)

        last_url, last_resp = self.gripper_ctrl(close_rad, spd=0, acc=close_acc, timeout_sec=timeout_sec)
        for _ in range(pulses - 1):
            time.sleep(pulse_delay)
            last_url, last_resp = self.gripper_ctrl(close_rad, spd=0, acc=close_acc, timeout_sec=timeout_sec)
        return last_url, last_resp

    def gripper_close(self, timeout_sec: float | None = None) -> Tuple[str, str]:
        effective = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        if os.environ.get("ROARM_GRIP_CLOSE_FORCE", "1").lower() not in ("0", "false", "no"):
            return self.gripper_close_force(timeout_sec=effective)
        return self.gripper_ctrl(3.14, spd=0, acc=0, timeout_sec=effective)

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
