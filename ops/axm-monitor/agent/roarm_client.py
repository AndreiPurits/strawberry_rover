"""RoArm-M3 HTTP client for fleet agent (socket HTTP → ESP32 on LAN)."""

from __future__ import annotations

import json
import socket
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
            if self.tcp_open(timeout_sec=connect_timeout):
                raise RoArmClientError(
                    f"HTTP read timeout for {url} — порт 80 открыт, но ESP не отвечает. "
                    "Закройте заводской UI http://{self.ip}/ в браузере (одно подключение)."
                ) from exc
            raise RoArmClientError(f"timeout for {url}") from exc
        except OSError as exc:
            raise RoArmClientError(f"request failed for {url}: {exc}") from exc

        raw = b"".join(chunks).decode("utf-8", errors="replace")
        if not raw.strip():
            raise RoArmClientError(
                f"empty HTTP response from {url} — закройте заводской UI http://{self.ip}/ в браузере"
            )
        _, _, body = raw.partition("\r\n\r\n")
        if not body and "\n\n" in raw:
            _, _, body = raw.partition("\n\n")
        return body.strip() or raw.strip()

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
