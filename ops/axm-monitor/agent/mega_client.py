"""Lightweight Arduino Mega serial client (Gecoma firmware)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None  # type: ignore
    SerialException = Exception  # type: ignore


def twist_to_pwm(linear_x: float, angular_z: float) -> Tuple[int, int, int, int]:
    linear_x = max(-1.0, min(1.0, linear_x))
    angular_z = max(-1.0, min(1.0, angular_z))
    left = max(-1.0, min(1.0, linear_x - angular_z))
    right = max(-1.0, min(1.0, linear_x + angular_z))

    def to_us(v: float) -> int:
        return int(1500 + v * 500.0)

    fl = rl = to_us(left)
    fr = rr = to_us(right)
    return fl, fr, rl, rr


def port_exists(port: str) -> bool:
    return bool(port) and os.path.exists(port)


def port_busy(port: str) -> bool:
    """True if another process holds the serial port (e.g. ROS driver)."""
    if serial is None or not port_exists(port):
        return False
    try:
        ser = serial.Serial(port, 115200, timeout=0.05)
        ser.close()
        return False
    except SerialException as exc:
        msg = str(exc).lower()
        return "busy" in msg or "lock" in msg or "permission" in msg


def _read_lines(ser: "serial.Serial", duration_s: float) -> list[str]:
    lines: list[str] = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="ignore").strip()
        if text:
            lines.append(text)
    return lines


def send_command(port: str, line: str, baud: int = 115200, read_s: float = 0.4) -> Dict[str, Any]:
    """Send one line to Mega; returns {ok, response, lines, error}."""
    if serial is None:
        return {"ok": False, "error": "pyserial_missing"}
    if not port_exists(port):
        return {"ok": False, "error": "port_missing", "port": port}
    try:
        with serial.Serial(port, baud, timeout=0.15) as ser:
            ser.reset_input_buffer()
            ser.write((line.strip() + "\n").encode("ascii"))
            ser.flush()
            lines = _read_lines(ser, read_s)
            resp = lines[-1] if lines else None
            return {"ok": True, "response": resp, "lines": lines}
    except SerialException as exc:
        return {"ok": False, "error": str(exc), "port": port}


def _reset_mega(ser: "serial.Serial") -> None:
    try:
        ser.dtr = False
        ser.rts = False
        time.sleep(0.05)
        ser.dtr = True
        time.sleep(0.35)
        ser.reset_input_buffer()
    except Exception:
        pass


def probe_mega(port: str, baud: int = 115200, reset: bool = True) -> Dict[str, Any]:
    """Quick connectivity check without holding the port long."""
    out: Dict[str, Any] = {
        "port": port,
        "connected": False,
        "armed": None,
        "pong": False,
        "status": None,
        "source": "serial_probe",
    }
    if not port_exists(port):
        out["error"] = "port_missing"
        return out
    if port_busy(port):
        out["connected"] = True
        out["source"] = "serial_locked"
        out["note"] = "Port held by local driver (ROS/agent)"
        return out
    if serial is None:
        out["error"] = "pyserial_missing"
        return out
    try:
        with serial.Serial(port, baud, timeout=0.3) as ser:
            if reset:
                _reset_mega(ser)
            _read_lines(ser, 0.4)
            ser.write(b"PING\n")
            ser.flush()
            lines = _read_lines(ser, 1.2)
            out["pong"] = "PONG" in lines
            out["connected"] = bool(out["pong"])
            ser.write(b"STATUS\n")
            ser.flush()
            for ln in _read_lines(ser, 1.0):
                if ln.startswith("{"):
                    out["status"] = ln
                    try:
                        data = json.loads(ln)
                        out["armed"] = data.get("armed")
                    except json.JSONDecodeError:
                        pass
                    break
    except SerialException as exc:
        out["error"] = str(exc)
    return out
