"""u-blox RTK / GNSS NMEA reader for fleet telemetry."""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None  # type: ignore
    SerialException = Exception  # type: ignore

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "connected": False,
    "port": "",
    "fix_quality": None,
    "fix_label": "нет",
    "satellites": None,
    "hdop": None,
    "lat": None,
    "lon": None,
    "alt_m": None,
    "speed_mps": None,
    "last_sentence": "",
    "updated_at": None,
    "error": None,
}
_thread: Optional[threading.Thread] = None
_stop = threading.Event()

_FIX_LABELS = {
    0: "нет",
    1: "GPS",
    2: "DGPS",
    4: "RTK fix",
    5: "RTK float",
    6: "оценка",
}


def _nmea_coord(raw: str, hemi: str) -> Optional[float]:
    raw = (raw or "").strip()
    if not raw or not hemi:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    deg = int(val // 100)
    minutes = val - deg * 100
    dec = deg + minutes / 60.0
    if hemi.upper() in ("S", "W"):
        dec = -dec
    return round(dec, 7)


def _parse_gga(parts: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(parts) < 10:
        return out
    quality = int(parts[6] or "0") if parts[6] else 0
    out["fix_quality"] = quality
    out["fix_label"] = _FIX_LABELS.get(quality, str(quality))
    out["satellites"] = int(parts[7]) if parts[7] else None
    out["hdop"] = float(parts[8]) if parts[8] else None
    out["alt_m"] = float(parts[9]) if parts[9] else None
    lat = _nmea_coord(parts[2], parts[3])
    lon = _nmea_coord(parts[4], parts[5])
    if lat is not None:
        out["lat"] = lat
    if lon is not None:
        out["lon"] = lon
    return out


def _parse_rmc(parts: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(parts) < 10:
        return out
    if parts[2] != "A":
        out["fix_label"] = "нет"
        out["fix_quality"] = 0
    lat = _nmea_coord(parts[3], parts[4])
    lon = _nmea_coord(parts[5], parts[6])
    if lat is not None:
        out["lat"] = lat
    if lon is not None:
        out["lon"] = lon
    if parts[7]:
        try:
            knots = float(parts[7])
            out["speed_mps"] = round(knots * 0.514444, 2)
        except ValueError:
            pass
    return out


def _handle_line(line: str) -> None:
    line = line.strip()
    if not line.startswith("$") or "," not in line:
        return
    body = line.split("*", 1)[0]
    parts = body.split(",")
    tag = parts[0][3:6]
    parsed: Dict[str, Any] = {}
    if tag == "GGA":
        parsed = _parse_gga(parts)
    elif tag == "RMC":
        parsed = _parse_rmc(parts)
    else:
        return
    with _lock:
        _state["last_sentence"] = line[:160]
        _state["updated_at"] = time.time()
        _state["connected"] = True
        _state["error"] = None
        _state.update(parsed)


def _reader_loop(port: str, baud: int) -> None:
    global _state
    if serial is None:
        with _lock:
            _state["error"] = "pyserial_missing"
        return
    while not _stop.is_set():
        try:
            with serial.Serial(port, baud, timeout=0.4) as ser:
                with _lock:
                    _state["port"] = port
                    _state["connected"] = True
                    _state["error"] = None
                while not _stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    try:
                        text = raw.decode("ascii", errors="ignore")
                    except Exception:
                        continue
                    _handle_line(text)
        except SerialException as exc:
            with _lock:
                _state["connected"] = False
                _state["error"] = str(exc)
            _stop.wait(2.0)
        except Exception as exc:
            with _lock:
                _state["connected"] = False
                _state["error"] = str(exc)
            _stop.wait(2.0)


def start_gnss_reader(
    port: Optional[str] = None,
    baud: int = 115200,
) -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    resolved = port or os.environ.get("RTK_PORT", "/dev/ttyACM0")
    if not os.path.exists(resolved):
        with _lock:
            _state["connected"] = False
            _state["port"] = resolved
            _state["error"] = "port_missing"
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_reader_loop,
        args=(resolved, baud),
        name="gnss-reader",
        daemon=True,
    )
    _thread.start()


def gnss_snapshot(max_age_s: float = 3.0) -> Dict[str, Any]:
    with _lock:
        snap = dict(_state)
    updated = snap.get("updated_at")
    if updated is not None and (time.time() - float(updated)) > max_age_s:
        snap["stale"] = True
    else:
        snap["stale"] = False
    if updated is not None:
        snap["age_s"] = round(time.time() - float(updated), 1)
    return snap
