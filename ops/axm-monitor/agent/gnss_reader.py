"""u-blox RTK / GNSS NMEA reader for fleet telemetry."""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import Any, Dict, List, Optional

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
    "baud": None,
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
    "nmea_count": 0,
    "error": None,
}
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_initial_baud = 38400
_fix_logged = False

_BAUD_CANDIDATES = (38400, 57600, 115200, 9600, 230400)

_FIX_LABELS = {
    0: "нет",
    1: "GPS",
    2: "DGPS",
    4: "RTK fix",
    5: "RTK float",
    6: "оценка",
}


def resolve_rtk_port(port: Optional[str] = None) -> str:
    """Prefer stable by-id path for u-blox USB."""
    if port and os.path.exists(port):
        return port
    env_port = os.environ.get("RTK_PORT", "")
    if env_port and os.path.exists(env_port):
        return env_port
    for pattern in (
        "/dev/serial/by-id/usb-u-blox_*",
        "/dev/serial/by-id/*u-blox*",
    ):
        for dev in sorted(glob.glob(pattern)):
            return dev
    for dev in sorted(glob.glob("/dev/ttyACM*")):
        return dev
    return port or env_port or "/dev/ttyACM0"


def probe_rtk_baud(port: str, candidates: Optional[List[int]] = None) -> Optional[int]:
    """Return baud rate with the most valid NMEA lines (read-only probe)."""
    if serial is None or not os.path.exists(port):
        return None
    best_baud: Optional[int] = None
    best_score = 0
    for baud in candidates or list(_BAUD_CANDIDATES):
        score = 0
        try:
            with serial.Serial(port, baud, timeout=0.25) as ser:
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("ascii", errors="ignore").strip()
                    if line.startswith("$") and "," in line and _nmea_type(line):
                        score += 1
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_baud = baud
    return best_baud if best_score > 0 else None


def _nmea_type(line: str) -> str:
    head = line.split(",", 1)[0]
    if not head.startswith("$") or len(head) < 6:
        return ""
    return head[-3:].upper()


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
    try:
        quality = int(parts[6] or "0") if parts[6] else 0
    except ValueError:
        quality = 0
    out["fix_quality"] = quality
    out["fix_label"] = _FIX_LABELS.get(quality, str(quality))
    try:
        out["satellites"] = int(parts[7]) if parts[7] else None
    except ValueError:
        out["satellites"] = None
    try:
        out["hdop"] = float(parts[8]) if parts[8] else None
    except ValueError:
        out["hdop"] = None
    try:
        out["alt_m"] = float(parts[9]) if parts[9] else None
    except ValueError:
        out["alt_m"] = None
    if quality <= 0:
        out["lat"] = None
        out["lon"] = None
    else:
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
        out["lat"] = None
        out["lon"] = None
    else:
        lat = _nmea_coord(parts[3], parts[4])
        lon = _nmea_coord(parts[5], parts[6])
        if lat is not None:
            out["lat"] = lat
        if lon is not None:
            out["lon"] = lon
        if out.get("fix_quality") is None:
            out["fix_quality"] = 1
            out["fix_label"] = _FIX_LABELS.get(1, "GPS")
    if parts[7]:
        try:
            knots = float(parts[7])
            out["speed_mps"] = round(knots * 0.514444, 2)
        except ValueError:
            pass
    return out


def _handle_line(line: str) -> None:
    global _fix_logged
    line = line.strip()
    if not line.startswith("$") or "," not in line:
        return
    body = line.split("*", 1)[0]
    parts = body.split(",")
    tag = _nmea_type(line)
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
        _state["nmea_count"] = int(_state.get("nmea_count") or 0) + 1
        _state.update(parsed)
        fix_q = int(_state.get("fix_quality") or 0)
        if fix_q > 0 and _state.get("lat") is not None and not _fix_logged:
            _fix_logged = True
            print(
                "[gnss] fix acquired "
                f"q={fix_q} {_state.get('fix_label')} "
                f"lat={_state.get('lat')} lon={_state.get('lon')} "
                f"sats={_state.get('satellites')}"
            )


def _drain_lines(ser: "serial.Serial", buf: bytearray) -> None:
    chunk = ser.read(ser.in_waiting or 256)
    if not chunk:
        chunk = ser.read(1)
    if not chunk:
        return
    buf.extend(chunk)
    while True:
        nl = buf.find(b"\n")
        cr = buf.find(b"\r")
        if nl < 0 and cr < 0:
            if len(buf) > 4096:
                del buf[:-256]
            return
        if nl < 0:
            cut = cr
        elif cr < 0:
            cut = nl
        else:
            cut = min(nl, cr)
        line = bytes(buf[:cut])
        del buf[: cut + 1]
        if buf[:1] in (b"\n", b"\r"):
            del buf[:1]
        try:
            _handle_line(line.decode("ascii", errors="ignore"))
        except Exception:
            pass


def _reader_loop(port: str, baud: int) -> None:
    global _state
    if serial is None:
        with _lock:
            _state["error"] = "pyserial_missing"
        return

    baud_idx = 0
    bauds = [baud] + [b for b in _BAUD_CANDIDATES if b != baud]
    current_baud = bauds[0]
    last_nmea_at = 0.0

    while not _stop.is_set():
        try:
            with serial.Serial(port, current_baud, timeout=0.35) as ser:
                buf = bytearray()
                with _lock:
                    _state["port"] = port
                    _state["baud"] = current_baud
                    _state["connected"] = True
                    _state["error"] = None
                print(f"[gnss] reader open port={port} baud={current_baud}")
                opened_at = time.time()
                while not _stop.is_set():
                    _drain_lines(ser, buf)
                    with _lock:
                        nmea_count = int(_state.get("nmea_count") or 0)
                    if nmea_count > 0:
                        last_nmea_at = time.time()
                    elif time.time() - opened_at > 8.0 and time.time() - last_nmea_at > 8.0:
                        baud_idx = (baud_idx + 1) % len(bauds)
                        current_baud = bauds[baud_idx]
                        print(f"[gnss] no NMEA — try baud {current_baud}")
                        break
                    _stop.wait(0.02)
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
    baud: int = 38400,
) -> None:
    global _thread, _initial_baud, _fix_logged
    if _thread and _thread.is_alive():
        return
    resolved = resolve_rtk_port(port)
    if not os.path.exists(resolved):
        with _lock:
            _state["connected"] = False
            _state["port"] = resolved
            _state["error"] = "port_missing"
        print(f"[gnss] port missing: {resolved}")
        return
    try:
        baud = int(os.environ.get("RTK_BAUD", str(baud)))
    except ValueError:
        baud = 38400
    if os.environ.get("AXM_RTK_AUTOBAUD", "true").lower() not in ("0", "false", "no"):
        probed = probe_rtk_baud(resolved)
        if probed:
            baud = probed
    _initial_baud = baud
    _fix_logged = False
    _stop.clear()
    _thread = threading.Thread(
        target=_reader_loop,
        args=(resolved, baud),
        name="gnss-reader",
        daemon=True,
    )
    _thread.start()


def gnss_snapshot(max_age_s: float = 15.0) -> Dict[str, Any]:
    with _lock:
        snap = dict(_state)
    updated = snap.get("updated_at")
    if updated is not None and (time.time() - float(updated)) > max_age_s:
        snap["stale"] = True
    else:
        snap["stale"] = False
    if updated is not None:
        snap["age_s"] = round(time.time() - float(updated), 1)
    if int(snap.get("fix_quality") or 0) <= 0:
        snap["lat"] = None
        snap["lon"] = None
    return snap
