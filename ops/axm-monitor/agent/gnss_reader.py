"""u-blox RTK / GNSS NMEA reader + RTCM injection for fleet telemetry."""
from __future__ import annotations

import glob
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None  # type: ignore
    SerialException = Exception  # type: ignore

from ntrip_client import ntrip_configured, ntrip_snapshot, start_ntrip_client

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
    "last_gga": "",
    "updated_at": None,
    "nmea_count": 0,
    "rtcm_injected_bytes": 0,
    "error": None,
}
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_initial_baud = 38400
_fix_logged = False
_rtk_fix_logged = False
_rtcm_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=256)

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


def probe_rtk_baud(
    port: str,
    candidates: Optional[List[int]] = None,
    min_score: int = 5,
) -> Optional[int]:
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
                    if line.startswith("$") and "," in line and _nmea_type(line) in ("GGA", "RMC"):
                        score += 1
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_baud = baud
    return best_baud if best_score >= min_score else None


def _nmea_type(line: str) -> str:
    head = line.split(",", 1)[0]
    if not head.startswith("$") or len(head) < 6:
        return ""
    return head[-3:].upper()


def _nmea_checksum(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"*{cs:02X}"


def _deg_to_nmea(value: float, is_lat: bool) -> tuple[str, str]:
    hemi = ("N", "S") if is_lat else ("E", "W")
    sign = 1.0 if value >= 0 else -1.0
    av = abs(value)
    deg = int(av)
    minutes = (av - deg) * 60.0
    if is_lat:
        return f"{deg:02d}{minutes:07.4f}", hemi[0 if sign >= 0 else 1]
    return f"{deg:03d}{minutes:07.4f}", hemi[0 if sign >= 0 else 1]


def build_gga_sentence(
    lat: float,
    lon: float,
    alt_m: float = 0.0,
    quality: int = 1,
    satellites: int = 8,
    hdop: float = 1.0,
) -> str:
    """Build GNGGA for NTRIP VRS casters."""
    tm = time.gmtime()
    tstr = f"{tm.tm_hour:02d}{tm.tm_min:02d}{tm.tm_sec:02d}.00"
    lat_s, ns = _deg_to_nmea(lat, True)
    lon_s, ew = _deg_to_nmea(lon, False)
    body = (
        f"GNGGA,{tstr},{lat_s},{ns},{lon_s},{ew},"
        f"{quality},{satellites},{hdop:.1f},{alt_m:.1f},M,0.0,M,,"
    )
    return f"${body}{_nmea_checksum(body)}"


def _gga_for_ntrip() -> Optional[str]:
    with _lock:
        last_gga = str(_state.get("last_gga") or "").strip()
        lat = _state.get("lat")
        lon = _state.get("lon")
        alt = _state.get("alt_m")
        quality = int(_state.get("fix_quality") or 0)
        sats = _state.get("satellites")
        hdop = _state.get("hdop")
    if last_gga.startswith("$") and "," in last_gga:
        parts = last_gga.split(",")
        if len(parts) > 5 and parts[2] and parts[4]:
            return last_gga
    if lat is None or lon is None:
        try:
            lat = float(os.environ.get("NTRIP_APPROX_LAT", ""))
            lon = float(os.environ.get("NTRIP_APPROX_LON", ""))
        except ValueError:
            return None
        quality = 1
        alt = alt or 0.0
    return build_gga_sentence(
        float(lat),
        float(lon),
        float(alt or 0.0),
        quality if quality > 0 else 1,
        int(sats or 8),
        float(hdop or 1.0),
    )


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
    lat = _nmea_coord(parts[2], parts[3])
    lon = _nmea_coord(parts[4], parts[5])
    if quality > 0:
        if lat is not None:
            out["lat"] = lat
        if lon is not None:
            out["lon"] = lon
    else:
        out["lat"] = None
        out["lon"] = None
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


def _normalize_nmea_line(line: str) -> str:
    line = line.strip()
    while line.startswith("$$"):
        line = "$" + line.lstrip("$")
    return line


def _handle_line(line: str) -> None:
    global _fix_logged, _rtk_fix_logged
    line = _normalize_nmea_line(line)
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
        if tag == "GGA":
            _state["last_gga"] = line[:160]
        _state["updated_at"] = time.time()
        _state["connected"] = True
        _state["error"] = None
        _state["nmea_count"] = int(_state.get("nmea_count") or 0) + 1
        if _state["nmea_count"] == 1:
            print(f"[gnss] first NMEA {line[:100]}")
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
        elif fix_q in (4, 5) and _state.get("lat") is not None and not _rtk_fix_logged:
            _rtk_fix_logged = True
            label = _state.get("fix_label")
            print(
                f"[gnss] RTK active q={fix_q} {label} "
                f"lat={_state.get('lat')} lon={_state.get('lon')}"
            )


def _rtcm_frame_len(buf: bytearray) -> Optional[int]:
    if len(buf) < 3 or buf[0] != 0xD3:
        return None
    length = ((buf[1] & 0x03) << 8) | buf[2]
    return 3 + length + 3


def _consume_buffer(buf: bytearray) -> None:
    while buf:
        if buf[0] == 0xD3:
            frame_len = _rtcm_frame_len(buf)
            if frame_len is None or len(buf) < frame_len:
                return
            del buf[:frame_len]
            continue
        if buf[0] in (0x0A, 0x0D):
            del buf[:1]
            continue
        if buf[0] != ord("$"):
            del buf[:1]
            continue
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


def _drain_serial(ser: "serial.Serial", buf: bytearray) -> None:
    chunk = ser.read(ser.in_waiting or 256)
    if not chunk:
        chunk = ser.read(1)
    if not chunk:
        return
    buf.extend(chunk)
    _consume_buffer(buf)


def _inject_rtcm(ser: "serial.Serial") -> None:
    injected = 0
    while True:
        try:
            chunk = _rtcm_queue.get_nowait()
        except queue.Empty:
            break
        if not chunk:
            continue
        ser.write(chunk)
        injected += len(chunk)
    if injected:
        with _lock:
            _state["rtcm_injected_bytes"] = int(_state.get("rtcm_injected_bytes") or 0) + injected


def _reader_loop(port: str, baud: int) -> None:
    global _state
    if serial is None:
        with _lock:
            _state["error"] = "pyserial_missing"
        return

    baud_idx = 0
    bauds = [baud] + [b for b in _BAUD_CANDIDATES if b != baud]
    current_baud = bauds[0]
    last_nmea_at = time.time()

    while not _stop.is_set():
        try:
            with serial.Serial(port, current_baud, timeout=0.35) as ser:
                buf = bytearray()
                with _lock:
                    _state["port"] = port
                    _state["baud"] = current_baud
                    _state["connected"] = True
                    _state["error"] = None
                ntrip_on = (
                    os.environ.get("AXM_GNSS_MODE", "gps").strip().lower() == "rtk"
                    and ntrip_configured()
                )
                print(
                    f"[gnss] reader open port={port} baud={current_baud}"
                    f"{' ntrip=on' if ntrip_on else ''}"
                )
                opened_at = time.time()
                last_nmea_at = opened_at
                while not _stop.is_set():
                    _drain_serial(ser, buf)
                    if ntrip_on:
                        _inject_rtcm(ser)
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
    global _thread, _initial_baud, _fix_logged, _rtk_fix_logged
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
        else:
            print(f"[gnss] autobaud skipped — using RTK_BAUD={baud}")
    _initial_baud = baud
    _fix_logged = False
    _rtk_fix_logged = False
    _stop.clear()
    if os.environ.get("AXM_GNSS_MODE", "gps").strip().lower() == "rtk" and ntrip_configured():
        start_ntrip_client(_rtcm_queue, _gga_for_ntrip)
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
    snap["ntrip"] = ntrip_snapshot()
    return snap
