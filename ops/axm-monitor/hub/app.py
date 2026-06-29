"""AXM fleet monitoring hub — central dashboard for rovers."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from webrtc_relay import WEBRTC_OK, close_relay, create_relay_answer
except ImportError:
    WEBRTC_OK = False

    def close_relay(_rover_id: str) -> None:
        return None

    async def create_relay_answer(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("webrtc_not_available")

STATIC_DIR = Path(__file__).resolve().parent / "static"
PRIVATE_DIR = Path(__file__).resolve().parent / "private"
SESSION_COOKIE = "axm_session"
SESSION_TTL_S = 60 * 60 * 12
SESSION_REMEMBER_TTL_S = 60 * 60 * 24 * 30
OPERATOR_LOCK_TTL_S = 120
MJPEG_BOUNDARY = b"frame"

app = FastAPI(title="AXM Fleet Monitor", version="0.8.0")

_sessions: Dict[str, float] = {}
_dashboard_clients: Set[WebSocket] = set()
_dashboard_ws_users: Dict[WebSocket, str] = {}
_rovers: Dict[str, Dict[str, Any]] = {}
_command_queues: Dict[str, List[Dict[str, Any]]] = {}
_command_log: List[Dict[str, Any]] = []
_camera_frames: Dict[str, Dict[str, Any]] = {}
_stereo_camera_frames: Dict[str, Dict[str, Any]] = {}
_operator_locks: Dict[str, Dict[str, Any]] = {}
_roarm_queue: List[Dict[str, Any]] = []
_roarm_waiters: Dict[str, asyncio.Future] = {}

ROARM_DEVICE_ID = "roarm-01"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


ROARM_HOME_FILE = Path(_env("AXM_ROARM_HOME_FILE", str(Path(__file__).resolve().parent / "data" / "roarm_home.json")))

_DEFAULT_ROARM_HOME: Dict[str, Any] = {
    "base": 0.0,
    "shoulder": -0.35,
    "elbow": 1.85,
    "wrist": 0.0,
    "roll": 0.0,
    "hand": 3.14,
    "spd": 0.0,
    "acc": 10.0,
    "use_preset_for_home": True,
    "note": "Поднятая позиция для ровера — подстройте под стенд и запишите T:502",
}


def _load_roarm_home() -> Dict[str, Any]:
    try:
        if ROARM_HOME_FILE.is_file():
            data = json.loads(ROARM_HOME_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out = dict(_DEFAULT_ROARM_HOME)
                out.update(data)
                return out
    except (OSError, json.JSONDecodeError):
        pass
    return dict(_DEFAULT_ROARM_HOME)


def _save_roarm_home(data: Dict[str, Any]) -> Dict[str, Any]:
    ROARM_HOME_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(_DEFAULT_ROARM_HOME)
    for key in ("base", "shoulder", "elbow", "wrist", "roll", "hand", "spd", "acc"):
        if key in data:
            merged[key] = float(data[key])
    if "use_preset_for_home" in data:
        merged["use_preset_for_home"] = bool(data["use_preset_for_home"])
    if isinstance(data.get("note"), str):
        merged["note"] = str(data["note"])[:500]
    merged["updated_at"] = time.time()
    ROARM_HOME_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


ROARM_POINTS_FILE = Path(
    _env("AXM_ROARM_POINTS_FILE", str(Path(__file__).resolve().parent / "data" / "roarm_points.json"))
)

_DEFAULT_ROARM_POINTS: List[Dict[str, Any]] = [
    {
        "id": "home",
        "name": "Home",
        "role": "home",
        "mode": "joints",
        "joints": {
            "base": 0.0,
            "shoulder": -0.35,
            "elbow": 1.85,
            "wrist": 0.0,
            "roll": 0.0,
            "hand": 3.14,
        },
        "xyz": {"x": 235.0, "y": 0.0, "z": 234.0, "t": 0.0, "r": 0.0, "g": 3.14},
        "joint_spd": 0.0,
        "joint_acc": 10.0,
        "xyz_spd": 0.25,
    },
    {
        "id": "berry_store",
        "name": "Хранилище ягод",
        "role": None,
        "mode": "xyz",
        "joints": None,
        "xyz": {"x": 235.0, "y": 0.0, "z": 234.0, "t": 0.0, "r": 0.0, "g": 3.14},
        "joint_spd": 0.0,
        "joint_acc": 10.0,
        "xyz_spd": 0.25,
    },
]


def _load_roarm_points() -> List[Dict[str, Any]]:
    try:
        if ROARM_POINTS_FILE.is_file():
            data = json.loads(ROARM_POINTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("points"), list):
                return list(data["points"])
            if isinstance(data, list):
                return list(data)
    except (OSError, json.JSONDecodeError):
        pass
    return [dict(p) for p in _DEFAULT_ROARM_POINTS]


def _save_roarm_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ROARM_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned: List[Dict[str, Any]] = []
    for raw in points:
        if not isinstance(raw, dict):
            continue
        pid = str(raw.get("id") or uuid.uuid4().hex[:12])
        name = str(raw.get("name") or "Point")[:64]
        mode = str(raw.get("mode") or "joints")
        if mode not in ("joints", "xyz"):
            mode = "joints"
        role = raw.get("role")
        if role is not None:
            role = str(role)[:32]
        joints = raw.get("joints") if isinstance(raw.get("joints"), dict) else None
        xyz = raw.get("xyz") if isinstance(raw.get("xyz"), dict) else None
        cleaned.append(
            {
                "id": pid,
                "name": name,
                "role": role,
                "mode": mode,
                "joints": joints,
                "xyz": xyz,
                "joint_spd": float(raw.get("joint_spd", 0)),
                "joint_acc": float(raw.get("joint_acc", 10)),
                "xyz_spd": float(raw.get("xyz_spd", 0.25)),
                "updated_at": float(raw.get("updated_at") or time.time()),
            }
        )
    ROARM_POINTS_FILE.write_text(
        json.dumps({"points": cleaned}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cleaned


def _sync_home_from_point(point: Dict[str, Any]) -> None:
    joints = point.get("joints") if isinstance(point.get("joints"), dict) else {}
    if not joints:
        return
    _save_roarm_home(
        {
            "base": joints.get("base", 0),
            "shoulder": joints.get("shoulder", 0),
            "elbow": joints.get("elbow", 1.57),
            "wrist": joints.get("wrist", 0),
            "roll": joints.get("roll", 0),
            "hand": joints.get("hand", 3.14),
            "spd": point.get("joint_spd", 0),
            "acc": point.get("joint_acc", 10),
            "use_preset_for_home": True,
            "note": f"HOME from point: {point.get('name')}",
        }
    )


AGENT_STALE_S = int(_env("AXM_AGENT_STALE_S", "600"))


def _cookie_secure() -> bool:
    return _env("AXM_COOKIE_SECURE", "true").lower() in ("1", "true", "yes")


def _admin_user() -> str:
    return _env("AXM_ADMIN_USER", "admin")


def _admin_password_hash() -> bytes:
    raw = _env("AXM_ADMIN_PASSWORD_HASH")
    if raw:
        return raw.encode()
    plain = _env("AXM_ADMIN_PASSWORD", "changeme")
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt())


def _agent_tokens() -> Dict[str, str]:
    out: Dict[str, str] = {}
    blob = _env("AXM_AGENT_TOKENS")
    if blob:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                out.update({str(k): str(v) for k, v in parsed.items()})
        except json.JSONDecodeError:
            pass
    for key, value in os.environ.items():
        if key.startswith("ROVER_") and key.endswith("_TOKEN") and value:
            rover_id = key[len("ROVER_") : -len("_TOKEN")].lower().replace("_", "-")
            out[rover_id] = value.strip()
    return out


def _normalize_username(username: str) -> str:
    user = (username or "").strip()
    if not re.fullmatch(r"[a-zA-Z0-9._-]{2,32}", user):
        raise ValueError("invalid_username")
    return user


def _auth_users() -> Optional[Dict[str, bytes]]:
    blob = _env("AXM_USERS_JSON")
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, bytes] = {}
    for key, val in parsed.items():
        if not val:
            continue
        out[str(key)] = val.encode() if isinstance(val, str) else bytes(val)
    return out or None


def _verify_password(username: str, password: str) -> bool:
    try:
        user = _normalize_username(username)
    except ValueError:
        return False
    users = _auth_users()
    if users is not None:
        phash = users.get(user)
        if not phash:
            return False
        try:
            return bcrypt.checkpw(password.encode(), phash)
        except ValueError:
            return False
    try:
        return bcrypt.checkpw(password.encode(), _admin_password_hash())
    except ValueError:
        return False


def _session_secret() -> bytes:
    raw = _env("AXM_SESSION_SECRET") or _env("AXM_ADMIN_PASSWORD_HASH") or _env("AXM_ADMIN_PASSWORD", "changeme")
    return raw.encode()


def _sign_session(username: str, expires_at: float) -> str:
    payload = f"{username}:{int(expires_at)}"
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _issue_session(username: str, remember: bool = False) -> tuple[str, int]:
    ttl = SESSION_REMEMBER_TTL_S if remember else SESSION_TTL_S
    expires = time.time() + ttl
    return _sign_session(username, expires), ttl


def _parse_session(token: Optional[str]) -> Optional[tuple[str, float]]:
    if not token or "." not in token:
        return None
    payload, sig = token.rsplit(".", 1)
    expected = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    if ":" in payload:
        user_part, exp_part = payload.split(":", 1)
        try:
            expires = float(exp_part)
        except ValueError:
            return None
        user = user_part
    else:
        try:
            expires = float(payload)
        except ValueError:
            return None
        user = _admin_user()
    if expires < time.time():
        return None
    return user, expires


def _session_valid(token: Optional[str]) -> bool:
    return _parse_session(token) is not None


def require_user(request: Request) -> str:
    parsed = _parse_session(request.cookies.get(SESSION_COOKIE))
    if not parsed:
        raise HTTPException(status_code=401, detail="login_required")
    return parsed[0]


def _verify_agent(rover_id: str, token: str) -> bool:
    expected = _agent_tokens().get(rover_id)
    if not expected:
        return False
    return secrets.compare_digest(expected, token)


def _operator_info(rover_id: str, current_user: Optional[str] = None) -> Dict[str, Any]:
    lock = _operator_locks.get(rover_id)
    now = time.time()
    if not lock or float(lock.get("expires", 0)) < now:
        return {"locked": False, "holder": None, "you": False}
    holder = str(lock.get("user") or "")
    return {
        "locked": True,
        "holder": holder,
        "you": bool(current_user and holder == current_user),
        "expires_in_s": max(0.0, float(lock["expires"]) - now),
    }


def _require_operator_lock(rover_id: str, user: str) -> None:
    info = _operator_info(rover_id, user)
    if info["locked"] and not info["you"]:
        raise HTTPException(
            status_code=409,
            detail=f"rover_controlled_by:{info['holder']}",
        )


def _link_status(rtt_ms: Optional[float], camera_age_ms: Optional[float]) -> str:
    if rtt_ms is None:
        return "red"
    if rtt_ms < 150 and (camera_age_ms is None or camera_age_ms < 300):
        return "green"
    if rtt_ms < 400 and (camera_age_ms is None or camera_age_ms < 800):
        return "yellow"
    return "red"


def _merge_telemetry(prev: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    if not new:
        return dict(prev or {})
    merged = dict(prev or {})
    for key, val in new.items():
        if key == "perception" and isinstance(val, dict):
            p = dict(merged.get("perception") or {})
            p.update(val)
            prev_p = (prev or {}).get("perception") or {}
            if not val.get("lidar_arc") and prev_p.get("lidar_arc"):
                p["lidar_arc"] = prev_p["lidar_arc"]
            if not val.get("lidar_guard") and prev_p.get("lidar_guard"):
                p["lidar_guard"] = prev_p["lidar_guard"]
            elif val.get("lidar_guard") and prev_p.get("lidar_guard"):
                ng = dict(val["lidar_guard"])
                min_fwd = ng.get("min_forward_m")
                thresh = float(ng.get("threshold_m") or 0.4)
                eff_fwd = ng.get("active_effective_forward")
                if eff_fwd is None:
                    eff_fwd = ng.get("active_effective")
                if (
                    not bool(eff_fwd)
                    and min_fwd is not None
                    and float(min_fwd) < thresh
                ):
                    ng["active_effective_forward"] = True
                    ng["active_effective"] = True
                    ng["active"] = True
                    ng["latched_forward"] = True
                    p["lidar_guard"] = ng
            merged["perception"] = p
        elif key == "mega" and isinstance(val, dict):
            m = dict(merged.get("mega") or {})
            for mk, mv in val.items():
                if mv is not None:
                    m[mk] = mv
            merged["mega"] = m
        elif key == "link" and isinstance(val, dict):
            merged["link"] = {**(merged.get("link") or {}), **val}
        elif key == "roarm" and isinstance(val, dict):
            merged["roarm"] = {**(merged.get("roarm") or {}), **val}
        elif key == "rtk" and isinstance(val, dict):
            merged["rtk"] = {**(merged.get("rtk") or {}), **val}
        else:
            merged[key] = val
    return merged


def _roarm_agent_rover_id() -> str:
    return _env("AXM_ROARM_AGENT_ROVER_ID", "rover-01")


def _roarm_public(current_user: Optional[str] = None) -> Dict[str, Any]:
    parent_id = _roarm_agent_rover_id()
    parent = _rovers.get(parent_id) or {}
    parent_last = float(parent.get("last_seen", 0))
    parent_online = bool(parent_last and (time.time() - parent_last) <= AGENT_STALE_S)
    roarm_t = dict((parent.get("telemetry") or {}).get("roarm") or {})
    reachable = bool(roarm_t.get("reachable"))
    return {
        "id": ROARM_DEVICE_ID,
        "name": _env("AXM_ROARM_NAME", "RoArm-01"),
        "kind": "roarm",
        "href": "/?device=roarm-01",
        "online": parent_online,
        "parent_online": parent_online,
        "parent_rover_id": parent_id,
        "last_seen": parent_last,
        "last_seen_ago_s": max(0.0, time.time() - parent_last) if parent_last else None,
        "telemetry": {"roarm": roarm_t},
        "meta": {"device": "roarm-m3"},
        "operator": {"locked": False, "holder": None, "you": False},
        "link": {
            "status": "green" if reachable else ("yellow" if parent_online and roarm_t.get("tcp_open") else ("yellow" if parent_online else "red"))
        },
    }


def _fleet_public(current_user: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [_rover_public(r, current_user) for r in sorted(_rovers.values(), key=lambda r: r.get("name", r["id"]))]
    if _env("AXM_ROARM_ENABLED", "true").lower() not in ("0", "false", "no"):
        rows.append(_roarm_public(current_user))
    return rows


def _rover_public(row: Dict[str, Any], current_user: Optional[str] = None) -> Dict[str, Any]:
    now = time.time()
    last = float(row.get("last_seen", 0))
    online = (now - last) <= AGENT_STALE_S
    cam = _camera_frames.get(row["id"]) or {}
    cam_age_ms = None
    if cam.get("updated_at"):
        cam_age_ms = round((now - float(cam["updated_at"])) * 1000.0, 1)
    link = dict((row.get("telemetry") or {}).get("link") or {})
    if cam_age_ms is not None:
        link["camera_age_ms"] = cam_age_ms
    link["status"] = _link_status(link.get("rtt_ms"), cam_age_ms)
    return {
        "id": row["id"],
        "name": row.get("name", row["id"]),
        "kind": "rover",
        "online": online,
        "last_seen": last,
        "last_seen_ago_s": max(0.0, now - last) if last else None,
        "telemetry": row.get("telemetry", {}),
        "meta": row.get("meta", {}),
        "last_commands": row.get("last_commands", []),
        "operator": _operator_info(row["id"], current_user),
        "link": link,
        "camera_live": bool(cam.get("bytes")),
        "stereo_camera_live": bool((_stereo_camera_frames.get(row["id"]) or {}).get("bytes")),
    }


async def _broadcast_dashboard() -> None:
    payload_generic = {
        "type": "fleet",
        "rovers": _fleet_public(None),
    }
    stale: List[WebSocket] = []
    for ws in _dashboard_clients:
        user = _dashboard_ws_users.get(ws)
        payload = (
            {
                "type": "fleet",
                "rovers": _fleet_public(user),
            }
            if user
            else payload_generic
        )
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _dashboard_clients.discard(ws)
        _dashboard_ws_users.pop(ws, None)


_DRIVE_ACTIONS = frozenset({"drive", "stop_drive", "command"})


def _enqueue_command(rover_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = {
        "id": uuid.uuid4().hex[:12],
        "action": action,
        "params": params or {},
        "queued_at": time.time(),
    }
    q = _command_queues.setdefault(rover_id, [])
    if action in _DRIVE_ACTIONS:
        q[:] = [c for c in q if c.get("action") not in _DRIVE_ACTIONS]
    q.append(cmd)
    entry = {"rover_id": rover_id, **cmd}
    _command_log.append(entry)
    _command_log[:] = _command_log[-100:]
    return cmd


class LoginBody(BaseModel):
    username: str
    password: str
    remember: bool = False


class HeartbeatBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    token: str
    name: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    command_results: List[Dict[str, Any]] = Field(default_factory=list)


class AgentPullBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    token: str


class WebRtcViewerOfferBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    sdp: str = Field(min_length=10)


class WebRtcIceBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    candidate: Dict[str, Any]


class RoArmRpcBody(BaseModel):
    op: str = Field(min_length=1, max_length=32)
    params: Dict[str, Any] = Field(default_factory=dict)


class RoArmHomeBody(BaseModel):
    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 1.57
    wrist: float = 0.0
    roll: float = 0.0
    hand: float = 3.14
    spd: float = 0.0
    acc: float = 10.0
    use_preset_for_home: bool = True
    note: str = ""


class RoArmPointBody(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    mode: str = Field(default="joints", pattern="^(joints|xyz)$")
    joints: Optional[Dict[str, float]] = None
    xyz: Optional[Dict[str, float]] = None
    joint_spd: float = 0.0
    joint_acc: float = 10.0
    xyz_spd: float = 0.25
    role: Optional[str] = None


class RoArmAgentPullBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    token: str


class RoArmAgentResultBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    token: str
    id: str = Field(min_length=8, max_length=64)
    result: Dict[str, Any] = Field(default_factory=dict)


class RoverCommandBody(BaseModel):
    action: str = Field(min_length=1, max_length=32)
    params: Dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    now = time.time()
    fleet: List[Dict[str, Any]] = []
    for row in sorted(_rovers.values(), key=lambda r: r.get("id", "")):
        last = float(row.get("last_seen", 0))
        ago = round(now - last, 1) if last else None
        fleet.append(
            {
                "id": row.get("id"),
                "name": row.get("name", row.get("id")),
                "last_seen_ago_s": ago,
                "online": bool(last and (now - last) <= AGENT_STALE_S),
            }
        )
    return {
        "ok": True,
        "rovers": len(_rovers),
        "online": sum(1 for f in fleet if f.get("online")),
        "fleet": fleet,
        "version": "0.5.4",
    }


@app.get("/login")
def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/api/login")
def api_login(body: LoginBody) -> JSONResponse:
    if not _verify_password(body.username, body.password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    try:
        username = _normalize_username(body.username)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token, ttl = _issue_session(username, body.remember)
    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=ttl,
        path="/",
    )
    return resp


@app.post("/api/logout")
def api_logout(request: Request) -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=_cookie_secure(),
        samesite="lax",
    )
    return resp


@app.get("/static/dashboard.html")
def static_dashboard_redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=302)


@app.get("/api/me")
def api_me(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True, "username": user}


@app.get("/")
def root(request: Request):
    parsed = _parse_session(request.cookies.get(SESSION_COOKIE))
    if not parsed:
        return RedirectResponse("/login", status_code=302)
    user, _ = parsed
    rows = sorted(_rovers.values(), key=lambda r: r.get("name", r["id"]))
    fleet = {"ok": True, "rovers": _fleet_public(user)}
    html = (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")
    boot = json.dumps(fleet, ensure_ascii=False)
    user_json = json.dumps(user, ensure_ascii=False)
    inject = (
        f'  <script>window.__AXM_BOOT_FLEET__={boot};</script>\n'
        f'  <script>window.__AXM_USER__={user_json};</script>\n'
        "  <script>\n"
        "  document.addEventListener('DOMContentLoaded',function(){\n"
        "    var d=window.__AXM_BOOT_FLEET__;\n"
        "    if(!d||!d.rovers||!d.rovers.length)return;\n"
        "    var sel=document.getElementById('rover-select');\n"
        "    var st=document.getElementById('rover-status');\n"
        "    if(!sel)return;\n"
        "    sel.innerHTML='<option value=\"\">— выберите ровер —</option>'+d.rovers.map(function(r){\n"
        "      return '<option value=\"'+r.id+'\">'+(r.name||r.id)+(r.online?' ●':' ○')+'</option>';\n"
        "    }).join('');\n"
        "    if(st)st.textContent=d.rovers.filter(function(r){return r.online;}).length+' online / '+d.rovers.length+' всего';\n"
        "  });\n"
        "  </script>\n"
    )
    html = html.replace("</head>", inject + "</head>", 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/rovers")
def api_rovers(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True, "rovers": _fleet_public(user)}


@app.get("/roarm")
def roarm_page(request: Request) -> RedirectResponse:
    parsed = _parse_session(request.cookies.get(SESSION_COOKIE))
    if not parsed:
        return RedirectResponse("/login?next=/roarm", status_code=302)
    return RedirectResponse("/?device=roarm-01", status_code=302)


@app.get("/roarm/{rest:path}")
def roarm_page_catchall(request: Request, rest: str) -> RedirectResponse:
    parsed = _parse_session(request.cookies.get(SESSION_COOKIE))
    if not parsed:
        return RedirectResponse("/login?next=/roarm", status_code=302)
    return RedirectResponse("/?device=roarm-01", status_code=302)


@app.get("/api/roarm/home")
async def api_roarm_home_get(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True, "home": _load_roarm_home()}


@app.put("/api/roarm/home")
async def api_roarm_home_put(body: RoArmHomeBody, user: str = Depends(require_user)) -> Dict[str, Any]:
    saved = _save_roarm_home(body.model_dump())
    return {"ok": True, "home": saved}


@app.get("/api/roarm/points")
async def api_roarm_points_get(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True, "points": _load_roarm_points()}


@app.post("/api/roarm/points")
async def api_roarm_points_post(body: RoArmPointBody, user: str = Depends(require_user)) -> Dict[str, Any]:
    points = _load_roarm_points()
    point = {
        "id": uuid.uuid4().hex[:12],
        "name": body.name.strip(),
        "role": body.role,
        "mode": body.mode,
        "joints": body.joints,
        "xyz": body.xyz,
        "joint_spd": body.joint_spd,
        "joint_acc": body.joint_acc,
        "xyz_spd": body.xyz_spd,
        "updated_at": time.time(),
    }
    if point["role"] is None and point["name"].strip().lower() in ("home", "дом"):
        point["role"] = "home"
    if point["role"] == "home":
        for p in points:
            p["role"] = None
    points.append(point)
    saved = _save_roarm_points(points)
    created = next((p for p in saved if p["id"] == point["id"]), point)
    if created.get("role") == "home":
        _sync_home_from_point(created)
    return {"ok": True, "point": created, "points": saved}


@app.delete("/api/roarm/points/{point_id}")
async def api_roarm_points_delete(point_id: str, user: str = Depends(require_user)) -> Dict[str, Any]:
    points = [p for p in _load_roarm_points() if str(p.get("id")) != str(point_id)]
    saved = _save_roarm_points(points)
    return {"ok": True, "points": saved}


@app.post("/api/roarm/points/{point_id}/set-home")
async def api_roarm_points_set_home(point_id: str, user: str = Depends(require_user)) -> Dict[str, Any]:
    points = _load_roarm_points()
    found = None
    for p in points:
        if str(p.get("id")) == str(point_id):
            p["role"] = "home"
            found = p
        else:
            p["role"] = None
    if found is None:
        raise HTTPException(status_code=404, detail="point_not_found")
    saved = _save_roarm_points(points)
    _sync_home_from_point(found)
    return {"ok": True, "point": found, "points": saved}


@app.post("/api/roarm/rpc")
async def api_roarm_rpc(body: RoArmRpcBody, user: str = Depends(require_user)) -> Dict[str, Any]:
    parent_id = _roarm_agent_rover_id()
    parent = _rovers.get(parent_id)
    if not parent or (time.time() - float(parent.get("last_seen", 0))) > AGENT_STALE_S:
        raise HTTPException(status_code=503, detail="roarm_gateway_offline")
    req_id = uuid.uuid4().hex
    _roarm_queue.append({"id": req_id, "op": body.op, "params": body.params, "user": user})
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _roarm_waiters[req_id] = fut
    timeout_s = float(_env("AXM_ROARM_RPC_TIMEOUT_S", "12"))
    if body.op in ("move_xyz", "move_xyz_direct", "home_joints", "home_joints_staged", "joint_move"):
        timeout_s = float(_env("AXM_ROARM_MOVE_RPC_TIMEOUT_S", "35"))
        if body.op == "home_joints_staged":
            timeout_s = float(_env("AXM_ROARM_STAGED_RPC_TIMEOUT_S", "120"))
    try:
        result = await asyncio.wait_for(fut, timeout=timeout_s)
        return {"ok": True, "id": req_id, **(result if isinstance(result, dict) else {"result": result})}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="roarm_timeout") from None
    finally:
        _roarm_waiters.pop(req_id, None)
        _roarm_queue[:] = [c for c in _roarm_queue if c.get("id") != req_id]


@app.post("/api/agents/roarm_pull")
async def agent_roarm_pull(body: RoArmAgentPullBody) -> Dict[str, Any]:
    if body.rover_id != _roarm_agent_rover_id():
        raise HTTPException(status_code=403, detail="roarm_agent_mismatch")
    if not _verify_agent(body.rover_id, body.token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")
    batch: List[Dict[str, Any]] = []
    while _roarm_queue and len(batch) < 8:
        batch.append(_roarm_queue.pop(0))
    return {"ok": True, "commands": batch}


@app.post("/api/agents/roarm_result")
async def agent_roarm_result(body: RoArmAgentResultBody) -> Dict[str, Any]:
    if body.rover_id != _roarm_agent_rover_id():
        raise HTTPException(status_code=403, detail="roarm_agent_mismatch")
    if not _verify_agent(body.rover_id, body.token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")
    fut = _roarm_waiters.get(body.id)
    if fut and not fut.done():
        fut.set_result(body.result)
    return {"ok": True}


@app.post("/api/rovers/{rover_id}/claim")
async def claim_operator(rover_id: str, user: str = Depends(require_user)) -> Dict[str, Any]:
    if rover_id == ROARM_DEVICE_ID:
        raise HTTPException(status_code=400, detail="use_roarm_page")
    if rover_id not in _rovers and rover_id not in _agent_tokens():
        raise HTTPException(status_code=404, detail="rover_not_found")
    info = _operator_info(rover_id, user)
    if info["locked"] and not info["you"]:
        raise HTTPException(status_code=409, detail=f"rover_controlled_by:{info['holder']}")
    _operator_locks[rover_id] = {
        "user": user,
        "expires": time.time() + OPERATOR_LOCK_TTL_S,
        "session_id": uuid.uuid4().hex[:12],
    }
    await _broadcast_dashboard()
    return {"ok": True, "operator": _operator_info(rover_id, user)}


@app.post("/api/rovers/{rover_id}/release")
async def release_operator(rover_id: str, user: str = Depends(require_user)) -> Dict[str, Any]:
    lock = _operator_locks.get(rover_id)
    if lock and lock.get("user") == user:
        _operator_locks.pop(rover_id, None)
    await _broadcast_dashboard()
    return {"ok": True, "operator": _operator_info(rover_id, user)}


@app.post("/api/rovers/{rover_id}/command")
async def rover_command(
    rover_id: str,
    body: RoverCommandBody,
    user: str = Depends(require_user),
) -> Dict[str, Any]:
    if rover_id == ROARM_DEVICE_ID:
        raise HTTPException(status_code=400, detail="use_roarm_page")
    if rover_id not in _rovers and rover_id not in _agent_tokens():
        raise HTTPException(status_code=404, detail="rover_not_found")
    _require_operator_lock(rover_id, user)
    lock = _operator_locks.get(rover_id)
    if lock and lock.get("user") == user:
        lock["expires"] = time.time() + OPERATOR_LOCK_TTL_S
    cmd = _enqueue_command(rover_id, body.action, body.params)
    await _broadcast_dashboard()
    return {"ok": True, "command": cmd}


@app.get("/api/rovers/{rover_id}/camera/mjpeg")
async def rover_camera_mjpeg(rover_id: str, user: str = Depends(require_user)) -> StreamingResponse:
    if rover_id not in _rovers and rover_id not in _agent_tokens():
        raise HTTPException(status_code=404, detail="rover_not_found")

    async def stream() -> Any:
        last_sent = 0.0
        while True:
            frame = _camera_frames.get(rover_id) or {}
            data = frame.get("bytes")
            stamp = float(frame.get("stamp") or 0.0)
            if data and stamp != last_sent:
                last_sent = stamp
                header = (
                    b"--" + MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                )
                yield header + data + b"\r\n"
            await asyncio.sleep(1.0 / 45.0)

    return StreamingResponse(
        stream(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/rovers/{rover_id}/camera/stereo/mjpeg")
async def rover_stereo_camera_mjpeg(rover_id: str, user: str = Depends(require_user)) -> StreamingResponse:
    if rover_id not in _rovers and rover_id not in _agent_tokens():
        raise HTTPException(status_code=404, detail="rover_not_found")

    async def stream() -> Any:
        last_sent = 0.0
        while True:
            frame = _stereo_camera_frames.get(rover_id) or {}
            data = frame.get("bytes")
            stamp = float(frame.get("stamp") or 0.0)
            if data and stamp != last_sent:
                last_sent = stamp
                header = (
                    b"--" + MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                )
                yield header + data + b"\r\n"
            await asyncio.sleep(1.0 / 30.0)

    return StreamingResponse(
        stream(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/agents/camera_frame")
async def agent_camera_frame(
    request: Request,
    rover_id: str = Query(...),
    token: str = Query(...),
) -> Dict[str, Any]:
    if not _verify_agent(rover_id, token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_frame")
    stamp_raw = request.headers.get("X-Frame-Stamp")
    try:
        stamp = float(stamp_raw) if stamp_raw else time.time()
    except ValueError:
        stamp = time.time()
    _camera_frames[rover_id] = {
        "bytes": body,
        "stamp": stamp,
        "updated_at": time.time(),
    }
    return {"ok": True}


@app.post("/api/agents/stereo_camera_frame")
async def agent_stereo_camera_frame(
    request: Request,
    rover_id: str = Query(...),
    token: str = Query(...),
) -> Dict[str, Any]:
    if not _verify_agent(rover_id, token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_frame")
    stamp_raw = request.headers.get("X-Frame-Stamp")
    try:
        stamp = float(stamp_raw) if stamp_raw else time.time()
    except ValueError:
        stamp = time.time()
    _stereo_camera_frames[rover_id] = {
        "bytes": body,
        "stamp": stamp,
        "updated_at": time.time(),
    }
    return {"ok": True}


@app.post("/api/agents/pull_commands")
async def agent_pull_commands(body: AgentPullBody) -> Dict[str, Any]:
    if not _verify_agent(body.rover_id, body.token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")
    pending = _command_queues.pop(body.rover_id, [])
    return {"ok": True, "commands": pending}


def _ice_servers() -> List[Dict[str, Any]]:
    raw = _env("AXM_ICE_SERVERS", "")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    servers: List[Dict[str, Any]] = [{"urls": _env("AXM_STUN_URL", "stun:stun.l.google.com:19302")}]
    turn_url = _env("AXM_TURN_URL", "")
    if turn_url:
        servers.append(
            {
                "urls": turn_url,
                "username": _env("AXM_TURN_USER", ""),
                "credential": _env("AXM_TURN_PASS", ""),
            }
        )
    return servers


@app.get("/api/webrtc/config")
def api_webrtc_config(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True, "iceServers": _ice_servers()}


def _get_camera_jpeg(rover_id: str) -> tuple:
    frame = _camera_frames.get(rover_id) or {}
    data = frame.get("bytes")
    stamp = float(frame.get("updated_at") or time.time())
    return data, stamp


@app.post("/api/webrtc/offer")
async def api_webrtc_offer(body: WebRtcViewerOfferBody, user: str = Depends(require_user)) -> Dict[str, Any]:
    if not WEBRTC_OK:
        raise HTTPException(status_code=503, detail="webrtc_not_available")
    if body.rover_id not in _rovers:
        raise HTTPException(status_code=404, detail="rover_not_found")
    try:
        answer_sdp = await create_relay_answer(
            body.rover_id,
            body.sdp,
            _get_camera_jpeg,
            _ice_servers(),
            fps=float(_env("AXM_WEBRTC_FPS", "15")),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"webrtc_failed:{exc}") from exc
    return {"ok": True, "answer": answer_sdp}


@app.post("/api/webrtc/ice")
async def api_webrtc_viewer_ice(body: WebRtcIceBody, user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"ok": True}


@app.post("/api/agents/heartbeat")
async def agent_heartbeat(body: HeartbeatBody) -> Dict[str, Any]:
    if not _verify_agent(body.rover_id, body.token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")

    row = _rovers.setdefault(
        body.rover_id,
        {"id": body.rover_id, "name": body.rover_id, "telemetry": {}, "meta": {}},
    )
    row["last_seen"] = time.time()
    if body.name:
        row["name"] = body.name
    prev = row.get("telemetry") or {}
    row["telemetry"] = _merge_telemetry(prev, body.telemetry or {})
    row["meta"] = body.meta
    if body.command_results:
        row["last_commands"] = body.command_results[-5:]
    await _broadcast_dashboard()

    return {"ok": True, "commands": [], "operator": _operator_info(body.rover_id)}


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket) -> None:
    await ws.accept()
    cookie = ws.cookies.get(SESSION_COOKIE)
    parsed = _parse_session(cookie)
    if not parsed:
        await ws.close(code=4401)
        return
    user, _ = parsed
    _dashboard_clients.add(ws)
    _dashboard_ws_users[ws] = user
    try:
        await ws.send_json(
            {
                "type": "fleet",
                "rovers": _fleet_public(user),
            }
        )
        while True:
            raw = await ws.receive_text()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg_type = str(msg.get("type") or "")
            if msg_type == "webrtc_ice":
                continue
            if msg_type != "drive":
                continue
            rover_id = str(msg.get("rover_id") or "").strip()
            if not rover_id:
                continue
            try:
                _require_operator_lock(rover_id, user)
            except HTTPException:
                continue
            lock = _operator_locks.get(rover_id)
            if lock and lock.get("user") == user:
                lock["expires"] = time.time() + OPERATOR_LOCK_TTL_S
            action = "stop_drive" if msg.get("stop") else "drive"
            params: Dict[str, Any] = {
                "forward": float(msg.get("forward", 0.0)),
                "turn": float(msg.get("turn", 0.0)),
                "speed_scale": float(msg.get("speed_scale", 1.0)),
            }
            if msg.get("lidar_override"):
                params["lidar_override"] = True
            _enqueue_command(rover_id, action, params)
    except WebSocketDisconnect:
        pass
    finally:
        _dashboard_clients.discard(ws)
        _dashboard_ws_users.pop(ws, None)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
