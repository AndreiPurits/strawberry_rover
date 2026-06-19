"""AXM fleet monitoring hub — central dashboard for rovers."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE = "axm_session"
SESSION_TTL_S = 60 * 60 * 12
SESSION_REMEMBER_TTL_S = 60 * 60 * 24 * 30
OPERATOR_LOCK_TTL_S = 120
MJPEG_BOUNDARY = b"frame"

app = FastAPI(title="AXM Fleet Monitor", version="0.5.3")

_sessions: Dict[str, float] = {}
_dashboard_clients: Set[WebSocket] = set()
_rovers: Dict[str, Dict[str, Any]] = {}
_command_queues: Dict[str, List[Dict[str, Any]]] = {}
_command_log: List[Dict[str, Any]] = []
_camera_frames: Dict[str, Dict[str, Any]] = {}
_stereo_camera_frames: Dict[str, Dict[str, Any]] = {}
_operator_locks: Dict[str, Dict[str, Any]] = {}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


AGENT_STALE_S = int(_env("AXM_AGENT_STALE_S", "600"))


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


def _verify_password(username: str, password: str) -> bool:
    if username != _admin_user():
        return False
    try:
        return bcrypt.checkpw(password.encode(), _admin_password_hash())
    except ValueError:
        return False


def _session_secret() -> bytes:
    raw = _env("AXM_SESSION_SECRET") or _env("AXM_ADMIN_PASSWORD_HASH") or _env("AXM_ADMIN_PASSWORD", "changeme")
    return raw.encode()


def _sign_session(expires_at: float) -> str:
    payload = str(int(expires_at))
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _issue_session(remember: bool = False) -> tuple[str, int]:
    ttl = SESSION_REMEMBER_TTL_S if remember else SESSION_TTL_S
    expires = time.time() + ttl
    return _sign_session(expires), ttl


def _session_valid(token: Optional[str]) -> bool:
    if not token or "." not in token:
        return False
    payload, sig = token.rsplit(".", 1)
    expected = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        expires = float(payload)
    except ValueError:
        return False
    return expires >= time.time()


def require_user(request: Request) -> str:
    if not _session_valid(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="login_required")
    return _admin_user()


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
    telemetry = dict(new or {})
    prev_p = (prev or {}).get("perception") or {}
    perception = dict(telemetry.get("perception") or {})
    if not perception.get("lidar_arc") and prev_p.get("lidar_arc"):
        perception["lidar_arc"] = prev_p["lidar_arc"]
    telemetry["perception"] = perception
    return telemetry


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
    payload = {
        "type": "fleet",
        "rovers": [_rover_public(r) for r in _rovers.values()],
    }
    stale: List[WebSocket] = []
    for ws in _dashboard_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _dashboard_clients.discard(ws)


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
    token, ttl = _issue_session(body.remember)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=_env("AXM_COOKIE_SECURE", "true").lower() in ("1", "true", "yes"),
        samesite="lax",
        max_age=ttl,
        path="/",
    )
    return resp


@app.post("/api/logout")
def api_logout(request: Request) -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/")
def root(request: Request):
    if not _session_valid(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/rovers")
def api_rovers(user: str = Depends(require_user)) -> Dict[str, Any]:
    rows = sorted(_rovers.values(), key=lambda r: r.get("name", r["id"]))
    return {"ok": True, "rovers": [_rover_public(r, user) for r in rows]}


@app.post("/api/rovers/{rover_id}/claim")
async def claim_operator(rover_id: str, user: str = Depends(require_user)) -> Dict[str, Any]:
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
    if not _session_valid(cookie):
        await ws.close(code=4401)
        return
    _dashboard_clients.add(ws)
    try:
        await ws.send_json(
            {"type": "fleet", "rovers": [_rover_public(r) for r in _rovers.values()]}
        )
        while True:
            raw = await ws.receive_text()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "drive":
                continue
            rover_id = str(msg.get("rover_id") or "").strip()
            if not rover_id:
                continue
            user = _admin_user()
            try:
                _require_operator_lock(rover_id, user)
            except HTTPException:
                continue
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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
