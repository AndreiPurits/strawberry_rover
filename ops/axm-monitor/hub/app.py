"""AXM fleet monitoring hub — central dashboard for rovers."""
from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE = "axm_session"
SESSION_TTL_S = 60 * 60 * 12
AGENT_STALE_S = 45

app = FastAPI(title="AXM Fleet Monitor", version="0.3.0")

_sessions: Dict[str, float] = {}
_dashboard_clients: Set[WebSocket] = set()
_rovers: Dict[str, Dict[str, Any]] = {}
_command_queues: Dict[str, List[Dict[str, Any]]] = {}
_command_log: List[Dict[str, Any]] = []


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


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


def _issue_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL_S
    return token


def _session_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    expires = _sessions.get(token)
    if expires is None:
        return False
    if expires < time.time():
        _sessions.pop(token, None)
        return False
    _sessions[token] = time.time() + SESSION_TTL_S
    return True


def require_user(request: Request) -> str:
    if not _session_valid(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="login_required")
    return _admin_user()


def _verify_agent(rover_id: str, token: str) -> bool:
    expected = _agent_tokens().get(rover_id)
    if not expected:
        return False
    return secrets.compare_digest(expected, token)


def _rover_public(row: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    last = float(row.get("last_seen", 0))
    online = (now - last) <= AGENT_STALE_S
    return {
        "id": row["id"],
        "name": row.get("name", row["id"]),
        "online": online,
        "last_seen": last,
        "last_seen_ago_s": max(0.0, now - last) if last else None,
        "telemetry": row.get("telemetry", {}),
        "meta": row.get("meta", {}),
        "last_commands": row.get("last_commands", []),
    }


async def _broadcast_dashboard() -> None:
    payload = {"type": "fleet", "rovers": [_rover_public(r) for r in _rovers.values()]}
    stale: List[WebSocket] = []
    for ws in _dashboard_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _dashboard_clients.discard(ws)


def _enqueue_command(rover_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = {
        "id": uuid.uuid4().hex[:12],
        "action": action,
        "params": params or {},
        "queued_at": time.time(),
    }
    _command_queues.setdefault(rover_id, []).append(cmd)
    entry = {"rover_id": rover_id, **cmd}
    _command_log.append(entry)
    _command_log[:] = _command_log[-100:]
    return cmd


class LoginBody(BaseModel):
    username: str
    password: str


class HeartbeatBody(BaseModel):
    rover_id: str = Field(min_length=1, max_length=64)
    token: str
    name: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    command_results: List[Dict[str, Any]] = Field(default_factory=list)


class RoverCommandBody(BaseModel):
    action: str = Field(min_length=1, max_length=32)
    params: Dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "rovers": len(_rovers), "version": "0.3.0"}


@app.get("/login")
def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/api/login")
def api_login(body: LoginBody) -> JSONResponse:
    if not _verify_password(body.username, body.password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = _issue_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=_env("AXM_COOKIE_SECURE", "true").lower() in ("1", "true", "yes"),
        samesite="lax",
        max_age=SESSION_TTL_S,
    )
    return resp


@app.post("/api/logout")
def api_logout(request: Request) -> JSONResponse:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        _sessions.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/")
def root(request: Request):
    if not _session_valid(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/rovers")
def api_rovers(_user: str = Depends(require_user)) -> Dict[str, Any]:
    rows = sorted(_rovers.values(), key=lambda r: r.get("name", r["id"]))
    return {"ok": True, "rovers": [_rover_public(r) for r in rows]}


@app.post("/api/rovers/{rover_id}/command")
async def rover_command(
    rover_id: str,
    body: RoverCommandBody,
    _user: str = Depends(require_user),
) -> Dict[str, Any]:
    if rover_id not in _rovers and rover_id not in _agent_tokens():
        raise HTTPException(status_code=404, detail="rover_not_found")
    cmd = _enqueue_command(rover_id, body.action, body.params)
    await _broadcast_dashboard()
    return {"ok": True, "command": cmd}


@app.post("/api/agents/heartbeat")
async def agent_heartbeat(body: HeartbeatBody) -> Dict[str, Any]:
    if not _verify_agent(body.rover_id, body.token):
        raise HTTPException(status_code=403, detail="invalid_agent_token")

    pending = _command_queues.pop(body.rover_id, [])
    row = _rovers.setdefault(
        body.rover_id,
        {"id": body.rover_id, "name": body.rover_id, "telemetry": {}, "meta": {}},
    )
    row["last_seen"] = time.time()
    if body.name:
        row["name"] = body.name
    row["telemetry"] = body.telemetry
    row["meta"] = body.meta
    if body.command_results:
        row["last_commands"] = body.command_results[-5:]
    await _broadcast_dashboard()

    return {"ok": True, "commands": pending}


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
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _dashboard_clients.discard(ws)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
