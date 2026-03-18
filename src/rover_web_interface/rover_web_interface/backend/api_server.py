import argparse
import asyncio
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

import rclpy
import uvicorn
from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rclpy.executors import SingleThreadedExecutor
from rclpy.utilities import remove_ros_args

from rover_web_interface.backend.ros_bridge import create_bridge_node

app = FastAPI(title="Rover Web Interface API", version="0.1.0")

_bridge = None
_executor: Optional[SingleThreadedExecutor] = None
_spin_thread: Optional[threading.Thread] = None
_ws_clients: List[WebSocket] = []
_broadcast_task: Optional[asyncio.Task] = None


def _frontend_dir() -> str:
    share = get_package_share_directory("rover_web_interface")
    return os.path.join(share, "frontend")


def _spin_executor() -> None:
    if _executor is None:
        return
    try:
        _executor.spin()
    except Exception:
        pass


async def _broadcast_loop() -> None:
    while True:
        if _bridge is not None and _ws_clients:
            state = _bridge.get_state_snapshot()
            stale: List[WebSocket] = []
            for ws in _ws_clients:
                try:
                    await ws.send_json({"type": "state", "data": state})
                except Exception:
                    stale.append(ws)
            for ws in stale:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)
        await asyncio.sleep(0.2)


@app.on_event("startup")
async def on_startup() -> None:
    global _bridge, _executor, _spin_thread, _broadcast_task
    _bridge = create_bridge_node()
    _executor = SingleThreadedExecutor()
    _executor.add_node(_bridge)
    _spin_thread = threading.Thread(target=_spin_executor, daemon=True)
    _spin_thread.start()
    _broadcast_task = asyncio.create_task(_broadcast_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _broadcast_task, _spin_thread, _executor, _bridge
    if _broadcast_task is not None:
        _broadcast_task.cancel()
        _broadcast_task = None
    if _executor is not None:
        _executor.shutdown()
        _executor = None
    if _bridge is not None:
        _bridge.destroy_node()
        _bridge = None
    if rclpy.ok():
        rclpy.shutdown()
    _spin_thread = None


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(_frontend_dir(), "index.html"))


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "bridge_active": _bridge is not None}


@app.get("/api/state")
def api_state() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return _bridge.get_state_snapshot()


@app.get("/api/control/state")
def api_control_state() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return _bridge.get_control_snapshot()


@app.get("/api/routes")
def api_routes() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return _bridge.get_route_snapshot()


@app.post("/api/routes/start")
def api_routes_start() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return {"ok": True, "routes": _bridge.start_route_recording()}


@app.post("/api/routes/stop")
def api_routes_stop() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return {"ok": True, "routes": _bridge.stop_route_recording()}


@app.post("/api/routes/save")
def api_routes_save() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return {"ok": True, "routes": _bridge.save_current_route()}


def _apply_control(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _bridge is None:
        raise HTTPException(status_code=503, detail="bridge_not_ready")

    action = str(payload.get("action", "")).strip().lower()
    if action == "start":
        return _bridge.start_control()
    if action == "stop":
        return _bridge.stop_control()
    if action == "set_mode":
        mode = str(payload.get("mode", "")).strip().lower()
        try:
            return _bridge.set_control_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if action == "command":
        cmd = payload.get("command", {})
        try:
            linear_x = float(cmd.get("linear_x", 0.0))
            angular_z = float(cmd.get("angular_z", 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid_command_values") from exc
        source = str(payload.get("source", "web"))
        return _bridge.apply_manual_command(linear_x=linear_x, angular_z=angular_z, source=source)
    if action == "zero":
        source = str(payload.get("source", "web"))
        return _bridge.publish_zero_cmd(source=source)

    raise HTTPException(status_code=400, detail="unknown_control_action")


@app.post("/api/control")
async def api_control(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    control = _apply_control(payload)
    return {"ok": True, "control": control}


@app.post("/api/control/start")
def api_control_start() -> Dict[str, Any]:
    control = _apply_control({"action": "start"})
    return {"ok": True, "control": control}


@app.post("/api/control/stop")
def api_control_stop() -> Dict[str, Any]:
    control = _apply_control({"action": "stop"})
    return {"ok": True, "control": control}


@app.post("/api/control/mode")
async def api_control_mode(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    control = _apply_control({"action": "set_mode", "mode": payload.get("mode", "")})
    return {"ok": True, "control": control}


@app.post("/api/control/command")
async def api_control_command(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    control = _apply_control(
        {
            "action": "command",
            "command": payload.get("command", {}),
            "source": payload.get("source", "web"),
        }
    )
    return {"ok": True, "control": control}


@app.get("/api/scan")
def api_scan() -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    return _bridge.get_scan_snapshot()


@app.get("/api/cameras/{camera_name}")
def api_camera(camera_name: str) -> Dict[str, Any]:
    if _bridge is None:
        return {"error": "bridge_not_ready"}
    if camera_name not in ("front", "bottom", "stereo"):
        return {"error": "unknown_camera"}
    return _bridge.get_camera_snapshot(camera_name)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            if raw == "hello":
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "control":
                payload = msg.get("data", {})
                try:
                    control = _apply_control(payload)
                    await websocket.send_json({"type": "control_ack", "data": control})
                except HTTPException as exc:
                    await websocket.send_json(
                        {"type": "control_error", "error": str(exc.detail)}
                    )
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


app.mount("/static", StaticFiles(directory=_frontend_dir()), name="static")


def main() -> None:
    cleaned = remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser(description="Rover web interface server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default="8080")
    args, _ = parser.parse_known_args(cleaned[1:])
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
