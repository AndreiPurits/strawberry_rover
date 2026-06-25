"""Background sequence runner on Orin (no hub round-trip per step)."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from roarm_client import RoArmClient
from sequence_manager import SequenceManager

_lock = threading.Lock()
_stop_flag = False
_thread: Optional[threading.Thread] = None
_state: Dict[str, Any] = {
    "running": False,
    "index": -1,
    "total": 0,
    "step_status": "",
    "result": None,
    "error": None,
    "log": [],
}


def _append_log(line: str) -> None:
    with _lock:
        log = list(_state.get("log") or [])
        log.append(line)
        _state["log"] = log[-80:]


def sequence_status() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def sequence_stop() -> Dict[str, Any]:
    global _stop_flag
    _stop_flag = True
    _append_log("stop requested")
    return {"ok": True, "running": bool(_state.get("running"))}


def _should_stop() -> bool:
    return _stop_flag


def _run_sequence(client: RoArmClient, steps: List[Dict[str, Any]]) -> None:
    global _stop_flag
    mgr = SequenceManager(client)

    def on_start(idx: int, step: Dict[str, Any], meta: Dict[str, Any]) -> None:
        with _lock:
            _state["index"] = idx
            _state["step_status"] = f"running:{step.get('type')}"
        _append_log(f"step {idx + 1}/{len(steps)} start {step.get('type')}")

    def on_done(idx: int, step: Dict[str, Any], url: str, resp: str, meta: Dict[str, Any]) -> None:
        with _lock:
            _state["step_status"] = f"done:{step.get('type')}"
        _append_log(f"step {idx + 1} ok")

    def on_error(idx: int, step: Dict[str, Any], err: str, meta: Dict[str, Any]) -> None:
        with _lock:
            _state["step_status"] = "error"
            _state["error"] = err
        _append_log(f"step {idx + 1} error: {err}")

    try:
        result = mgr.run(steps, on_start, on_done, on_error, _should_stop)
        with _lock:
            _state["result"] = result
            _state["running"] = False
            _state["step_status"] = result or "done"
        _append_log(f"sequence finished: {result}")
    except Exception as exc:
        with _lock:
            _state["result"] = "error"
            _state["running"] = False
            _state["error"] = str(exc)
            _state["step_status"] = "error"
        _append_log(f"sequence crash: {exc}")


def sequence_start(client: RoArmClient, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    global _stop_flag, _thread
    if not steps:
        return {"ok": False, "error": "empty_sequence"}
    with _lock:
        if _state.get("running"):
            return {"ok": False, "error": "sequence_already_running"}
    _stop_flag = False
    with _lock:
        _state.update(
            {
                "running": True,
                "index": -1,
                "total": len(steps),
                "step_status": "starting",
                "result": None,
                "error": None,
                "log": [],
            }
        )
    _thread = threading.Thread(
        target=_run_sequence,
        args=(client, steps),
        name="roarm-sequence",
        daemon=True,
    )
    _thread.start()
    return {"ok": True, "running": True, "total": len(steps)}
