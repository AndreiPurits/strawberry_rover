"""Simple sequence executor for RoArm local GUI."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal

from roarm_client import RoArmClient, RoArmClientError

RunResult = Literal["done", "stopped", "error"]


class SequenceManager:
    """Run HOME/TARGET step lists using RoArmClient."""

    def __init__(self, client: RoArmClient) -> None:
        self._client = client

    def run(
        self,
        sequence: List[Dict[str, Any]],
        on_step_start: Callable[[int, Dict[str, Any], Dict[str, Any]], None],
        on_step_done: Callable[[int, Dict[str, Any], str, str, Dict[str, Any]], None],
        on_step_error: Callable[[int, Dict[str, Any], str, Dict[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> RunResult:
        for idx, step in enumerate(sequence):
            if should_stop():
                return "stopped"

            step_type = str(step.get("type", "target")).lower()
            step_meta: Dict[str, Any] = {
                "step_type": step_type,
                "json": {},
                "timeout_sec": 0.0,
                "start_time": datetime.now().isoformat(timespec="seconds"),
            }
            on_step_start(idx, step, step_meta)
            t0 = time.monotonic()
            try:
                if step_type == "home":
                    timeout_sec = 10.0
                    cmd = {"T": 100}
                    step_meta["json"] = cmd
                    step_meta["timeout_sec"] = timeout_sec
                    url, response = self._client.home(timeout_sec=timeout_sec)
                elif step_type == "target":
                    timeout_sec = 30.0
                    cmd = {
                        "T": 104,
                        "x": float(step["x"]),
                        "y": float(step["y"]),
                        "z": float(step["z"]),
                        "t": float(step.get("t", 0.0)),
                        "r": float(step.get("r", 0.0)),
                        "g": float(step.get("g", 3.14)),
                        "spd": float(step.get("spd", 0.25)),
                    }
                    step_meta["json"] = cmd
                    step_meta["timeout_sec"] = timeout_sec
                    url, response = self._client.move_xyz(
                        x=cmd["x"],
                        y=cmd["y"],
                        z=cmd["z"],
                        t=cmd["t"],
                        r=cmd["r"],
                        g=cmd["g"],
                        spd=cmd["spd"],
                        timeout_sec=timeout_sec,
                    )
                elif step_type == "gripper":
                    timeout_sec = 5.0
                    action = str(step.get("action", "open")).lower()
                    if action == "close":
                        cmd = {"T": 106, "cmd": 3.14, "spd": 0, "acc": 0}
                        step_meta["json"] = cmd
                        step_meta["timeout_sec"] = timeout_sec
                        url, response = self._client.gripper_close(timeout_sec=timeout_sec)
                    else:
                        cmd = {"T": 106, "cmd": 1.08, "spd": 0, "acc": 0}
                        step_meta["json"] = cmd
                        step_meta["timeout_sec"] = timeout_sec
                        url, response = self._client.gripper_open(timeout_sec=timeout_sec)
                elif step_type == "delay":
                    delay_only = float(step.get("sec", 1.0))
                    step_meta["json"] = {"type": "delay", "sec": delay_only}
                    step_meta["timeout_sec"] = 0.0
                    time.sleep(max(0.0, delay_only))
                    url, response = "delay://local", f"delay {delay_only}s"
                else:
                    raise RoArmClientError(f"Unsupported sequence step type: {step_type}")

                step_meta["finish_time"] = datetime.now().isoformat(timespec="seconds")
                step_meta["duration_sec"] = round(time.monotonic() - t0, 3)
                on_step_done(idx, step, url, response, step_meta)
            except (RoArmClientError, ValueError, KeyError) as exc:
                step_meta["finish_time"] = datetime.now().isoformat(timespec="seconds")
                step_meta["duration_sec"] = round(time.monotonic() - t0, 3)
                on_step_error(idx, step, str(exc), step_meta)
                return "error"

            if step_type == "home":
                post_sleep = 1.5
            elif step_type == "target":
                post_sleep = 1.5
            elif step_type == "gripper":
                post_sleep = 0.8
            else:
                post_sleep = 0.0

            waited = 0.0
            while waited < post_sleep:
                if should_stop():
                    return "stopped"
                dt = min(0.1, post_sleep - waited)
                time.sleep(dt)
                waited += dt
        return "done"

