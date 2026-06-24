"""Tank (skid-steer) drive: UI forward/turn -> track speeds or cmd_vel."""
from __future__ import annotations

from typing import Tuple


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def ui_to_tracks(
    forward: float,
    turn: float,
    *,
    forward_sign: float = -1.0,
    turn_sign: float = 1.0,
) -> Tuple[float, float]:
    """
    Web UI: W/S = forward, A/D = turn (A=+1 left, D=-1 right).
    Turn = left and right tracks opposite when forward=0.
    """
    f = _clamp(forward_sign * forward)
    t = _clamp(turn_sign * turn)
    left = _clamp(f - t)
    right = _clamp(f + t)
    return left, right


def ui_to_cmd_vel(
    forward: float,
    turn: float,
    *,
    forward_sign: float = -1.0,
    turn_sign: float = 1.0,
) -> Tuple[float, float]:
    """Same as ui_to_tracks but as (linear_x, angular_z) for /cmd_vel."""
    f = _clamp(forward_sign * forward)
    t = _clamp(turn_sign * turn)
    return f, t


def tracks_to_pwm(left: float, right: float) -> Tuple[int, int, int, int]:
    def to_us(v: float) -> int:
        return int(1500 + _clamp(v) * 500.0)

    fl = rl = to_us(left)
    fr = rr = to_us(right)
    return fl, fr, rl, rr


def ui_to_pwm(
    forward: float,
    turn: float,
    *,
    forward_sign: float = -1.0,
    turn_sign: float = 1.0,
) -> Tuple[int, int, int, int]:
    left, right = ui_to_tracks(forward, turn, forward_sign=forward_sign, turn_sign=turn_sign)
    return tracks_to_pwm(left, right)
