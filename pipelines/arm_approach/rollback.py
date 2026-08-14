"""Rollback along recorded joint history until the berry reappears."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def push_history(
    history: List[Dict[str, Any]],
    q: Dict[str, float],
    berry: Optional[Dict[str, Any]],
    *,
    max_len: int = 32,
) -> None:
    history.append({"q": dict(q), "berry": dict(berry) if berry else None})
    if len(history) > max_len:
        del history[0 : len(history) - max_len]


def rollback_until_visible(
    history: List[Dict[str, Any]],
    *,
    visible: bool,
) -> Tuple[Optional[Dict[str, float]], str]:
    """If the target is not visible, return the previous recorded joints."""
    if visible:
        return None, "visible"
    if len(history) < 2:
        return None, "no_history"
    prev = history[-2]
    return dict(prev["q"]), "rollback"
