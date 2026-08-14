"""Manipulator approach v1 — types and config."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

Point2D = Tuple[float, float]
BBoxXYXY = Tuple[float, float, float, float]
JOINT_KEYS = ("base", "shoulder", "elbow")


class ApproachState(str, Enum):
    SEARCH = "SEARCH"
    LOCK = "LOCK"
    PLAN = "PLAN"
    VERIFY = "VERIFY"
    STANDOFF = "STANDOFF"
    CENTER = "CENTER"
    REACH = "REACH"
    ROLLBACK = "ROLLBACK"
    REJECT = "REJECT"


class ApproachStatus(str, Enum):
    OK = "OK"
    INCOMPLETE = "INCOMPLETE"
    REJECT = "REJECT"


@dataclass
class BerryLock:
    px: float
    py: float
    depth_m: float
    conf: float
    bbox: Optional[BBoxXYXY] = None


@dataclass
class ApproachPlan:
    q_start: Dict[str, float]
    q_target: Dict[str, float]
    delta_q: Dict[str, float]
    berry_lock: BerryLock
    prediction: Dict[str, float]
    target_image: Dict[str, float]
    reason: str
    confidence: float
    warnings: List[str] = field(default_factory=list)
    adjustments: List[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    ok: bool
    reasons: List[str]
    conf: float = 0.0
    in_frame: bool = False
    depth_ok: bool = False
    progress_ok: bool = False


@dataclass
class ApproachResult:
    status: ApproachStatus
    state: ApproachState
    reason: str
    lock: Optional[BerryLock] = None
    plan: Optional[ApproachPlan] = None
    verify: Optional[VerifyReport] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["state"] = self.state.value
        return d


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "arm_approach_v1.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"invalid config: {path}")
    return cfg
