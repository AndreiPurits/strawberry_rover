"""Peduncle grasp association v1 — types and config."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


Point2D = Tuple[float, float]
BBoxXYXY = Tuple[float, float, float, float]


class GraspState(str, Enum):
    SEARCH_BERRIES = "SEARCH_BERRIES"
    SELECT_TARGET = "SELECT_TARGET"
    DETECT_OR_ESTIMATE_CALYX = "DETECT_OR_ESTIMATE_CALYX"
    PLAN_APPROACH = "PLAN_APPROACH"
    APPROACH_AND_TRACK = "APPROACH_AND_TRACK"
    CAPTURE_CLOSEUP = "CAPTURE_CLOSEUP"
    DETECT_PEDUNCLE_LOCAL = "DETECT_PEDUNCLE_LOCAL"
    ASSOCIATE = "ASSOCIATE"
    CHANGE_VIEWPOINT = "CHANGE_VIEWPOINT"
    VERIFY = "VERIFY"
    READY_TO_CUT = "READY_TO_CUT"
    REJECT_TARGET = "REJECT_TARGET"


class PerceptionStatus(str, Enum):
    OK = "OK"
    NEED_NEW_VIEW = "NEED_NEW_VIEW"
    REJECT = "REJECT"
    INCOMPLETE = "INCOMPLETE"


@dataclass
class CalyxEstimate:
    point_2d: Point2D
    confidence: float
    source: str  # model | geometry | unknown
    mask: Optional[Any] = None  # np.ndarray or None


@dataclass
class PeduncleCandidateFeatures:
    model_confidence: float
    base_to_calyx_distance: float
    normalized_base_distance: float
    calyx_contact: Optional[float]
    direction_consistency: Optional[float]
    continuity: Optional[float]
    depth_consistency: Optional[float]
    leaf_occlusion: Optional[float]
    crossing_penalty: Optional[float]
    obb_length_norm: float


@dataclass
class PeduncleCandidate:
    points: List[Point2D]
    conf: float
    base: Point2D
    tip: Point2D
    features: PeduncleCandidateFeatures
    score: float
    available_weight_sum: float


@dataclass
class CutProposal:
    point_2d: Point2D
    point_3d: Optional[Tuple[float, float, float]]
    confidence: float
    clearance_ok: bool
    reason: str


@dataclass
class BerryTarget:
    track_id: int
    bbox: BBoxXYXY
    mask: Optional[Any]
    center_2d: Point2D
    center_3d: Optional[Tuple[float, float, float]]
    berry_height_px: float
    ripeness: Optional[str]
    confidence: float
    occlusion_score: Optional[float]
    calyx: Optional[CalyxEstimate] = None
    peduncle_candidates: List[PeduncleCandidate] = field(default_factory=list)
    selected_peduncle: Optional[PeduncleCandidate] = None
    association_confidence: float = 0.0
    cut: Optional[CutProposal] = None


@dataclass
class PerceptionResult:
    status: PerceptionStatus
    state: GraspState
    reason: str
    suggested_view: Optional[str] = None  # LEFT|RIGHT|UP|DOWN
    berry: Optional[BerryTarget] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["state"] = self.state.value
        # drop non-JSON masks
        berry = d.get("berry")
        if isinstance(berry, dict):
            berry["mask"] = None if berry.get("mask") is None else "present"
            calyx = berry.get("calyx")
            if isinstance(calyx, dict) and calyx.get("mask") is not None:
                calyx["mask"] = "present"
        return d


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "peduncle_grasp_v1.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"invalid config: {path}")
    return cfg
