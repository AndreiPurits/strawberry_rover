"""Peduncle v3: calyx pose → OBB → LogReg association (TARGET / NO_ASSOC)."""

from .pipeline import PeduncleV3Pipeline, PeduncleV3Result, StageState
from .association import AssociationScorer

__all__ = [
    "PeduncleV3Pipeline",
    "PeduncleV3Result",
    "StageState",
    "AssociationScorer",
]
