"""JSON interchange for immutable raw hand-eye observations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from .handeye import HandEyeObservation
from .se3 import RigidTransform


SCHEMA = "roarm_m3_handeye_observations_v1"


def observation_to_dict(observation: HandEyeObservation) -> Dict[str, object]:
    return {
        "id": observation.observation_id,
        "split": observation.split,
        "T_base_link5": observation.transform_base_link5.to_dict(),
        "T_camera_board": observation.transform_camera_board.to_dict(),
        "reprojection_rmse_px": observation.reprojection_rmse_px,
        "q_rad": None if observation.q_rad is None else list(observation.q_rad),
    }


def _transform(value: Mapping[str, object]) -> RigidTransform:
    if value.get("units") != "mm":
        raise ValueError("only millimetre transforms are accepted")
    return RigidTransform(
        str(value["parent"]), str(value["child"]), np.asarray(value["matrix"], dtype=np.float64)
    )


def observation_from_dict(value: Mapping[str, object]) -> HandEyeObservation:
    return HandEyeObservation(
        observation_id=str(value["id"]),
        split=str(value.get("split", "train")),
        transform_base_link5=_transform(value["T_base_link5"]),
        transform_camera_board=_transform(value["T_camera_board"]),
        reprojection_rmse_px=(
            None
            if value.get("reprojection_rmse_px") is None
            else float(value["reprojection_rmse_px"])
        ),
        q_rad=(
            None
            if value.get("q_rad") is None
            else tuple(float(item) for item in value["q_rad"])
        ),
    )


def load_dataset(path: Path) -> List[HandEyeObservation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported observation schema")
    return [observation_from_dict(item) for item in payload.get("observations", [])]


def save_dataset(path: Path, observations: Iterable[HandEyeObservation], metadata=None) -> None:
    payload = {
        "schema": SCHEMA,
        "metadata": dict(metadata or {}),
        "observations": [observation_to_dict(item) for item in observations],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
