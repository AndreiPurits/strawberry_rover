#!/usr/bin/env python3
"""Offline quantitative validation for the canonical RoArm-M3 model.

This program imports no arm transport code and cannot send servo commands.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipelines.roarm_m3_kinematics import (  # noqa: E402
    GEOMETRY,
    HARD_LIMITS,
    JointVector,
    Pose5,
    fk,
    ik,
    jacobian,
    jacobian_condition,
    pose_error,
)


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def validate_t105(fixture: Path) -> Dict[str, Any]:
    document = json.loads(fixture.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for sample in document["samples"]:
        fb = sample["feedback"]
        predicted = fk([fb["b"], fb["s"], fb["e"], fb["t"], fb["r"]])
        expected = Pose5(fb["x"], fb["y"], fb["z"], fb["tit"], fb["r"])
        error = pose_error(predicted, expected)
        rows.append(
            {
                "id": sample["id"],
                "error": {
                    "x_mm": float(error[0]),
                    "y_mm": float(error[1]),
                    "z_mm": float(error[2]),
                    "pitch_rad": float(error[3]),
                    "roll_rad": float(error[4]),
                    "xyz_norm_mm": float(np.linalg.norm(error[:3])),
                },
            }
        )
    unique_joint_states = {
        tuple(sample["feedback"][name] for name in ("b", "s", "e", "t", "r"))
        for sample in document["samples"]
    }
    return {
        "samples": len(rows),
        "unique_joint_states": len(unique_joint_states),
        "source_limitation": (
            "Existing repository logs save joint states but not paired firmware x/y/z/tit; "
            "Phase A therefore has one unique read-only T:105 pose (two identical replies)."
        ),
        "xyz_norm_mm": stats(row["error"]["xyz_norm_mm"] for row in rows),
        "x_abs_mm": stats(abs(row["error"]["x_mm"]) for row in rows),
        "y_abs_mm": stats(abs(row["error"]["y_mm"]) for row in rows),
        "z_abs_mm": stats(abs(row["error"]["z_mm"]) for row in rows),
        "pitch_abs_rad": stats(abs(row["error"]["pitch_rad"]) for row in rows),
        "roll_abs_rad": stats(abs(row["error"]["roll_rad"]) for row in rows),
        "rows": rows,
    }


def random_q(rng: np.random.Generator) -> JointVector:
    # Stay inside the soft/intersection envelope used by both firmware and URDF.
    return JointVector(
        base=float(rng.uniform(-1.20, 1.20)),
        shoulder=float(rng.uniform(-1.20, 1.20)),
        elbow=float(rng.uniform(0.40, 2.90)),
        wrist=float(rng.uniform(-1.20, 1.20)),
        roll=float(rng.uniform(-1.20, 1.20)),
    )


def validate_jacobian(rng: np.random.Generator, n: int, step: float) -> Dict[str, Any]:
    element_errors: List[float] = []
    per_sample_max: List[float] = []
    worst: Dict[str, Any] = {}
    for _ in range(n):
        q = random_q(rng).as_array()
        analytic = jacobian(q)
        numeric = np.zeros((5, 5), dtype=np.float64)
        for column in range(5):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[column] += step
            q_minus[column] -= step
            numeric[:, column] = pose_error(fk(q_minus), fk(q_plus)) / (2.0 * step)
        error = np.abs(analytic - numeric)
        sample_max = float(np.max(error))
        element_errors.extend(map(float, error.reshape(-1)))
        per_sample_max.append(sample_max)
        if not worst or sample_max > worst["max_abs_element_error"]:
            row, col = np.unravel_index(int(np.argmax(error)), error.shape)
            worst = {
                "q": list(map(float, q)),
                "row": int(row),
                "column": int(col),
                "analytic": float(analytic[row, col]),
                "numeric": float(numeric[row, col]),
                "max_abs_element_error": sample_max,
            }
    return {
        "samples": n,
        "central_difference_step_rad": step,
        "absolute_element_error": stats(element_errors),
        "per_sample_max_abs_error": stats(per_sample_max),
        "worst": worst,
    }


def validate_ik(rng: np.random.Generator, n: int) -> Dict[str, Any]:
    xyz_residuals: List[float] = []
    orientation_residuals: List[float] = []
    seed_joint_errors: List[float] = []
    conditions: List[float] = []
    candidate_counts: Counter = Counter()
    failures: Counter = Counter()
    hard_invalid_selected = 0
    worst: Dict[str, Any] = {}

    for _ in range(n):
        original = random_q(rng)
        target = fk(original)
        result = ik(target, seed_q=original)
        candidate_counts[len(result.candidates)] += 1
        if not result.ok:
            failures[result.reason] += 1
            continue
        selected = result.selected
        assert selected is not None
        residual = pose_error(fk(selected.q), target)
        xyz = float(np.linalg.norm(residual[:3]))
        orientation = float(np.linalg.norm(residual[3:]))
        q_error = float(np.linalg.norm(selected.q.as_array() - original.as_array()))
        xyz_residuals.append(xyz)
        orientation_residuals.append(orientation)
        seed_joint_errors.append(q_error)
        conditions.append(float(selected.jacobian_condition))
        if not selected.hard_ok:
            hard_invalid_selected += 1
        if not worst or xyz > worst["xyz_residual_mm"]:
            worst = {
                "q_original": original.as_dict(),
                "q_selected": selected.q.as_dict(),
                "xyz_residual_mm": xyz,
                "orientation_residual_rad": orientation,
                "joint_distance_to_seed_rad": q_error,
            }
    success = len(xyz_residuals)
    return {
        "samples": n,
        "success": success,
        "failure": n - success,
        "failure_reasons": dict(failures),
        "candidate_count_distribution": {str(k): v for k, v in sorted(candidate_counts.items())},
        "selected_outside_hard_limits": hard_invalid_selected,
        "xyz_residual_mm": stats(xyz_residuals),
        "orientation_residual_rad": stats(orientation_residuals),
        "joint_distance_to_seed_rad": stats(seed_joint_errors),
        "selected_condition": stats(conditions),
        "worst": worst,
    }


def scan_conditioning(rng: np.random.Generator, n: int) -> Dict[str, Any]:
    conditions: List[float] = []
    worst_rows: List[Dict[str, Any]] = []
    for _ in range(n):
        q = random_q(rng)
        cond = jacobian_condition(q)
        conditions.append(cond)
        row = {"condition": float(cond), "q": q.as_dict(), "pose": fk(q).as_dict()}
        if len(worst_rows) < 10:
            worst_rows.append(row)
            worst_rows.sort(key=lambda value: value["condition"], reverse=True)
        elif cond > worst_rows[-1]["condition"]:
            worst_rows[-1] = row
            worst_rows.sort(key=lambda value: value["condition"], reverse=True)
    return {
        "samples": n,
        "translation_normalization_mm": GEOMETRY.max_reach_mm,
        "condition_number": stats(conditions),
        "analytic_singular_sets": [
            "rho(q)=0 (base rotation cannot create Cartesian translation)",
            "sin(elbow + d3 - d2)=0 (main planar links collinear)",
        ],
        "exact_elbow_singularities_rad": [GEOMETRY.d2_rad - GEOMETRY.d3_rad],
        "operational_hard_elbow_range_rad": list(HARD_LIMITS["elbow"]),
        "worst_samples": worst_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REPO / "tests/data/roarm_m3_t105_samples.json",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--jacobian-samples", type=int, default=2000)
    parser.add_argument("--ik-samples", type=int, default=10000)
    parser.add_argument("--conditioning-samples", type=int, default=50000)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    report = {
        "schema": "roarm_m3_phase_a_validation_v1",
        "offline_only": True,
        "seed": args.seed,
        "geometry": {
            "l2_mm": GEOMETRY.l2_mm,
            "d2_rad": GEOMETRY.d2_rad,
            "l3_mm": GEOMETRY.l3_mm,
            "d3_rad": GEOMETRY.d3_rad,
            "tool_mm": GEOMETRY.tool_mm,
            "tool_offset_rad": GEOMETRY.tool_offset_rad,
        },
        "t105_fk": validate_t105(args.fixture),
        "jacobian_finite_difference": validate_jacobian(
            rng, args.jacobian_samples, args.finite_difference_step
        ),
        "ik_fk_round_trip": validate_ik(rng, args.ik_samples),
        "conditioning_scan": scan_conditioning(rng, args.conditioning_samples),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
