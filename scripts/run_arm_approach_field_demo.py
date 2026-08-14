#!/usr/bin/env python3
"""Offline one-shot approach demo: plan and verify from recorded locks (no actuators)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pipelines.arm_approach import load_config  # noqa: E402
from pipelines.arm_approach.pipeline import ManipulatorApproachPipeline  # noqa: E402
from arm_approach_geometry import berry_error  # noqa: E402


def _sample_learned() -> dict:
    return {
        "schema": "roarm_berry_learned_v2",
        "demos": [
            {
                "source": "fixture",
                "success": True,
                "q_start": {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
                "berry_start": {"px": 410.0, "py": 220.0, "depth_m": 0.42, "conf": 0.9},
                "q_success": {"base": -0.12, "shoulder": 0.55, "elbow": 1.45},
                "berry_success": {"px": 320.0, "py": 230.0, "depth_m": 0.13, "conf": 0.88},
                "delta_q": {"base": -0.12, "shoulder": 0.15, "elbow": -0.15},
            }
        ],
        "local_jacobian": {
            "quality_ok": True,
            "matrix": [
                [0.0, 0.0, 900.0],
                [140.0, -80.0, 0.0],
                [-0.20, 0.12, 0.0],
            ],
        },
    }


def _sample_cases() -> list:
    return [
        {
            "name": "near_demo",
            "q_start": {"base": 0.02, "shoulder": 0.41, "elbow": 1.58},
            "berry_before": {"px": 400.0, "py": 222.0, "depth_m": 0.40, "conf": 0.91},
            "berry_after": {"px": 318.0, "py": 228.0, "depth_m": 0.13, "conf": 0.87},
            "q_end": {"base": -0.11, "shoulder": 0.54, "elbow": 1.46},
        },
        {
            "name": "no_progress",
            "q_start": {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
            "berry_before": {"px": 400.0, "py": 220.0, "depth_m": 0.40, "conf": 0.9},
            "berry_after": {"px": 398.0, "py": 221.0, "depth_m": 0.39, "conf": 0.9},
            "q_end": {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
        },
    ]


def main() -> int:
    cfg = load_config(REPO / "config" / "arm_approach_v1.yaml")
    pipe = ManipulatorApproachPipeline(cfg)
    learned = _sample_learned()
    out = REPO / "runs" / "arm_approach_v1"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in _sample_cases():
        plan_res = pipe.plan(case["q_start"], case["berry_before"], learned)
        plan = plan_res.plan
        q_target = plan.q_target if plan else case["q_start"]
        q_end = q_target if case["name"] == "near_demo" else case["q_end"]
        verify_res = pipe.verify_move(
            case["berry_before"],
            case["berry_after"],
            case["q_start"],
            q_target,
            q_end,
            expected_px=(plan.target_image["px"] if plan else None),
            expected_py=(plan.target_image["py"] if plan else None),
        )
        err = berry_error(case["berry_before"], plan.target_image if plan else {}).tolist()
        rec = {
            "name": case["name"],
            "plan": plan_res.to_dict(),
            "verify": verify_res.to_dict(),
            "error": err,
        }
        (out / f"{case['name']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary.append(
            {
                "name": case["name"],
                "plan_reason": plan_res.reason,
                "verify": verify_res.reason,
                "status": verify_res.status.value,
            }
        )
        print(case["name"], plan_res.reason, verify_res.reason)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUT", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
