#!/usr/bin/env python3
"""Offline-only RoArm-M3 hand-eye calibration. Never sends robot commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.roarm_calibration.handeye import solve_handeye  # noqa: E402
from pipelines.roarm_calibration.io import load_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=8)
    parser.add_argument("--min-validation", type=int, default=3)
    args = parser.parse_args()

    observations = load_dataset(args.dataset)
    result = solve_handeye(
        observations, min_train=args.min_train, min_validation=args.min_validation
    )
    payload = {"status": result.status, "method": result.method, "report": result.report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.status, "method": result.method, "KIN_GATE": "FAIL"}))
    # A solved transform is still not motion-authorized until independent
    # known-point validation is entered by a later, explicit procedure.
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
