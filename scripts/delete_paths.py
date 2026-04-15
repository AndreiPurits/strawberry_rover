#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete a list of file paths (safety-scoped to this repo).")
    ap.add_argument("json_list", type=Path, help="JSON file containing an array of absolute paths")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root: Path = args.repo_root.resolve()
    allowed_prefixes = [
        (repo_root / "data" / "raw").resolve(),
        (repo_root / "data" / "normalized").resolve(),
    ]

    items = json.loads(args.json_list.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("JSON must be a list of paths")

    deleted = 0
    missing = 0
    rejected = 0
    for raw in items:
        try:
            p = Path(raw).resolve()
        except Exception:
            rejected += 1
            continue

        if not any(str(p).startswith(str(pref) + "/") or p == pref for pref in allowed_prefixes):
            rejected += 1
            continue

        if not p.exists():
            missing += 1
            continue

        if args.dry_run:
            deleted += 1
            continue

        if p.is_dir():
            # We never expect directories in the list
            rejected += 1
            continue

        p.unlink()
        deleted += 1

    print(f"deleted={deleted} missing={missing} rejected={rejected} total={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

