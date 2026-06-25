#!/usr/bin/env bash
# Read-only RTK / GNSS diagnostics (does NOT move the rover).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AXM_PYTHON:-$REPO_ROOT/.venv_cuda/bin/python3}"
export PYTHONPATH="$REPO_ROOT/ops/axm-monitor/agent${PYTHONPATH:+:$PYTHONPATH}"

echo "=== RTK diagnostics (read-only) ==="
"$PYTHON" - <<'PY'
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops", "axm-monitor", "agent"))
from gnss_reader import gnss_snapshot, probe_rtk_baud, resolve_rtk_port

port = resolve_rtk_port(os.environ.get("RTK_PORT"))
print(f"port: {port} exists={os.path.exists(port)}")
if os.path.exists(port):
    baud = probe_rtk_baud(port)
    print(f"probed_baud: {baud}")
print("fleet-agent snapshot (if agent running, port may be busy):")
print(json.dumps(gnss_snapshot(), indent=2, ensure_ascii=False))
PY
