#!/usr/bin/env bash
# Flash Arduino Mega 2560 from terminal (no Arduino IDE).
# Usage:
#   bash scripts/install_arduino_cli.sh   # once
#   ./scripts/flash_mega.sh
#
# Requires: dialout group, Mega on /dev/ttyUSB0 (CH340) or /dev/ttyACM0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKETCH_DIR="$REPO_ROOT/arduino/MEGA_Rover_Gecoma"
FQBN="arduino:avr:mega"
PORT="${MEGA_PORT:-/dev/ttyUSB0}"

CLI="$REPO_ROOT/.local_prefix/bin/arduino-cli"
if [ ! -x "$CLI" ]; then
  CLI="$(command -v arduino-cli || true)"
fi
if [ -z "$CLI" ]; then
  echo "arduino-cli not found. Run: bash scripts/install_arduino_cli.sh"
  exit 1
fi

if ! groups | grep -q dialout; then
  echo "ERROR: not in dialout group. Run: newgrp dialout"
  exit 1
fi

if [ ! -e "$PORT" ]; then
  echo "ERROR: port $PORT not found. Set MEGA_PORT=/dev/ttyACM0 if needed."
  exit 1
fi

echo "==> Compile $SKETCH_DIR"
"$CLI" compile --fqbn "$FQBN" "$SKETCH_DIR"

echo "==> Upload to $PORT (Mega may reset)"
"$CLI" upload --fqbn "$FQBN" --port "$PORT" "$SKETCH_DIR"

echo "==> Done. Verify: ./scripts/check_mega.sh"
