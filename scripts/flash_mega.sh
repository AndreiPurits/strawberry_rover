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
PORT="${MEGA_PORT:-/dev/ttyUSB1}"

CLI="$REPO_ROOT/.local_prefix/bin/arduino-cli"
if [ ! -x "$CLI" ]; then
  CLI="$(command -v arduino-cli || true)"
fi

PIO="${PLATFORMIO_BIN:-$REPO_ROOT/.venv_cuda/bin/pio}"
use_pio=false

if [ -z "$CLI" ] || ! "$CLI" core list 2>/dev/null | grep -q 'arduino:avr'; then
  if [ -x "$PIO" ]; then
    echo "WARN: arduino-cli/avr unavailable — using PlatformIO ($PIO)"
    use_pio=true
  elif [ -z "$CLI" ]; then
    echo "ERROR: need arduino-cli or PlatformIO (.venv_cuda/bin/pio)"
    exit 1
  fi
fi

if ! groups | grep -q dialout; then
  echo "ERROR: not in dialout group. Run: newgrp dialout"
  exit 1
fi

if [ ! -e "$PORT" ]; then
  echo "ERROR: port $PORT not found. Set MEGA_PORT=/dev/ttyACM0 if needed."
  exit 1
fi

if [ "$use_pio" = false ]; then
  echo "==> Compile $SKETCH_DIR"
  "$CLI" compile --fqbn "$FQBN" "$SKETCH_DIR"

  echo "==> Upload to $PORT (Mega may reset)"
  "$CLI" upload --fqbn "$FQBN" --port "$PORT" "$SKETCH_DIR"
else
  echo "==> PlatformIO upload $SKETCH_DIR -> $PORT"
  (cd "$SKETCH_DIR" && MEGA_PORT="$PORT" "$PIO" run -t upload)
fi

echo "==> Done. Verify: ./scripts/check_mega.sh"
