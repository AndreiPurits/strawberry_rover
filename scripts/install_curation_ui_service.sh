#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="${REPO_ROOT}/scripts/curation-ui.service"
UNIT_DST_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DST_DIR}/curation-ui.service"

mkdir -p "${UNIT_DST_DIR}"
cp -f "${UNIT_SRC}" "${UNIT_DST}"

systemctl --user daemon-reload
systemctl --user enable --now curation-ui.service

echo "Installed and started: curation-ui.service"
echo "Open: http://127.0.0.1:7860/?dataset=classifier"
