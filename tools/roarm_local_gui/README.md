# RoArm-M3 Local GUI (No ROS)

Standalone local diagnostic tool for manual RoArm-M3 testing.

UI is split into:
- `MAIN`: operator workflow (connection, system, gripper, target point, status)
- `ADVANCED`: joint/axis/direct controls and debug log
- `Sequence`: build/run step-by-step target/home sequences

## Requirements

- Python 3.8+
- `curl` available in system
- PyQt5:

```bash
python3 -m pip install PyQt5
```

## Run

```bash
cd tools/roarm_local_gui
python3 roarm_test_gui.py
```

Default IP is `192.168.1.87` and can be edited in the Connection block.

## Notes

- This tool is fully independent of ROS.
- Transport is raw URL + `curl` via subprocess:
  - Status: `GET http://<ip>/js`
  - Commands: `GET http://<ip>/js?json={...}` with `curl -g`
- XYZ reachability is warning-only (no hard blocking).
- Tested workspace history is stored in:
  - `tools/roarm_local_gui/tested_workspace.json`
- Sequences can be saved/loaded from:
  - `tools/roarm_local_gui/sequences.json`

