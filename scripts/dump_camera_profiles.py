#!/usr/bin/env python3
import os
import subprocess
import sys


def main() -> int:
    """
    Dump Orbbec Gemini stream profiles using the vendor SDK (libobsensor) via the
    orbbec_camera package tool. This is the only reliable way to get the *actual*
    device profile list (no log parsing, no guessing).
    """
    exe = ["ros2", "run", "orbbec_camera", "dump_camera_profiles_node"]

    env = os.environ.copy()
    # If called from desktop, DISPLAY may be missing; keep it consistent with other launchers.
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))

    try:
        p = subprocess.run(exe, env=env, check=False)
    except FileNotFoundError:
        sys.stderr.write("ERROR: `ros2` not found in PATH. Did you source ROS2 Foxy?\n")
        return 2

    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

