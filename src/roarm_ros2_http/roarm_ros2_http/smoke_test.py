"""Minimal transport smoke test for RoArm-M3 without ROS2."""

import subprocess


def main() -> None:
    ip = "192.168.1.87"

    print("=== STATUS ===")
    status_url = f"http://{ip}/js"
    r = subprocess.run(
        ["curl", status_url],
        capture_output=True,
        text=True,
        timeout=5,
    )
    print(r.stdout)

    print("=== HOME ===")
    cmd = '{"T":100}'
    home_url = f"http://{ip}/js?json={cmd}"
    r = subprocess.run(
        ["curl", "-g", home_url],
        capture_output=True,
        text=True,
        timeout=5,
    )
    print(r.stdout)


if __name__ == "__main__":
    main()

