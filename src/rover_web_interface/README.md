## rover_web_interface

Minimal Stage 6 web visualization layer for Strawberry Rover.

### Backend

- FastAPI server with ROS2 bridge node (`rclpy`)
- REST endpoints:
  - `/api/health`
  - `/api/state`
  - `/api/control/state`
  - `/api/control`
  - `/api/control/start`
  - `/api/control/stop`
  - `/api/control/mode`
  - `/api/control/command`
  - `/api/routes`
  - `/api/routes/start`
  - `/api/routes/stop`
  - `/api/routes/save`
  - `/api/scan`
  - `/api/cameras/front`
  - `/api/cameras/bottom`
  - `/api/cameras/stereo`
- WebSocket endpoint:
  - `/ws` (live state stream + control commands)

Control payload contract (frontend -> backend):

```json
{
  "action": "start | stop | set_mode | command | zero",
  "mode": "manual | auto",
  "command": { "linear_x": 0.45, "angular_z": 0.9 },
  "source": "keyboard"
}
```

`/cmd_vel` publishing rules in Stage 6.1:
- manual commands are accepted only when `started=true` and `mode=manual`;
- stop always publishes zero `Twist`;
- switching to `auto` publishes zero `Twist`.
- keyboard source: `W/S/A/D`, `Space` (`source=keyboard`)
- joystick source: browser Gamepad API (`source=joystick`)

### Frontend

- Static dashboard page served by FastAPI:
  - `/`
- Features:
  - field map with beds
  - rover pose, heading, route trail
  - route recording overlay (draft + saved routes)
  - telemetry panel
  - camera panels (front/bottom/stereo)
  - single-screen no-scroll operator layout (map + cameras + controls + telemetry)
  - mock sensor + analytics layers
  - web control: Start/Stop, Manual/Auto
  - route control: Start Recording / Stop Recording / Save Route
  - keyboard control: `W/S/A/D`, `Space`
  - joystick control (Gamepad API): left stick Y = forward/backward, left/right stick X = turn
  - gamepad status + live axes + current control source

### Launch

```bash
ros2 launch rover_web_interface web_interface.launch.py
```

Optional host/port:

```bash
ros2 launch rover_web_interface web_interface.launch.py host:=0.0.0.0 port:=8080
```

Port conflict handling:

```bash
ros2 launch rover_web_interface web_interface.launch.py auto_cleanup_before_start:=true
```

`auto_cleanup_before_start` is enabled by default and cleans up stale `rover_web_server` process before launch.
