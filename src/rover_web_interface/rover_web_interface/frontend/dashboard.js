/* global window, document, fetch, WebSocket, atob */

const stateStore = {
  state: null,
  roverScreen: { x: 0, y: 0 },
  ws: null,
  keysDown: new Set(),
  lastManualCmd: { linear_x: 0.0, angular_z: 0.0 },
  gamepad: {
    connected: false,
    index: null,
    axes: { leftX: 0.0, leftY: 0.0, rightX: 0.0 },
    active: false,
  },
  lastControlSource: "n/a",
  gamepadLoopHandle: null,
};

const GAMEPAD_DEADZONE = 0.18;
const LINEAR_SPEED_MAX = 0.55;
const ANGULAR_SPEED_MAX = 1.1;

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function applyDeadzone(value, deadzone) {
  if (Math.abs(value) < deadzone) return 0.0;
  const sign = value >= 0 ? 1 : -1;
  const scaled = (Math.abs(value) - deadzone) / (1.0 - deadzone);
  return sign * scaled;
}

function worldToCanvas(x, y, bounds, canvas) {
  const xNorm = (x - bounds.minX) / Math.max(1e-6, bounds.maxX - bounds.minX);
  const yNorm = (y - bounds.minY) / Math.max(1e-6, bounds.maxY - bounds.minY);
  return {
    x: xNorm * canvas.width,
    y: canvas.height - (yNorm * canvas.height),
  };
}

function computeBounds(state) {
  const beds = state?.field?.beds || [];
  if (!beds.length) {
    return { minX: -12, maxX: 12, minY: -1, maxY: 9 };
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  beds.forEach((b) => {
    minX = Math.min(minX, b.x - b.length / 2);
    maxX = Math.max(maxX, b.x + b.length / 2);
    minY = Math.min(minY, b.y - b.width / 2);
    maxY = Math.max(maxY, b.y + b.width / 2);
  });
  return { minX: minX - 1.0, maxX: maxX + 1.0, minY: minY - 1.0, maxY: maxY + 1.0 };
}

function drawFieldMap(canvasId, state, centerOnRover = false) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !state) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const bounds = computeBounds(state);
  if (centerOnRover && state.rover?.pose) {
    const cx = state.rover.pose.x;
    const cy = state.rover.pose.y;
    const w = 16;
    const h = 8;
    bounds.minX = cx - w / 2;
    bounds.maxX = cx + w / 2;
    bounds.minY = cy - h / 2;
    bounds.maxY = cy + h / 2;
  }

  const showSensors = document.getElementById("layerSensors")?.checked ?? true;
  const showAnalytics = document.getElementById("layerAnalytics")?.checked ?? true;
  const showTrail = document.getElementById("layerTrail")?.checked ?? true;

  // Beds layer
  (state.field?.beds || []).forEach((bed) => {
    const p0 = worldToCanvas(bed.x - bed.length / 2, bed.y - bed.width / 2, bounds, canvas);
    const p1 = worldToCanvas(bed.x + bed.length / 2, bed.y + bed.width / 2, bounds, canvas);
    const x = Math.min(p0.x, p1.x);
    const y = Math.min(p0.y, p1.y);
    const w = Math.abs(p1.x - p0.x);
    const h = Math.abs(p1.y - p0.y);
    ctx.fillStyle = "rgba(40, 130, 40, 0.55)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(130, 190, 130, 0.8)";
    ctx.strokeRect(x, y, w, h);
  });

  // Sensor layer (mock)
  if (showSensors) {
    (state.field?.sensor_grid || []).forEach((s) => {
      const p = worldToCanvas(s.x, s.y, bounds, canvas);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(120, 190, 255, 0.9)";
      ctx.fill();
    });
  }

  // Route trail
  if (showTrail) {
    const trail = state.rover?.route_trail || [];
    if (trail.length > 1) {
      ctx.beginPath();
      trail.forEach((pt, i) => {
        const p = worldToCanvas(pt.x, pt.y, bounds, canvas);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.strokeStyle = "rgba(255, 220, 120, 0.85)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Rover icon
  const rover = state.rover || {};
  if (rover.pose) {
    const p = worldToCanvas(rover.pose.x, rover.pose.y, bounds, canvas);
    stateStore.roverScreen = p;
    const heading = rover.heading_rad || 0;
    const len = 18;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 140, 70, 0.95)";
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + len * Math.cos(heading), p.y - len * Math.sin(heading));
    ctx.strokeStyle = "rgba(255, 235, 150, 1.0)";
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  if (showAnalytics) {
    const a = state.analytics || {};
    document.getElementById("fieldStats").textContent =
      `Analytics: berries/day=${(a.berries_collected_today || 0).toFixed(1)} | ` +
      `work time=${(a.working_time_hours || 0).toFixed(2)}h | ` +
      `energy=${(a.energy_consumption_kwh || 0).toFixed(2)}kWh | ` +
      `avg speed=${(a.avg_harvest_speed_berries_per_hour || 0).toFixed(1)}/h`;
  }
}

function controlSnapshot(state) {
  const c = state?.control || {};
  return {
    mode: c.mode || "auto",
    started: Boolean(c.started),
    manualAllowed: Boolean(c.manual_allowed),
    lastCommand: c.last_command || {},
  };
}

function renderControl(state) {
  const c = controlSnapshot(state);
  const modeClass = c.mode === "manual" ? "mode-manual" : "mode-auto";
  const runClass = c.started ? "status-running" : "status-stopped";
  const modeLabel = c.mode === "manual" ? "manual" : "auto";
  const runLabel = c.started ? "running" : "stopped";
  const last = c.lastCommand || {};
  stateStore.lastControlSource = last.source || "n/a";
  const statusEl = document.getElementById("controlStatus");
  if (statusEl) {
    statusEl.innerHTML = `
      <span class="badge ${modeClass}">Mode: ${modeLabel}</span>
      <span class="badge ${runClass}">Status: ${runLabel}</span><br>
      last cmd: vx=${Number(last.linear_x || 0).toFixed(2)},
      wz=${Number(last.angular_z || 0).toFixed(2)},
      src=${last.source || "n/a"}
    `;
  }
  const modeBtn = document.getElementById("btnModeToggle");
  if (modeBtn) {
    modeBtn.textContent = c.mode === "manual" ? "Switch to Auto" : "Switch to Manual";
  }
  const sourceEl = document.getElementById("controlSource");
  if (sourceEl) {
    sourceEl.textContent = `Control source: ${stateStore.lastControlSource}`;
  }
}

function renderGamepadStatus() {
  const statusEl = document.getElementById("gamepadStatus");
  if (!statusEl) return;
  const gp = stateStore.gamepad;
  if (!gp.connected) {
    statusEl.textContent = "Gamepad: disconnected";
    return;
  }
  statusEl.textContent = `Gamepad: connected | lx=${gp.axes.leftX.toFixed(2)} ly=${gp.axes.leftY.toFixed(2)} rx=${gp.axes.rightX.toFixed(2)} | active=${gp.active}`;
}

function renderTelemetry(state) {
  const t = state.telemetry || {};
  const r = state.rover || {};
  const p = r.pose || {};
  document.getElementById("telemetryPanel").innerHTML = `
    position: x=${(p.x || 0).toFixed(2)}, y=${(p.y || 0).toFixed(2)}<br>
    heading: ${(r.heading_deg || 0).toFixed(1)} deg<br>
    current row: ${t.current_row || 0}<br>
    speed: ${(t.speed_mps || 0).toFixed(2)} m/s<br>
    angular velocity: ${(t.angular_velocity_rps || 0).toFixed(2)} rad/s<br>
    battery (mock): ${(t.battery_level_pct || 0).toFixed(1)} %<br>
    avg energy (mock): ${(t.avg_energy_consumption_wh || 0).toFixed(1)} Wh<br>
    battery temp (mock): ${(t.battery_temperature_c || 0).toFixed(1)} C<br>
    berries collected (mock): ${(t.berries_collected || 0).toFixed(1)}<br>
    collection speed/h (mock): ${(t.collection_speed_per_hour || 0).toFixed(1)}<br>
    berry density (mock): ${(t.berry_density_estimate || 0).toFixed(2)}<br>
    nav state: ${r.nav_state || "unknown"}
  `;
}

function drawBgrToCanvas(canvas, frame) {
  if (!canvas || !frame || !frame.data_b64) return;
  const ctx = canvas.getContext("2d");
  const width = frame.width;
  const height = frame.height;
  const raw = atob(frame.data_b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  const imageData = ctx.createImageData(width, height);
  let si = 0;
  let di = 0;
  while (si + 2 < bytes.length && di + 3 < imageData.data.length) {
    const b = bytes[si];
    const g = bytes[si + 1];
    const r = bytes[si + 2];
    imageData.data[di] = r;
    imageData.data[di + 1] = g;
    imageData.data[di + 2] = b;
    imageData.data[di + 3] = 255;
    si += 3;
    di += 4;
  }
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.putImageData(imageData, 0, 0);
}

async function fetchCamera(name, canvasId) {
  const res = await fetch(`/api/cameras/${name}`);
  const frame = await res.json();
  drawBgrToCanvas(document.getElementById(canvasId), frame);
}

async function fetchScan() {
  const res = await fetch("/api/scan");
  const data = await res.json();
  const count = (data.points || []).length;
  document.getElementById("scanPanel").textContent = `Front LiDAR (/scan): ${count} sampled points`;
}

async function postControl(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : "{}",
  });
  return res.json();
}

function sendWsControl(payload) {
  const ws = stateStore.ws;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "control", data: payload }));
    return true;
  }
  return false;
}

function canManualControl() {
  const c = controlSnapshot(stateStore.state);
  return c.mode === "manual" && c.started && c.manualAllowed;
}

function currentKeyboardCommand() {
  const keys = stateStore.keysDown;
  let linear = 0.0;
  let angular = 0.0;

  if (keys.has("KeyW")) linear += 0.45;
  if (keys.has("KeyS")) linear -= 0.45;
  if (keys.has("KeyA")) angular += 0.9;
  if (keys.has("KeyD")) angular -= 0.9;

  return { linear_x: linear, angular_z: angular };
}

function sendManualKeyboardCommand() {
  if (!canManualControl()) return;
  const cmd = currentKeyboardCommand();
  stateStore.lastManualCmd = cmd;
  if (!sendWsControl({ action: "command", command: cmd, source: "keyboard" })) {
    // Fallback if WS reconnect is in progress.
    postControl("/api/control/command", { command: cmd, source: "keyboard" }).catch(() => {});
  }
}

function stopManualKeyboard(source = "keyboard_keyup") {
  stateStore.lastManualCmd = { linear_x: 0.0, angular_z: 0.0 };
  if (!canManualControl()) return;
  if (!sendWsControl({ action: "zero", source })) {
    postControl("/api/control/command", {
      command: { linear_x: 0.0, angular_z: 0.0 },
      source,
    }).catch(() => {});
  }
}

function currentGamepadCommand() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let idx = stateStore.gamepad.index;
  if (idx === null) {
    for (let i = 0; i < pads.length; i += 1) {
      if (pads[i]) {
        idx = i;
        stateStore.gamepad.index = i;
        break;
      }
    }
  }
  const pad = idx !== null ? pads[idx] : null;
  if (!pad) {
    stateStore.gamepad.connected = false;
    stateStore.gamepad.active = false;
    renderGamepadStatus();
    return { connected: false, command: { linear_x: 0.0, angular_z: 0.0 } };
  }

  const rawLeftX = Number(pad.axes[0] || 0.0);
  const rawLeftY = Number(pad.axes[1] || 0.0);
  const rawRightX = Number(pad.axes[2] || 0.0);
  const leftX = applyDeadzone(rawLeftX, GAMEPAD_DEADZONE);
  const leftY = applyDeadzone(rawLeftY, GAMEPAD_DEADZONE);
  const rightX = applyDeadzone(rawRightX, GAMEPAD_DEADZONE);
  const turn = Math.abs(rightX) > 1e-3 ? rightX : leftX;
  const linear = clamp(-leftY * LINEAR_SPEED_MAX, -LINEAR_SPEED_MAX, LINEAR_SPEED_MAX);
  const angular = clamp(-turn * ANGULAR_SPEED_MAX, -ANGULAR_SPEED_MAX, ANGULAR_SPEED_MAX);

  stateStore.gamepad.connected = true;
  stateStore.gamepad.axes = { leftX, leftY, rightX };
  stateStore.gamepad.active = Math.abs(linear) > 1e-3 || Math.abs(angular) > 1e-3;
  renderGamepadStatus();

  return {
    connected: true,
    command: { linear_x: linear, angular_z: angular },
  };
}

function sendManualJoystickCommand(cmd) {
  if (!canManualControl()) return;
  stateStore.lastManualCmd = cmd;
  if (!sendWsControl({ action: "command", command: cmd, source: "joystick" })) {
    postControl("/api/control/command", { command: cmd, source: "joystick" }).catch(() => {});
  }
}

function stopManualJoystick(source = "joystick_disconnect") {
  if (!canManualControl()) return;
  if (!sendWsControl({ action: "zero", source })) {
    postControl("/api/control/command", {
      command: { linear_x: 0.0, angular_z: 0.0 },
      source,
    }).catch(() => {});
  }
}

function gamepadLoop() {
  const snap = controlSnapshot(stateStore.state);
  if (snap.mode === "manual" && snap.started) {
    const gp = currentGamepadCommand();
    if (gp.connected) {
      if (stateStore.gamepad.active) {
        sendManualJoystickCommand(gp.command);
      } else if (stateStore.lastControlSource === "joystick") {
        stopManualJoystick("joystick_idle");
      }
    } else if (stateStore.lastControlSource === "joystick") {
      stopManualJoystick("joystick_disconnect");
    }
  } else {
    // Keep status panel fresh even if manual input disabled.
    currentGamepadCommand();
  }
  stateStore.gamepadLoopHandle = window.requestAnimationFrame(gamepadLoop);
}

function renderArmCameraMocks(state) {
  const container = document.getElementById("armCams");
  if (!container) return;
  const cams = state.telemetry?.arm_cameras || [];
  container.innerHTML = "";
  cams.forEach((cam) => {
    const d = document.createElement("div");
    d.className = "mini";
    d.textContent = `${cam.label} (${cam.status})`;
    container.appendChild(d);
  });
}

function onState(state) {
  stateStore.state = state;
  drawFieldMap("fieldCanvas", state, false);
  drawFieldMap("miniMapCanvas", state, true);
  renderTelemetry(state);
  renderArmCameraMocks(state);
  renderControl(state);
}

function connectWs() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  stateStore.ws = ws;
  ws.onopen = () => { ws.send("hello"); };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "state") onState(msg.data);
    if (msg.type === "control_ack" && stateStore.state) {
      stateStore.state.control = msg.data;
      renderControl(stateStore.state);
    }
  };
  ws.onclose = () => {
    stateStore.ws = null;
    setTimeout(connectWs, 1500);
  };
}

function initUi() {
  window.addEventListener("gamepadconnected", (evt) => {
    stateStore.gamepad.connected = true;
    stateStore.gamepad.index = evt.gamepad.index;
    stateStore.gamepad.active = false;
    renderGamepadStatus();
  });
  window.addEventListener("gamepaddisconnected", (evt) => {
    if (stateStore.gamepad.index === evt.gamepad.index) {
      stateStore.gamepad.connected = false;
      stateStore.gamepad.index = null;
      stateStore.gamepad.active = false;
      renderGamepadStatus();
      if (stateStore.lastControlSource === "joystick") {
        stopManualJoystick("joystick_disconnect");
      }
    }
  });

  document.getElementById("btnStart").addEventListener("click", async () => {
    await postControl("/api/control/start");
  });
  document.getElementById("btnStop").addEventListener("click", async () => {
    stateStore.keysDown.clear();
    stateStore.gamepad.active = false;
    await postControl("/api/control/stop");
  });
  document.getElementById("btnModeToggle").addEventListener("click", async () => {
    const mode = controlSnapshot(stateStore.state).mode === "manual" ? "auto" : "manual";
    stateStore.keysDown.clear();
    if (mode === "auto" && stateStore.lastControlSource === "joystick") {
      stopManualJoystick("mode_auto");
    }
    await postControl("/api/control/mode", { mode });
  });

  window.addEventListener("keydown", (evt) => {
    if (["INPUT", "TEXTAREA"].includes((evt.target?.tagName || "").toUpperCase())) return;
    if (evt.code === "Space") {
      evt.preventDefault();
      stateStore.keysDown.clear();
      stopManualKeyboard("keyboard_space");
      return;
    }
    if (!["KeyW", "KeyA", "KeyS", "KeyD"].includes(evt.code)) return;
    evt.preventDefault();
    stateStore.keysDown.add(evt.code);
    sendManualKeyboardCommand();
  });

  window.addEventListener("keyup", (evt) => {
    if (!["KeyW", "KeyA", "KeyS", "KeyD"].includes(evt.code)) return;
    evt.preventDefault();
    stateStore.keysDown.delete(evt.code);
    const cmd = currentKeyboardCommand();
    if (Math.abs(cmd.linear_x) < 1e-6 && Math.abs(cmd.angular_z) < 1e-6) {
      stopManualKeyboard("keyboard_keyup");
    } else {
      sendManualKeyboardCommand();
    }
  });

  window.addEventListener("blur", () => {
    stateStore.keysDown.clear();
    stopManualKeyboard("window_blur");
    if (stateStore.lastControlSource === "joystick") {
      stopManualJoystick("window_blur");
    }
  });

  document.getElementById("openDetail").addEventListener("click", () => {
    const panel = document.getElementById("detailPanel");
    panel.style.display = panel.style.display === "none" ? "block" : "none";
  });
  document.getElementById("fieldCanvas").addEventListener("click", (evt) => {
    const c = evt.currentTarget;
    const rect = c.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) * c.width) / rect.width;
    const y = ((evt.clientY - rect.top) * c.height) / rect.height;
    const dx = x - stateStore.roverScreen.x;
    const dy = y - stateStore.roverScreen.y;
    if ((dx * dx) + (dy * dy) < 18 * 18) {
      document.getElementById("detailPanel").style.display = "block";
    }
  });
  renderGamepadStatus();
}

function startPolling() {
  setInterval(() => fetchCamera("front", "camFront"), 1000);
  setInterval(() => fetchCamera("bottom", "camBottom"), 1000);
  setInterval(() => fetchCamera("stereo", "camStereo"), 1000);
  setInterval(fetchScan, 1200);
}

async function bootstrap() {
  initUi();
  connectWs();
  startPolling();
  gamepadLoop();
  const initial = await (await fetch("/api/state")).json();
  if (!initial.error) onState(initial);
}

bootstrap();
