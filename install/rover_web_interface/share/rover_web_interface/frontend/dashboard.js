/* global window, document, fetch, WebSocket, atob */

const stateStore = {
  state: null,
  control: {
    mode: "auto",
    started: false,
    manualAllowed: false,
    lastCommand: {},
  },
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
  routeRecording: false,
  selectedRouteId: null,
  routeEditorRouteId: null,
};

const GAMEPAD_DEADZONE = 0.18;
const LINEAR_SPEED_MAX = 0.55;
const ANGULAR_SPEED_MAX = 1.1;
const MAX_LIDAR_RANGE = 6.0; // meters (fallback/manual scale)
const LIDAR_AUTO_SCALE = true;
const LIDAR_AUTO_PERCENTILE = 0.9;
const LIDAR_AUTO_MIN_RANGE = 2.0;
const LIDAR_AUTO_MAX_RANGE = 8.0;
const LIDAR_FILL_RATIO = 0.88;

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function applyDeadzone(value, deadzone) {
  if (Math.abs(value) < deadzone) return 0.0;
  const sign = value >= 0 ? 1 : -1;
  const scaled = (Math.abs(value) - deadzone) / (1.0 - deadzone);
  return sign * scaled;
}

function percentile(values, p) {
  if (!Array.isArray(values) || values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = clamp((sorted.length - 1) * p, 0, sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const t = pos - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * t;
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

  // Route trail (historical rover path)
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

  // Saved routes stay visible.
  const routes = state.routes || {};
  const activeRouteId = routes.active_route_id || null;
  (routes.saved_routes || []).forEach((route) => {
    const points = route.points || [];
    if (points.length < 2) return;
    ctx.beginPath();
    points.forEach((pt, i) => {
      const p = worldToCanvas(pt.x, pt.y, bounds, canvas);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    const isActive = route.id === activeRouteId;
    ctx.strokeStyle = isActive ? "rgba(80, 220, 255, 0.98)" : "rgba(172, 136, 255, 0.9)";
    ctx.lineWidth = isActive ? 4 : 2.2;
    ctx.stroke();
  });

  // Current recording route is highlighted on map in real time.
  const recordingPoints = routes.current_route?.points || [];
  if (recordingPoints.length > 1) {
    ctx.beginPath();
    recordingPoints.forEach((pt, i) => {
      const p = worldToCanvas(pt.x, pt.y, bounds, canvas);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.strokeStyle = routes.recording ? "rgba(255, 90, 90, 0.95)" : "rgba(90, 170, 255, 0.8)";
    ctx.lineWidth = 3;
    ctx.stroke();
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
    const fieldStatsEl = document.getElementById("fieldStats");
    if (fieldStatsEl) {
      fieldStatsEl.textContent =
        `Analytics: berries/day=${(a.berries_collected_today || 0).toFixed(1)} | ` +
        `work time=${(a.working_time_hours || 0).toFixed(2)}h | ` +
        `energy=${(a.energy_consumption_kwh || 0).toFixed(2)}kWh | ` +
        `avg speed=${(a.avg_harvest_speed_berries_per_hour || 0).toFixed(1)}/h`;
    }
  }
}

function controlSnapshot(state) {
  const c = state?.control || stateStore.control || {};
  return {
    mode: c.mode || "auto",
    started: Boolean(c.started),
    manualAllowed: Boolean(c.manual_allowed ?? c.manualAllowed),
    lastCommand: c.last_command || c.lastCommand || {},
  };
}

function renderControl(state) {
  const c = controlSnapshot(state);
  const last = c.lastCommand || {};
  stateStore.lastControlSource = last.source || "n/a";
  const modeBtn = document.getElementById("btnModeToggle");
  if (modeBtn) {
    modeBtn.textContent = c.mode === "manual" ? "Switch to Auto" : "Switch to Manual";
    modeBtn.classList.toggle("active", c.mode === "manual");
  }
  const startBtn = document.getElementById("btnStart");
  const stopBtn = document.getElementById("btnStop");
  if (startBtn) {
    startBtn.classList.toggle("active", c.started);
    startBtn.disabled = c.started;
  }
  if (stopBtn) {
    stopBtn.disabled = !c.started;
    stopBtn.classList.toggle("active", !c.started);
  }
}

function renderRouteUi(state) {
  const routes = state?.routes || {};
  const recording = Boolean(routes.recording);
  const currentCount = Number(routes.current_route?.point_count || 0);
  const savedCount = Number((routes.saved_routes || []).length);
  const activeRouteId = routes.active_route_id || null;
  if (activeRouteId) {
    stateStore.selectedRouteId = activeRouteId;
  }
  const statusEl = document.getElementById("routeRecordStatus");
  if (statusEl) {
    statusEl.textContent = recording ? `Recording: ON (${currentCount} pts)` : `Recording: idle (${currentCount} pts)`;
    statusEl.className = recording ? "recording-on" : "recording-off";
  }
  const routeStats = document.getElementById("routeStats");
  if (routeStats) {
    routeStats.textContent = `Routes: ${savedCount} saved | draft points: ${currentCount}`;
  }

  const btnStart = document.getElementById("btnRouteStart");
  const btnStop = document.getElementById("btnRouteStop");
  const btnSave = document.getElementById("btnRouteSave");
  if (btnStart) {
    btnStart.disabled = recording;
    btnStart.classList.toggle("active", recording);
  }
  if (btnStop) {
    btnStop.disabled = !recording;
  }
  if (btnSave) {
    btnSave.disabled = recording || currentCount < 2;
  }

  const routeList = document.getElementById("routeList");
  if (routeList) {
    const saved = routes.saved_routes || [];
    if (!saved.length) {
      routeList.textContent = "No saved routes";
    } else {
      routeList.innerHTML = "";
      saved.forEach((route) => {
        const d = document.createElement("div");
        const isActive = route.id === (routes.active_route_id || stateStore.selectedRouteId);
        d.className = `route-item ${isActive ? "active" : ""}`;
        const created = route.created_at ? new Date(route.created_at * 1000).toLocaleTimeString() : "n/a";
        d.innerHTML = `
          <div><strong>${route.name || route.id}</strong> <span class="tiny">(${route.id})</span></div>
          <div class="tiny">pts: ${route.point_count || 0} | created: ${created}</div>
        `;
        d.addEventListener("click", async () => {
          stateStore.selectedRouteId = route.id;
          await postJson("/api/routes/select", { route_id: route.id });
        });
        routeList.appendChild(d);
      });
    }
  }

  const activeRoute = routes.active_route || null;
  const routeNameInput = document.getElementById("routeNameInput");
  const routeNotesInput = document.getElementById("routeNotesInput");
  const routeRowsInput = document.getElementById("routeRowsInput");
  const routeSpacingInput = document.getElementById("routeSpacingInput");
  const selectedRouteInfo = document.getElementById("selectedRouteInfo");
  if (activeRoute) {
    const meta = activeRoute.metadata || {};
    if (stateStore.routeEditorRouteId !== activeRoute.id) {
      if (routeNameInput) routeNameInput.value = activeRoute.name || "";
      if (routeNotesInput) routeNotesInput.value = meta.notes || "";
      if (routeRowsInput) routeRowsInput.value = Number(meta.row_count ?? 0);
      if (routeSpacingInput) routeSpacingInput.value = Number(meta.spacing_m ?? 0).toFixed(2);
      stateStore.routeEditorRouteId = activeRoute.id;
    }
    if (selectedRouteInfo) {
      selectedRouteInfo.textContent = `Selected: ${activeRoute.name || activeRoute.id}`;
    }
  } else {
    stateStore.routeEditorRouteId = null;
    if (selectedRouteInfo) {
      selectedRouteInfo.textContent = "Selected: none";
    }
  }

  const editable = Boolean(activeRoute);
  const btnRouteSelect = document.getElementById("btnRouteSelect");
  const btnRouteDelete = document.getElementById("btnRouteDelete");
  const btnRouteTrim = document.getElementById("btnRouteTrim");
  const btnRouteRename = document.getElementById("btnRouteRename");
  const btnRouteMetaSave = document.getElementById("btnRouteMetaSave");
  const btnRowAdd = document.getElementById("btnRowAdd");
  const btnRowRemove = document.getElementById("btnRowRemove");
  if (btnRouteSelect) btnRouteSelect.disabled = !stateStore.selectedRouteId;
  if (btnRouteDelete) btnRouteDelete.disabled = !editable;
  if (btnRouteTrim) btnRouteTrim.disabled = !editable;
  if (btnRouteRename) btnRouteRename.disabled = !editable;
  if (btnRouteMetaSave) btnRouteMetaSave.disabled = !editable;
  if (btnRowAdd) btnRowAdd.disabled = !editable;
  if (btnRowRemove) btnRowRemove.disabled = !editable;
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
  const srcWidth = frame.width;
  const srcHeight = frame.height;
  const raw = atob(frame.data_b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  const imageData = ctx.createImageData(srcWidth, srcHeight);
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
  if (canvas.width === 0 || canvas.height === 0) return;

  // Draw with "contain" behavior to preserve the full frame.
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = srcWidth;
  frameCanvas.height = srcHeight;
  const frameCtx = frameCanvas.getContext("2d");
  frameCtx.putImageData(imageData, 0, 0);

  const destW = canvas.width;
  const destH = canvas.height;
  const srcAspect = srcWidth / Math.max(1, srcHeight);
  const destAspect = destW / Math.max(1, destH);

  let drawW = destW;
  let drawH = destH;
  if (srcAspect > destAspect) {
    drawH = Math.max(1, Math.round(destW / srcAspect));
  } else if (srcAspect < destAspect) {
    drawW = Math.max(1, Math.round(destH * srcAspect));
  }
  const dx = Math.round((destW - drawW) / 2);
  const dy = Math.round((destH - drawH) / 2);

  ctx.fillStyle = "#000";
  ctx.clearRect(0, 0, destW, destH);
  ctx.fillRect(0, 0, destW, destH);
  ctx.drawImage(frameCanvas, 0, 0, srcWidth, srcHeight, dx, dy, drawW, drawH);
}

async function fetchCamera(name, canvasId) {
  const res = await fetch(`/api/cameras/${name}`);
  const frame = await res.json();
  drawBgrToCanvas(document.getElementById(canvasId), frame);
}

async function fetchScan() {
  const res = await fetch("/api/scan");
  const data = await res.json();
  const scanCanvas = document.getElementById("scanCanvas");
  if (!scanCanvas) return;
  const ctx = scanCanvas.getContext("2d");
  const w = scanCanvas.width;
  const h = scanCanvas.height;
  ctx.fillStyle = "#090c10";
  ctx.fillRect(0, 0, w, h);
  const points = data.points || [];
  const validRanges = points
    .map((p) => Number(p.r))
    .filter((r) => Number.isFinite(r) && r > 0.0);
  let renderRange = MAX_LIDAR_RANGE;
  if (LIDAR_AUTO_SCALE && validRanges.length > 0) {
    const p90 = percentile(validRanges, LIDAR_AUTO_PERCENTILE);
    if (Number.isFinite(p90)) {
      renderRange = clamp(p90, LIDAR_AUTO_MIN_RANGE, LIDAR_AUTO_MAX_RANGE);
    }
  }

  const cx = Math.round(w * 0.5);
  const cy = Math.round(h * 0.5);
  const canvasRadius = Math.min(w, h) * 0.5 * LIDAR_FILL_RATIO;
  const scale = canvasRadius / Math.max(1e-6, renderRange);

  ctx.strokeStyle = "rgba(120,160,210,0.45)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    const r = (i * renderRange / 4.0) * scale;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(80, 255, 235, 0.98)";
  points.forEach((p) => {
    const a = Number(p.a || 0);
    const r = Number(p.r || 0);
    if (!Number.isFinite(r) || r <= 0.0) return;
    const rr = Math.min(r, renderRange); // clamp distant points to visualization boundary
    const x = cx + (Math.cos(a) * rr * scale);
    const y = cy - (Math.sin(a) * rr * scale);
    if (x >= 0 && x < w && y >= 0 && y < h) {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  ctx.fillStyle = "rgba(255, 210, 95, 0.98)";
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();
}

async function postControl(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : "{}",
  });
  return res.json();
}

function applyControlResponse(resp) {
  if (!resp || !resp.control) return;
  if (!stateStore.state) stateStore.state = {};
  stateStore.state.control = resp.control;
  stateStore.control = controlSnapshot(stateStore.state);
  renderControl(stateStore.state);
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
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
  const cmd = currentKeyboardCommand();
  stateStore.lastManualCmd = cmd;
  if (!sendWsControl({ action: "command", command: cmd, source: "keyboard" })) {
    // Fallback if WS reconnect is in progress.
    postControl("/api/control/command", { command: cmd, source: "keyboard" }).catch(() => {});
  }
}

function stopManualKeyboard(source = "keyboard_keyup") {
  stateStore.lastManualCmd = { linear_x: 0.0, angular_z: 0.0 };
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
  stateStore.lastManualCmd = cmd;
  if (!sendWsControl({ action: "command", command: cmd, source: "joystick" })) {
    postControl("/api/control/command", { command: cmd, source: "joystick" }).catch(() => {});
  }
}

function stopManualJoystick(source = "joystick_disconnect") {
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

function onState(state) {
  stateStore.state = state;
  stateStore.control = controlSnapshot(state);
  drawFieldMap("fieldCanvas", state, false);
  renderTelemetry(state);
  renderControl(state);
  renderRouteUi(state);
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
    const resp = await postControl("/api/control/start");
    applyControlResponse(resp);
  });
  document.getElementById("btnStop").addEventListener("click", async () => {
    stateStore.keysDown.clear();
    stateStore.gamepad.active = false;
    const resp = await postControl("/api/control/stop");
    applyControlResponse(resp);
  });
  document.getElementById("btnModeToggle").addEventListener("click", async () => {
    const mode = controlSnapshot(stateStore.state).mode === "manual" ? "auto" : "manual";
    stateStore.keysDown.clear();
    if (mode === "auto" && stateStore.lastControlSource === "joystick") {
      stopManualJoystick("mode_auto");
    }
    const resp = await postControl("/api/control/mode", { mode });
    applyControlResponse(resp);
  });
  document.getElementById("btnRouteStart").addEventListener("click", async () => {
    await postControl("/api/routes/start");
  });
  document.getElementById("btnRouteStop").addEventListener("click", async () => {
    await postControl("/api/routes/stop");
  });
  document.getElementById("btnRouteSave").addEventListener("click", async () => {
    await postControl("/api/routes/save");
  });
  document.getElementById("btnRouteSelect").addEventListener("click", async () => {
    if (!stateStore.selectedRouteId) return;
    await postJson("/api/routes/select", { route_id: stateStore.selectedRouteId });
  });
  document.getElementById("btnRouteDelete").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    await postJson("/api/routes/delete", { route_id: routeId });
  });
  document.getElementById("btnRouteRename").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    const activeName = stateStore.state?.routes?.active_route?.name || "";
    const name = (window.prompt("Route name", activeName) || "").trim();
    if (!routeId || !name) return;
    await postJson("/api/routes/rename", { route_id: routeId, name });
  });
  document.getElementById("btnRouteMetaSave").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    const activeMeta = stateStore.state?.routes?.active_route?.metadata || {};
    const notes = String(activeMeta.notes || "");
    const rowCount = Number(activeMeta.row_count || 0);
    const spacing = Number(activeMeta.spacing_m || 0);
    await postJson("/api/routes/metadata", {
      route_id: routeId,
      metadata: {
        notes,
        row_count: rowCount,
        spacing_m: spacing,
      },
    });
  });
  document.getElementById("btnRowAdd").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    const active = stateStore.state?.routes?.active_route || {};
    const rows = active.metadata?.rows || [];
    const idx = rows.length + 1;
    await postJson("/api/routes/rows/add", {
      route_id: routeId,
      row: {
        row_id: `row_${idx}`,
        label: `Row ${idx}`,
        length_m: Number(active.metadata?.bed_length_m || 22.0),
      },
    });
  });
  document.getElementById("btnRowRemove").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    const active = stateStore.state?.routes?.active_route || {};
    const rows = active.metadata?.rows || [];
    if (!rows.length) return;
    await postJson("/api/routes/rows/remove", {
      route_id: routeId,
      row_index: rows.length - 1,
    });
  });
  document.getElementById("btnRouteTrim").addEventListener("click", async () => {
    const routeId = stateStore.state?.routes?.active_route_id;
    if (!routeId) return;
    await postJson("/api/routes/trim_last", { route_id: routeId, points_to_trim: 20 });
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
