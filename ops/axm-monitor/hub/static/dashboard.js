const roverSelect = document.getElementById("rover-select");
const roverList = document.getElementById("rover-list");
const roverStatus = document.getElementById("rover-status");
const emptyState = document.getElementById("empty-state");
const roverPanel = document.getElementById("rover-panel");
const roarmPanel = document.getElementById("roarm-panel");
const panelTitle = document.getElementById("panel-title");
const panelSub = document.getElementById("panel-sub");
const panelBadge = document.getElementById("panel-badge");
const linkIndicator = document.getElementById("link-indicator");
const lidarGuardBadge = document.getElementById("lidar-guard-badge");
const lidarGuardBanner = document.getElementById("lidar-guard-banner");
const lidarBlock = document.getElementById("lidar-block");
const operatorBadge = document.getElementById("operator-badge");
const modeJoystick = document.getElementById("mode-joystick");
const modeAuto = document.getElementById("mode-auto");
const joystickPanel = document.getElementById("joystick-panel");
const autoPanel = document.getElementById("auto-panel");
const controlsRail = document.querySelector(".controls-rail");
const lidarLockOverlay = document.getElementById("lidar-lock-overlay");

const keysDown = new Set();
const keysLogical = new Set();
const DRIVE_KEY_CODES = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD",
  "KeyЦ", "KeyФ", "KeyЫ", "KeyВ",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "ShiftLeft", "ShiftRight", "KeyE", "Space",
]);
let lastFleet = [];
let selectedId = localStorage.getItem("axm_rover_id") || "";
let uiDriveMode = localStorage.getItem("axm_drive_mode") || "joystick";

const DRIVE_INTERVAL_MS = 40;
const GAMEPAD_DEADZONE = 0.12;

let lastBumperSpeedAt = 0;
let targetFwd = 0;
let targetTurn = 0;
let smoothFwd = 0;
let smoothTurn = 0;
let pendingOverride = false;
let fleetWs = null;

let lastDriveSentAt = 0;
let sessionStarted = false;
let operatorLocked = false;
let operatorRenewTimer = null;

let currentUser = window.__AXM_USER__ || localStorage.getItem("axm_saved_username") || "admin";

function localUsername() {
  return currentUser;
}

function speedStorageKey() {
  return `axm_speed_pct:${localUsername()}`;
}

function loadSpeedPct() {
  const saved = Number(localStorage.getItem(speedStorageKey()));
  if (Number.isFinite(saved) && saved >= 10 && saved <= 100) return saved;
  const legacy = Number(localStorage.getItem("axm_speed_pct"));
  if (Number.isFinite(legacy) && legacy >= 10 && legacy <= 100) return legacy;
  return 35;
}

function saveSpeedPct() {
  localStorage.setItem(speedStorageKey(), String(speedPct));
}

/** WS broadcast used to omit `you`; reconcile with logged-in user. */
function normalizeOperator(op) {
  if (!op) return { locked: false, holder: null, you: false };
  const holder = op.holder;
  if (op.locked && holder && holder === localUsername()) {
    return { ...op, you: true };
  }
  return op;
}
let lidarGuardActive = false;
let lidarOverrideActive = false;
let speedPct = loadSpeedPct();

const gamepadState = { index: null, connected: false, active: false, name: "" };
let mjpegUrl = "";
let stereoMjpegUrl = "";
let lastMegaSticky = {};
let webrtcPc = null;
let webrtcPollTimer = null;
let webrtcRoverId = "";
let webrtcIceServers = null;
const drivePad = document.getElementById("drive-pad");

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function applyDeadzone(v, dz) {
  return Math.abs(v) < dz ? 0 : v;
}

function fmtAgo(sec) {
  if (sec == null) return "—";
  if (sec < 5) return "сейчас";
  if (sec < 60) return `${Math.round(sec)} с`;
  return `${Math.round(sec / 60)} мин`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function getRover(id) {
  return lastFleet.find((r) => r.id === id) || null;
}

function selectedRover() {
  return selectedId ? getRover(selectedId) : null;
}

async function claimOperator() {
  if (!selectedId) return { ok: false };
  const res = await fetch(`/api/rovers/${encodeURIComponent(selectedId)}/claim`, {
    method: "POST",
    credentials: "same-origin",
  });
  if (res.status === 401) {
    location.href = "/login";
    return { ok: false };
  }
  if (res.status === 409) {
    const body = await res.json().catch(() => ({}));
    return { ok: false, conflict: true, detail: body.detail || "locked" };
  }
  if (!res.ok) return { ok: false };
  operatorLocked = true;
  return { ok: true, ...(await res.json()) };
}

async function releaseOperator() {
  if (!selectedId) return;
  if (operatorRenewTimer) {
    clearInterval(operatorRenewTimer);
    operatorRenewTimer = null;
  }
  await fetch(`/api/rovers/${encodeURIComponent(selectedId)}/release`, {
    method: "POST",
    credentials: "same-origin",
  });
  operatorLocked = false;
}

function startOperatorRenew() {
  if (operatorRenewTimer) clearInterval(operatorRenewTimer);
  operatorRenewTimer = setInterval(() => {
    if (sessionStarted && selectedId) claimOperator().catch(() => {});
  }, 45000);
}

async function sendCommand(action, params = {}) {
  if (!selectedId) return null;
  const res = await fetch(`/api/rovers/${encodeURIComponent(selectedId)}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, params }),
  });
  if (res.status === 401) {
    location.href = "/login";
    return null;
  }
  if (res.status === 404) return null;
  if (res.status === 409) {
    const body = await res.json().catch(() => ({}));
    alert(`Ровер занят: ${body.detail || "другой оператор"}`);
    sessionStarted = false;
    operatorLocked = false;
    renderPanel();
    return null;
  }
  return res.json();
}

function lidarForwardBlocked() {
  return lidarGuardActive && !lidarOverrideActive;
}

function lidarLockEngaged() {
  const kb = readKeyboardTargets();
  const backIntent = kb.fwd < 0;
  return lidarForwardBlocked() && !backIntent;
}

function readOverrideFromInputs() {
  if (keysLogical.has("override")) return true;
  if (!navigator.getGamepads) return false;
  const pads = navigator.getGamepads();
  for (let i = 0; i < pads.length; i += 1) {
    const pad = pads[i];
    if (pad && gamepadR2Pressed(pad)) return true;
  }
  return false;
}

function syncLidarGuardUi() {
  const override = readOverrideFromInputs();
  lidarOverrideActive = Boolean(override && lidarGuardActive);
  applyLidarLockUi();
  updateDriveKeyHighlight();
  renderSessionHint();
  if (lidarGuardBadge && lidarGuardActive) {
    lidarGuardBadge.classList.remove("hidden");
    if (lidarOverrideActive) {
      lidarGuardBadge.textContent = "OVERRIDE";
      lidarGuardBadge.classList.add("override");
      lidarGuardBadge.classList.remove("stop-active");
    } else {
      lidarGuardBadge.textContent = "СТОП";
      lidarGuardBadge.classList.remove("override");
      lidarGuardBadge.classList.add("stop-active");
    }
  }
  if (lidarGuardBanner) {
    if (lidarLockEngaged()) {
      lidarGuardBanner.classList.remove("hidden");
      lidarGuardBanner.textContent = "СТОП";
    } else {
      lidarGuardBanner.classList.add("hidden");
      lidarGuardBanner.textContent = "";
    }
  }
}

function applyLidarLockUi() {
  const locked = lidarLockEngaged();
  if (lidarBlock) lidarBlock.classList.toggle("guard-active", lidarForwardBlocked());
  if (controlsRail) controlsRail.classList.toggle("lidar-locked", locked);
  if (lidarLockOverlay) lidarLockOverlay.classList.toggle("hidden", !locked);
}

function forwardGuardActive(guard) {
  if (!guard) return false;
  if (guard.active_effective_forward != null) return Boolean(guard.active_effective_forward);
  if (guard.latched_forward != null) return Boolean(guard.latched_forward);
  if (guard.active_forward != null) return Boolean(guard.active_forward);
  return Boolean(guard.active_effective ?? guard.active);
}

function applyLidarGuard(fwd, override) {
  if (override) return fwd;
  if (fwd > 0 && lidarGuardActive) return 0;
  return fwd;
}

function sendDriveFast(fwd, turn, override, stop = false) {
  if (!selectedId || !sessionStarted || uiDriveMode !== "joystick") return;
  const scale = speedPct / 100;
  fwd = applyLidarGuard(fwd, override);
  const payload = {
    type: "drive",
    rover_id: selectedId,
    forward: stop ? 0 : fwd,
    turn: stop ? 0 : turn,
    speed_scale: scale,
    stop,
    lidar_override: Boolean(override),
  };
  if (fleetWs && fleetWs.readyState === WebSocket.OPEN) {
    fleetWs.send(JSON.stringify(payload));
    return;
  }
  const params = { forward: payload.forward, turn: payload.turn, speed_scale: scale };
  if (override) params.lidar_override = true;
  if (stop || (fwd === 0 && turn === 0)) sendCommand("stop_drive");
  else sendCommand("drive", params);
}

function smoothRamp() {
  return 0.05 + 0.12 * (speedPct / 100);
}

function readKeyboardTargets() {
  let fwd = 0;
  let turn = 0;
  if (keysLogical.has("fwd")) fwd = 1;
  if (keysLogical.has("back")) fwd = -1;
  if (keysLogical.has("left")) turn = 1;
  if (keysLogical.has("right")) turn = -1;
  return { fwd, turn, override: keysLogical.has("override"), active: fwd !== 0 || turn !== 0 };
}

function applyGamepadSpeedBumper(pad) {
  if (!pad || !sessionStarted) return;
  const buttons = pad.buttons || [];
  const lb = Boolean(buttons[4]?.pressed);
  const rb = Boolean(buttons[5]?.pressed);
  if (!lb && !rb) return;
  const now = Date.now();
  if (now - lastBumperSpeedAt < 120) return;
  lastBumperSpeedAt = now;
  if (lb) speedPct = clamp(speedPct - 5, 10, 100);
  if (rb) speedPct = clamp(speedPct + 5, 10, 100);
  saveSpeedPct();
  const speedSlider = document.getElementById("speed-slider");
  const speedLabel = document.getElementById("speed-label");
  if (speedSlider) speedSlider.value = String(speedPct);
  if (speedLabel) speedLabel.textContent = `${speedPct}%`;
}

function readGamepadAxes(pad) {
  const dz = GAMEPAD_DEADZONE;
  let lx = applyDeadzone(Number(pad.axes[0] || 0), dz);
  let ly = applyDeadzone(Number(pad.axes[1] || 0), dz);
  if (Math.abs(lx) < 0.03 && Math.abs(ly) < 0.03 && pad.axes.length > 3) {
    lx = applyDeadzone(Number(pad.axes[2] || 0), dz);
    ly = applyDeadzone(Number(pad.axes[3] || 0), dz);
  }
  let fwd = clamp(-ly, -1, 1);
  let turn = clamp(-lx, -1, 1);
  const buttons = pad.buttons || [];
  if (buttons[12]?.pressed) fwd = 1;
  if (buttons[13]?.pressed) fwd = -1;
  if (buttons[14]?.pressed) turn = 1;
  if (buttons[15]?.pressed) turn = -1;
  const rtRaw = buttons[7]?.value;
  const rt = rtRaw != null ? Number(rtRaw) : buttons[7]?.pressed ? 1 : 0;
  const ltRaw = buttons[6]?.value;
  const lt = ltRaw != null ? Number(ltRaw) : buttons[6]?.pressed ? 1 : 0;
  if (Number(rt) > 0.15) fwd = Math.max(fwd, Number(rt));
  if (Number(lt) > 0.15) fwd = Math.min(fwd, -Number(lt));
  return { fwd, turn };
}

function pollGamepad() {
  if (!navigator.getGamepads) return null;
  const pads = navigator.getGamepads();
  for (let i = 0; i < pads.length; i += 1) {
    const pad = pads[i];
    if (!pad) continue;
    gamepadState.index = i;
    gamepadState.connected = true;
    gamepadState.name = pad.id || "Gamepad";
    const { fwd, turn } = readGamepadAxes(pad);
    const override = gamepadR2Pressed(pad);
    const active = Math.abs(fwd) > 0.05 || Math.abs(turn) > 0.05;
    gamepadState.active = active;
    return { pad, fwd, turn, override, active, connected: true };
  }
  gamepadState.connected = false;
  gamepadState.active = false;
  gamepadState.name = "";
  return null;
}

function wakeGamepads() {
  if (!navigator.getGamepads) return;
  const pads = navigator.getGamepads();
  for (let i = 0; i < pads.length; i += 1) {
    const pad = pads[i];
    if (!pad) continue;
    gamepadState.index = i;
    gamepadState.connected = true;
    gamepadState.name = pad.id || "Gamepad";
    renderGamepadStatus();
    return;
  }
  gamepadState.connected = false;
  gamepadState.name = "";
  renderGamepadStatus();
}

function driveTick() {
  if (!sessionStarted || uiDriveMode !== "joystick") {
    smoothFwd = 0;
    smoothTurn = 0;
    return;
  }
  const gp = pollGamepad();
  const kb = readKeyboardTargets();
  if (gp?.pad) applyGamepadSpeedBumper(gp.pad);

  const kbFwdAxis = keysLogical.has("fwd") || keysLogical.has("back");
  const kbTurnAxis = keysLogical.has("left") || keysLogical.has("right");

  let fwd = kb.fwd;
  let turn = kb.turn;
  let override = kb.override || Boolean(gp?.override);
  // Per-axis: WASD/arrow turn beats gamepad stick drift when a pad is plugged in.
  if (gp && !kbFwdAxis && (gp.active || Math.abs(gp.fwd) > 0.05)) fwd = gp.fwd;
  if (gp && !kbTurnAxis && (gp.active || Math.abs(gp.turn) > 0.05)) turn = gp.turn;
  pendingOverride = override;
  syncLidarGuardUi();
  targetFwd = fwd;
  targetTurn = turn;
  const ramp = smoothRamp();
  smoothFwd += (targetFwd - smoothFwd) * ramp;
  smoothTurn += (targetTurn - smoothTurn) * ramp;
  renderGamepadStatus();
  const now = Date.now();
  if (now - lastDriveSentAt < DRIVE_INTERVAL_MS) return;
  lastDriveSentAt = now;
  const moving = Math.abs(smoothFwd) > 0.03 || Math.abs(smoothTurn) > 0.03;
  if (!moving) {
    if (pendingOverride) sendDriveFast(0, 0, true, false);
    else sendDriveFast(0, 0, false, true);
  } else sendDriveFast(smoothFwd, smoothTurn, pendingOverride, false);
}

function queueDrive(fwd, turn, override = false) {
  targetFwd = fwd;
  targetTurn = turn;
  pendingOverride = override;
}

function renderSessionHint() {
  const el = document.getElementById("session-hint");
  if (!el) return;
  const r = selectedRover();
  const op = normalizeOperator(r?.operator);
  if (!selectedId || !r?.online) {
    el.textContent = "Выберите online-ровер";
    el.className = "session-hint muted blocked";
    return;
  }
  if (op.locked && !op.you) {
    el.textContent = `Ровер занят оператором ${op.holder || "?"}`;
    el.className = "session-hint muted blocked";
    return;
  }
  if (!sessionStarted) {
    if (uiDriveMode === "joystick") {
      el.textContent = "Manual — ARM… (WASD после подключения)";
    } else {
      el.textContent = "Нажмите Manual — затем WASD / стрелки / Xbox";
    }
    el.className = "session-hint muted";
    return;
  }
  if (lidarOverrideActive) {
    el.textContent = "OVERRIDE: Shift/E/Space — проезд вперед разрешён";
    el.className = "session-hint ready";
    return;
  }
  if (lidarGuardActive) {
    el.textContent = "LiDAR СТОП вперед · S/▼ — назад · Shift/E/Space — override";
    el.className = "session-hint guard-stop";
    return;
  }
  el.textContent = "Manual: WASD / стрелки / Xbox · LB/RB = скорость";
  el.className = "session-hint ready";
}

function logicalKey(e) {
  const k = (e.key || "").toLowerCase();
  if (k === "w" || k === "ц" || e.code === "ArrowUp" || e.code === "KeyW") return "fwd";
  if (k === "s" || k === "ы" || e.code === "ArrowDown" || e.code === "KeyS") return "back";
  if (k === "a" || k === "ф" || e.code === "ArrowLeft" || e.code === "KeyA") return "left";
  if (k === "d" || k === "в" || e.code === "ArrowRight" || e.code === "KeyD") return "right";
  if (e.code === "ShiftLeft" || e.code === "ShiftRight" || e.code === "KeyE" || e.code === "Space")
    return "override";
  return null;
}

function updateDriveKeyHighlight() {
  const activeHints = new Set();
  if (keysLogical.has("fwd")) activeHints.add("w").add("arrowup");
  if (keysLogical.has("back")) activeHints.add("s").add("arrowdown");
  if (keysLogical.has("left")) activeHints.add("a").add("arrowleft");
  if (keysLogical.has("right")) activeHints.add("d").add("arrowright");
  const fwdBlocked = lidarForwardBlocked();

  document.querySelectorAll("[data-key-hint]").forEach((btn) => {
    const hints = (btn.getAttribute("data-key-hint") || "").split(",").filter(Boolean);
    const on = hints.some((h) => activeHints.has(h));
    btn.classList.toggle("key-active", on);
  });
  document.querySelectorAll("[data-fwd]").forEach((btn) => {
    const f = Number(btn.getAttribute("data-fwd") || 0);
    btn.disabled = fwdBlocked && f > 0;
  });
}

function isRoarmDevice(r) {
  return r && (r.kind === "roarm" || r.id === "roarm-01");
}

function isRoarmSelected() {
  return selectedId === "roarm-01";
}

function renderRoverList() {
  const rovers = [...lastFleet].sort((a, b) => {
    if (a.online !== b.online) return a.online ? -1 : 1;
    return (a.name || a.id).localeCompare(b.name || b.id);
  });

  roverSelect.innerHTML =
    `<option value="">— выберите устройство —</option>` +
    rovers
      .map(
        (r) =>
          `<option value="${escapeHtml(r.id)}" ${r.id === selectedId ? "selected" : ""}>${escapeHtml(r.name || r.id)} ${r.online ? "●" : "○"}</option>`
      )
      .join("");

  roverList.innerHTML = rovers
    .map((r) => {
      const t = r.telemetry || {};
      const isArm = isRoarmDevice(r);
      const agent = isArm
        ? "RoArm · Orin proxy"
        : t.agent === "orin"
          ? "Orin"
          : t.hostname || r.id;
      const cls = [
        r.id === selectedId ? "active" : "",
        r.online ? "online" : "offline",
        isArm && r.online && !((t.roarm || {}).reachable) ? "online-warn" : "",
        isArm ? "device-roarm" : "",
      ]
        .filter(Boolean)
        .join(" ");
      const armReach = isArm ? Boolean((t.roarm || {}).reachable) : false;
      const tcpOpen = isArm ? Boolean((t.roarm || {}).tcp_open) : false;
      const statusLine = isArm
        ? r.online
          ? armReach
            ? "online"
            : tcpOpen
              ? "HTTP busy — закройте UI лапы"
              : "proxy · arm offline"
          : `offline${r.last_seen_ago_s != null ? ` · ${Math.round(r.last_seen_ago_s)}s` : ""}`
        : r.online
          ? "online"
          : `offline${r.last_seen_ago_s != null ? ` · ${Math.round(r.last_seen_ago_s)}s` : ""}`;
      return `<li class="${cls}" data-id="${escapeHtml(r.id)}" data-kind="${isArm ? "roarm" : "rover"}">
        <strong>${escapeHtml(r.name || r.id)}</strong>
        <span>${escapeHtml(agent)}</span>
        <em>${statusLine}</em>
      </li>`;
    })
    .join("");

  roverList.querySelectorAll("li").forEach((li) => {
    li.onclick = () => selectRover(li.getAttribute("data-id"));
  });

  const onlineN = rovers.filter((r) => r.online).length;
  roverStatus.textContent = `${onlineN} online / ${rovers.length} всего`;

  if (!selectedId && rovers.length) {
    const pick =
      rovers.find((r) => r.online && !isRoarmDevice(r)) ||
      rovers.find((r) => !isRoarmDevice(r));
    if (pick) selectRover(pick.id, false);
  }
}

function selectRover(id, persist = true) {
  const r = getRover(id);
  if (!r) return;

  if (isRoarmDevice(r)) {
    if (id !== selectedId) {
      sessionStarted = false;
      operatorLocked = false;
      mjpegUrl = "";
      stopWebRtcFront();
    }
    selectedId = id || "";
    if (persist) localStorage.setItem("axm_rover_id", selectedId);
    roverSelect.value = selectedId;
    renderRoverList();
    renderPanel();
    window.RoArmPanel?.onSelect?.(r);
    return;
  }

  if (id !== selectedId) {
    sessionStarted = false;
    operatorLocked = false;
    mjpegUrl = "";
    stopWebRtcFront();
  }
  selectedId = id || "";
  if (persist) localStorage.setItem("axm_rover_id", selectedId);
  roverSelect.value = selectedId;
  renderRoverList();
  renderPanel();
  if (selectedId) {
    const r = getRover(selectedId);
    if (r?.online) startWebRtcFront(selectedId);
  }
}

function lidarRadii(maxDisplayM) {
  const minR = 6;
  const maxR = 94;
  const ringMaxM = Math.min(2, maxDisplayM || 2);
  return { cx: 100, cy: 102, minR, maxR, ringMaxM };
}

function distToRadius(distM, maxDisplayM, radii) {
  const clamped = Math.min(Math.max(distM, 0.05), maxDisplayM);
  return radii.minR + (clamped / maxDisplayM) * (radii.maxR - radii.minR);
}

function renderLidarRings(maxDisplayM) {
  const g = document.getElementById("lidar-rings");
  if (!g) return;
  g.innerHTML = "";
  const { cx, cy, minR, maxR, ringMaxM } = lidarRadii(maxDisplayM);
  [1, 2].forEach((m) => {
    if (m > ringMaxM) return;
    const r = minR + (m / ringMaxM) * (maxR - minR);
    const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    ring.setAttribute("cx", String(cx));
    ring.setAttribute("cy", String(cy));
    ring.setAttribute("r", String(r));
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "#24304a");
    ring.setAttribute("stroke-width", "1");
    ring.setAttribute("stroke-dasharray", "3 4");
    g.appendChild(ring);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(cx + 4));
    label.setAttribute("y", String(cy - r + 3));
    label.setAttribute("fill", "#5a6888");
    label.setAttribute("font-size", "8");
    label.textContent = `${m}m`;
    g.appendChild(label);
  });
}

function renderLidarGuardZone(arc, guard) {
  const g = document.getElementById("lidar-guard-zone");
  if (!g) return;
  g.innerHTML = "";
  if (!arc?.connected || !guard) return;

  const maxDisplayM = Math.min(2, arc.display_max_m || arc.range_max_m || 2);
  const radii = lidarRadii(maxDisplayM);
  const { cx, cy, minR } = radii;
  const thresholdM = Number(guard.threshold_m || 0.4);
  const lookaheadM = 2.0;
  const halfDeg = Number(guard.guard_half_angle_deg || 17.4);
  const halfRad = (halfDeg * Math.PI) / 180;
  const center = -Math.PI / 2;
  const makeZonePath = (halfAngleRad, distM, className) => {
    const a0 = center - halfAngleRad;
    const a1 = center + halfAngleRad;
    const r = distToRadius(distM, maxDisplayM, radii);
    const x0 = cx + minR * Math.cos(a0);
    const y0 = cy + minR * Math.sin(a0);
    const x1 = cx + minR * Math.cos(a1);
    const y1 = cy + minR * Math.sin(a1);
    const x2 = cx + r * Math.cos(a1);
    const y2 = cy + r * Math.sin(a1);
    const x3 = cx + r * Math.cos(a0);
    const y3 = cy + r * Math.sin(a0);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      `M ${x0} ${y0} A ${minR} ${minR} 0 0 1 ${x1} ${y1} L ${x2} ${y2} A ${r} ${r} 0 0 0 ${x3} ${y3} Z`
    );
    path.setAttribute("class", className);
    g.appendChild(path);
  };

  // Guard corridor = fixed ~40° cone ahead; side points are ignored for STOP.
  makeZonePath(halfRad, lookaheadM, "lidar-guard-zone");
  // Near danger zone (red) inside corridor.
  makeZonePath(halfRad, Math.min(thresholdM, lookaheadM), "lidar-danger-zone");
}

function renderLidarArc(arc, guard) {
  const g = document.getElementById("lidar-sectors");
  if (!g) return;
  g.innerHTML = "";
  if (!arc || !arc.connected || !arc.sectors || !arc.sectors.length) {
    const hint = document.createElementNS("http://www.w3.org/2000/svg", "text");
    hint.setAttribute("x", "100");
    hint.setAttribute("y", "60");
    hint.setAttribute("text-anchor", "middle");
    hint.setAttribute("fill", "#5a6888");
    hint.setAttribute("font-size", "9");
    hint.textContent = arc?.connected ? "нет препятствий впереди" : "LiDAR — нет данных";
    g.appendChild(hint);
    return;
  }

  const maxDisplayM = Math.min(2, arc.display_max_m || arc.range_max_m || 2);
  renderLidarRings(maxDisplayM);
  renderLidarGuardZone(arc, guard);

  const n = arc.sectors.length;
  const radii = lidarRadii(maxDisplayM);
  const { cx, cy, minR, maxR } = radii;
  const fov = (arc.fov_deg || 160) * (Math.PI / 180);
  const start = -Math.PI / 2 - fov / 2;
  const step = fov / n;

  arc.sectors.forEach((sec, i) => {
    const a0 = start + i * step + 0.03;
    const a1 = start + (i + 1) * step - 0.03;
    const dist = sec.dist_m;
    if (dist == null) return;

    const r = distToRadius(dist, maxDisplayM, radii);
    // Sector thickness for visualization (reduced to avoid elongated blobs).
    const band = Math.max(1.6, ((maxR - minR) / maxDisplayM) * 0.09);

    const x0o = cx + (r + band) * Math.cos(a0);
    const y0o = cy + (r + band) * Math.sin(a0);
    const x1o = cx + (r + band) * Math.cos(a1);
    const y1o = cy + (r + band) * Math.sin(a1);
    const x1i = cx + Math.max(minR, r - band) * Math.cos(a1);
    const y1i = cy + Math.max(minR, r - band) * Math.sin(a1);
    const x0i = cx + Math.max(minR, r - band) * Math.cos(a0);
    const y0i = cy + Math.max(minR, r - band) * Math.sin(a0);

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      `M ${x0i} ${y0i} L ${x0o} ${y0o} A ${r + band} ${r + band} 0 0 1 ${x1o} ${y1o} L ${x1i} ${y1i} A ${Math.max(minR, r - band)} ${Math.max(minR, r - band)} 0 0 0 ${x0i} ${y0i} Z`
    );
    const guardIdx = new Set((guard?.sector_indices || []).map((x) => Number(x)));
    const inGuard = guardIdx.has(i);
    const guardActive = forwardGuardActive(guard);
    const blocked = guardActive && inGuard && dist != null && dist < (guard.threshold_m || 0.4);
    path.setAttribute("class", `lidar-sector level-${blocked ? 3 : sec.level || 0}`);
    g.appendChild(path);
  });
}

function mergeMega(m) {
  const out = { ...lastMegaSticky };
  for (const [k, v] of Object.entries(m || {})) {
    if (v != null && v !== "") out[k] = v;
  }
  lastMegaSticky = out;
  return out;
}

async function loadIceServers() {
  if (webrtcIceServers) return webrtcIceServers;
  try {
    const res = await fetch("/api/webrtc/config", { credentials: "same-origin" });
    if (!res.ok) return [{ urls: "stun:stun.l.google.com:19302" }];
    const data = await res.json();
    webrtcIceServers = data.iceServers || [{ urls: "stun:stun.l.google.com:19302" }];
  } catch (_) {
    webrtcIceServers = [{ urls: "stun:stun.l.google.com:19302" }];
  }
  return webrtcIceServers;
}

function stopWebRtcFront() {
  if (webrtcPollTimer) {
    clearInterval(webrtcPollTimer);
    webrtcPollTimer = null;
  }
  if (webrtcPc) {
    webrtcPc.close();
    webrtcPc = null;
  }
  webrtcRoverId = "";
  const video = document.getElementById("front-camera-webrtc");
  if (video) {
    video.srcObject = null;
    video.classList.add("hidden");
  }
}

async function pollWebRtcSession(roverId) {
  if (!webrtcPc || webrtcRoverId !== roverId) return;
  try {
    const res = await fetch(`/api/webrtc/session/${encodeURIComponent(roverId)}`, {
      credentials: "same-origin",
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.answer && webrtcPc.signalingState !== "stable") {
      await webrtcPc.setRemoteDescription({ type: "answer", sdp: data.answer });
    }
    for (const cand of data.agent_ice || []) {
      try {
        await webrtcPc.addIceCandidate(cand);
      } catch (_) {}
    }
  } catch (_) {}
}

async function startWebRtcFront(roverId) {
  if (!roverId) return;
  if (webrtcRoverId === roverId && webrtcPc) return;
  stopWebRtcFront();
  webrtcRoverId = roverId;

  const video = document.getElementById("front-camera-webrtc");
  const img = document.getElementById("front-camera");
  const ph = document.getElementById("camera-placeholder");
  const status = document.getElementById("cam-status");
  if (!video || !ph) return;

  ph.textContent = "WebRTC — подключение…";
  ph.classList.remove("hidden");
  img?.classList.add("hidden");
  video.classList.add("hidden");

  const iceServers = await loadIceServers();
  const pc = new RTCPeerConnection({ iceServers });
  webrtcPc = pc;

  pc.ontrack = (ev) => {
    if (!video) return;
    video.srcObject = ev.streams[0] || new MediaStream([ev.track]);
    video.classList.remove("hidden");
    ph.classList.add("hidden");
    img?.classList.add("hidden");
    if (status) status.textContent = "WebRTC · live";
  };
  pc.onconnectionstatechange = () => {
    if (!status) return;
    if (pc.connectionState === "connected") {
      status.textContent = "WebRTC · connected";
      setTimeout(() => {
        if (!videoHasFrames()) {
          stopWebRtcFront();
          renderFrontCameraMjpeg(getRover(roverId));
        }
      }, 2500);
    } else if (pc.connectionState === "failed") {
      status.textContent = "WebRTC — ошибка, MJPEG…";
      stopWebRtcFront();
      renderFrontCameraMjpeg(getRover(roverId));
    }
  };

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.onicecandidate = (ev) => {
    if (!ev.candidate) return;
    fetch("/api/webrtc/ice", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rover_id: roverId,
        candidate: {
          candidate: ev.candidate.candidate,
          sdpMid: ev.candidate.sdpMid,
          sdpMLineIndex: ev.candidate.sdpMLineIndex,
        },
      }),
    }).catch(() => {});
  };

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const res = await fetch("/api/webrtc/offer", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rover_id: roverId, sdp: offer.sdp }),
    });
    if (!res.ok) {
      if (status) status.textContent = "WebRTC недоступен — MJPEG";
      renderFrontCameraMjpeg(getRover(roverId));
      return;
    }
    const data = await res.json();
    if (data.answer) {
      await pc.setRemoteDescription({ type: "answer", sdp: data.answer });
    }
  } catch (err) {
    if (status) status.textContent = "WebRTC недоступен — MJPEG";
    renderFrontCameraMjpeg(getRover(roverId));
  }
}

function renderFrontCameraMjpeg(r) {
  const img = document.getElementById("front-camera");
  const video = document.getElementById("front-camera-webrtc");
  const ph = document.getElementById("camera-placeholder");
  const status = document.getElementById("cam-status");
  if (!img || !ph || !status) return;

  const online = Boolean(r?.online);
  const live = Boolean(r?.camera_live);
  if (online && live && selectedId) {
    const url = `/api/rovers/${encodeURIComponent(selectedId)}/camera/mjpeg`;
    if (mjpegUrl !== url) {
      mjpegUrl = url;
      img.src = url;
    }
    video?.classList.add("hidden");
    img.classList.remove("hidden");
    ph.classList.add("hidden");
    const age = r.link?.camera_age_ms;
    status.textContent = age != null ? `MJPEG · ${Math.round(age)} ms` : "MJPEG · резерв";
  } else {
    if (mjpegUrl) {
      img.removeAttribute("src");
      mjpegUrl = "";
    }
    img.classList.add("hidden");
    if (!video || video.classList.contains("hidden")) {
      ph.classList.remove("hidden");
      ph.textContent = online ? "Камера — ожидание" : "Камера — нет сигнала";
    }
    if (status && (!video || video.classList.contains("hidden"))) {
      status.textContent = online ? "Передняя · ожидание" : "Передняя · offline";
    }
  }
}

function renderFrontCamera(r) {
  renderFrontCameraMjpeg(r);
}

function videoHasFrames() {
  const video = document.getElementById("front-camera-webrtc");
  if (!video || video.classList.contains("hidden")) return false;
  return video.readyState >= 2 && video.videoWidth > 0;
}

let rtkMap = null;
let rtkMarker = null;
let rtkTrail = null;
const rtkTrack = [];

function ensureRtkMap() {
  if (rtkMap || typeof L === "undefined") return;
  const el = document.getElementById("rtk-map");
  if (!el) return;
  rtkMap = L.map(el, {
    zoomControl: false,
    attributionControl: false,
    dragging: true,
    scrollWheelZoom: false,
  }).setView([0, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20,
    opacity: 0.85,
  }).addTo(rtkMap);
  rtkTrail = L.polyline([], { color: "#5b8cff", weight: 2, opacity: 0.7 }).addTo(rtkMap);
  setTimeout(() => rtkMap?.invalidateSize(), 200);
}

function renderRtkMap(rtk) {
  ensureRtkMap();
  const label = document.getElementById("s-rtk");
  if (!rtk) {
    if (label) label.textContent = "нет данных";
    return;
  }
  const fixQ = Number(rtk.fix_quality ?? rtk.fix ?? 0);
  const hasFix = fixQ > 0 && !rtk.stale;
  const lat = hasFix ? rtk.lat : null;
  const lon = hasFix ? rtk.lon : null;
  const fix = rtk.fix_label || (rtk.fix != null ? `fix${rtk.fix}` : "—");
  const sats = rtk.satellites != null ? `${rtk.satellites} sat` : "";
  const hdop = rtk.hdop != null ? `hdop ${rtk.hdop}` : "";
  const age = rtk.age_s != null ? `${rtk.age_s}s` : "";
  const ntrip = rtk.ntrip || {};
  let ntripMeta = "";
  if (ntrip.enabled) {
    if (ntrip.connected) {
      const rate = ntrip.rtcm_rate_bps != null ? `${ntrip.rtcm_rate_bps} B/s` : "RTCM";
      ntripMeta = `NTRIP ✓ ${rate}`;
    } else {
      ntripMeta = `NTRIP ✗ ${ntrip.error || "offline"}`;
    }
  }
  if (label) {
    if (lat != null && lon != null) {
      label.textContent = `${fix} · ${lat.toFixed(5)}, ${lon.toFixed(5)} ${sats} ${ntripMeta}`.trim();
      label.title = label.textContent;
    } else if (rtk.connected && !rtk.stale) {
      const parts = ["ожидание fix…", fix !== "—" ? fix : "", sats, hdop, ntripMeta, age].filter(Boolean);
      label.textContent = parts.join(" · ");
      label.title = rtk.last_sentence ? String(rtk.last_sentence) : label.textContent;
    } else if (rtk.connected) {
      label.textContent = `NMEA устарели · ${sats} ${hdop}`.trim();
    } else {
      const err = rtk.error ? String(rtk.error) : "нет связи";
      label.textContent = `${err}${rtk.port ? ` · ${rtk.port}` : ""}`;
    }
  }
  if (!rtkMap || lat == null || lon == null) {
    if (rtkMarker && rtkMap.hasLayer(rtkMarker)) rtkMap.removeLayer(rtkMarker);
    return;
  }
  if (!rtkMarker) {
    rtkMarker = L.circleMarker([lat, lon], {
      radius: 5,
      color: "#3ddc97",
      fillColor: "#3ddc97",
      fillOpacity: 0.9,
    }).addTo(rtkMap);
  } else if (!rtkMap.hasLayer(rtkMarker)) {
    rtkMarker.addTo(rtkMap);
  }
  const pt = [lat, lon];
  rtkMarker.setLatLng(pt);
  const last = rtkTrack[rtkTrack.length - 1];
  if (!last || Math.hypot(last[0] - lat, last[1] - lon) > 1e-6) {
    rtkTrack.push(pt);
    if (rtkTrack.length > 200) rtkTrack.shift();
    rtkTrail.setLatLngs(rtkTrack);
  }
  if (rtkTrack.length <= 2) rtkMap.setView(pt, 18);
}

function renderStereoCamera(r) {
  const img = document.getElementById("stereo-camera");
  const ph = document.getElementById("stereo-camera-placeholder");
  const status = document.getElementById("stereo-cam-status");
  if (!img || !ph) return;

  const online = Boolean(r?.online);
  const live = Boolean(r?.stereo_camera_live);
  const stereo = r?.telemetry?.perception?.stereo || {};
  const fps = stereo.stream_fps ?? stereo.hub_fps ?? 10;
  const mean = stereo.brightness_mean;
  const inRange = stereo.brightness_ok !== false;
  const tuning = Boolean(stereo.tuning);

  if (online && live && selectedId) {
    const url = `/api/rovers/${encodeURIComponent(selectedId)}/camera/stereo/mjpeg`;
    if (stereoMjpegUrl !== url) {
      stereoMjpegUrl = url;
      img.src = url;
    }
    img.classList.remove("hidden");
    ph.classList.add("hidden");
    if (status) {
      let label = `${fps} fps`;
      if (mean != null) {
        label += ` · ярк ${Math.round(mean)}`;
        if (tuning) label += " ⟳";
      }
      status.textContent = label;
      status.classList.toggle("warn", mean != null && !inRange);
      status.classList.toggle("ok", mean != null && inRange);
    }
  } else {
    if (stereoMjpegUrl) {
      img.removeAttribute("src");
      stereoMjpegUrl = "";
    }
    img.classList.add("hidden");
    ph.classList.remove("hidden");
    ph.textContent = online ? "ожидание…" : "offline";
    if (status) {
      status.textContent = online ? "нет потока" : "—";
      status.classList.remove("warn", "ok");
    }
  }
}

function renderLinkIndicator(r) {
  if (!linkIndicator) return;
  const link = r?.link || {};
  const status = link.status || "red";
  const rtt = link.rtt_ms;
  const age = link.camera_age_ms;
  linkIndicator.className = `link-badge link-${status}`;
  const rttTxt = rtt != null ? `${Math.round(rtt)} ms` : "—";
  const ageTxt = age != null ? `${Math.round(age)} ms` : "—";
  linkIndicator.textContent = rtt != null ? `${Math.round(rtt)}ms` : "—";
  linkIndicator.title = `RTT ${rttTxt}, кадр ${ageTxt}`;
}

function renderLidarGuardBadge(guard) {
  if (!lidarGuardBadge) return;
  lidarGuardActive = forwardGuardActive(guard);
  syncLidarGuardUi();
  if (!lidarGuardActive) {
    lidarGuardBadge.classList.add("hidden");
    lidarGuardBadge.classList.remove("override", "stop-active");
  }
}

function renderOperatorBadge(op) {
  if (!operatorBadge) return;
  op = normalizeOperator(op);
  if (!op?.locked) {
    operatorBadge.classList.add("hidden");
    operatorBadge.classList.remove("locked-you", "locked-other");
    return;
  }
  operatorBadge.classList.remove("hidden");
  operatorBadge.classList.toggle("locked-you", Boolean(op.you));
  operatorBadge.classList.toggle("locked-other", !op.you);
  operatorBadge.textContent = op.you
    ? "Вы — оператор"
    : `Занят: ${op.holder || "?"}`;
}

function renderPerception(r) {
  const t = r?.telemetry || {};
  const p = t.perception || {};
  renderLidarArc(p.lidar_arc, p.lidar_guard);
  renderFrontCamera(r);
  renderStereoCamera(r);
  renderRtkMap(t.rtk);
  renderLidarGuardBadge(p.lidar_guard);
}

function updateStartStopButtons(mode, canControl, online) {
  const startBtn = document.getElementById("btn-start");
  const stopBtn = document.getElementById("btn-stop");
  const manual = mode === "joystick";
  // In Manual mode Start/Stop are redundant — Manual toggles session + ARM.
  if (startBtn) startBtn.disabled = manual || !canControl;
  if (stopBtn) stopBtn.disabled = manual || !online;
}

function renderPanel() {
  const r = selectedRover();
  if (!r) {
    emptyState.classList.remove("hidden");
    roverPanel.classList.add("hidden");
    roarmPanel?.classList.add("hidden");
    return;
  }

  emptyState.classList.add("hidden");

  if (isRoarmDevice(r)) {
    roverPanel.classList.add("hidden");
    roarmPanel?.classList.remove("hidden");
    window.RoArmPanel?.onFleetUpdate?.(r);
    return;
  }

  roarmPanel?.classList.add("hidden");
  roverPanel.classList.remove("hidden");

  const t = r.telemetry || {};
  const m = mergeMega(t.mega || {});
  const agentLabel = t.agent === "orin" ? "Jetson Orin" : t.hostname || "—";

  panelTitle.textContent = r.name || r.id;
  panelSub.textContent = agentLabel;
  panelBadge.textContent = r.online ? "ON" : "OFF";
  panelBadge.className = `badge ${r.online ? "online" : "offline"}`;
  const topRover = document.getElementById("topbar-rover");
  if (topRover) topRover.textContent = r.online ? `${r.name || r.id} · online` : `${r.name || r.id} · offline`;

  renderLinkIndicator(r);
  renderOperatorBadge(r.operator);

  document.getElementById("t-mega").textContent =
    t.arduino_connected === true ? "OK" : t.arduino_connected === false ? "нет" : "—";
  const megaArmed =
    m.armed === true || m.mega_armed === true || t.armed === true;
  document.getElementById("t-arm").textContent =
    megaArmed ? "ARM" : m.armed === false || m.mega_armed === false || t.armed === false ? "DIS" : "—";
  const armEl = document.getElementById("t-arm");
  if (armEl) {
    armEl.classList.toggle(
      "mega-dis",
      !megaArmed &&
        (m.armed === false || m.mega_armed === false || t.armed === false)
    );
    armEl.title = !megaArmed && uiDriveMode === "joystick"
      ? "Нажмите Manual ещё раз для повторного ARM"
      : "";
  }

  const motorPercent = (us) => {
    const n = Number(us);
    if (!Number.isFinite(n)) return "—";
    return `${Math.round(clamp((n - 1500) / 5, -100, 100))}`;
  };
  document.getElementById("s-left").textContent = motorPercent(m.left_us);
  document.getElementById("s-right").textContent = motorPercent(m.right_us);
  document.getElementById("s-speed").textContent =
    m.speed_mps != null ? `${m.speed_mps}` : "0";
  document.getElementById("s-current-a0").textContent =
    m.current_a0 != null ? `${m.current_a0}` : "—";
  document.getElementById("s-current-d22").textContent =
    m.current_d22 != null ? `${m.current_d22}` : m.current_d0 != null ? `${m.current_d0}` : "—";
  document.getElementById("s-temp").textContent =
    m.temp_c != null ? `${m.temp_c}°` : "—";
  document.getElementById("s-humidity").textContent =
    m.humidity_pct != null ? `${m.humidity_pct}%` : "—";
  document.getElementById("s-vibration").textContent =
    m.vibration_d24 != null ? (m.vibration_d24 ? "H" : "L") : "—";
  renderPerception(r);
  if (rtkMap) setTimeout(() => rtkMap.invalidateSize(), 150);
  document.getElementById("t-ago").textContent = fmtAgo(r.last_seen_ago_s);

  const mode = t.drive_mode || uiDriveMode;
  if (t.session_active === true && mode === "joystick") {
    sessionStarted = true;
  }
  applyModeUi(mode, false);

  const online = r.online;
  const op = normalizeOperator(r.operator);
  const canControl = online && (!op.locked || op.you);
  updateStartStopButtons(mode, canControl, online);
  modeJoystick.disabled = !online;
  modeAuto.disabled = !online;
  document.querySelectorAll("[data-fwd]").forEach((b) => {
    b.disabled = !canControl || mode !== "joystick" || !sessionStarted;
  });
  renderSessionHint();
  updateDriveKeyHighlight();
}

function applySpeedUi() {
  const speedSlider = document.getElementById("speed-slider");
  const speedLabel = document.getElementById("speed-label");
  if (speedSlider) speedSlider.value = String(speedPct);
  if (speedLabel) speedLabel.textContent = `${speedPct}%`;
}

async function enterManualMode() {
  applyModeUi("joystick", true);
  queueDrive(0, 0);
  targetFwd = 0;
  targetTurn = 0;
  smoothFwd = 0;
  smoothTurn = 0;
  await sendCommand("stop_drive");

  const r = selectedRover();
  const online = Boolean(r?.online);
  const op = normalizeOperator(r?.operator);
  const canControl = online && (!op.locked || op.you);
  if (!canControl) {
    renderPanel();
    return;
  }

  wakeGamepads();
  const claim = await claimOperator();
  if (!claim.ok) {
    if (claim.conflict) alert("Ровер уже управляется другим оператором");
    renderPanel();
    return;
  }
  await sendCommand("session_start");
  sessionStarted = true;
  startOperatorRenew();
  drivePad?.focus();
  renderPanel();
}

function applyModeUi(mode, notifyAgent = true) {
  uiDriveMode = mode;
  localStorage.setItem("axm_drive_mode", mode);
  modeJoystick.classList.toggle("active", mode === "joystick");
  modeAuto.classList.toggle("active", mode === "auto");
  joystickPanel.classList.toggle("hidden", mode !== "joystick");
  autoPanel.classList.toggle("hidden", mode !== "auto");
  if (notifyAgent && selectedId) {
    sendCommand("set_drive_mode", { mode });
  }
  const r = selectedRover();
  const online = Boolean(r?.online);
  const op = normalizeOperator(r?.operator);
  const canControl = online && (!op.locked || op.you);
  updateStartStopButtons(mode, canControl, online);
  document.querySelectorAll("[data-fwd]").forEach((b) => {
    b.disabled = !canControl || mode !== "joystick" || !sessionStarted;
  });
  renderSessionHint();
}

modeJoystick.onclick = () => {
  enterManualMode();
};

modeAuto.onclick = async () => {
  if (sessionStarted) {
    queueDrive(0, 0);
    smoothFwd = 0;
    smoothTurn = 0;
    await sendCommand("stop_drive");
    await sendCommand("session_stop");
    sessionStarted = false;
    await releaseOperator();
  }
  applyModeUi("auto", true);
  renderPanel();
};

roverSelect.onchange = () => selectRover(roverSelect.value);

document.getElementById("btn-start").onclick = () => enterManualMode();

document.getElementById("btn-stop").onclick = async () => {
  sessionStarted = false;
  queueDrive(0, 0);
  await sendCommand("session_stop");
  await releaseOperator();
  renderPanel();
};

document.querySelectorAll("[data-fwd]").forEach((btn) => {
  btn.onclick = () => {
    if (uiDriveMode !== "joystick" || !selectedId || !sessionStarted) return;
    const fwd = Number(btn.getAttribute("data-fwd") || 0);
    const turn = Number(btn.getAttribute("data-turn") || 0);
    queueDrive(fwd, turn, false);
  };
});

function keyboardDrive() {
  syncLidarGuardUi();
}

function renderGamepadStatus() {
  // UI status line removed; gamepad wakes on Manual / driveTick.
}

function gamepadR2Pressed(pad) {
  const buttons = pad.buttons || [];
  if (buttons[7]?.pressed) return true;
  if (buttons[5]?.pressed) return true;
  if (pad.axes.length > 5 && pad.axes[5] > 0.5) return true;
  return false;
}

function onKeyDown(e) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target?.tagName || "").toUpperCase())) return;
  const lk = logicalKey(e);
  if (!lk && !DRIVE_KEY_CODES.has(e.code)) return;
  e.preventDefault();
  e.stopPropagation();
  if (lk) keysLogical.add(lk);
  keysDown.add(e.code);
  keyboardDrive();
}

function onKeyUp(e) {
  const lk = logicalKey(e);
  if (lk) keysLogical.delete(lk);
  if (!DRIVE_KEY_CODES.has(e.code) && !lk) return;
  e.preventDefault();
  keysDown.delete(e.code);
  keyboardDrive();
}

function onWindowBlur() {
  keysDown.clear();
  keysLogical.clear();
  updateDriveKeyHighlight();
  targetFwd = 0;
  targetTurn = 0;
  if (sessionStarted) queueDrive(0, 0, false);
}

document.addEventListener("keydown", onKeyDown, true);
document.addEventListener("keyup", onKeyUp, true);
window.addEventListener("blur", onWindowBlur);
drivePad?.addEventListener("click", () => drivePad.focus());

const speedSlider = document.getElementById("speed-slider");
const speedLabel = document.getElementById("speed-label");
if (speedSlider) {
  applySpeedUi();
  speedSlider.oninput = () => {
    speedPct = Number(speedSlider.value);
    saveSpeedPct();
    if (speedLabel) speedLabel.textContent = `${speedPct}%`;
  };
}

let lastFleetRenderAt = 0;

function renderFleet(data) {
  const now = Date.now();
  if (now - lastFleetRenderAt < 250) return;
  lastFleetRenderAt = now;
  lastFleet = data.rovers || [];
  renderRoverList();
  renderPanel();
  if (isRoarmSelected()) {
    window.RoArmPanel?.onFleetUpdate?.(getRover("roarm-01"));
  }
}

async function loadFleet() {
  const statusEl = document.getElementById("rover-status");
  let timeoutId;
  const signal =
    typeof AbortSignal !== "undefined" && AbortSignal.timeout
      ? AbortSignal.timeout(10000)
      : (() => {
          const ctrl = new AbortController();
          timeoutId = setTimeout(() => ctrl.abort(), 10000);
          return ctrl.signal;
        })();
  try {
    const res = await fetch("/api/rovers", {
      credentials: "same-origin",
      signal,
    });
    if (timeoutId) clearTimeout(timeoutId);
    if (res.status === 401) {
      location.href = "/login";
      return;
    }
    if (!res.ok) {
      if (statusEl) statusEl.textContent = `Ошибка сервера ${res.status}`;
      return;
    }
    renderFleet(await res.json());
  } catch (err) {
    if (timeoutId) clearTimeout(timeoutId);
    if (statusEl && (!lastFleet || lastFleet.length === 0)) {
      statusEl.textContent = "Нет связи с сервером — Ctrl+F5 или /login";
    }
  }
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/dashboard`);
  fleetWs = ws;
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === "fleet") renderFleet(data);
    } catch (_) {}
  };
  ws.onclose = (ev) => {
    fleetWs = null;
    if (ev.code === 4401) {
      location.href = "/login";
      return;
    }
    setTimeout(connectWs, 2000);
  };
}

document.getElementById("logout").onclick = () => {
  const stop = sessionStarted;
  const rid = selectedId;
  sessionStarted = false;
  if (stop && rid) {
    fetch(`/api/rovers/${encodeURIComponent(rid)}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "session_stop", params: {} }),
    }).catch(() => {});
    fetch(`/api/rovers/${encodeURIComponent(rid)}/release`, { method: "POST" }).catch(() => {});
  }
  fetch("/api/logout", { method: "POST", credentials: "same-origin" })
    .catch(() => {})
    .finally(() => {
      location.href = "/login";
    });
};

async function initUser() {
  if (window.__AXM_USER__) {
    currentUser = window.__AXM_USER__;
    localStorage.setItem("axm_saved_username", currentUser);
  } else {
    try {
      const res = await fetch("/api/me", { credentials: "same-origin" });
      if (res.ok) {
        const data = await res.json();
        if (data.username) {
          currentUser = data.username;
          localStorage.setItem("axm_saved_username", currentUser);
        }
      }
    } catch (_) {}
  }
  const badge = document.getElementById("user-badge");
  if (badge) badge.textContent = localUsername();
  speedPct = loadSpeedPct();
  applySpeedUi();
}

if (window.__AXM_BOOT_FLEET__) {
  renderFleet(window.__AXM_BOOT_FLEET__);
}

initUser().then(() => {
  loadFleet();
  connectWs();
  const urlDevice = new URLSearchParams(location.search).get("device");
  if (urlDevice) selectRover(urlDevice);
});
setInterval(loadFleet, 3000);
setInterval(() => {
  if (!sessionStarted) wakeGamepads();
}, 2000);
setInterval(driveTick, DRIVE_INTERVAL_MS);

window.addEventListener("gamepadconnected", (evt) => {
  gamepadState.index = evt.gamepad.index;
  gamepadState.connected = true;
  gamepadState.name = evt.gamepad.id || "Gamepad";
  renderGamepadStatus();
});
window.addEventListener("gamepaddisconnected", (evt) => {
  if (gamepadState.index === evt.gamepad.index) {
    gamepadState.index = null;
    gamepadState.connected = false;
    gamepadState.active = false;
    gamepadState.name = "";
    queueDrive(0, 0);
    renderGamepadStatus();
  }
});
renderGamepadStatus();
wakeGamepads();
