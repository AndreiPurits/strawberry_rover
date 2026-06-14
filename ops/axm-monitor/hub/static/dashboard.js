const roverSelect = document.getElementById("rover-select");
const roverList = document.getElementById("rover-list");
const roverStatus = document.getElementById("rover-status");
const emptyState = document.getElementById("empty-state");
const roverPanel = document.getElementById("rover-panel");
const panelTitle = document.getElementById("panel-title");
const panelSub = document.getElementById("panel-sub");
const panelBadge = document.getElementById("panel-badge");
const modeJoystick = document.getElementById("mode-joystick");
const modeAuto = document.getElementById("mode-auto");
const modeHint = document.getElementById("mode-hint");
const joystickPanel = document.getElementById("joystick-panel");
const autoPanel = document.getElementById("auto-panel");

const keysDown = new Set();
let lastFleet = [];
let selectedId = localStorage.getItem("axm_rover_id") || "";
let uiDriveMode = localStorage.getItem("axm_drive_mode") || "joystick";

const DRIVE_INTERVAL_MS = 50;
const GAMEPAD_DEADZONE = 0.18;
const LINEAR_SPEED_MAX = 0.55;
const ANGULAR_SPEED_MAX = 1.1;

let lastDriveSentAt = 0;
let pendingDrive = null;
let driveTimer = null;
let sessionStarted = false;

const gamepadState = { index: null, connected: false, active: false };

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
  return res.json();
}

function flushDrive() {
  driveTimer = null;
  if (!pendingDrive || !selectedId || uiDriveMode !== "joystick" || !sessionStarted) return;
  const now = Date.now();
  if (now - lastDriveSentAt < DRIVE_INTERVAL_MS) {
    driveTimer = setTimeout(flushDrive, DRIVE_INTERVAL_MS - (now - lastDriveSentAt));
    return;
  }
  lastDriveSentAt = now;
  const { lx, az } = pendingDrive;
  pendingDrive = null;
  if (lx === 0 && az === 0) sendCommand("stop_drive");
  else sendCommand("drive", { linear_x: lx, angular_z: az });
}

function queueDrive(lx, az) {
  pendingDrive = { lx, az };
  if (!driveTimer) flushDrive();
}

function renderRoverList() {
  const rovers = [...lastFleet].sort((a, b) => {
    if (a.online !== b.online) return a.online ? -1 : 1;
    return (a.name || a.id).localeCompare(b.name || b.id);
  });

  roverSelect.innerHTML =
    `<option value="">— выберите ровер —</option>` +
    rovers
      .map(
        (r) =>
          `<option value="${escapeHtml(r.id)}" ${r.id === selectedId ? "selected" : ""}>${escapeHtml(r.name || r.id)} ${r.online ? "●" : "○"}</option>`
      )
      .join("");

  roverList.innerHTML = rovers
    .map((r) => {
      const t = r.telemetry || {};
      const agent = t.agent === "orin" ? "Orin" : t.hostname || r.id;
      return `<li class="${r.id === selectedId ? "active" : ""} ${r.online ? "online" : "offline"}" data-id="${escapeHtml(r.id)}">
        <strong>${escapeHtml(r.name || r.id)}</strong>
        <span>${escapeHtml(agent)}</span>
        <em>${r.online ? "online" : "offline"}</em>
      </li>`;
    })
    .join("");

  roverList.querySelectorAll("li").forEach((li) => {
    li.onclick = () => selectRover(li.getAttribute("data-id"));
  });

  const onlineN = rovers.filter((r) => r.online).length;
  roverStatus.textContent = `${onlineN} online / ${rovers.length} всего`;

  if (!selectedId && rovers.length) {
    const pick = rovers.find((r) => r.online) || rovers[0];
    selectRover(pick.id, false);
  }
}

function selectRover(id, persist = true) {
  selectedId = id || "";
  if (persist) localStorage.setItem("axm_rover_id", selectedId);
  roverSelect.value = selectedId;
  renderRoverList();
  renderPanel();
}

function renderPanel() {
  const r = selectedRover();
  if (!r) {
    emptyState.classList.remove("hidden");
    roverPanel.classList.add("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  roverPanel.classList.remove("hidden");

  const t = r.telemetry || {};
  const m = t.mega || {};
  const agentLabel = t.agent === "orin" ? "Jetson Orin" : t.hostname || "—";

  panelTitle.textContent = r.name || r.id;
  panelSub.textContent = `${agentLabel} · ${t.mega_port || "Mega"}`;
  panelBadge.textContent = r.online ? "ONLINE" : "OFFLINE";
  panelBadge.className = `badge ${r.online ? "online" : "offline"}`;

  document.getElementById("t-mega").textContent =
    t.arduino_connected === true ? "подключена" : t.arduino_connected === false ? "нет" : "—";
  document.getElementById("t-arm").textContent =
    m.armed === true ? "ARM" : m.armed === false ? "DISARM" : "—";
  document.getElementById("t-speed").textContent =
    m.speed_mps != null ? `${m.speed_mps} m/s (${m.linear_pct ?? 0}%)` : "0 m/s";
  document.getElementById("t-left").textContent =
    m.left_us != null ? `${m.left_us} µs (${m.left_pct ?? 0}%)` : "—";
  document.getElementById("t-right").textContent =
    m.right_us != null ? `${m.right_us} µs (${m.right_pct ?? 0}%)` : "—";
  document.getElementById("t-current").textContent = m.current_a0 != null ? String(m.current_a0) : "—";
  document.getElementById("t-d0").textContent = m.current_d0 != null ? String(m.current_d0) : "—";
  document.getElementById("t-ago").textContent = fmtAgo(r.last_seen_ago_s);

  const mode = t.drive_mode || uiDriveMode;
  applyModeUi(mode, false);

  const online = r.online;
  document.getElementById("btn-start").disabled = !online || mode !== "joystick";
  document.getElementById("btn-stop").disabled = !online;
  modeJoystick.disabled = !online;
  modeAuto.disabled = !online;
  document.querySelectorAll("[data-drive]").forEach((b) => {
    b.disabled = !online || mode !== "joystick";
  });
}

function applyModeUi(mode, notifyAgent = true) {
  uiDriveMode = mode;
  localStorage.setItem("axm_drive_mode", mode);
  modeJoystick.classList.toggle("active", mode === "joystick");
  modeAuto.classList.toggle("active", mode === "auto");
  joystickPanel.classList.toggle("hidden", mode !== "joystick");
  autoPanel.classList.toggle("hidden", mode !== "auto");
  modeHint.textContent =
    mode === "joystick"
      ? "Подключите Xbox к PC, откройте эту страницу в Chrome, нажмите Start. Левый стик — езда."
      : "Автоматический режим — заглушка, ручное управление отключено.";
  if (notifyAgent && selectedId) {
    sendCommand("set_drive_mode", { mode });
  }
  const r = selectedRover();
  const online = Boolean(r?.online);
  document.getElementById("btn-start").disabled = !online || mode !== "joystick";
  document.querySelectorAll("[data-drive]").forEach((b) => {
    b.disabled = !online || mode !== "joystick";
  });
}

modeJoystick.onclick = () => applyModeUi("joystick");
modeAuto.onclick = () => applyModeUi("auto");

roverSelect.onchange = () => selectRover(roverSelect.value);

document.getElementById("btn-start").onclick = async () => {
  await sendCommand("session_start");
  sessionStarted = true;
};
document.getElementById("btn-stop").onclick = async () => {
  sessionStarted = false;
  queueDrive(0, 0);
  await sendCommand("session_stop");
};

document.querySelectorAll("[data-drive]").forEach((btn) => {
  btn.onclick = async () => {
    if (uiDriveMode !== "joystick" || !selectedId || !sessionStarted) return;
    const [lx, az] = btn.getAttribute("data-drive").split(",").map(Number);
    queueDrive(lx, az);
  };
});

function keyboardDrive() {
  if (!selectedId || uiDriveMode !== "joystick" || !sessionStarted) return;
  let lx = 0;
  let az = 0;
  if (keysDown.has("w") || keysDown.has("arrowup")) lx = 0.5;
  if (keysDown.has("s") || keysDown.has("arrowdown")) lx = -0.5;
  if (keysDown.has("a") || keysDown.has("arrowleft")) az = 0.7;
  if (keysDown.has("d") || keysDown.has("arrowright")) az = -0.7;
  queueDrive(lx, az);
}

function renderGamepadStatus() {
  const el = document.getElementById("gamepad-status");
  if (!el) return;
  if (!gamepadState.connected) {
    el.textContent = "Геймпад: не подключён (USB/BT → нажмите A в Chrome на этой вкладке)";
    return;
  }
  el.textContent = gamepadState.active
    ? "Геймпад: активен — левый стик Y / X"
    : "Геймпад: подключён — нажмите Start на сайте";
}

function currentGamepadCommand() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let idx = gamepadState.index;
  if (idx === null) {
    for (let i = 0; i < pads.length; i += 1) {
      if (pads[i]) {
        idx = i;
        gamepadState.index = i;
        break;
      }
    }
  }
  const pad = idx !== null ? pads[idx] : null;
  if (!pad) {
    gamepadState.connected = false;
    gamepadState.active = false;
    renderGamepadStatus();
    return { lx: 0, az: 0, connected: false };
  }
  const leftX = applyDeadzone(Number(pad.axes[0] || 0), GAMEPAD_DEADZONE);
  const leftY = applyDeadzone(Number(pad.axes[1] || 0), GAMEPAD_DEADZONE);
  const rightX = applyDeadzone(Number(pad.axes[2] || 0), GAMEPAD_DEADZONE);
  const turn = Math.abs(rightX) > 1e-3 ? rightX : leftX;
  const lx = clamp(-leftY * LINEAR_SPEED_MAX, -LINEAR_SPEED_MAX, LINEAR_SPEED_MAX);
  const az = clamp(-turn * ANGULAR_SPEED_MAX, -ANGULAR_SPEED_MAX, ANGULAR_SPEED_MAX);
  gamepadState.connected = true;
  gamepadState.active = Math.abs(lx) > 1e-3 || Math.abs(az) > 1e-3;
  renderGamepadStatus();
  return { lx, az, connected: true };
}

function gamepadLoop() {
  if (selectedId && uiDriveMode === "joystick" && sessionStarted) {
    const gp = currentGamepadCommand();
    if (gp.connected && gp.active) queueDrive(gp.lx, gp.az);
    else if (gp.connected && !gp.active) queueDrive(0, 0);
  } else {
    currentGamepadCommand();
  }
  requestAnimationFrame(gamepadLoop);
}

document.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (!["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k)) return;
  e.preventDefault();
  keysDown.add(k);
  keyboardDrive();
});

document.addEventListener("keyup", (e) => {
  keysDown.delete(e.key.toLowerCase());
  keyboardDrive();
});

function renderFleet(data) {
  lastFleet = data.rovers || [];
  renderRoverList();
  renderPanel();
}

async function loadFleet() {
  const res = await fetch("/api/rovers");
  if (res.status === 401) {
    location.href = "/login";
    return;
  }
  if (res.ok) renderFleet(await res.json());
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/dashboard`);
  ws.onmessage = (ev) => {
    try {
      renderFleet(JSON.parse(ev.data));
    } catch (_) {}
  };
  ws.onclose = () => setTimeout(connectWs, 2000);
}

document.getElementById("logout").onclick = async () => {
  await fetch("/api/logout", { method: "POST" });
  location.href = "/login";
};

loadFleet();
connectWs();
setInterval(loadFleet, 10000);
requestAnimationFrame(gamepadLoop);

window.addEventListener("gamepadconnected", (evt) => {
  gamepadState.index = evt.gamepad.index;
  gamepadState.connected = true;
  renderGamepadStatus();
});
window.addEventListener("gamepaddisconnected", (evt) => {
  if (gamepadState.index === evt.gamepad.index) {
    gamepadState.index = null;
    gamepadState.connected = false;
    gamepadState.active = false;
    queueDrive(0, 0);
    renderGamepadStatus();
  }
});
renderGamepadStatus();
