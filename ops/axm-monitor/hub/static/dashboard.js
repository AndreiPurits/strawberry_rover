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
      ? "Ручное управление: WASD, стрелки или кнопки ниже."
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

document.getElementById("btn-start").onclick = () => sendCommand("session_start");
document.getElementById("btn-stop").onclick = () => sendCommand("session_stop");

document.querySelectorAll("[data-drive]").forEach((btn) => {
  btn.onclick = async () => {
    if (uiDriveMode !== "joystick" || !selectedId) return;
    const [lx, az] = btn.getAttribute("data-drive").split(",").map(Number);
    if (lx === 0 && az === 0) await sendCommand("stop_drive");
    else await sendCommand("drive", { linear_x: lx, angular_z: az });
  };
});

function keyboardDrive() {
  if (!selectedId || uiDriveMode !== "joystick") return;
  let lx = 0;
  let az = 0;
  if (keysDown.has("w") || keysDown.has("arrowup")) lx = 0.5;
  if (keysDown.has("s") || keysDown.has("arrowdown")) lx = -0.5;
  if (keysDown.has("a") || keysDown.has("arrowleft")) az = 0.7;
  if (keysDown.has("d") || keysDown.has("arrowright")) az = -0.7;
  if (lx === 0 && az === 0) sendCommand("stop_drive");
  else sendCommand("drive", { linear_x: lx, angular_z: az });
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
