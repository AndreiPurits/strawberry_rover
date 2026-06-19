const roverSelect = document.getElementById("rover-select");
const roverList = document.getElementById("rover-list");
const roverStatus = document.getElementById("rover-status");
const emptyState = document.getElementById("empty-state");
const roverPanel = document.getElementById("rover-panel");
const panelTitle = document.getElementById("panel-title");
const panelSub = document.getElementById("panel-sub");
const panelBadge = document.getElementById("panel-badge");
const linkIndicator = document.getElementById("link-indicator");
const lidarGuardBadge = document.getElementById("lidar-guard-badge");
const operatorBadge = document.getElementById("operator-badge");
const modeJoystick = document.getElementById("mode-joystick");
const modeAuto = document.getElementById("mode-auto");
const joystickPanel = document.getElementById("joystick-panel");
const autoPanel = document.getElementById("auto-panel");

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
let lidarGuardActive = false;
let lidarOverrideActive = false;

const gamepadState = { index: null, connected: false, active: false, name: "" };
let mjpegUrl = "";
let stereoMjpegUrl = "";
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
  await fetch(`/api/rovers/${encodeURIComponent(selectedId)}/release`, {
    method: "POST",
  });
  operatorLocked = false;
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

function applyLidarGuard(fwd, override) {
  if (fwd > 0 && lidarGuardActive && !override) return 0;
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
  localStorage.setItem("axm_speed_pct", String(speedPct));
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
    lidarOverrideActive = override && lidarGuardActive;
    return { pad, fwd, turn, override, active, connected: true };
  }
  gamepadState.connected = false;
  gamepadState.active = false;
  gamepadState.name = "";
  lidarOverrideActive = false;
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

  let fwd = kb.fwd;
  let turn = kb.turn;
  let override = kb.override;
  if (gp && (gp.active || Math.abs(gp.fwd) > 0.05 || Math.abs(gp.turn) > 0.05)) {
    fwd = gp.fwd;
    turn = gp.turn;
  }
  if (gp?.override) override = true;
  pendingOverride = override;
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
  if (!moving) sendDriveFast(0, 0, pendingOverride, true);
  else sendDriveFast(smoothFwd, smoothTurn, pendingOverride, false);
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
  const op = r?.operator || {};
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
    el.textContent = "Нажмите Start → клик по ▲◀▶▼ → WASD / стрелки / геймпад";
    el.className = "session-hint muted";
    return;
  }
  el.textContent = "Управление активно · Shift/E/Пробел или R2 = снять LiDAR-стоп · LB/RB = скорость";
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

  document.querySelectorAll("[data-key-hint]").forEach((btn) => {
    const hints = (btn.getAttribute("data-key-hint") || "").split(",").filter(Boolean);
    const on = hints.some((h) => activeHints.has(h));
    btn.classList.toggle("key-active", on);
  });
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
        <em>${r.online ? "online" : `offline${r.last_seen_ago_s != null ? ` · ${Math.round(r.last_seen_ago_s)}s` : ""}`}</em>
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
  if (id !== selectedId) {
    sessionStarted = false;
    operatorLocked = false;
    mjpegUrl = "";
  }
  selectedId = id || "";
  if (persist) localStorage.setItem("axm_rover_id", selectedId);
  roverSelect.value = selectedId;
  renderRoverList();
  renderPanel();
}

function renderLidarRings(maxDisplayM) {
  const g = document.getElementById("lidar-rings");
  if (!g) return;
  g.innerHTML = "";
  const cx = 100;
  const cy = 102;
  const minR = 16;
  const maxR = 78;
  [1, 2, 3].forEach((m) => {
    if (m > maxDisplayM) return;
    const r = minR + (m / maxDisplayM) * (maxR - minR);
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

function renderLidarArc(arc) {
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

  const maxDisplayM = arc.display_max_m || arc.range_max_m || 4;
  renderLidarRings(maxDisplayM);

  const n = arc.sectors.length;
  const cx = 100;
  const cy = 102;
  const minR = 16;
  const maxR = 78;
  const fov = (arc.fov_deg || 160) * (Math.PI / 180);
  const start = -Math.PI / 2 - fov / 2;
  const step = fov / n;

  arc.sectors.forEach((sec, i) => {
    const a0 = start + i * step + 0.03;
    const a1 = start + (i + 1) * step - 0.03;
    const dist = sec.dist_m;
    if (dist == null) return;

    const clamped = Math.min(Math.max(dist, 0.05), maxDisplayM);
    const r = minR + (clamped / maxDisplayM) * (maxR - minR);
    const band = Math.max(4, ((maxR - minR) / maxDisplayM) * 0.22);

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
    path.setAttribute("class", `lidar-sector level-${sec.level || 0}`);
    g.appendChild(path);
  });
}

function renderStereoCamera(r) {
  const img = document.getElementById("stereo-camera");
  const ph = document.getElementById("stereo-camera-placeholder");
  const status = document.getElementById("stereo-cam-status");
  if (!img || !ph || !status) return;

  const online = Boolean(r?.online);
  const live = Boolean(r?.stereo_camera_live);
  if (online && live && selectedId) {
    const url = `/api/rovers/${encodeURIComponent(selectedId)}/camera/stereo/mjpeg`;
    if (stereoMjpegUrl !== url) {
      stereoMjpegUrl = url;
      img.src = url;
    }
    img.classList.remove("hidden");
    ph.classList.add("hidden");
    status.textContent = "MJPEG · RealSense";
  } else {
    if (stereoMjpegUrl) {
      img.removeAttribute("src");
      stereoMjpegUrl = "";
    }
    img.classList.add("hidden");
    ph.classList.remove("hidden");
    status.textContent = online ? "Стерео — ожидание потока" : "Стерео — нет сигнала";
  }
}

function renderFrontCamera(r) {
  const img = document.getElementById("front-camera");
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
    img.classList.remove("hidden");
    ph.classList.add("hidden");
    const age = r.link?.camera_age_ms;
    status.textContent =
      age != null ? `MJPEG · age ${Math.round(age)} ms` : "MJPEG · live";
  } else {
    if (mjpegUrl) {
      img.removeAttribute("src");
      mjpegUrl = "";
    }
    img.classList.add("hidden");
    ph.classList.remove("hidden");
    status.textContent = online ? "Камера — ожидание потока" : "Камера — нет сигнала";
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
  linkIndicator.textContent = `● RTT ${rttTxt} · frame ${ageTxt}`;
  linkIndicator.title = `Связь: ${status} (RTT ${rttTxt}, age кадра ${ageTxt})`;
}

function renderLidarGuardBadge(guard) {
  if (!lidarGuardBadge) return;
  lidarGuardActive = Boolean(guard?.active);
  if (!lidarGuardActive) {
    lidarGuardBadge.classList.add("hidden");
    lidarGuardBadge.classList.remove("override");
    return;
  }
  lidarGuardBadge.classList.remove("hidden");
  if (lidarOverrideActive) {
    lidarGuardBadge.textContent = "LiDAR OVERRIDE (R2)";
    lidarGuardBadge.classList.add("override");
  } else {
    const m = guard.min_forward_m;
    lidarGuardBadge.textContent = "LiDAR СТОП — Shift/E/R2 чтобы ехать";
    lidarGuardBadge.classList.remove("override");
  }
}

function renderOperatorBadge(op) {
  if (!operatorBadge) return;
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
  renderLidarArc(p.lidar_arc);
  renderFrontCamera(r);
  renderStereoCamera(r);
  renderLidarGuardBadge(p.lidar_guard);
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

  renderLinkIndicator(r);
  renderOperatorBadge(r.operator);

  document.getElementById("t-mega").textContent =
    t.arduino_connected === true ? "подключена" : t.arduino_connected === false ? "нет" : "—";
  document.getElementById("t-arm").textContent =
    m.armed === true ? "ARM — моторы разрешены" : m.armed === false ? "DISARM — стоп" : "—";

  const trackDir = (us) => (us > 1505 ? "вперёд" : us < 1495 ? "назад" : "нейтраль");
  const trackLine = (us, power, pct) => {
    if (us == null) return "—";
    const p = power != null ? Math.abs(Number(power)) : Math.abs(pct ?? 0);
    return `${us} µs · ${trackDir(us)} · мощность ${p}%`;
  };
  document.getElementById("s-left").textContent = trackLine(m.left_us, m.left_power_pct, m.left_pct);
  document.getElementById("s-right").textContent = trackLine(m.right_us, m.right_power_pct, m.right_pct);
  document.getElementById("s-speed").textContent =
    m.speed_mps != null
      ? `~${m.speed_mps} m/s (средняя мощность ${Math.abs(m.linear_pct ?? 0)}%)`
      : "0 m/s";
  document.getElementById("s-current-a0").textContent =
    m.current_a0 != null ? `ADC ${m.current_a0} (0–1023, без калибр.)` : "—";
  document.getElementById("s-current-d22").textContent =
    m.current_d22 != null ? `D22=${m.current_d22} (цифра)` : m.current_d0 != null ? `D0=${m.current_d0}` : "—";
  document.getElementById("s-temp").textContent =
    m.temp_c != null ? `${m.temp_c} °C` : m.dht_ok === false ? "нет ответа (D23)" : "—";
  document.getElementById("s-humidity").textContent =
    m.humidity_pct != null ? `${m.humidity_pct} %` : m.dht_ok === false ? "—" : "—";
  document.getElementById("s-vibration").textContent =
    m.vibration_d24 != null
      ? m.vibration_d24
        ? "сработал (HIGH)"
        : "нет (LOW)"
      : "—";
  const rtk = t.rtk || {};
  const rtkEl = document.getElementById("s-rtk");
  if (rtkEl) {
    if (rtk.lat != null && rtk.lon != null) {
      const fix = rtk.fix_label || "—";
      const sats = rtk.satellites != null ? `${rtk.satellites} спутн.` : "—";
      rtkEl.textContent = `${fix} · ${rtk.lat.toFixed(5)}, ${rtk.lon.toFixed(5)} · ${sats}`;
    } else if (rtk.connected) {
      rtkEl.textContent = "подключён — ожидание fix";
    } else {
      rtkEl.textContent = rtk.error ? String(rtk.error) : "нет данных";
    }
  }
  renderPerception(r);
  document.getElementById("t-ago").textContent = fmtAgo(r.last_seen_ago_s);

  const mode = t.drive_mode || uiDriveMode;
  applyModeUi(mode, false);

  const online = r.online;
  const op = r.operator || {};
  const canControl = online && (!op.locked || op.you);
  document.getElementById("btn-start").disabled = !canControl || mode !== "joystick";
  document.getElementById("btn-stop").disabled = !online;
  modeJoystick.disabled = !online;
  modeAuto.disabled = !online;
  document.querySelectorAll("[data-fwd]").forEach((b) => {
    b.disabled = !canControl || mode !== "joystick" || !sessionStarted;
  });
  renderSessionHint();
  updateDriveKeyHighlight();
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
  const op = r?.operator || {};
  const canControl = online && (!op.locked || op.you);
  document.getElementById("btn-start").disabled = !canControl || mode !== "joystick";
  document.querySelectorAll("[data-fwd]").forEach((b) => {
    b.disabled = !canControl || mode !== "joystick" || !sessionStarted;
  });
  renderSessionHint();
}

modeJoystick.onclick = () => applyModeUi("joystick");
modeAuto.onclick = () => applyModeUi("auto");

roverSelect.onchange = () => selectRover(roverSelect.value);

document.getElementById("btn-start").onclick = async () => {
  wakeGamepads();
  const claim = await claimOperator();
  if (!claim.ok) {
    if (claim.conflict) alert("Ровер уже управляется другим оператором");
    return;
  }
  await sendCommand("session_start");
  sessionStarted = true;
  drivePad?.focus();
  renderPanel();
};

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
  updateDriveKeyHighlight();
}

function renderGamepadStatus() {
  const el = document.getElementById("gamepad-status");
  if (!el) return;
  if (!navigator.getGamepads) {
    el.textContent = "Геймпад: браузер не поддерживает Gamepad API";
    return;
  }
  if (uiDriveMode !== "joystick") {
    if (gamepadState.connected) {
      el.textContent = `Геймпад: ${gamepadState.name} — режим AUTO, езда заблокирована`;
    } else {
      el.textContent = "Геймпад: режим AUTO — переключите на Joystick для ручного управления";
    }
    return;
  }
  if (!sessionStarted) {
    if (gamepadState.connected) {
      el.textContent = `Геймпад: ${gamepadState.name} — нажмите Start на сайте`;
    } else {
      el.textContent = "Геймпад: USB/BT к этому ПК → кнопка «Разбудить» или Start";
    }
    return;
  }
  if (!gamepadState.connected) {
    el.textContent = "Геймпад: не виден — подключите к ПК с браузером, нажмите «Разбудить»";
    return;
  }
  const guardHint = lidarGuardActive ? " · R2 = снять стоп" : "";
  const speedHint = " · LB/RB = скорость";
  el.textContent = gamepadState.active
    ? `Геймпад: ${gamepadState.name} — стик / D-pad / RT${guardHint}${speedHint}`
    : `Геймпад: ${gamepadState.name} — двигайте стик${guardHint}${speedHint}`;
}

function gamepadR2Pressed(pad) {
  const buttons = pad.buttons || [];
  if (buttons[7]?.pressed) return true;
  if (buttons[5]?.pressed) return true;
  if (pad.axes.length > 5 && pad.axes[5] > 0.5) return true;
  return false;
}

document.getElementById("btn-wake-gamepad")?.addEventListener("click", () => {
  wakeGamepads();
  drivePad?.focus();
});

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
  speedSlider.value = String(speedPct);
  if (speedLabel) speedLabel.textContent = `${speedPct}%`;
  speedSlider.oninput = () => {
    speedPct = Number(speedSlider.value);
    localStorage.setItem("axm_speed_pct", String(speedPct));
    if (speedLabel) speedLabel.textContent = `${speedPct}%`;
  };
}

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
  fleetWs = ws;
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === "fleet") renderFleet(data);
    } catch (_) {}
  };
  ws.onclose = () => {
    fleetWs = null;
    setTimeout(connectWs, 2000);
  };
}

document.getElementById("logout").onclick = async () => {
  if (sessionStarted) {
    await sendCommand("session_stop");
    await releaseOperator();
  }
  await fetch("/api/logout", { method: "POST" });
  location.href = "/login";
};

loadFleet();
connectWs();
setInterval(loadFleet, 10000);
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
