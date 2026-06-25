const logEl = document.getElementById("roarm-log");
const jsonEl = document.getElementById("roarm-json");
const statusEl = document.getElementById("roarm-status");
let busy = false;

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (!logEl) return;
  logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 6000);
}

async function rpc(op, params = {}) {
  if (busy) {
    log("занято — подождите");
    return null;
  }
  busy = true;
  document.querySelectorAll(".btn").forEach((b) => (b.disabled = true));
  try {
    const res = await fetch("/api/roarm/rpc", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op, params }),
    });
    if (res.status === 401) {
      location.href = "/login?next=/roarm";
      return null;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      log(`${op} ERROR ${data.detail || res.status}`);
      return data;
    }
    log(`${op} OK`);
    return data;
  } catch (err) {
    log(`${op} FAIL ${err}`);
    return null;
  } finally {
    busy = false;
    document.querySelectorAll(".btn").forEach((b) => (b.disabled = false));
  }
}

function setOnline(ok, hint = "") {
  if (!statusEl) return;
  statusEl.textContent = ok ? `online ${hint}`.trim() : `offline ${hint}`.trim();
  statusEl.classList.toggle("online", ok);
  statusEl.classList.toggle("offline", !ok);
}

async function refreshStatus() {
  const data = await rpc("status");
  if (!data) {
    setOnline(false, "нет ответа");
    return;
  }
  if (data.ok === false || data.error) {
    setOnline(false, data.error || "");
    if (jsonEl) jsonEl.textContent = JSON.stringify(data, null, 2);
    return;
  }
  const st = data.status || data.result?.status || data;
  setOnline(true);
  if (jsonEl) jsonEl.textContent = JSON.stringify(st, null, 2);
}

function num(id) {
  return Number(document.getElementById(id)?.value || 0);
}

document.getElementById("btn-refresh")?.addEventListener("click", refreshStatus);
document.getElementById("btn-home")?.addEventListener("click", () => rpc("home"));
document.getElementById("btn-torque-on")?.addEventListener("click", () => rpc("torque_on"));
document.getElementById("btn-torque-off")?.addEventListener("click", () => rpc("torque_off"));
document.getElementById("btn-grip-open")?.addEventListener("click", () => rpc("gripper_open"));
document.getElementById("btn-grip-close")?.addEventListener("click", () => rpc("gripper_close"));
document.getElementById("btn-move")?.addEventListener("click", () =>
  rpc("move_xyz", {
    x: num("mv-x"),
    y: num("mv-y"),
    z: num("mv-z"),
    t: num("mv-t"),
    r: num("mv-r"),
    g: num("mv-g"),
    spd: num("mv-spd"),
  })
);

document.getElementById("btn-logout")?.addEventListener("click", () => {
  fetch("/api/logout", { method: "POST", credentials: "same-origin" })
    .catch(() => {})
    .finally(() => {
      location.href = "/login";
    });
});

refreshStatus();
setInterval(refreshStatus, 15000);
