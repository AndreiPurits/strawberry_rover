const grid = document.getElementById("rover-grid");
const onlineCount = document.getElementById("online-count");
const totalCount = document.getElementById("total-count");

function fmtAgo(sec) {
  if (sec == null) return "—";
  if (sec < 5) return "сейчас";
  if (sec < 60) return `${Math.round(sec)} с назад`;
  return `${Math.round(sec / 60)} мин назад`;
}

function renderFleet(data) {
  const rovers = data.rovers || [];
  totalCount.textContent = String(rovers.length);
  onlineCount.textContent = String(rovers.filter((r) => r.online).length);
  grid.innerHTML = rovers.length
    ? rovers.map(renderCard).join("")
    : `<div class="empty">Роверы пока не подключены. Запустите fleet-agent на Orin.</div>`;
}

function renderCard(r) {
  const t = r.telemetry || {};
  const statusClass = r.online ? "online" : "offline";
  return `
    <article class="card rover-card ${statusClass}">
      <header>
        <h2>${escapeHtml(r.name || r.id)}</h2>
        <span class="badge ${statusClass}">${r.online ? "ONLINE" : "OFFLINE"}</span>
      </header>
      <dl>
        <div><dt>ID</dt><dd>${escapeHtml(r.id)}</dd></div>
        <div><dt>Сигнал</dt><dd>${fmtAgo(r.last_seen_ago_s)}</dd></div>
        <div><dt>Arduino</dt><dd>${escapeHtml(String(t.arduino_connected ?? "—"))}</dd></div>
        <div><dt>ARM</dt><dd>${escapeHtml(String(t.armed ?? "—"))}</dd></div>
        <div><dt>Хост</dt><dd>${escapeHtml(String(t.hostname ?? "—"))}</dd></div>
        <div><dt>Web UI</dt><dd>${t.local_web_url ? `<a href="${escapeHtml(t.local_web_url)}" target="_blank" rel="noopener">локально</a>` : "—"}</dd></div>
      </dl>
      <pre class="telemetry">${escapeHtml(JSON.stringify(t, null, 2))}</pre>
    </article>
  `;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadFleet() {
  const res = await fetch("/api/rovers");
  if (res.status === 401) {
    location.href = "/login";
    return;
  }
  renderFleet(await res.json());
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

document.getElementById("logout").addEventListener("click", async () => {
  await fetch("/api/logout", { method: "POST" });
  location.href = "/login";
});

loadFleet();
connectWs();
