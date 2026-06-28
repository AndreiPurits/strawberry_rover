/** RoArm operator panel — embedded in main dashboard (sidebar navigation). */
(function () {
  const ROARM_ID = "roarm-01";
  const LS_KEY = "axm_roarm_sequence_v1";

  let savedPoints = [];
  let selectedPointId = null;
  let lastFeedback = null;

  const ptListEl = () => document.getElementById("pt-list");
  const ptStatusEl = () => document.getElementById("pt-status");

  function setPtStatus(msg) {
    const el = ptStatusEl();
    if (el) el.textContent = msg;
  }

  function feedbackToPose(fb) {
    if (!fb || typeof fb !== "object") return null;
    const pick = (...keys) => {
      for (const k of keys) {
        if (fb[k] !== undefined && fb[k] !== null && Number.isFinite(Number(fb[k]))) return Number(fb[k]);
      }
      return undefined;
    };
    const joints = {
      base: pick("b", "base"),
      shoulder: pick("s", "shoulder"),
      elbow: pick("e", "elbow"),
      wrist: pick("t", "wrist"),
      roll: pick("r", "roll"),
      hand: pick("g", "hand"),
    };
    const xyz = {
      x: pick("x"),
      y: pick("y"),
      z: pick("z"),
      t: pick("tit", "t"),
      r: pick("r"),
      g: pick("g"),
    };
    if (Object.values(joints).every((v) => v === undefined) && Object.values(xyz).every((v) => v === undefined)) {
      return null;
    }
    return { joints, xyz };
  }

  function pointSummary(pt) {
    if (pt.mode === "xyz" && pt.xyz) {
      const x = pt.xyz.x ?? "—";
      const y = pt.xyz.y ?? "—";
      const z = pt.xyz.z ?? "—";
      return `XYZ ${x}/${y}/${z}`;
    }
    if (pt.joints) {
      const j = pt.joints;
      return `J S=${Number(j.shoulder ?? 0).toFixed(2)} E=${Number(j.elbow ?? 0).toFixed(2)}`;
    }
    return pt.mode || "—";
  }

  function renderPointsList() {
    const list = ptListEl();
    const countEl = document.getElementById("pt-count");
    if (!list) return;
    list.innerHTML = "";
    if (countEl) countEl.textContent = savedPoints.length ? `${savedPoints.length} шт.` : "";
    savedPoints.forEach((pt) => {
      const li = document.createElement("li");
      li.dataset.id = pt.id;
      if (pt.id === selectedPointId) li.classList.add("selected");
      const name = document.createElement("span");
      name.textContent = pt.name || pt.id;
      li.appendChild(name);
      if (pt.role === "home") {
        const badge = document.createElement("span");
        badge.className = "pt-badge";
        badge.textContent = "HOME";
        li.appendChild(badge);
      }
      const meta = document.createElement("span");
      meta.className = "pt-meta";
      meta.textContent = pointSummary(pt);
      li.appendChild(meta);
      li.addEventListener("click", () => {
        selectedPointId = pt.id;
        renderPointsList();
        const nameInput = document.getElementById("pt-name");
        if (nameInput) nameInput.value = pt.name || "";
        const modeSel = document.getElementById("pt-mode");
        if (modeSel) modeSel.value = pt.mode || "joints";
      });
      list.appendChild(li);
    });
  }

  async function loadPoints() {
    try {
      const res = await fetch("/api/roarm/points", { credentials: "same-origin" });
      if (res.status === 401) {
        location.href = "/login";
        return;
      }
      const data = await res.json().catch(() => ({}));
      savedPoints = Array.isArray(data.points) ? data.points : [];
      if (!selectedPointId && savedPoints.length) {
        const homePt = savedPoints.find((p) => p.role === "home");
        selectedPointId = (homePt || savedPoints[0]).id;
      }
      renderPointsList();
    } catch (err) {
      setPtStatus(`Ошибка загрузки точек: ${err}`);
    }
  }

  async function postPoint(body) {
    const res = await fetch("/api/roarm/points", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 401) {
      location.href = "/login";
      return null;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setPtStatus(`Ошибка: ${data.detail || res.status}`);
      return null;
    }
    savedPoints = data.points || savedPoints;
    if (data.point) selectedPointId = data.point.id;
    renderPointsList();
    return data.point;
  }

  async function captureFromArm() {
    setPtStatus("Читаю позицию T:105…");
    const data = await rpc("feedback", {}, { blocking: true });
    if (!data || data.ok === false) {
      setPtStatus("Не удалось считать позицию");
      return null;
    }
    const fb = data.feedback || data.status || data.result?.feedback;
    lastFeedback = fb;
    if (jsonEl) jsonEl.textContent = JSON.stringify(fb, null, 2);
    return feedbackToPose(fb);
  }

  async function savePointFromArm() {
    const name = String(document.getElementById("pt-name")?.value || "").trim();
    if (!name) {
      setPtStatus("Введите название точки");
      return;
    }
    const pose = await captureFromArm();
    if (!pose) return;
    const mode = String(document.getElementById("pt-mode")?.value || "joints");
    const pt = await postPoint({
      name,
      mode,
      joints: pose.joints,
      xyz: pose.xyz,
      role: name.toLowerCase() === "home" ? "home" : null,
    });
    if (pt) {
      setPtStatus(`Сохранено: ${pt.name}`);
      log(`point saved: ${pt.name}`);
    }
  }

  async function savePointFromForm() {
    const name = String(document.getElementById("pt-name")?.value || "").trim();
    if (!name) {
      setPtStatus("Введите название точки");
      return;
    }
    const pt = await postPoint({
      name,
      mode: "xyz",
      xyz: {
        x: num("mv-x"),
        y: num("mv-y"),
        z: num("mv-z"),
        t: num("mv-t"),
        r: num("mv-r"),
        g: num("mv-g"),
      },
      xyz_spd: num("mv-spd"),
      role: name.toLowerCase() === "home" ? "home" : null,
    });
    if (pt) {
      setPtStatus(`Сохранено из формы: ${pt.name}`);
      log(`point saved from form: ${pt.name}`);
    }
  }

  function selectedPoint() {
    return savedPoints.find((p) => p.id === selectedPointId) || null;
  }

  async function goToPoint(pt) {
    if (!pt) {
      setPtStatus("Выберите точку в списке");
      return null;
    }
    if (pt.mode === "xyz" && pt.xyz) {
      setPtStatus(`→ ${pt.name} (T:104)`);
      return rpc("move_xyz", {
        x: pt.xyz.x ?? 0,
        y: pt.xyz.y ?? 0,
        z: pt.xyz.z ?? 0,
        t: pt.xyz.t ?? 0,
        r: pt.xyz.r ?? 0,
        g: pt.xyz.g ?? 3.14,
        spd: pt.xyz_spd ?? 0.25,
      });
    }
    if (pt.joints) {
      setPtStatus(`→ ${pt.name} (T:102)`);
      const j = pt.joints;
      return rpc("home_joints", {
        base: j.base ?? 0,
        shoulder: j.shoulder ?? 0,
        elbow: j.elbow ?? 1.57,
        wrist: j.wrist ?? 0,
        roll: j.roll ?? 0,
        hand: j.hand ?? 3.14,
        spd: pt.joint_spd ?? 0,
        acc: pt.joint_acc ?? 10,
      });
    }
    setPtStatus("У точки нет координат");
    return null;
  }

  async function goHome() {
    const homePt = savedPoints.find((p) => p.role === "home");
    if (homePt) {
      setPtStatus("HOME → сохранённая точка");
      return goToPoint(homePt);
    }
    setPtStatus("HOME → T:100 (нет точки ★ HOME)");
    return rpc("home");
  }

  async function setSelectedAsHome() {
    const pt = selectedPoint();
    if (!pt) {
      setPtStatus("Выберите точку");
      return;
    }
    const res = await fetch(`/api/roarm/points/${encodeURIComponent(pt.id)}/set-home`, {
      method: "POST",
      credentials: "same-origin",
    });
    if (res.status === 401) {
      location.href = "/login";
      return;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setPtStatus(`Ошибка: ${data.detail || res.status}`);
      return;
    }
    savedPoints = data.points || savedPoints;
    renderPointsList();
    setPtStatus(`★ HOME = ${pt.name}`);
    log(`home point set: ${pt.name}`);
  }

  async function commitSelectedToServo() {
    const pt = selectedPoint();
    if (!pt) {
      setPtStatus("Выберите точку");
      return;
    }
    const ok = window.confirm(
      `Записать «${pt.name}» в сервоприводы (T:502)?\n\n` +
        "Рука перейдёт в позицию → T:502 запомнит её для включения питания."
    );
    if (!ok) return;
    if (pt.role !== "home") {
      await setSelectedAsHome();
    }
    const moved = await goToPoint(pt);
    if (!moved || moved.ok === false) {
      setPtStatus("Не удалось перейти — T:502 отменён");
      return;
    }
    await new Promise((r) => setTimeout(r, 4000));
    setPtStatus("T:502…");
    const data = await rpc("set_servo_middle", {});
    if (data && data.ok !== false) {
      setPtStatus(`HOME записан в серво: ${pt.name}`);
      log(`servo middle: ${pt.name}`);
    } else {
      setPtStatus("Ошибка T:502");
    }
  }

  async function deleteSelectedPoint() {
    const pt = selectedPoint();
    if (!pt) {
      setPtStatus("Выберите точку");
      return;
    }
    if (!window.confirm(`Удалить точку «${pt.name}»?`)) return;
    const res = await fetch(`/api/roarm/points/${encodeURIComponent(pt.id)}`, {
      method: "DELETE",
      credentials: "same-origin",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setPtStatus(`Ошибка: ${data.detail || res.status}`);
      return;
    }
    savedPoints = data.points || [];
    selectedPointId = savedPoints[0]?.id || null;
    renderPointsList();
    setPtStatus("Точка удалена");
  }


  const logEl = document.getElementById("roarm-log");
  const jsonEl = document.getElementById("roarm-json");
  const reachEl = document.getElementById("reachability");
  const seqTable = document.querySelector("#seq-table tbody");
  const seqProgress = document.getElementById("seq-progress");
  const panelRoot = document.getElementById("roarm-panel");

  let motionBusy = false;
  let sequencePoll = null;
  let initialized = false;

  const FPS_EXAMPLE = [
    { type: "home" },
    { type: "target", x: 58, y: 200, z: -97, t: 1.57, r: 1.57, g: 0 },
    { type: "target", x: 58, y: 200, z: -97, t: 1.57, r: 1.57, g: 3.14 },
    { type: "target", x: -388, y: 110, z: 67, t: 1.57, r: 1.57, g: 3.14 },
    { type: "target", x: -388, y: 110, z: 67, t: 1.57, r: 1.57, g: 0 },
    { type: "home" },
  ];

  function panelButtons() {
    return panelRoot ? panelRoot.querySelectorAll(".btn") : [];
  }

  function log(msg, target = logEl) {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (!target) return;
    target.textContent = `${line}\n${target.textContent}`.slice(0, 8000);
  }

  function seqLog(msg) {
    log(msg);
  }

  async function rpc(op, params = {}, opts = {}) {
    const blocking = opts.blocking !== false;
    if (blocking && motionBusy) {
      log("занято — подождите");
      return null;
    }
    if (blocking) {
      motionBusy = true;
      panelButtons().forEach((b) => { b.disabled = true; });
    }
    try {
      const res = await fetch("/api/roarm/rpc", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op, params }),
      });
      if (res.status === 401) { location.href = "/login"; return null; }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail || data.error || res.status;
        log(`${op} ERROR ${detail}`);
        if (detail === "roarm_gateway_offline") log("Orin offline — проверьте fleet-agent");
        return data;
      }
      if (blocking && op !== "sequence_status") log(`${op} OK`);
      return data;
    } catch (err) {
      log(`${op} FAIL ${err}`);
      return null;
    } finally {
      if (blocking) {
        motionBusy = false;
        panelButtons().forEach((b) => { b.disabled = false; });
      }
    }
  }

  function num(id) {
    return Number(document.getElementById(id)?.value || 0);
  }

  function reachabilityText(x, y, z) {
    const r = Math.sqrt(x * x + y * y);
    const reasons = [];
    if (r < 80) reasons.push("r < 80");
    if (r > 420) reasons.push("r > 420");
    if (z < -100) reasons.push("z < -100");
    if (z > 380) reasons.push("z > 380");
    if (reasons.length) return { text: `risky: ${reasons.join("; ")}`, cls: "risky" };
    return { text: "within soft heuristic zone", cls: "ok" };
  }

  function updateReachability() {
    const info = reachabilityText(num("mv-x"), num("mv-y"), num("mv-z"));
    if (reachEl) {
      reachEl.textContent = `Reachability: ${info.text}`;
      reachEl.className = `reachability ${info.cls}`;
    }
  }

  async function refreshStatus() {
    const data = await rpc("status", {}, { blocking: false });
    if (!data) return;
    if (data.ok === false || data.error) {
      if (jsonEl) jsonEl.textContent = JSON.stringify(data, null, 2);
      return;
    }
    const st = data.status || data.feedback || data.result?.status || data.result?.feedback;
    if (st && typeof st === "object") lastFeedback = st;
    if (jsonEl) jsonEl.textContent = JSON.stringify(st || data, null, 2);
  }

  function emptyRow(type = "target") {
    if (type === "home") {
      return { type: "home", x: "", y: "", z: "", t: "", r: "", g: "", mode: "100", reach: "home step", status: "" };
    }
    if (type === "gripper") {
      return { type: "gripper", x: "", y: "", z: "", t: "", r: "", g: "", mode: "open", reach: "gripper step", status: "" };
    }
    if (type === "delay") {
      return { type: "delay", x: "1", y: "", z: "", t: "", r: "", g: "", mode: "delay", reach: "delay step", status: "" };
    }
    return {
      type: "target",
      x: String(num("mv-x")),
      y: String(num("mv-y")),
      z: String(num("mv-z")),
      t: String(num("mv-t")),
      r: String(num("mv-r")),
      g: String(num("mv-g")),
      mode: "104",
      reach: "",
      status: "",
    };
  }

  function renderSeqTable(rows) {
    if (!seqTable) return;
    seqTable.innerHTML = "";
    rows.forEach((row, idx) => {
      const tr = document.createElement("tr");
      tr.dataset.idx = String(idx);
      ["type", "x", "y", "z", "t", "r", "g", "mode"].forEach((c) => {
        const td = document.createElement("td");
        const input = document.createElement("input");
        input.value = row[c] ?? "";
        input.dataset.col = c;
        input.addEventListener("change", () => refreshRowReach(idx));
        td.appendChild(input);
        tr.appendChild(td);
      });
      const reachTd = document.createElement("td");
      reachTd.className = "reach-cell";
      reachTd.textContent = row.reach || "";
      tr.appendChild(reachTd);
      const statusTd = document.createElement("td");
      statusTd.className = "status-cell";
      statusTd.textContent = row.status || "";
      tr.appendChild(statusTd);
      tr.addEventListener("click", () => {
        seqTable.querySelectorAll("tr").forEach((r) => r.classList.remove("selected"));
        tr.classList.add("selected");
      });
      seqTable.appendChild(tr);
      refreshRowReach(idx);
    });
  }

  function getSelectedIdx() {
    const sel = seqTable?.querySelector("tr.selected");
    return sel ? Number(sel.dataset.idx) : -1;
  }

  function readTableRows() {
    const rows = [];
    seqTable?.querySelectorAll("tr").forEach((tr) => {
      const row = { type: "", x: "", y: "", z: "", t: "", r: "", g: "", mode: "" };
      tr.querySelectorAll("input").forEach((inp) => {
        row[inp.dataset.col] = inp.value;
      });
      row.reach = tr.querySelector(".reach-cell")?.textContent || "";
      row.status = tr.querySelector(".status-cell")?.textContent || "";
      rows.push(row);
    });
    return rows;
  }

  function refreshRowReach(idx) {
    const tr = seqTable?.querySelector(`tr[data-idx="${idx}"]`);
    if (!tr) return;
    const vals = {};
    tr.querySelectorAll("input").forEach((inp) => {
      vals[inp.dataset.col] = inp.value;
    });
    const reachCell = tr.querySelector(".reach-cell");
    const t = String(vals.type || "").toLowerCase();
    if (t === "home") {
      reachCell.textContent = "home step";
      return;
    }
    if (t === "gripper") {
      reachCell.textContent = "gripper step";
      return;
    }
    if (t === "delay") {
      reachCell.textContent = "delay step";
      return;
    }
    try {
      reachCell.textContent = reachabilityText(Number(vals.x), Number(vals.y), Number(vals.z)).text;
    } catch {
      reachCell.textContent = "invalid point";
    }
  }

  function rowsToSequence(rows) {
    const seq = [];
    rows.forEach((vals, i) => {
      const t = String(vals.type || "target").toLowerCase();
      if (t === "home") {
        seq.push({ type: "home" });
        return;
      }
      if (t === "gripper") {
        let action = String(vals.mode || "open").toLowerCase();
        if (action !== "open" && action !== "close") action = "open";
        seq.push({ type: "gripper", action });
        return;
      }
      if (t === "delay") {
        const sec = Number(vals.x || 1);
        if (!Number.isFinite(sec)) throw new Error(`Invalid delay row ${i + 1}`);
        seq.push({ type: "delay", sec });
        return;
      }
      seq.push({
        type: "target",
        x: Number(vals.x),
        y: Number(vals.y),
        z: Number(vals.z),
        t: Number(vals.t || 0),
        r: Number(vals.r || 0),
        g: Number(vals.g || 3.14),
      });
    });
    return seq;
  }

  function loadSequenceJson(raw) {
    const rows = [];
    raw.forEach((step) => {
      if (step.type === "home") {
        rows.push(emptyRow("home"));
        return;
      }
      if (step.type === "gripper") {
        const r = emptyRow("gripper");
        r.mode = step.action || "open";
        rows.push(r);
        return;
      }
      if (step.type === "delay") {
        const r = emptyRow("delay");
        r.x = String(step.sec ?? 1);
        rows.push(r);
        return;
      }
      rows.push({
        type: "target",
        x: String(step.x ?? 235),
        y: String(step.y ?? 0),
        z: String(step.z ?? 234),
        t: String(step.t ?? 0),
        r: String(step.r ?? 0),
        g: String(step.g ?? 3.14),
        mode: "104",
        reach: "",
        status: "",
      });
    });
    renderSeqTable(rows);
  }

  function updateSeqUiFromStatus(st) {
    const idx = Number(st.index ?? -1);
    const total = Number(st.total ?? 0);
    if (seqProgress) {
      seqProgress.textContent = st.running
        ? `step ${idx + 1}/${total} · ${st.step_status || ""}`
        : st.result
          ? `finished: ${st.result}`
          : "";
    }
    seqTable?.querySelectorAll("tr").forEach((tr, i) => {
      const cell = tr.querySelector(".status-cell");
      if (!cell) return;
      if (!st.running && !st.result) {
        if (!cell.textContent) cell.textContent = "";
        return;
      }
      if (i < idx) cell.textContent = "done";
      else if (i === idx && st.running) cell.textContent = st.step_status?.startsWith("error") ? "error" : "running";
      else if (st.result && i <= idx) cell.textContent = st.step_status === "error" && i === idx ? "error" : "done";
    });
    (st.log || []).slice(-5).forEach((line) => {
      if (!logEl?.textContent.includes(line)) seqLog(line);
    });
  }

  async function pollSequenceStatus() {
    const data = await rpc("sequence_status", {}, { blocking: false });
    if (!data) return;
    updateSeqUiFromStatus(data);
    if (!data.running) {
      clearInterval(sequencePoll);
      sequencePoll = null;
      motionBusy = false;
      panelButtons().forEach((b) => {
        b.disabled = false;
      });
      seqLog(`sequence end: ${data.result || "done"}${data.error ? ` (${data.error})` : ""}`);
    }
  }

  function renderFleetStatus(device) {
    if (!device) return;
    const rt = (device.telemetry || {}).roarm || {};
    const tcpOk = Boolean(rt.tcp_open);
    const proxyOk = Boolean(device.parent_online ?? device.online);
    const armOk = Boolean(rt.reachable);
    const ip = rt.ip || "";

    const proxyBadge = document.getElementById("roarm-proxy-badge");
    const armBadge = document.getElementById("roarm-arm-badge");
    const ipBadge = document.getElementById("roarm-ip-badge");
    const title = document.getElementById("roarm-panel-title");
    const sub = document.getElementById("roarm-panel-sub");

    if (title) title.textContent = device.name || ROARM_ID;
    if (sub) sub.textContent = armOk ? `RoArm-M3 · ${ip}` : `Orin proxy · ${ip || "—"}`;
    if (ipBadge) ipBadge.textContent = ip ? `${ip}` : "";

    if (proxyBadge) {
      proxyBadge.textContent = proxyOk ? "Orin online" : "Orin offline";
      proxyBadge.className = `link-badge ${proxyOk ? "link-green" : "link-red"}`;
    }
    if (armBadge) {
      if (armOk) {
        armBadge.textContent = "ARM HTTP OK";
        armBadge.className = "badge online";
        armBadge.title = "";
      } else if (tcpOk) {
        armBadge.textContent = "HTTP busy";
        armBadge.className = "badge warn";
        armBadge.title =
          rt.error ||
          "Порт 80 открыт, но JSON не пришёл — перезагрузите RoArm или проверьте Wi‑Fi";
      } else if (proxyOk) {
        armBadge.textContent = "ARM offline";
        armBadge.className = "badge warn";
        armBadge.title = rt.error || "Нет TCP :80 до RoArm";
      } else {
        armBadge.textContent = "ARM HTTP";
        armBadge.className = "badge offline";
      }
    }

    const topRover = document.getElementById("topbar-rover");
    if (topRover) {
      topRover.textContent = proxyOk
        ? armOk
          ? `${device.name || ROARM_ID} · online`
          : `${device.name || ROARM_ID} · proxy online, arm offline`
        : `${device.name || ROARM_ID} · offline`;
    }
  }

  function bindOnce() {
    if (initialized) return;
    initialized = true;

    document.getElementById("mv-show-trg")?.addEventListener("change", (e) => {
      document.getElementById("mv-trg-fields")?.classList.toggle("hidden", !e.target.checked);
    });
    ["mv-x", "mv-y", "mv-z"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", updateReachability);
    });

    document.getElementById("btn-refresh")?.addEventListener("click", refreshStatus);
    document.getElementById("btn-home")?.addEventListener("click", () => goHome());
    document.getElementById("pt-save-arm")?.addEventListener("click", () => savePointFromArm());
    document.getElementById("pt-save-form")?.addEventListener("click", () => savePointFromForm());
    document.getElementById("pt-go")?.addEventListener("click", () => goToPoint(selectedPoint()));
    document.getElementById("pt-set-home")?.addEventListener("click", () => setSelectedAsHome());
    document.getElementById("pt-servo")?.addEventListener("click", () => commitSelectedToServo());
    document.getElementById("pt-del")?.addEventListener("click", () => deleteSelectedPoint());
    document.getElementById("btn-torque-on")?.addEventListener("click", () => rpc("torque_on"));
    document.getElementById("btn-torque-off")?.addEventListener("click", () => rpc("torque_off"));
    document.getElementById("btn-grip-open")?.addEventListener("click", () => rpc("gripper_open"));
    document.getElementById("btn-grip-close")?.addEventListener("click", () => rpc("gripper_close"));
    document.getElementById("btn-check")?.addEventListener("click", updateReachability);
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

    document.getElementById("seq-add-target")?.addEventListener("click", () => {
      renderSeqTable([...readTableRows(), emptyRow("target")]);
    });
    document.getElementById("seq-add-home")?.addEventListener("click", () => {
      renderSeqTable([...readTableRows(), emptyRow("home")]);
    });
    document.getElementById("seq-add-gripper")?.addEventListener("click", () => {
      renderSeqTable([...readTableRows(), emptyRow("gripper")]);
    });
    document.getElementById("seq-add-delay")?.addEventListener("click", () => {
      renderSeqTable([...readTableRows(), emptyRow("delay")]);
    });
    document.getElementById("seq-del")?.addEventListener("click", () => {
      const idx = getSelectedIdx();
      if (idx < 0) return;
      const rows = readTableRows();
      rows.splice(idx, 1);
      renderSeqTable(rows);
    });
    document.getElementById("seq-up")?.addEventListener("click", () => {
      const idx = getSelectedIdx();
      if (idx <= 0) return;
      const rows = readTableRows();
      [rows[idx - 1], rows[idx]] = [rows[idx], rows[idx - 1]];
      renderSeqTable(rows);
      seqTable?.querySelector(`tr[data-idx="${idx - 1}"]`)?.classList.add("selected");
    });
    document.getElementById("seq-down")?.addEventListener("click", () => {
      const idx = getSelectedIdx();
      const rows = readTableRows();
      if (idx < 0 || idx >= rows.length - 1) return;
      [rows[idx + 1], rows[idx]] = [rows[idx], rows[idx + 1]];
      renderSeqTable(rows);
      seqTable?.querySelector(`tr[data-idx="${idx + 1}"]`)?.classList.add("selected");
    });

    document.getElementById("seq-save")?.addEventListener("click", () => {
      try {
        const seq = rowsToSequence(readTableRows());
        localStorage.setItem(LS_KEY, JSON.stringify(seq, null, 2));
        seqLog(`saved ${seq.length} steps`);
      } catch (e) {
        seqLog(`save error: ${e}`);
      }
    });
    document.getElementById("seq-load")?.addEventListener("click", () => {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) {
        seqLog("nothing saved");
        return;
      }
      try {
        loadSequenceJson(JSON.parse(raw));
        seqLog("loaded from localStorage");
      } catch (e) {
        seqLog(`load error: ${e}`);
      }
    });
    document.getElementById("seq-load-test")?.addEventListener("click", () => {
      loadSequenceJson([
        { type: "home" },
        { type: "target", x: num("mv-x"), y: num("mv-y"), z: num("mv-z"), t: num("mv-t"), r: num("mv-r"), g: num("mv-g") },
        { type: "home" },
      ]);
      seqLog("test: Home → Target → Home");
    });
    document.getElementById("seq-load-fps")?.addEventListener("click", () => {
      loadSequenceJson(FPS_EXAMPLE);
      seqLog("loaded FPS example");
    });

    document.getElementById("seq-run")?.addEventListener("click", async () => {
      let sequence;
      try {
        sequence = rowsToSequence(readTableRows());
      } catch (e) {
        seqLog(`parse error: ${e}`);
        return;
      }
      if (!sequence.length) {
        seqLog("empty sequence");
        return;
      }
      renderSeqTable(readTableRows().map((r) => ({ ...r, status: "" })));
      seqLog(`starting ${sequence.length} steps on Orin…`);
      motionBusy = true;
      panelButtons().forEach((b) => {
        b.disabled = true;
      });
      const data = await rpc("sequence_start", { steps: sequence });
      if (!data || data.ok === false) {
        seqLog(`start failed: ${data?.error || "unknown"}`);
        motionBusy = false;
        panelButtons().forEach((b) => {
          b.disabled = false;
        });
        return;
      }
      if (sequencePoll) clearInterval(sequencePoll);
      sequencePoll = setInterval(pollSequenceStatus, 500);
      pollSequenceStatus();
    });

    document.getElementById("seq-stop")?.addEventListener("click", async () => {
      await rpc("sequence_stop", {}, { blocking: false });
      seqLog("stop sent");
    });

    updateReachability();
    renderSeqTable([]);
  }

  window.RoArmPanel = {
    id: ROARM_ID,
    init() {
      bindOnce();
    },
    onSelect(device) {
      bindOnce();
      renderFleetStatus(device);
      refreshStatus();
      loadPoints();
    },
    onFleetUpdate(device) {
      if (!device) return;
      renderFleetStatus(device);
    },
  };

  window.RoArmPanel.init();
})();
