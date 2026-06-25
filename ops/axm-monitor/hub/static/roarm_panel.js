/** RoArm operator panel — embedded in main dashboard (sidebar navigation). */
(function () {
  const ROARM_ID = "roarm-01";
  const LS_KEY = "axm_roarm_sequence_v1";

  const logEl = document.getElementById("roarm-log");
  const seqLogEl = document.getElementById("seq-log");
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
    log(msg, seqLogEl);
  }

  async function rpc(op, params = {}, opts = {}) {
    const blocking = opts.blocking !== false;
    if (blocking && motionBusy) {
      log("занято — подождите");
      return null;
    }
    if (blocking) {
      motionBusy = true;
      panelButtons().forEach((b) => {
        if (!b.classList.contains("tab")) b.disabled = true;
      });
    }
    try {
      const res = await fetch("/api/roarm/rpc", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op, params }),
      });
      if (res.status === 401) {
        location.href = "/login";
        return null;
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail || res.status;
        log(`${op} ERROR ${detail}`);
        if (detail === "roarm_gateway_offline") {
          log("Orin offline на hub — проверьте fleet-agent");
        }
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
        panelButtons().forEach((b) => {
          if (!b.classList.contains("tab")) b.disabled = false;
        });
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
    const st = data.status || data.result?.status || data;
    if (jsonEl) jsonEl.textContent = JSON.stringify(st, null, 2);
  }

  function switchTab(name) {
    document.querySelectorAll("#roarm-panel .roarm-tab-panel").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll("#roarm-panel .roarm-tabs .tab").forEach((t) => t.classList.remove("active"));
    document.getElementById(`roarm-tab-${name}`)?.classList.add("active");
    document.querySelector(`#roarm-panel .roarm-tabs .tab[data-tab="${name}"]`)?.classList.add("active");
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
      if (!seqLogEl?.textContent.includes(line)) seqLog(line);
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
        if (!b.classList.contains("tab")) b.disabled = false;
      });
      seqLog(`sequence end: ${data.result || "done"}${data.error ? ` (${data.error})` : ""}`);
    }
  }

  function renderFleetStatus(device) {
    if (!device) return;
    const rt = (device.telemetry || {}).roarm || {};
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
      } else if (proxyOk) {
        armBadge.textContent = "ARM HTTP нет";
        armBadge.className = "badge warn";
        armBadge.title = rt.error || "С Orin порт 80 на RoArm недоступен (client isolation?)";
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

    document.querySelectorAll("#roarm-panel .roarm-tabs .tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    document.getElementById("mv-show-trg")?.addEventListener("change", (e) => {
      document.getElementById("mv-trg-fields")?.classList.toggle("hidden", !e.target.checked);
    });
    ["mv-x", "mv-y", "mv-z"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", updateReachability);
    });

    document.getElementById("btn-refresh")?.addEventListener("click", refreshStatus);
    document.getElementById("btn-home")?.addEventListener("click", () => rpc("home"));
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
        if (!b.classList.contains("tab")) b.disabled = true;
      });
      const data = await rpc("sequence_start", { steps: sequence });
      if (!data || data.ok === false) {
        seqLog(`start failed: ${data?.error || "unknown"}`);
        motionBusy = false;
        panelButtons().forEach((b) => {
          if (!b.classList.contains("tab")) b.disabled = false;
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
    },
    onFleetUpdate(device) {
      if (!device) return;
      renderFleetStatus(device);
    },
  };

  window.RoArmPanel.init();
})();
