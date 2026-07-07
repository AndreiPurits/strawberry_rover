/** RoArm operator panel — embedded in main dashboard (sidebar navigation). */
(function () {
  const ROARM_ID = "roarm-01";
  const LS_KEY = "axm_roarm_sequence_v1";

  let stereoMjpegUrl = "";
  let lastApproachLogLine = "";

  let savedPoints = [];
  let selectedPointId = null;
  let lastFeedback = null;
  let lastStatusAt = 0;
  const STATUS_MIN_INTERVAL_MS = 4000;
  let syncingJointSliders = false;
  const jointSendTimers = {};

  const JOINT_DEFS = [
    { id: 1, key: "b", label: "Основание", min: -3.14, max: 3.14 },
    { id: 2, key: "s", label: "Плечо", min: -1.57, max: 1.57 },
    { id: 3, key: "e", label: "Локоть", min: -0.5, max: 3.5 },
    { id: 4, key: "t", label: "Запястье", min: -3.14, max: 3.14 },
    { id: 5, key: "r", label: "Кисть", min: -3.14, max: 3.14 },
    { id: 6, key: "g", label: "Захват", min: 1.08, max: 3.14 },
  ];

  const JOINT_KEYS = ["base", "shoulder", "elbow", "wrist", "roll", "hand"];

  function isHomePointName(name) {
    const n = String(name || "").trim().toLowerCase();
    return n === "home" || n === "дом";
  }

  function getHomePoint() {
    return (
      savedPoints.find((p) => p.role === "home") ||
      savedPoints.find((p) => isHomePointName(p.name))
    );
  }

  function normalizeJoints(raw) {
    if (!raw || typeof raw !== "object") return null;
    const j = {
      base: raw.base ?? raw.b,
      shoulder: raw.shoulder ?? raw.s,
      elbow: raw.elbow ?? raw.e,
      wrist: raw.wrist ?? raw.t,
      roll: raw.roll ?? raw.r,
      hand: raw.hand ?? raw.g,
    };
    const vals = Object.values(j);
    if (vals.every((v) => v === undefined || v === null)) return null;
    return {
      base: Number(j.base ?? 0),
      shoulder: Number(j.shoulder ?? 0),
      elbow: Number(j.elbow ?? 1.57),
      wrist: Number(j.wrist ?? 0),
      roll: Number(j.roll ?? 0),
      hand: Number(j.hand ?? 3.14),
    };
  }

  function getJointsFromSliders() {
    const joints = {};
    JOINT_DEFS.forEach((joint, idx) => {
      const key = JOINT_KEYS[idx];
      const slider = document.getElementById(`j-slider-${joint.id}`);
      joints[key] = Number(slider?.value ?? 0);
    });
    return joints;
  }

  function updateStereoCamera(device) {
    const img = document.getElementById("roarm-stereo-camera");
    const ph = document.getElementById("roarm-stereo-placeholder");
    const status = document.getElementById("roarm-cam-status");
    const live = Boolean(device?.stereo_camera_live);
    const stereo = (device?.telemetry?.perception || {}).stereo || {};
    const fps = stereo.stream_fps ?? stereo.hub_fps;

    if (live) {
      const url = `/api/rovers/${encodeURIComponent(ROARM_ID)}/camera/stereo/mjpeg`;
      if (stereoMjpegUrl !== url) {
        stereoMjpegUrl = url;
        if (img) {
          img.onload = () => {
            img.classList.remove("hidden");
            ph?.classList.add("hidden");
          };
          img.src = `${url}?t=${Date.now()}`;
        }
      }
      if (status) {
        status.textContent = fps ? `live · ${Number(fps).toFixed(1)} fps` : "live";
      }
      ph?.classList.add("hidden");
    } else {
      if (img) {
        img.classList.add("hidden");
        img.removeAttribute("src");
      }
      stereoMjpegUrl = "";
      ph?.classList.remove("hidden");
      if (status) status.textContent = "нет кадра";
    }
  }

  function updateApproachUi(device) {
    const berry = (device?.telemetry?.roarm || {}).strawberry || {};

    const statusEl = document.getElementById("roarm-approach-status");
    const resultEl = document.getElementById("roarm-approach-result");
    if (statusEl) {
      statusEl.textContent = berry.status_text || (berry.valid ? "Клубника найдена" : "Поиск клубники…");
    }
    if (resultEl) {
      const parts = [];
      if (berry.valid && Array.isArray(berry.detections)) {
        parts.push(`✓ ${berry.count || berry.detections.length} berry`);
        berry.detections.slice(0, 3).forEach((d, i) => {
          parts.push(`#${i + 1} conf ${Number(d.conf).toFixed(2)}`);
        });
      } else {
        parts.push(`✗ ${berry.status || "no detections"}`);
      }
      resultEl.textContent = parts.filter(Boolean).join(" · ");
    }

    const line = berry.status_text || "";
    if (line && line !== lastApproachLogLine) {
      lastApproachLogLine = line;
      seqLog(`detect: ${line}`);
    }
    (berry.log || []).forEach((entry) => {
      if (entry && !logEl?.textContent.includes(entry)) seqLog(entry);
    });
  }

  function resolveSequenceSteps(sequence) {
    const homePt = getHomePoint();
    const hj = homePt ? normalizeJoints(homePt.joints) : null;
    return sequence.map((step) => {
      if (step.type !== "home" || !hj) return step;
      return {
        type: "home_joints_staged",
        base: hj.base,
        shoulder: hj.shoulder,
        elbow: hj.elbow,
        wrist: hj.wrist,
        roll: hj.roll,
        hand: hj.hand,
        spd: homePt.joint_spd ?? 0,
        acc: homePt.joint_acc ?? 10,
      };
    });
  }

  const ptListEl = () => document.getElementById("pt-list");
  const ptStatusEl = () => document.getElementById("pt-status");

  function setPtStatus(msg) {
    const el = ptStatusEl();
    if (el) el.textContent = msg;
  }

  function pickFeedback(fb, ...keys) {
    if (!fb || typeof fb !== "object") return undefined;
    for (const k of keys) {
      if (fb[k] !== undefined && fb[k] !== null && Number.isFinite(Number(fb[k]))) {
        return Number(fb[k]);
      }
    }
    return undefined;
  }

  function trgFieldsVisible() {
    return Boolean(document.getElementById("mv-show-trg")?.checked);
  }

  /** T/R/G for T:104 — when hidden, use live arm orientation (not form fields). */
  function getOrientationParams() {
    if (trgFieldsVisible()) {
      return { t: num("mv-t"), r: num("mv-r"), g: num("mv-g") };
    }
    const fb = lastFeedback;
    return {
      t: pickFeedback(fb, "tit", "t") ?? 0,
      r: pickFeedback(fb, "r") ?? 0,
      g: pickFeedback(fb, "g") ?? 3.14,
    };
  }

  function getMoveXyzParams() {
    const ori = getOrientationParams();
    return {
      x: num("mv-x"),
      y: num("mv-y"),
      z: num("mv-z"),
      t: ori.t,
      r: ori.r,
      g: ori.g,
      spd: num("mv-spd"),
    };
  }

  function formatRad(v) {
    return Number(v).toFixed(2);
  }

  function buildJointSliders() {
    const root = document.getElementById("joint-sliders");
    if (!root || root.childElementCount) return;
    JOINT_DEFS.forEach((joint) => {
      const row = document.createElement("div");
      row.className = "joint-slider-row";

      const label = document.createElement("label");
      label.textContent = joint.label;
      label.setAttribute("for", `j-slider-${joint.id}`);

      const slider = document.createElement("input");
      slider.type = "range";
      slider.id = `j-slider-${joint.id}`;
      slider.min = String(joint.min);
      slider.max = String(joint.max);
      slider.step = "0.01";
      slider.value = "0";

      const val = document.createElement("span");
      val.className = "joint-val";
      val.id = `j-val-${joint.id}`;
      val.textContent = "0.00";

      const updateVal = () => {
        val.textContent = formatRad(slider.value);
      };

      slider.addEventListener("input", () => {
        updateVal();
        if (syncingJointSliders) return;
        clearTimeout(jointSendTimers[joint.id]);
        jointSendTimers[joint.id] = setTimeout(() => {
          rpc(
            "joint_move",
            { joint: joint.id, rad: Number(slider.value), spd: 0, acc: 10 },
            { blocking: false }
          );
        }, 80);
      });

      slider.addEventListener("change", () => {
        updateVal();
        if (syncingJointSliders) return;
        clearTimeout(jointSendTimers[joint.id]);
        rpc(
          "joint_move",
          { joint: joint.id, rad: Number(slider.value), spd: 0, acc: 10 },
          { blocking: false }
        );
      });

      row.appendChild(label);
      row.appendChild(slider);
      row.appendChild(val);
      root.appendChild(row);
    });
  }

  function applyArmFeedback(fb) {
    if (!fb || typeof fb !== "object") return;
    lastFeedback = fb;
    if (!motionBusy) syncJointSlidersFromFeedback(fb);
    if (jsonEl) jsonEl.textContent = JSON.stringify(fb, null, 2);
  }

  function syncJointSlidersFromFeedback(fb) {
    if (!fb || typeof fb !== "object") return;
    syncingJointSliders = true;
    JOINT_DEFS.forEach((joint) => {
      const rad = pickFeedback(fb, joint.key);
      if (rad === undefined) return;
      const slider = document.getElementById(`j-slider-${joint.id}`);
      const val = document.getElementById(`j-val-${joint.id}`);
      if (slider) slider.value = String(Math.min(joint.max, Math.max(joint.min, rad)));
      if (val) val.textContent = formatRad(slider?.value ?? rad);
    });
    syncingJointSliders = false;
  }

  function feedbackToPose(fb) {
    if (!fb || typeof fb !== "object") return null;
    const pick = (...keys) => pickFeedback(fb, ...keys);
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
    const j = normalizeJoints(pt.joints);
    if (j) {
      return `J B=${j.base.toFixed(2)} S=${j.shoulder.toFixed(2)} E=${j.elbow.toFixed(2)}`;
    }
    return "—";
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
        const homePt = getHomePoint();
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
    syncJointSlidersFromFeedback(fb);
    return feedbackToPose(fb);
  }

  async function savePoint() {
    const name = String(document.getElementById("pt-name")?.value || "").trim();
    if (!name) {
      setPtStatus("Введите название точки");
      return;
    }
    const joints = getJointsFromSliders();
    const role = isHomePointName(name) ? "home" : null;
    const pt = await postPoint({
      name,
      mode: "joints",
      joints,
      xyz: null,
      role,
    });
    if (pt) {
      setPtStatus(`Сохранено: ${pt.name}${role ? " (HOME)" : ""}`);
      log(`point saved: ${pt.name} joints=${JSON.stringify(joints)}`);
    }
  }

  function selectedPoint() {
    return savedPoints.find((p) => p.id === selectedPointId) || null;
  }

  async function goToPoint(pt, opts = {}) {
    if (!pt) {
      setPtStatus("Выберите точку в списке");
      return null;
    }
    const j = normalizeJoints(pt.joints);
    if (!j) {
      setPtStatus("У точки нет шарниров — сохраните заново");
      return null;
    }
    const staged = opts.staged !== false;
    const op = staged ? "home_joints_staged" : "home_joints";
    setPtStatus(
      staged
        ? `→ ${pt.name} (основание → плечо → локоть…)`
        : `→ ${pt.name} (T:102)`
    );
    const data = await rpc(op, {
      ...j,
      spd: pt.joint_spd ?? 0,
      acc: pt.joint_acc ?? 10,
    });
    if (data && data.ok !== false) {
      syncJointSlidersFromFeedback({ b: j.base, s: j.shoulder, e: j.elbow, t: j.wrist, r: j.roll, g: j.hand });
    }
    return data;
  }

  async function goHome() {
    await loadPoints();
    const homePt = getHomePoint();
    if (homePt) {
      const j = normalizeJoints(homePt.joints);
      if (!j) {
        setPtStatus(`HOME «${homePt.name}» без шарниров — пересохраните точку`);
        log("HOME ERROR no joints on home point");
        return null;
      }
      setPtStatus(`HOME → ${homePt.name} (T:102 staged)`);
      log(`HOME → ${homePt.name} joints B=${j.base} S=${j.shoulder} E=${j.elbow}`);
      return goToPoint(homePt);
    }
    setPtStatus("HOME → T:100 заводская (нет точки Home/Дом — см. ★ HOME)");
    log("HOME → T:100 factory (no saved home point)");
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
    setPtStatus(`★ HOME = ${pt.name} · кнопка HOME на панели → сюда`);
    log(`home point set: ${pt.name}`);
    if (
      window.confirm(
        `«${pt.name}» — HOME для кнопки на сайте.\n\n` +
          "Чтобы при ВКЛЮЧЕНИИ питания лапа вставала сюда же — нужен T:502 в сервоприводы.\n\n" +
          "Сейчас перейти в точку и записать T:502?"
      )
    ) {
      await commitSelectedToServo();
    }
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
        if (op === "status" && (res.status === 504 || detail === "roarm_timeout")) {
          return { ok: false, error: "roarm_timeout", silent: true };
        }
        log(`${op} ERROR ${detail}`);
        if (detail === "roarm_gateway_offline") log("Orin offline — проверьте fleet-agent");
        return { ...data, ok: false, error: detail };
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
      reachEl.textContent = `Достижимость: ${info.text}`;
      reachEl.className = `reachability ${info.cls}`;
    }
  }

  async function refreshStatus(force = false) {
    const now = Date.now();
    if (!force && now - lastStatusAt < STATUS_MIN_INTERVAL_MS) return;
    if (!force && motionBusy) return;
    lastStatusAt = now;
    const data = await rpc("status", {}, { blocking: false });
    if (!data) return;
    if (data.silent || data.error === "roarm_timeout") {
      if (lastFeedback && jsonEl) {
        jsonEl.textContent = JSON.stringify(lastFeedback, null, 2);
      }
      return;
    }
    if (data.ok === false || data.error) {
      if (jsonEl) jsonEl.textContent = JSON.stringify(data, null, 2);
      return;
    }
    const st = data.status || data.feedback || data.result?.status || data.result?.feedback;
    if (st && typeof st === "object") {
      applyArmFeedback(st);
      return;
    }
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
    const ori = getOrientationParams();
    return {
      type: "target",
      x: String(num("mv-x")),
      y: String(num("mv-y")),
      z: String(num("mv-z")),
      t: String(ori.t),
      r: String(ori.r),
      g: String(ori.g),
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

    if (rt.arm && typeof rt.arm === "object") {
      applyArmFeedback(rt.arm);
    }

    updateStereoCamera(device);
    updateApproachUi(device);
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

    document.getElementById("btn-refresh")?.addEventListener("click", () => refreshStatus(true));
    document.getElementById("btn-home")?.addEventListener("click", () => goHome());
    document.getElementById("pt-save")?.addEventListener("click", () => savePoint());
    document.getElementById("pt-go")?.addEventListener("click", () => goToPoint(selectedPoint()));
    document.getElementById("pt-set-home")?.addEventListener("click", () => setSelectedAsHome());
    document.getElementById("pt-servo")?.addEventListener("click", () => commitSelectedToServo());
    document.getElementById("pt-del")?.addEventListener("click", () => deleteSelectedPoint());
    document.getElementById("btn-torque-on")?.addEventListener("click", () => rpc("torque_on"));
    document.getElementById("btn-torque-off")?.addEventListener("click", () => rpc("torque_off"));
    document.getElementById("btn-grip-open")?.addEventListener("click", () => rpc("gripper_open"));
    document.getElementById("btn-grip-close")?.addEventListener("click", () => rpc("gripper_close"));
    document.getElementById("btn-check")?.addEventListener("click", updateReachability);
    document.getElementById("btn-move")?.addEventListener("click", () => rpc("move_xyz", getMoveXyzParams()));

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
      const mv = getMoveXyzParams();
      loadSequenceJson([
        { type: "home" },
        { type: "target", x: mv.x, y: mv.y, z: mv.z, t: mv.t, r: mv.r, g: mv.g },
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
        sequence = resolveSequenceSteps(rowsToSequence(readTableRows()));
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

    buildJointSliders();
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
      refreshStatus(true);
      loadPoints();
      updateStereoCamera(device);
      updateApproachUi(device);
    },
    onFleetUpdate(device) {
      if (!device) return;
      renderFleetStatus(device);
      updateStereoCamera(device);
      updateApproachUi(device);
      if (!savedPoints.length) loadPoints();
    },
  };

  window.RoArmPanel.init();
})();
