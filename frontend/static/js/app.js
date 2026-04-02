/* ─── API base URL ─────────────────────────────────────────────────────────── */
const API_BASE = window.API_BASE || "http://localhost:8000";

/* ─── Utility helpers ─────────────────────────────────────────────────────── */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function setVisible(el, visible) {
  if (!el) return;
  el.classList.toggle("hidden", !visible);
}

function showAlert(container, msg, type = "error") {
  if (!container) return;
  container.innerHTML = `<div class="alert alert-${type}">⚠ ${msg}</div>`;
  container.classList.remove("hidden");
}

function clearAlert(container) {
  if (!container) return;
  container.innerHTML = "";
  container.classList.add("hidden");
}

function confBar(value) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "#10b981" : pct >= 50 ? "#f59e0b" : "#ef4444";
  return `
    <div class="conf-bar-wrap">
      <div class="conf-bar">
        <div class="conf-bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <span class="conf-value" style="color:${color}">${pct}%</span>
    </div>`;
}

/* ─── Upload zone drag-and-drop ───────────────────────────────────────────── */
function initUploadZone(zoneEl, inputEl, fileNameEl) {
  if (!zoneEl || !inputEl) return;
  ["dragenter", "dragover"].forEach(e =>
    zoneEl.addEventListener(e, ev => { ev.preventDefault(); zoneEl.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(e =>
    zoneEl.addEventListener(e, ev => { ev.preventDefault(); zoneEl.classList.remove("dragover"); })
  );
  zoneEl.addEventListener("drop", ev => {
    const files = ev.dataTransfer?.files;
    if (files?.length) {
      inputEl.files = files;
      inputEl.dispatchEvent(new Event("change"));
    }
  });
  zoneEl.addEventListener("click", () => inputEl.click());
  inputEl.addEventListener("change", () => {
    const name = inputEl.files[0]?.name;
    if (fileNameEl && name) fileNameEl.textContent = `✓ ${name}`;
  });
}

/* ─── Image upload page ───────────────────────────────────────────────────── */
function initImagePage() {
  const form       = $("#image-form");
  const input      = $("#image-input");
  const zone       = $("#upload-zone");
  const fileNameEl = $("#file-name");
  const submitBtn  = $("#submit-btn");
  const alertBox   = $("#alert-box");
  const resultSec  = $("#result-section");
  const loadOverlay= $("#loading-overlay");

  if (!form) return;
  initUploadZone(zone, input, fileNameEl);

  form.addEventListener("submit", async e => {
    e.preventDefault();
    clearAlert(alertBox);
    if (!input.files[0]) { showAlert(alertBox, "Please select an image file."); return; }

    submitBtn.disabled = true;
    loadOverlay && loadOverlay.classList.add("active");
    setVisible(resultSec, false);

    const fd = new FormData();
    fd.append("file", input.files[0]);

    try {
      const res = await fetch(`${API_BASE}/api/image`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      renderImageResult(data);
      setVisible(resultSec, true);
    } catch (err) {
      showAlert(alertBox, err.message);
    } finally {
      submitBtn.disabled = false;
      loadOverlay && loadOverlay.classList.remove("active");
    }
  });

  function renderImageResult(data) {
    // Show annotated image
    const img = $("#result-img");
    if (img) img.src = `${API_BASE}${data.result_image_url}`;

    // Stats
    const statsEl = $("#stats");
    if (statsEl) statsEl.innerHTML = `
      <div class="stat-badge"><span class="stat-value">${data.plates_count}</span><span class="stat-label">Plates</span></div>
      <div class="stat-badge"><span class="stat-value">${data.inference_time_ms.toFixed(0)}</span><span class="stat-label">ms</span></div>
    `;

    // Plates list
    const listEl = $("#plates-list");
    if (!listEl) return;
    if (!data.plates.length) {
      listEl.innerHTML = "<p class='text-muted text-sm'>No plates detected.</p>";
      return;
    }
    listEl.innerHTML = data.plates.map((p, i) => `
      <div class="card mb-3">
        <div class="flex items-center gap-3 mb-2">
          <span class="text-muted text-sm">#${i + 1}</span>
          <span class="plate-tag">${p.plate_text || "—"}</span>
        </div>
        <div class="mb-1 text-sm text-muted">Detection confidence</div>
        ${confBar(p.confidence)}
        <div class="mt-2 mb-1 text-sm text-muted">OCR confidence</div>
        ${confBar(p.ocr_confidence)}
        <div class="mt-2 text-sm text-muted font-mono">BBox: [${p.bbox.join(", ")}]</div>
      </div>
    `).join("");
  }
}

/* ─── Video upload page ───────────────────────────────────────────────────── */
function initVideoPage() {
  const form       = $("#video-form");
  const input      = $("#video-input");
  const zone       = $("#upload-zone");
  const fileNameEl = $("#file-name");
  const submitBtn  = $("#submit-btn");
  const alertBox   = $("#alert-box");
  const resultSec  = $("#result-section");
  const loadOverlay= $("#loading-overlay");

  if (!form) return;
  initUploadZone(zone, input, fileNameEl);

  form.addEventListener("submit", async e => {
    e.preventDefault();
    clearAlert(alertBox);
    if (!input.files[0]) { showAlert(alertBox, "Please select a video file."); return; }

    submitBtn.disabled = true;
    loadOverlay && loadOverlay.classList.add("active");
    setVisible(resultSec, false);

    const fd = new FormData();
    fd.append("file", input.files[0]);

    try {
      const res = await fetch(`${API_BASE}/api/video`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      renderVideoResult(data);
      setVisible(resultSec, true);
    } catch (err) {
      showAlert(alertBox, err.message);
    } finally {
      submitBtn.disabled = false;
      loadOverlay && loadOverlay.classList.remove("active");
    }
  });

  function renderVideoResult(data) {
    const vid = $("#result-video");
    if (vid) {
      vid.src = `${API_BASE}${data.result_video_url}`;
      vid.load();
    }

    const statsEl = $("#stats");
    if (statsEl) statsEl.innerHTML = `
      <div class="stat-badge"><span class="stat-value">${data.plates_detected}</span><span class="stat-label">Detections</span></div>
      <div class="stat-badge"><span class="stat-value">${data.unique_plates.length}</span><span class="stat-label">Unique</span></div>
      <div class="stat-badge"><span class="stat-value">${data.processed_frames}</span><span class="stat-label">Frames</span></div>
      <div class="stat-badge"><span class="stat-value">${(data.inference_time_ms / 1000).toFixed(1)}s</span><span class="stat-label">Time</span></div>
    `;

    const listEl = $("#plates-list");
    if (!listEl) return;
    if (!data.unique_plates.length) {
      listEl.innerHTML = "<p class='text-muted text-sm'>No plates detected.</p>";
      return;
    }
    listEl.innerHTML = data.unique_plates.map(p => `
      <div class="flex items-center gap-2 mb-2">
        <span class="plate-tag">${p}</span>
      </div>
    `).join("");
  }
}

/* ─── Live camera page ────────────────────────────────────────────────────── */
function initStreamPage() {
  const startBtn   = $("#start-btn");
  const stopBtn    = $("#stop-btn");
  const streamImg  = $("#stream-img");
  const statusEl   = $("#stream-status");
  const camSelect  = $("#camera-source");
  const ipInput    = $("#ip-url");

  if (!startBtn || !streamImg) return;

  let streaming = false;

  startBtn.addEventListener("click", () => {
    const sourceType = camSelect?.value || "webcam";
    let url;
    if (sourceType === "webcam") {
      url = `${API_BASE}/api/stream/webcam`;
    } else {
      const ipUrl = ipInput?.value?.trim();
      if (!ipUrl) { alert("Please enter a camera URL."); return; }
      url = `${API_BASE}/api/stream/ip?url=${encodeURIComponent(ipUrl)}`;
    }
    streamImg.src = url;
    streaming = true;
    setVisible(startBtn, false);
    setVisible(stopBtn, true);
    if (statusEl) {
      statusEl.innerHTML = `<span class="dot green"></span> LIVE`;
      statusEl.className = "stream-status live";
    }
  });

  stopBtn?.addEventListener("click", () => {
    streamImg.src = "";
    streaming = false;
    setVisible(startBtn, true);
    setVisible(stopBtn, false);
    if (statusEl) {
      statusEl.innerHTML = `<span class="dot red"></span> OFFLINE`;
      statusEl.className = "stream-status offline";
    }
  });
}

/* ─── History page ────────────────────────────────────────────────────────── */
async function initHistoryPage() {
  const tableBody = $("#history-tbody");
  const alertBox  = $("#alert-box");
  if (!tableBody) return;

  try {
    const res = await fetch(`${API_BASE}/api/history?limit=100`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to load history");

    if (!data.length) {
      tableBody.innerHTML = `<tr><td colspan="7" class="text-muted" style="text-align:center;padding:2rem">No detection history yet.</td></tr>`;
      return;
    }

    tableBody.innerHTML = data.map(r => `
      <tr>
        <td>${r.id}</td>
        <td><span class="badge-type">${r.source_type}</span></td>
        <td class="font-mono">${r.plate_text ? `<span class="plate-tag" style="font-size:0.9rem">${r.plate_text}</span>` : "—"}</td>
        <td>${r.confidence != null ? (r.confidence * 100).toFixed(1) + "%" : "—"}</td>
        <td>${r.plates_count}</td>
        <td class="text-muted text-sm">${new Date(r.created_at).toLocaleString()}</td>
        <td>
          <button class="btn btn-ghost text-sm" onclick="deleteRecord(${r.id}, this)">🗑</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    showAlert(alertBox, err.message);
  }
}

window.deleteRecord = async function(id, btn) {
  if (!confirm("Delete this record?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/history/${id}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) throw new Error("Delete failed");
    btn.closest("tr").remove();
  } catch (err) {
    alert(err.message);
  }
};

/* ─── Bootstrap the right page ────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  // Mark active nav link
  const path = location.pathname.split("/").pop() || "index.html";
  $$(".nav-link").forEach(a => {
    if (a.getAttribute("href")?.includes(path)) a.classList.add("active");
  });

  initImagePage();
  initVideoPage();
  initStreamPage();
  initHistoryPage();
});
