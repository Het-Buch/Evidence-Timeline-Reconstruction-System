const configNode = document.getElementById("app-config");
const appConfig = configNode ? JSON.parse(configNode.textContent) : {};

const api = appConfig.api || {};
const classColors = appConfig.classColors || {};

let currentResult = null;

document.addEventListener("DOMContentLoaded", () => {
  bindUi();
  checkStatus();
  loadHistory();
});

function bindUi() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("video-file");
  const fpsRange = document.getElementById("fps-range");
  const fpsLabel = document.getElementById("fps-label");
  const analyzeBtn = document.getElementById("analyze-btn");
  const refreshBtn = document.getElementById("refresh-history-btn");

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("drag");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag");
    if (event.dataTransfer.files.length) {
      fileInput.files = event.dataTransfer.files;
      updateFileName(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      updateFileName(fileInput.files[0]);
    }
  });

  fpsRange.addEventListener("input", () => {
    fpsLabel.textContent = `${fpsRange.value} frame/sec`;
  });

  analyzeBtn.addEventListener("click", startAnalysis);
  refreshBtn.addEventListener("click", loadHistory);
}

function updateFileName(file) {
  document.getElementById("file-name").textContent = `Selected: ${file.name}`;
}

async function checkStatus() {
  try {
    const response = await fetch(api.status || "/api/status");
    const data = await response.json();
    const badge = document.getElementById("status-badge");
    if (data.model_loaded) {
      badge.textContent = `Ready: ${data.model_name} on ${String(data.device).toUpperCase()}`;
      badge.classList.add("ready");
    } else {
      badge.textContent = "No model loaded. Train a model first.";
      badge.classList.remove("ready");
    }
  } catch (error) {
    document.getElementById("status-badge").textContent = "Status unavailable";
  }
}

async function startAnalysis() {
  const btn = document.getElementById("analyze-btn");
  const progressWrap = document.getElementById("progress-wrap");
  const progressFill = document.getElementById("prog-fill");
  const progressLabel = document.getElementById("prog-label");
  const fileInput = document.getElementById("video-file");
  const pathInput = document.getElementById("video-path");
  const fps = document.getElementById("fps-range").value;

  if (!fileInput.files.length && !pathInput.value.trim()) {
    toast("Upload a video or provide a file path.", "warn");
    return;
  }

  btn.disabled = true;
  progressWrap.style.display = "block";
  progressFill.style.width = "0%";
  progressLabel.textContent = "Preparing analysis...";

  let progress = 0;
  const timer = setInterval(() => {
    progress = Math.min(progress + Math.random() * 9, 90);
    progressFill.style.width = `${progress}%`;
    progressLabel.textContent = progress < 30 ? "Uploading video..." : progress < 60 ? "Extracting frames..." : "Running inference...";
  }, 350);

  try {
    const payload = new FormData();
    if (fileInput.files.length) {
      payload.append("video", fileInput.files[0]);
    } else {
      payload.append("video_path", pathInput.value.trim());
    }
    payload.append("fps_sample", fps);

    const response = await fetch(api.analyze || "/api/analyze", { method: "POST", body: payload });
    const data = await response.json();

    clearInterval(timer);
    progressFill.style.width = "100%";

    if (!response.ok || data.error) {
      toast(data.error || "Analysis failed.", "error");
      return;
    }

    currentResult = data;
    renderResults(data);
    toast("Analysis complete.", "success");
    loadHistory();
  } catch (error) {
    clearInterval(timer);
    toast(`Network error: ${error.message}`, "error");
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      progressWrap.style.display = "none";
      progressFill.style.width = "0%";
    }, 1000);
  }
}

function renderResults(data) {
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("results-panel").style.display = "grid";

  const stats = [
    { label: "Duration", value: formatDuration(data.duration_sec) },
    { label: "Frames Analyzed", value: data.sampled_frames },
    { label: "Anomaly %", value: `${data.anomaly_pct}%` },
    { label: "Dominant Class", value: data.dominant_class },
  ];

  document.getElementById("stats-row").innerHTML = stats.map((stat) => `
    <article class="stat-card">
      <div class="stat-val">${stat.value}</div>
      <div class="stat-label">${stat.label}</div>
    </article>
  `).join("");

  renderTimeline(data.timeline || [], data.duration_sec || 0);

  document.getElementById("events-list").innerHTML = (data.events && data.events.length)
    ? data.events.map((event) => `
        <div class="event-card" style="border-left-color:${event.color}">
          <div>
            <div class="event-class" style="color:${event.color}">${event.class}</div>
            <div class="event-time">${event.start} → ${event.end}</div>
          </div>
          <div class="event-dur">${event.duration_sec}s</div>
        </div>
      `).join("")
    : '<div class="empty-state">No anomaly events detected. The video appears normal.</div>';

  const classCounts = data.class_counts || {};
  const total = Object.values(classCounts).reduce((sum, value) => sum + value, 0);
  const sorted = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);
  document.getElementById("class-dist").innerHTML = sorted.map(([name, count]) => {
    const percentage = total > 0 ? (count / total * 100).toFixed(1) : 0;
    const color = classColors[name] || "#67e8f9";
    return `
      <div class="prob-row">
        <div class="prob-name">${name}</div>
        <div class="prob-track"><div class="prob-fill" style="width:${percentage}%;background:${color}"></div></div>
        <div class="prob-val">${percentage}%</div>
      </div>
    `;
  }).join("");

  document.getElementById("frame-preds").innerHTML = renderFrameTable(data.timeline || []);
}

function renderTimeline(timeline, duration) {
  const bar = document.getElementById("timeline-bar");
  const labels = document.getElementById("timeline-labels");
  bar.innerHTML = "";

  if (!timeline.length) {
    labels.innerHTML = "";
    return;
  }

  const segments = [];
  let index = 0;
  while (index < timeline.length) {
    const cls = timeline[index].class_name;
    let next = index;
    while (next < timeline.length && timeline[next].class_name === cls) {
      next += 1;
    }
    segments.push({
      cls,
      startTs: timeline[index].timestamp_sec,
      endTs: next < timeline.length ? timeline[next].timestamp_sec : duration,
      color: classColors[cls] || "#67e8f9",
    });
    index = next;
  }

  segments.forEach((segment) => {
    const width = ((segment.endTs - segment.startTs) / Math.max(duration, 1)) * 100;
    const element = document.createElement("div");
    element.className = "timeline-seg";
    element.style.width = `${width}%`;
    element.style.background = segment.color;
    element.title = `${segment.cls}\n${formatDuration(segment.startTs)} → ${formatDuration(segment.endTs)}`;
    element.textContent = width > 10 ? segment.cls : "";
    bar.appendChild(element);
  });

  labels.innerHTML = [0, 0.25, 0.5, 0.75, 1].map((point) => `<span>${formatDuration(point * duration)}</span>`).join("");
}

function renderFrameTable(timeline) {
  if (!timeline.length) {
    return '<div class="empty-state">No frame predictions available.</div>';
  }

  const rows = timeline.map((frame) => `
    <tr>
      <td>${frame.timestamp_fmt}</td>
      <td style="color:${frame.color};font-weight:${frame.class_name === "Normal" ? 400 : 700}">${frame.class_name}</td>
      <td style="text-align:right">${(frame.confidence * 100).toFixed(1)}%</td>
    </tr>
  `).join("");

  return `
    <table class="frame-table">
      <thead>
        <tr><th>Time</th><th>Class</th><th style="text-align:right">Confidence</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadHistory() {
  try {
    const response = await fetch(api.history || "/api/history");
    const items = await response.json();
    const historyList = document.getElementById("history-list");

    if (!items.length) {
      historyList.innerHTML = '<div class="empty-state">No history yet.</div>';
      return;
    }

    historyList.innerHTML = items.map((item) => `
      <article class="history-item" data-session-id="${item.session_id}">
        <div>
          <div class="history-name"><span class="dot" style="background:${classColors[item.dominant_class] || "#67e8f9"}"></span>${item.video_name || "Unknown"}</div>
          <div class="history-meta">${item.analyzed_at} · ${item.dominant_class} · ${item.anomaly_pct}% anomaly</div>
        </div>
        <div class="history-meta">${formatDuration(item.duration_sec)}</div>
      </article>
    `).join("");

    historyList.querySelectorAll("[data-session-id]").forEach((element) => {
      element.addEventListener("click", () => loadResult(element.dataset.sessionId));
    });
  } catch (error) {
    toast("Could not load analysis history.", "warn");
  }
}

async function loadResult(sessionId) {
  try {
    const response = await fetch(`/api/result/${sessionId}`);
    const data = await response.json();
    if (!data.error) {
      currentResult = data;
      renderResults(data);
      toast(`Loaded ${data.video_name}`, "info");
    }
  } catch (error) {
    toast("Could not load saved result.", "error");
  }
}

function formatDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function toast(message, type = "info") {
  const node = document.getElementById("toast");
  const icons = { success: "✅", error: "❌", warn: "⚠️", info: "ℹ️" };
  node.textContent = `${icons[type] || ""} ${message}`;
  node.classList.add("show");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => node.classList.remove("show"), 3200);
}
