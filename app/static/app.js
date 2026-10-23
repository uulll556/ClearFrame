const state = {
  videos: [], total: 0, offset: 0, limit: 36,
  selected: new Set(), settings: null, stats: null,
  currentVideo: null, currentView: "library", toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function formatBytes(bytes = 0) {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function formatDuration(seconds = 0) {
  const value = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

const statusLabels = {
  pending: "\u5f85\u5904\u7406", queued: "\u5df2\u5165\u961f", processing: "\u5904\u7406\u4e2d",
  completed: "\u5df2\u5b8c\u6210", failed: "\u5931\u8d25",
};

async function loadStats() {
  state.stats = await api("/api/stats");
  const statuses = state.stats.statuses || {};
  $("#totalCount").textContent = state.stats.total.toLocaleString();
  $("#totalSize").textContent = formatBytes(state.stats.bytes);
  $("#pendingCount").textContent = (statuses.pending || 0).toLocaleString();
  $("#queuedCount").textContent = ((statuses.queued || 0) + (statuses.processing || 0)).toLocaleString();
  $("#completedCount").textContent = (statuses.completed || 0).toLocaleString();
  $("#failedLabel").textContent = `${statuses.failed || 0} \u4e2a\u5931\u8d25\u4efb\u52a1`;
  $("#runningLabel").textContent = state.stats.worker_running ? "\u5904\u7406\u5668\u6b63\u5728\u8fd0\u884c" : "\u5904\u7406\u5668\u5df2\u6682\u505c";
  const workerButton = $("#workerButton");
  workerButton.querySelector("span:last-child").textContent = state.stats.worker_running ? "\u6682\u505c\u961f\u5217" : "\u542f\u52a8\u961f\u5217";
  workerButton.querySelector(".btn-icon").textContent = state.stats.worker_running ? "\u2161" : "\u25b6";
  renderCategories(state.stats.categories || []);
}

function renderCategories(categories) {
  const select = $("#categoryFilter");
  const current = select.value;
  select.innerHTML = `<option value="">\u5168\u90e8\u5206\u7c7b</option>` + categories.map(item =>
    `<option value="${escapeHtml(item.category)}">${escapeHtml(item.category)} (${item.count})</option>`
  ).join("");
  select.value = current;
}

async function loadVideos(reset = false) {
  if (reset) state.offset = 0;
  const params = new URLSearchParams({
    limit: state.limit, offset: state.offset,
    search: $("#searchInput").value.trim(),
    category: $("#categoryFilter").value,
    status: $("#statusFilter").value,
  });
  const result = await api(`/api/videos?${params}`);
  state.videos = result.items;
  state.total = result.total;
  renderVideos();
}

function renderVideos() {
  const grid = $("#videoGrid");
  $("#emptyState").classList.toggle("hidden", state.total > 0 || !!$("#searchInput").value);
  grid.innerHTML = state.videos.map(video => {
    const selected = state.selected.has(video.id);
    const progress = Math.round((video.progress || 0) * 100);
    return `<article class="video-card ${selected ? "selected" : ""}" data-id="${video.id}">
      <div class="thumb-wrap">
        <input class="card-check" type="checkbox" ${selected ? "checked" : ""} aria-label="select">
        <img loading="lazy" src="/api/videos/${video.id}/thumbnail" alt="">
        <span class="status-badge ${video.status}">${statusLabels[video.status] || video.status}</span>
        <span class="duration-badge">${formatDuration(video.duration_seconds)}</span>
      </div>
      ${video.status === "processing" ? `<div class="progress-track"><div class="progress-bar" style="width:${progress}%"></div></div>` : ""}
      <div class="card-body">
        <h3 class="card-title">${escapeHtml(video.title || video.filename)}</h3>
        <div class="card-meta"><span>${escapeHtml(video.category)}</span><span>${formatBytes(video.size_bytes)}</span></div>
      </div>
    </article>`;
  }).join("");
  $$(".video-card").forEach(card => {
    card.addEventListener("click", event => {
      const id = Number(card.dataset.id);
      if (event.target.classList.contains("card-check")) {
        toggleSelection(id, event.target.checked);
      } else {
        openDrawer(id);
      }
    });
  });
  const pages = Math.max(1, Math.ceil(state.total / state.limit));
  const page = Math.floor(state.offset / state.limit) + 1;
  $("#pageInfo").textContent = `${page} / ${pages}`;
  $("#prevPage").disabled = state.offset === 0;
  $("#nextPage").disabled = state.offset + state.limit >= state.total;
  updateSelectionUi();
}

function toggleSelection(id, selected) {
  selected ? state.selected.add(id) : state.selected.delete(id);
  renderVideos();
}

function updateSelectionUi() {
  $("#selectionText").textContent = `\u5df2\u9009\u62e9 ${state.selected.size} \u9879`;
  $("#queueSelectedButton").disabled = state.selected.size === 0;
}

async function queueVideos(videoIds = [], category = null, status = "pending") {
  const result = await api("/api/jobs/queue", {
    method: "POST", body: JSON.stringify({ video_ids: videoIds, category, status }),
  });
  showToast(`\u5df2\u5c06 ${result.queued} \u6761\u7d20\u6750\u6dfb\u52a0\u5230\u961f\u5217`);
  state.selected.clear();
  await Promise.all([loadStats(), loadVideos()]);
}

async function scanLibrary() {
  const button = $("#scanButton");
  button.disabled = true;
  button.querySelector(".btn-icon").textContent = "...";
  try {
    const result = await api("/api/scan", { method: "POST" });
    showToast(`\u626b\u63cf\u5b8c\u6210\uff1a${result.imported + result.discovered} \u6761\uff0c${result.missing} \u6761\u7f3a\u5931`);
    await Promise.all([loadStats(), loadVideos(true)]);
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.querySelector(".btn-icon").textContent = "\u21bb"; }
}

async function openDrawer(videoId) {
  try {
    const video = await api(`/api/videos/${videoId}`);
    state.currentVideo = video;
    $("#drawerTitle").textContent = video.title || video.filename;
    $("#drawerCategory").textContent = video.category;
    $("#drawerDuration").textContent = formatDuration(video.duration_seconds);
    $("#drawerSize").textContent = formatBytes(video.size_bytes);
    $("#previewVideo").src = `/api/videos/${video.id}/media`;
    const values = {
      x: video.mask_x ?? state.settings.mask_x,
      y: video.mask_y ?? state.settings.mask_y,
      width: video.mask_width ?? state.settings.mask_width,
      height: video.mask_height ?? state.settings.mask_height,
    };
    setRegionControls(values);
    $("#detailDrawer").classList.add("open");
    $("#drawerBackdrop").classList.add("open");
    if (video.mask_x == null) detectCurrentRegion(false);
  } catch (error) { showToast(error.message, true); }
}

function closeDrawer() {
  $("#detailDrawer").classList.remove("open");
  $("#drawerBackdrop").classList.remove("open");
  $("#previewVideo").pause();
  $("#previewVideo").removeAttribute("src");
}

function getRegionControls() {
  return {
    x: Number($("#maskX").value), y: Number($("#maskY").value),
    width: Number($("#maskW").value), height: Number($("#maskH").value),
  };
}

function setRegionControls(region) {
  $("#maskX").value = region.x;
  $("#maskY").value = region.y;
  $("#maskW").value = region.width;
  $("#maskH").value = region.height;
  updateMaskPreview();
}

function updateMaskPreview() {
  const region = getRegionControls();
  if (region.x + region.width > 1) {
    region.x = 1 - region.width;
    $("#maskX").value = region.x;
  }
  if (region.y + region.height > 1) {
    region.y = 1 - region.height;
    $("#maskY").value = region.y;
  }
  const box = $("#maskBox");
  box.style.left = `${region.x * 100}%`;
  box.style.top = `${region.y * 100}%`;
  box.style.width = `${region.width * 100}%`;
  box.style.height = `${region.height * 100}%`;
  $("#xValue").textContent = `${(region.x * 100).toFixed(1)}%`;
  $("#yValue").textContent = `${(region.y * 100).toFixed(1)}%`;
  $("#wValue").textContent = `${(region.width * 100).toFixed(1)}%`;
  $("#hValue").textContent = `${(region.height * 100).toFixed(1)}%`;
}

async function saveCurrentRegion(showMessage = true) {
  if (!state.currentVideo) return;
  const region = getRegionControls();
  state.currentVideo = await api(`/api/videos/${state.currentVideo.id}/region`, {
    method: "PUT", body: JSON.stringify(region),
  });
  if (showMessage) showToast("\u906e\u7f69\u8303\u56f4\u5df2\u4fdd\u5b58");
}

async function detectCurrentRegion(showMessage = true) {
  if (!state.currentVideo) return;
  const button = $("#resetRegion");
  button.disabled = true;
  button.textContent = "\u5b9a\u4f4d\u4e2d...";
  try {
    state.currentVideo = await api(`/api/videos/${state.currentVideo.id}/detect-region`, {
      method: "POST",
    });
    setRegionControls({
      x: state.currentVideo.mask_x, y: state.currentVideo.mask_y,
      width: state.currentVideo.mask_width, height: state.currentVideo.mask_height,
    });
    if (showMessage) {
      const score = Number(state.currentVideo.detection_score || 0).toFixed(2);
      showToast(`\u667a\u80fd\u5b9a\u4f4d\u5b8c\u6210\uff0c\u7f6e\u4fe1\u5ea6 ${score}`);
    }
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.textContent = "\u667a\u80fd\u5b9a\u4f4d"; }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  const form = $("#settingsForm");
  Object.entries(state.settings).forEach(([key, value]) => {
    if (form.elements[key] && key !== "api_key") form.elements[key].value = value;
  });
  $("#keyState").textContent = state.settings.api_key_configured ? "(\u5df2\u914d\u7f6e)" : "(\u672a\u914d\u7f6e)";
  const providers = { mock: "Mock", propainter: "ProPainter", local: "\u672c\u5730\u4fee\u590d", generic: "Generic API" };
  $("#providerName").textContent = providers[state.settings.provider] || state.settings.provider;
  $("#providerDot").classList.toggle(
    "online",
    state.settings.provider === "propainter" ||
    state.settings.provider === "local" ||
      (state.settings.provider === "generic" && state.settings.api_key_configured),
  );
  $("#providerHint").textContent = state.settings.provider === "mock"
    ? "\u6f14\u793a\u6a21\u5f0f\u4e0d\u4f1a\u4fee\u590d\u753b\u9762"
    : state.settings.provider === "propainter"
      ? "GPU \u89c6\u9891\u4fee\u590d\u5f15\u64ce\u5df2\u5c31\u7eea"
    : state.settings.provider === "local"
      ? "\u672c\u5730\u8865\u5168\u5f15\u64ce\u5df2\u5c31\u7eea"
      : (state.settings.api_key_configured ? "\u8fdc\u7a0b\u6a21\u578b\u5df2\u5c31\u7eea" : "\u8bf7\u914d\u7f6e API Key");
}

async function saveSettings(event) {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.currentTarget));
  ["concurrency", "max_retries", "poll_interval_seconds"].forEach(key => {
    data[key] = Number(data[key]);
  });
  if (!data.api_key) delete data.api_key;
  try {
    state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(data) });
    showToast("\u8bbe\u7f6e\u5df2\u4fdd\u5b58");
    await loadSettings();
  } catch (error) { showToast(error.message, true); }
}

async function toggleWorker() {
  try {
    const running = state.stats?.worker_running;
    await api(running ? "/api/jobs/pause" : "/api/jobs/start", { method: "POST" });
    showToast(running ? "\u961f\u5217\u5df2\u6682\u505c" : "\u961f\u5217\u5df2\u542f\u52a8");
    await loadStats();
  } catch (error) { showToast(error.message, true); }
}

async function loadQueue() {
  const result = await api("/api/videos?limit=200");
  const jobs = result.items.filter(video => video.status !== "pending");
  $("#queueList").innerHTML = jobs.length ? jobs.map(video => {
    const progress = Math.round((video.progress || 0) * 100);
    return `<article class="queue-item">
      <img src="/api/videos/${video.id}/thumbnail" alt="">
      <div class="queue-name"><strong>${escapeHtml(video.title || video.filename)}</strong><small>${escapeHtml(video.category)} &middot; ${formatBytes(video.size_bytes)}</small></div>
      <div class="queue-progress"><span>${video.status === "processing" ? `${progress}%` : statusLabels[video.status]}</span><div class="progress-track"><div class="progress-bar" style="width:${video.status === "completed" ? 100 : progress}%"></div></div></div>
      <div class="queue-status ${video.status}" title="${escapeHtml(video.error || "")}">${statusLabels[video.status] || video.status}</div>
    </article>`;
  }).join("") : `<div class="empty-state"><div class="empty-icon">\u21c4</div><h2>\u961f\u5217\u662f\u7a7a\u7684</h2><p>\u5728\u7d20\u6750\u5e93\u4e2d\u9009\u62e9\u89c6\u9891\u5e76\u6dfb\u52a0\u5230\u961f\u5217\u3002</p></div>`;
}

function switchView(view) {
  state.currentView = view;
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.view === view));
  $("#libraryView").classList.toggle("hidden", view !== "library");
  $("#queueView").classList.toggle("hidden", view !== "queue");
  $("#settingsView").classList.toggle("hidden", view !== "settings");
  const titles = { library: "\u7d20\u6750\u5e93", queue: "\u4efb\u52a1\u961f\u5217", settings: "\u63a5\u53e3\u8bbe\u7f6e" };
  $("#pageTitle").textContent = titles[view];
  $(".search-box").classList.toggle("hidden", view !== "library");
  if (view === "queue") loadQueue().catch(error => showToast(error.message, true));
  if (view === "settings") loadSettings().catch(error => showToast(error.message, true));
}

function bindEvents() {
  let searchTimer;
  $("#searchInput").addEventListener("input", () => {
    clearTimeout(searchTimer); searchTimer = setTimeout(() => loadVideos(true), 300);
  });
  $("#categoryFilter").addEventListener("change", () => loadVideos(true));
  $("#statusFilter").addEventListener("change", () => loadVideos(true));
  $("#scanButton").addEventListener("click", scanLibrary);
  $("#workerButton").addEventListener("click", toggleWorker);
  $("#prevPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadVideos(); });
  $("#nextPage").addEventListener("click", () => { state.offset += state.limit; loadVideos(); });
  $("#selectPageButton").addEventListener("click", () => {
    const allSelected = state.videos.every(video => state.selected.has(video.id));
    state.videos.forEach(video => allSelected ? state.selected.delete(video.id) : state.selected.add(video.id));
    renderVideos();
  });
  $("#queueSelectedButton").addEventListener("click", () => queueVideos([...state.selected]));
  $("#closeDrawer").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  ["#maskX", "#maskY", "#maskW", "#maskH"].forEach(id => $(id).addEventListener("input", updateMaskPreview));
  $("#resetRegion").addEventListener("click", () => detectCurrentRegion(true));
  $("#saveRegion").addEventListener("click", () => saveCurrentRegion(true).catch(error => showToast(error.message, true)));
  $("#queueCurrent").addEventListener("click", async () => {
    try { await saveCurrentRegion(false); await queueVideos([state.currentVideo.id]); closeDrawer(); }
    catch (error) { showToast(error.message, true); }
  });
  $("#settingsForm").addEventListener("submit", saveSettings);
  $("#retryFailedButton").addEventListener("click", () => queueVideos([], null, "failed").then(loadQueue));
  $$(".nav-item").forEach(item => item.addEventListener("click", () => switchView(item.dataset.view)));
}

async function init() {
  bindEvents();
  try {
    await Promise.all([loadSettings(), loadStats()]);
    if (state.stats.total === 0) await scanLibrary();
    else await loadVideos();
    setInterval(async () => {
      try {
        await loadStats();
        if (state.currentView === "library") await loadVideos();
        if (state.currentView === "queue") await loadQueue();
      } catch (_) {}
    }, 3000);
  } catch (error) { showToast(error.message, true); }
}

init();
