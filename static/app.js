/* Chart digitalization UI: uploads an image, then draws the extracted series,
   an optional grid and an optional numeric scale on top of the original picture. */

const SERIES_SLOTS = [
  "--series-1", "--series-2", "--series-3", "--series-4",
  "--series-5", "--series-6", "--series-7", "--series-8",
];

// The overlay sits on a foreign image whose own line is usually blue or black,
// so series colors start at the orange slot to stay distinguishable from it.
const DEFAULT_SLOT = 1;

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

// How many hand-drawn line edits Ctrl+Z can walk back.
const MAX_EDIT_HISTORY = 10;

// Distance (CSS px) within which a pointer snaps to an extracted datapoint.
const SNAP_TOLERANCE = 10;

const state = {
  chart: null,
  image: null,
  ticks: { x: [], y: [] },
  hoverIndex: null,
  hiddenSeries: new Set(),
  // Index of the series whose popup menu is open, or null.
  menuIndex: null,
  modify: {
    active: false,
    seriesIndex: null,
    // hiddenSeries as it was before modify mode hid everything else, restored on exit.
    hiddenBefore: null,
    // First clicked point of the line being drawn; null between lines.
    // {x, y, anchor} -- anchor is a datapoint index, or null for a free point.
    first: null,
    cursor: null,
    // The datapoint the pointer would snap to right now, or null.
    snap: null,
    // The settled two-point span awaiting Enter/Del:
    // {from, to, lo, hi, unbroken}, where lo/hi are datapoint indices.
    selection: null,
  },
  // Corrections currently applied on the server, and the snapshots undo/redo walks.
  edits: [],
  editHistory: [[]],
  editIndex: 0,
  zoom: { scale: 1, tx: 0, ty: 0 },
  viewScale: 1,
  lastView: { width: 0, height: 0 },
  isPanning: false,
  panPointerId: null,
  areaSelect: {
    kind: null,
    active: false,
    dragging: false,
    pointerId: null,
    // Set only while drawing a brand-new rectangle (handle === null).
    start: null,
    current: null,
    // The rectangle once a first drag completes -- editable (via handle drag)
    // until the user confirms or cancels it.
    rect: null,
    // Which handle is being dragged ('nw'/'n'/.../'move'), or null while
    // drawing a brand-new rectangle.
    handle: null,
    dragAnchor: null,
    rectAtDragStart: null,
  },
};

// Distance (CSS px, independent of zoom) within which a pointer counts as
// "on" a pending rectangle's edge/corner rather than its interior.
const HANDLE_TOLERANCE = 8;
const HANDLE_CURSORS = {
  nw: "nwse-resize", se: "nwse-resize",
  ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize",
  e: "ew-resize", w: "ew-resize",
  move: "move",
};

// Drag-to-select is shared by the legend area and the plot area: only the
// endpoint and the status messages differ between the two.
const AREA_KINDS = {
  legend: {
    endpoint: "legend-area",
    payloadKey: "legend_area",
    selectButtonId: "legend-area-select",
    resetButtonId: "legend-area-reset",
    applyMessage: "Searching the highlighted area for a legend…",
    resetMessage: "Resetting to the default legend area…",
  },
  chart: {
    endpoint: "chart-area",
    payloadKey: "chart_area",
    selectButtonId: "chart-area-select",
    resetButtonId: "chart-area-reset",
    applyMessage: "Re-extracting with the highlighted plot area…",
    resetMessage: "Resetting the plot area to auto-detection…",
  },
};

const el = (id) => document.getElementById(id);
const canvas = el("overlay-canvas");
const ctx = canvas.getContext("2d");

const cssVar = (name) =>
  getComputedStyle(document.body).getPropertyValue(name).trim();

const seriesColor = (index) =>
  cssVar(SERIES_SLOTS[(DEFAULT_SLOT + index) % SERIES_SLOTS.length]);

/* ---------------------------------------------------------------- upload */

function setStatus(message, kind = "") {
  const status = el("status");
  status.textContent = message;
  status.className = `status${kind ? ` is-${kind}` : ""}`;
}

async function uploadImage(file) {
  if (!file) return;
  setStatus(`Digitalizing ${file.name}`, "busy");
  el("browse-button").disabled = true;

  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/charts", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    await showChart(payload);
    setStatus(`Extracted ${countValues(payload)} points from ${file.name}.`);
  } catch (error) {
    setStatus(error.message || "Extraction failed.", "error");
  } finally {
    el("browse-button").disabled = false;
  }
}

function countValues(chart) {
  return chart.series.reduce(
    (total, s) => total + s.points.filter((p) => p.y_value !== null).length, 0);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the uploaded image"));
    image.src = url;
  });
}

/* --------------------------------------------------------------- results */

async function showChart(chart) {
  state.chart = chart;
  state.image = await loadImage(chart.image.url);
  state.hoverIndex = null;
  // Before clearing hiddenSeries: exiting modify mode restores the set it saved.
  exitModifyMode();
  closeSeriesMenu();
  state.hiddenSeries = new Set();
  resetEditHistory();
  resetZoom();

  el("result-panel").hidden = false;
  el("result-file").textContent = chart.filename;
  el("download-csv").href = `/api/charts/${chart.id}/series.csv`;
  renderBadges(chart);
  renderSeriesToggles(chart);
  renderTable(chart);
  updateAreaHints(chart);
  el("scale-hint").textContent = chart.axes.y_is_log
    ? "Values read off a logarithmic y-axis."
    : "Values read off a linear y-axis.";

  await refreshTicks();
  render();
}

/* Refresh from a re-extraction (e.g. after picking a new legend or plot area)
   without reloading the image or resetting zoom/pan -- only the chart data changed. */
async function updateChart(chart) {
  state.chart = chart;
  renderBadges(chart);
  renderSeriesToggles(chart);
  renderTable(chart);
  updateAreaHints(chart);
  el("scale-hint").textContent = chart.axes.y_is_log
    ? "Values read off a logarithmic y-axis."
    : "Values read off a linear y-axis.";

  await refreshTicks();
  render();
}

function updateAreaHints(chart) {
  const legendHint = el("legend-area-hint");
  if (!chart.legend_matches) {
    legendHint.textContent = "No legend matched in this area. Drag a rectangle over the legend to search there.";
  } else {
    legendHint.textContent = `${chart.legend_matches} series name${chart.legend_matches === 1 ? "" : "s"} matched from the legend.`;
  }

  const a = chart.chart_area;
  el("chart-area-hint").textContent =
    `Plot area: ${Math.round(a.x2 - a.x1)}×${Math.round(a.y2 - a.y1)} px. ` +
    `Drag "Select area…" if detection got the boundary wrong.`;
}

function renderBadges(chart) {
  const { axes, series, chart_area: area } = chart;
  const badges = [
    `${series.length} series`,
    `${countValues(chart)} points`,
    axes.x_is_datetime ? "datetime x-axis" : "numeric x-axis",
    axes.y_is_log ? "log y-axis" : "linear y-axis",
    `range ${formatNumber(axes.y_min)} – ${formatNumber(axes.y_max)}`,
    `plot area ${area.x2 - area.x1}×${area.y2 - area.y1} px`,
  ];
  el("badges").innerHTML = badges
    .map((text) => `<span class="badge">${text}</span>`)
    .join("");
}

const seriesLabel = (index) =>
  (state.chart && state.chart.series[index] && state.chart.series[index].name) ||
  `Series ${index + 1}`;

/* One chip per extracted series; click opens its hide/modify menu. */
function renderSeriesToggles(chart) {
  const container = el("series-toggles");
  container.innerHTML = chart.series
    .map(
      (series, index) => `<button type="button" class="series-toggle"
        data-index="${index}" aria-haspopup="menu" aria-pressed="true"
        style="--dot: ${seriesColor(index)}">
        <span class="dot"></span>
        <span class="name">${series.name || `Series ${index + 1}`}</span>
      </button>`,
    )
    .join("");
  syncSeriesToggles();
}

/* Reflect state.hiddenSeries on the chips: they are rebuilt from scratch on every
   chart update, and hiding is driven from the popup menu and by modify mode. */
function syncSeriesToggles() {
  for (const button of el("series-toggles").querySelectorAll(".series-toggle")) {
    const index = Number(button.dataset.index);
    button.setAttribute("aria-pressed", String(!state.hiddenSeries.has(index)));
  }
}

function setSeriesHidden(index, hidden) {
  if (hidden) state.hiddenSeries.add(index);
  else state.hiddenSeries.delete(index);
  syncSeriesToggles();
  render();
}

/* Bound once in bindControls (not here): renderSeriesToggles rebuilds the
   container's innerHTML on every chart update, but the container element
   itself persists, so a listener added here would accumulate one copy per
   update instead of being replaced. */
function handleSeriesToggleClick(event) {
  const button = event.target.closest(".series-toggle");
  if (!button) return;
  const index = Number(button.dataset.index);
  if (state.menuIndex === index) return closeSeriesMenu();
  openSeriesMenu(index, button);
}

/* ------------------------------------------------------------ series menu */

function openSeriesMenu(index, button) {
  const menu = el("series-menu");
  state.menuIndex = index;
  el("series-menu-title").textContent = seriesLabel(index);
  el("series-menu-hide").textContent = state.hiddenSeries.has(index) ? "Show" : "Hide";
  menu.hidden = false;

  const anchor = button.getBoundingClientRect();
  const left = Math.min(anchor.left, window.innerWidth - menu.offsetWidth - 8);
  const top = Math.min(anchor.bottom + 4, window.innerHeight - menu.offsetHeight - 8);
  menu.style.left = `${Math.max(8, left)}px`;
  menu.style.top = `${Math.max(8, top)}px`;
}

function closeSeriesMenu() {
  state.menuIndex = null;
  el("series-menu").hidden = true;
}

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toPrecision(4).replace(/\.?0+$/, "")}B`;
  if (abs >= 1e6) return `${(value / 1e6).toPrecision(4).replace(/\.?0+$/, "")}M`;
  if (abs >= 1e3) return `${(value / 1e3).toPrecision(4).replace(/\.?0+$/, "")}k`;
  return String(Number(value.toPrecision(4)));
}

function formatX(value) {
  if (typeof value !== "string") return formatNumber(value);
  return value.slice(0, 10);
}

/* ----------------------------------------------------------------- ticks */

async function refreshTicks() {
  const source = el("opt-grid").value;
  const needsTicks = source !== "off" || el("opt-scale").checked;
  if (!state.chart || !needsTicks) {
    state.ticks = { x: [], y: [] };
    return;
  }
  const params = new URLSearchParams({
    source: source === "off" ? "generated" : source,
    x_count: el("opt-xticks").value,
    y_count: el("opt-yticks").value,
  });
  const response = await fetch(`/api/charts/${state.chart.id}/ticks?${params}`);
  state.ticks = response.ok ? await response.json() : { x: [], y: [] };
}

/* --------------------------------------------------------------- drawing */

function render() {
  if (!state.chart || !state.image) return;

  const image = state.image;
  const wrap = el("canvas-wrap");
  const fullscreen = el("result-panel").classList.contains("is-fullscreen");
  const naturalRatio = image.naturalHeight / image.naturalWidth;

  // Fullscreen fits the image inside the available box (letterboxed); otherwise it's width-driven.
  let cssWidth = wrap.clientWidth;
  let cssHeight = cssWidth * naturalRatio;
  if (fullscreen && wrap.clientHeight > 0 && cssHeight > wrap.clientHeight) {
    cssHeight = wrap.clientHeight;
    cssWidth = cssHeight / naturalRatio;
  }

  const scale = cssWidth / image.naturalWidth;
  const ratio = window.devicePixelRatio || 1;

  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;

  state.viewScale = scale;
  state.lastView = { width: cssWidth, height: cssHeight };

  const zoom = state.zoom;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.setTransform(ratio * zoom.scale, 0, 0, ratio * zoom.scale, ratio * zoom.tx, ratio * zoom.ty);
  ctx.drawImage(image, 0, 0, cssWidth, cssHeight);

  const view = {
    scale,
    width: cssWidth,
    height: cssHeight,
    x: (px) => px * scale,
    y: (py) => py * scale,
  };

  const dim = Number(el("opt-dim").value) / 100;
  if (dim > 0) {
    ctx.fillStyle = cssVar("--surface-1");
    ctx.globalAlpha = dim;
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    ctx.globalAlpha = 1;
  }

  if (el("opt-grid").value !== "off") drawGrid(view);
  if (el("opt-area").checked) drawPlotArea(view);
  if (el("opt-legend-area").checked) drawLegendArea(view);
  drawAreaDragPreview(view);
  if (el("opt-scale").checked) drawNumericScale(view);
  if (el("opt-series").checked) drawSeries(view);
  drawModifyPreview(view);

  updateConfirmPopup();
  updateEditToolbar();
}

function drawPlotArea(view) {
  const a = state.chart.chart_area;
  ctx.save();
  ctx.strokeStyle = cssVar("--text-muted");
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(
    view.x(a.x1), view.y(a.y1),
    view.x(a.x2 - a.x1), view.y(a.y2 - a.y1),
  );
  ctx.restore();
}

function strokeRectArea(view, area, color, dash, fillAlpha = 0) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dash);
  const x = view.x(area.x1), y = view.y(area.y1);
  const w = view.x(area.x2 - area.x1), h = view.y(area.y2 - area.y1);
  if (fillAlpha > 0) {
    ctx.globalAlpha = fillAlpha;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 1;
  }
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

/* The area currently searched for a legend (default top-left corner unless the
   user highlighted one). */
function drawLegendArea(view) {
  if (state.chart.legend_area) {
    strokeRectArea(view, state.chart.legend_area, cssVar("--series-2"), [6, 3]);
  }
}

/* The rectangle currently being selected for the legend or plot area: either
   a brand-new one still being dragged out, or a settled "pending" one (drawn
   with resize handles) awaiting Confirm/Cancel. */
function drawAreaDragPreview(view) {
  const select = state.areaSelect;
  if (!select.active) return;
  if (select.rect) {
    strokeRectArea(view, select.rect, cssVar("--series-1"), [4, 3], 0.12);
    drawAreaHandles(view, select.rect);
  } else if (select.dragging && select.start && select.current) {
    const area = normalizeRect({
      x1: select.start.x, y1: select.start.y,
      x2: select.current.x, y2: select.current.y,
    });
    strokeRectArea(view, area, cssVar("--series-1"), [4, 3], 0.12);
  }
}

/* Small squares at the 4 corners + 4 edge midpoints of a pending rectangle,
   marking where a drag resizes rather than moves it (see hitTestHandle). */
function drawAreaHandles(view, rect) {
  const midX = (rect.x1 + rect.x2) / 2;
  const midY = (rect.y1 + rect.y2) / 2;
  const points = [
    [rect.x1, rect.y1], [rect.x2, rect.y1], [rect.x1, rect.y2], [rect.x2, rect.y2],
    [midX, rect.y1], [midX, rect.y2], [rect.x1, midY], [rect.x2, midY],
  ];
  ctx.save();
  ctx.fillStyle = cssVar("--series-1");
  ctx.strokeStyle = cssVar("--surface-1");
  ctx.lineWidth = 1;
  for (const [x, y] of points) {
    const cx = view.x(x), cy = view.y(y);
    ctx.beginPath();
    ctx.rect(cx - 4, cy - 4, 8, 8);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function normalizeRect(rect) {
  return {
    x1: Math.min(rect.x1, rect.x2),
    y1: Math.min(rect.y1, rect.y2),
    x2: Math.max(rect.x1, rect.x2),
    y2: Math.max(rect.y1, rect.y2),
  };
}

function resizeRect(rect, handle, dx, dy) {
  const r = { ...rect };
  if (handle === "move") {
    r.x1 += dx; r.x2 += dx;
    r.y1 += dy; r.y2 += dy;
    return r;
  }
  if (handle.includes("n")) r.y1 += dy;
  if (handle.includes("s")) r.y2 += dy;
  if (handle.includes("w")) r.x1 += dx;
  if (handle.includes("e")) r.x2 += dx;
  return r;
}

function drawGrid(view) {
  const a = state.chart.chart_area;
  ctx.save();
  ctx.strokeStyle = cssVar("--text-muted");
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1;

  for (const tick of state.ticks.x) {
    const x = Math.round(view.x(tick.pixel)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, view.y(a.y1));
    ctx.lineTo(x, view.y(a.y2));
    ctx.stroke();
  }
  for (const tick of state.ticks.y) {
    const y = Math.round(view.y(tick.pixel)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(view.x(a.x1), y);
    ctx.lineTo(view.x(a.x2), y);
    ctx.stroke();
  }
  ctx.restore();
}

/* Labels get an opaque chip so they stay readable over whatever is underneath. */
function drawLabel(text, centerX, centerY, align) {
  const width = ctx.measureText(text).width + 8;
  const height = 15;
  const left = align === "right" ? centerX - width + 4 : centerX - width / 2;

  ctx.fillStyle = cssVar("--surface-1");
  ctx.strokeStyle = cssVar("--border");
  ctx.lineWidth = 1;
  ctx.fillRect(left, centerY - height / 2, width, height);
  ctx.strokeRect(left + 0.5, centerY - height / 2 + 0.5, width - 1, height - 1);

  ctx.fillStyle = cssVar("--text-secondary");
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, centerX, centerY);
}

/* The scale is drawn along the top and right edges of the plot: the source chart
   keeps its own labels on the left and bottom, so the two never overlap. */
function drawNumericScale(view) {
  const a = state.chart.chart_area;
  const left = view.x(a.x1);
  const right = view.x(a.x2);
  const top = view.y(a.y1);
  const bottom = view.y(a.y2);

  ctx.save();
  ctx.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

  let lastTop = Infinity;
  for (const tick of [...state.ticks.y].sort((p, q) => q.pixel - p.pixel)) {
    const y = Math.min(Math.max(view.y(tick.pixel), top + 9), bottom - 9);
    if (y > lastTop - 16) continue;
    lastTop = y;
    drawLabel(tick.label, right - 5, y, "right");
  }

  // keep the x row clear of the y column in the top-right corner
  const yColumnWidth = state.ticks.y.length
    ? Math.max(...state.ticks.y.map((t) => ctx.measureText(t.label).width)) + 14
    : 0;

  let lastRight = -Infinity;
  for (const tick of state.ticks.x) {
    const x = view.x(tick.pixel);
    const halfWidth = ctx.measureText(tick.label).width / 2 + 6;
    if (x - halfWidth < Math.max(lastRight, left)) continue;
    if (x + halfWidth > right - yColumnWidth) continue;
    lastRight = x + halfWidth;
    drawLabel(tick.label, x, top + 11, "center");
  }
  ctx.restore();
}

function drawSeries(view) {
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  state.chart.series.forEach((series, index) => {
    if (state.hiddenSeries.has(index)) return;
    // A surface-coloured halo under the stroke keeps it legible on busy images.
    for (const pass of [
      { color: cssVar("--surface-1"), width: 4.5, alpha: 0.65 },
      { color: seriesColor(index), width: 2, alpha: 1 },
    ]) {
      ctx.strokeStyle = pass.color;
      ctx.lineWidth = pass.width;
      ctx.globalAlpha = pass.alpha;
      ctx.beginPath();
      let penDown = false;
      for (const point of series.points) {
        if (point.y_pixel === null) {
          penDown = false;
          continue;
        }
        const x = view.x(point.x_pixel);
        const y = view.y(point.y_pixel);
        if (penDown) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        penDown = true;
      }
      ctx.stroke();
    }
  });

  ctx.globalAlpha = 1;
  drawHoverMarker(view);
  ctx.restore();
}

function drawHoverMarker(view) {
  const index = state.hoverIndex;
  if (index === null) return;
  const a = state.chart.chart_area;
  const anchor = state.chart.series[0].points[index];
  if (!anchor) return;

  const x = view.x(anchor.x_pixel);
  ctx.strokeStyle = cssVar("--text-muted");
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x, view.y(a.y1));
  ctx.lineTo(x, view.y(a.y2));
  ctx.stroke();
  ctx.setLineDash([]);

  state.chart.series.forEach((series, seriesIndex) => {
    if (state.hiddenSeries.has(seriesIndex)) return;
    const point = series.points[index];
    if (!point || point.y_pixel === null) return;
    const y = view.y(point.y_pixel);
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = seriesColor(seriesIndex);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = cssVar("--surface-1");
    ctx.stroke();
  });
}

/* ------------------------------------------------------------------- zoom */

// Keeps the zoomed image from being panned past its edges into empty canvas.
function clampZoom(scale, tx, ty) {
  const { width, height } = state.lastView;
  if (scale <= 1) return { scale: 1, tx: 0, ty: 0 };
  const minTx = width - width * scale;
  const minTy = height - height * scale;
  return {
    scale,
    tx: Math.min(0, Math.max(minTx, tx)),
    ty: Math.min(0, Math.max(minTy, ty)),
  };
}

function updateZoomUI() {
  el("zoom-level").textContent = `${Math.round(state.zoom.scale * 100)}%`;
  canvas.classList.toggle("is-zoomed", state.zoom.scale > 1);
}

function zoomAt(rectX, rectY, factor) {
  const zoom = state.zoom;
  const newScale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom.scale * factor));
  if (newScale === zoom.scale) return;
  const userX = (rectX - zoom.tx) / zoom.scale;
  const userY = (rectY - zoom.ty) / zoom.scale;
  const newTx = rectX - newScale * userX;
  const newTy = rectY - newScale * userY;
  state.zoom = clampZoom(newScale, newTx, newTy);
  updateZoomUI();
  render();
}

function resetZoom() {
  state.zoom = { scale: 1, tx: 0, ty: 0 };
  updateZoomUI();
}

function handleWheelZoom(event) {
  if (!state.chart) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
}

function startPan(event) {
  if (state.modify.active) return handleModifyClick(event);
  if (state.areaSelect.active) return startAreaInteraction(event);
  if (state.zoom.scale <= 1) return;
  state.isPanning = true;
  state.panPointerId = event.pointerId;
  state.panStart = { x: event.clientX, y: event.clientY, ...state.zoom };
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("is-panning");
  handlePointerLeave();
}

function panTo(event) {
  const start = state.panStart;
  const dx = event.clientX - start.x;
  const dy = event.clientY - start.y;
  state.zoom = clampZoom(start.scale, start.tx + dx, start.ty + dy);
  render();
}

function endPan(event) {
  if (state.areaSelect.dragging) return finishAreaInteraction(event);
  if (!state.isPanning) return;
  state.isPanning = false;
  canvas.classList.remove("is-panning");
  if (state.panPointerId !== null) canvas.releasePointerCapture(state.panPointerId);
  state.panPointerId = null;
}

/* ------------------------------------------------------------- full screen */

function setFullscreen(on) {
  el("result-panel").classList.toggle("is-fullscreen", on);
  el("fullscreen-toggle").textContent = on ? "Exit full screen" : "Full screen";
  render();
}

function toggleFullscreen() {
  setFullscreen(!el("result-panel").classList.contains("is-fullscreen"));
}

/* Client (viewport) coordinates -> original-image pixel coordinates, undoing
   both the zoom/pan transform and the CSS-to-natural-size scale. Used to turn
   an area drag into the same pixel space as chart_area/legend_area. */
function imagePointAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const zoom = state.zoom;
  const userX = (clientX - rect.left - zoom.tx) / zoom.scale;
  const userY = (clientY - rect.top - zoom.ty) / zoom.scale;
  return { x: userX / state.viewScale, y: userY / state.viewScale };
}

/* Inverse of imagePointAt: an image-pixel coordinate -> client (viewport)
   coordinates, for hit-testing a pending rectangle's handles against a raw
   pointer event and for positioning the confirm popup. */
function clientPointFor(imageX, imageY) {
  const rect = canvas.getBoundingClientRect();
  const zoom = state.zoom;
  return {
    x: rect.left + zoom.tx + imageX * state.viewScale * zoom.scale,
    y: rect.top + zoom.ty + imageY * state.viewScale * zoom.scale,
  };
}

/* Which handle of `rect` (image coords) client point (clientX, clientY) is
   on, if any: a corner ('nw'/'ne'/'sw'/'se'), an edge ('n'/'s'/'e'/'w'), the
   interior ('move'), or null if outside the rectangle entirely. Tolerance is
   in CSS px so handles stay a constant visual size regardless of zoom. */
function hitTestHandle(clientX, clientY, rect) {
  const topLeft = clientPointFor(rect.x1, rect.y1);
  const bottomRight = clientPointFor(rect.x2, rect.y2);
  const t = HANDLE_TOLERANCE;
  if (
    clientX < topLeft.x - t || clientX > bottomRight.x + t ||
    clientY < topLeft.y - t || clientY > bottomRight.y + t
  ) {
    return null;
  }

  const near = (a, b) => Math.abs(a - b) <= t;
  const onLeft = near(clientX, topLeft.x);
  const onRight = near(clientX, bottomRight.x);
  const onTop = near(clientY, topLeft.y);
  const onBottom = near(clientY, bottomRight.y);

  if (onTop && onLeft) return "nw";
  if (onTop && onRight) return "ne";
  if (onBottom && onLeft) return "sw";
  if (onBottom && onRight) return "se";
  if (onTop) return "n";
  if (onBottom) return "s";
  if (onLeft) return "w";
  if (onRight) return "e";
  return "move";
}

/* ------------------------------------------------------------------- areas */
/* Drag-to-select for the legend area and the plot area: only one can be
   active at a time, since both are drawn with the same pointer gesture on
   the same canvas. A drag first draws out a rectangle; releasing settles it
   into a "pending" state where its edges/corners can be dragged to resize,
   or its interior dragged to move it, until the user confirms or cancels. */

// The area currently in effect for `kind`, as a draggable rect -- so turning
// on select mode starts from something already there instead of a blank
// canvas the user has to draw from scratch.
function currentAreaRect(kind) {
  const area = state.chart && state.chart[AREA_KINDS[kind].payloadKey];
  return area ? { x1: area.x1, y1: area.y1, x2: area.x2, y2: area.y2 } : null;
}

function setAreaSelectMode(kind, active) {
  const previousKind = state.areaSelect.kind;
  if (previousKind && previousKind !== kind) resetSelectButton(previousKind);

  Object.assign(state.areaSelect, {
    kind: active ? kind : null,
    active,
    dragging: false,
    start: null,
    current: null,
    rect: active ? currentAreaRect(kind) : null,
    handle: null,
    dragAnchor: null,
    rectAtDragStart: null,
  });
  canvas.classList.toggle("is-selecting-area", active);
  canvas.style.cursor = "";

  const button = el(AREA_KINDS[kind].selectButtonId);
  button.setAttribute("aria-pressed", String(active));
  button.textContent = active ? "Drag to select…" : "Select area…";
  render();
}

function resetSelectButton(kind) {
  const button = el(AREA_KINDS[kind].selectButtonId);
  button.setAttribute("aria-pressed", "false");
  button.textContent = "Select area…";
}

function startAreaInteraction(event) {
  const select = state.areaSelect;

  if (select.rect) {
    const handle = hitTestHandle(event.clientX, event.clientY, select.rect);
    if (handle) {
      select.dragging = true;
      select.handle = handle;
      select.pointerId = event.pointerId;
      select.dragAnchor = imagePointAt(event.clientX, event.clientY);
      select.rectAtDragStart = { ...select.rect };
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    // Clicked outside the pending rectangle: abandon it and draw a new one.
    select.rect = null;
  }

  select.dragging = true;
  select.handle = null;
  select.pointerId = event.pointerId;
  select.start = imagePointAt(event.clientX, event.clientY);
  select.current = select.start;
  canvas.setPointerCapture(event.pointerId);
  render();
}

function updateAreaInteraction(event) {
  const select = state.areaSelect;
  if (select.handle) {
    const point = imagePointAt(event.clientX, event.clientY);
    const dx = point.x - select.dragAnchor.x;
    const dy = point.y - select.dragAnchor.y;
    select.rect = normalizeRect(resizeRect(select.rectAtDragStart, select.handle, dx, dy));
  } else {
    select.current = imagePointAt(event.clientX, event.clientY);
  }
  render();
}

// Hover feedback (resize/move cursor) while a pending rectangle exists but
// nothing is currently being dragged.
function updateAreaHover(event) {
  const select = state.areaSelect;
  if (!select.rect) {
    canvas.style.cursor = "";
    return;
  }
  const handle = hitTestHandle(event.clientX, event.clientY, select.rect);
  canvas.style.cursor = handle ? HANDLE_CURSORS[handle] || "crosshair" : "crosshair";
}

function finishAreaInteraction(event) {
  const select = state.areaSelect;
  const { pointerId, handle } = select;
  select.dragging = false;
  if (pointerId !== null) canvas.releasePointerCapture(pointerId);
  select.pointerId = null;

  if (handle) {
    // Finished resizing/moving an already-pending rectangle: stays pending.
    select.handle = null;
    select.dragAnchor = select.rectAtDragStart = null;
    render();
    return;
  }

  const { start, current } = select;
  select.start = select.current = null;
  // Too small to be a deliberate selection (e.g. a stray click): stay in
  // select mode with no pending rectangle, ready to try again.
  if (!start || !current || Math.abs(current.x - start.x) < 6 || Math.abs(current.y - start.y) < 6) {
    render();
    return;
  }
  select.rect = normalizeRect({ x1: start.x, y1: start.y, x2: current.x, y2: current.y });
  render();
}

async function confirmAreaSelection() {
  const select = state.areaSelect;
  if (!select.rect || !select.kind) return;
  const { kind, rect } = select;
  setAreaSelectMode(kind, false);
  await applyArea(kind, rect);
}

function cancelAreaSelection() {
  const select = state.areaSelect;
  select.rect = null;
  select.handle = null;
  select.dragAnchor = select.rectAtDragStart = null;
  render();
}

/* Shows/hides and positions the Confirm/Cancel popup under the pending
   rectangle. Hidden while a handle drag is in progress so it doesn't jump
   around under the pointer; it reappears once the drag settles. */
function updateConfirmPopup() {
  const popup = el("area-confirm");
  const select = state.areaSelect;
  if (!select.active || !select.rect || select.handle) {
    popup.hidden = true;
    return;
  }
  popup.hidden = false;

  const wrap = el("canvas-wrap");
  const wrapRect = wrap.getBoundingClientRect();
  const anchor = clientPointFor((select.rect.x1 + select.rect.x2) / 2, select.rect.y2);
  const left = anchor.x - wrapRect.left - popup.offsetWidth / 2;
  const top = anchor.y - wrapRect.top + 10;
  popup.style.left = `${Math.max(8, Math.min(left, wrapRect.width - popup.offsetWidth - 8))}px`;
  popup.style.top = `${Math.max(8, Math.min(top, wrapRect.height - popup.offsetHeight - 8))}px`;
}

async function applyArea(kind, area) {
  await sendArea(kind, area, AREA_KINDS[kind].applyMessage);
}

async function resetArea(kind) {
  await sendArea(kind, null, AREA_KINDS[kind].resetMessage);
}

async function sendArea(kind, area, busyMessage) {
  if (!state.chart) return;
  setStatus(busyMessage, "busy");
  try {
    const response = await fetch(`/api/charts/${state.chart.id}/${AREA_KINDS[kind].endpoint}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(area),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    await updateChart(payload);
    setStatus(`Extracted ${countValues(payload)} points from ${payload.filename}.`);
  } catch (error) {
    setStatus(error.message || "Area update failed.", "error");
  }
}

/* ----------------------------------------------------------- modify mode */
/* Redrawing one series by hand. Every other series is hidden so the target line
   is unambiguous, and each pair of clicks replaces the values across the x span
   they cover with the straight line between them -- neither point has to sit on
   the extracted line. Panning is suspended while modifying, since pointerdown is
   what places a point; the zoom buttons and wheel still work. */

function enterModifyMode(index) {
  exitModifyMode();
  if (state.areaSelect.active) setAreaSelectMode(state.areaSelect.kind, false);
  closeSeriesMenu();
  handlePointerLeave();

  state.modify = {
    active: true,
    seriesIndex: index,
    hiddenBefore: new Set(state.hiddenSeries),
    first: null,
    cursor: null,
    snap: null,
    selection: null,
  };
  state.hiddenSeries = new Set(
    state.chart.series.map((_, i) => i).filter((i) => i !== index),
  );
  syncSeriesToggles();
  // Drawing against a hidden overlay would be guesswork.
  el("opt-series").checked = true;

  el("modify-series").textContent = seriesLabel(index);
  el("modify-bar").hidden = false;
  canvas.classList.add("is-modifying");
  updateModifyHint();
  render();
}

function exitModifyMode() {
  if (!state.modify.active) return;
  state.hiddenSeries = state.modify.hiddenBefore || new Set();
  state.modify = {
    active: false, seriesIndex: null, hiddenBefore: null,
    first: null, cursor: null, snap: null, selection: null,
  };
  syncSeriesToggles();
  el("modify-bar").hidden = true;
  canvas.classList.remove("is-modifying");
  render();
}

function updateModifyHint() {
  const { first, snap, selection } = state.modify;
  if (selection) {
    const count = selection.hi - selection.lo + 1;
    const broken = selection.unbroken ? "" : " (already broken here)";
    el("modify-hint").textContent =
      `${count} point${count === 1 ? "" : "s"} selected${broken} — Enter draws a line, Del removes them.`;
    return;
  }
  if (snap) {
    el("modify-hint").textContent = first
      ? "Click to end on the highlighted datapoint."
      : "Click to start from the highlighted datapoint.";
    return;
  }
  el("modify-hint").textContent = first
    ? "Click the second point."
    : "Click two points to select a segment.";
}

/* The extracted datapoint close enough to the pointer to snap to, or null.
   There is one datapoint per x pixel column, so proximity has to be measured in
   2D -- by x alone every cursor position would sit on one. Distances are compared
   in CSS px so the snap radius stays constant regardless of zoom. */
function snapCandidateAt(clientX, clientY) {
  const points = state.chart.series[state.modify.seriesIndex].points;
  if (!points.length) return null;

  // Columns are 1 image px apart, so the tolerance in CSS px converts directly
  // to how many columns on either side of the pointer could be within reach.
  const scale = state.viewScale * state.zoom.scale;
  const reach = Math.max(1, Math.ceil(SNAP_TOLERANCE / scale));
  const middle = Math.round(imagePointAt(clientX, clientY).x - points[0].x_pixel);

  let best = null;
  for (let i = middle - reach; i <= middle + reach; i += 1) {
    const point = points[i];
    if (!point || point.y_pixel === null) continue;
    const at = clientPointFor(point.x_pixel, point.y_pixel);
    const distance = Math.hypot(at.x - clientX, at.y - clientY);
    if (distance <= SNAP_TOLERANCE && (!best || distance < best.distance)) {
      best = { index: i, x: point.x_pixel, y: point.y_pixel, distance };
    }
  }
  return best;
}

/* Where a click lands: on the datapoint under the pointer if one is in range
   (anchored, so it stays tied to that datapoint), else wherever was clicked. */
function modifyPointAt(event) {
  const snap = snapCandidateAt(event.clientX, event.clientY);
  if (snap) return { x: snap.x, y: snap.y, anchor: snap.index };
  const free = imagePointAt(event.clientX, event.clientY);
  return { x: free.x, y: free.y, anchor: null };
}

function handleModifyClick(event) {
  const modify = state.modify;
  const point = modifyPointAt(event);
  if (!modify.first) {
    modify.selection = null;
    modify.first = point;
    modify.cursor = { x: point.x, y: point.y };
  } else {
    modify.selection = buildSelection(modify.first, point);
    modify.first = null;
  }
  updateModifyHint();
  render();
}

/* Which datapoint a placed point sits on: the one it is anchored to, else the
   column nearest its x (datapoints are one per x pixel column). */
function datapointIndexOf(point) {
  if (point.anchor !== null) return point.anchor;
  const points = state.chart.series[state.modify.seriesIndex].points;
  return Math.round(point.x - points[0].x_pixel);
}

/* The datapoint range two placed points cover, and whether the extracted line
   runs unbroken across it -- only an unbroken run is lit up as a segment. */
function buildSelection(from, to) {
  const points = state.chart.series[state.modify.seriesIndex].points;
  if (!points.length) return null;
  const a = datapointIndexOf(from);
  const b = datapointIndexOf(to);
  const lo = Math.max(0, Math.min(a, b));
  const hi = Math.min(points.length - 1, Math.max(a, b));

  let unbroken = hi > lo;
  for (let i = lo; i <= hi && unbroken; i += 1) {
    if (points[i].y_pixel === null) unbroken = false;
  }
  return { from, to, lo, hi, unbroken };
}

function clearSelection() {
  state.modify.selection = null;
  updateModifyHint();
  render();
}

/* Enter and Del both consume the selection; the edit itself carries the two
   points, so an anchored end still resolves against its datapoint server-side. */
function applySelection(kind) {
  const { selection, seriesIndex } = state.modify;
  if (!selection) return;
  const { from, to } = selection;
  clearSelection();
  applyEdit({
    kind,
    series_index: seriesIndex,
    x1: from.x, y1: from.y, anchor1: from.anchor,
    x2: to.x, y2: to.y, anchor2: to.anchor,
  });
}

/* The line being drawn: a ring on the datapoint the pointer would snap to, a
   marker on the placed start point, and a dashed rubber band between them. */
function drawModifyPreview(view) {
  const { active, first, cursor, snap, selection } = state.modify;
  if (!active) return;
  ctx.save();
  if (selection) {
    drawSelectionHighlight(view, selection);
    drawModifyMarker(view, selection.from, selection.from.anchor !== null);
    drawModifyMarker(view, selection.to, selection.to.anchor !== null);
  }
  if (snap) drawModifyMarker(view, snap, true);
  if (first) {
    const end = snap || cursor;
    if (end) {
      ctx.strokeStyle = cssVar("--series-1");
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(view.x(first.x), view.y(first.y));
      ctx.lineTo(view.x(end.x), view.y(end.y));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    drawModifyMarker(view, first, first.anchor !== null);
  }
  ctx.restore();
}

/* The selected run of the extracted line, as a thick translucent glow tracing the
   exact datapoints in range. Only drawn for an unbroken run: a span with gaps in
   it is not one continuous segment, so lighting it up would overstate what is
   selected. */
function drawSelectionHighlight(view, selection) {
  if (!selection.unbroken) return;
  const points = state.chart.series[state.modify.seriesIndex].points;
  ctx.save();
  ctx.strokeStyle = cssVar("--series-1");
  ctx.globalAlpha = 0.3;
  ctx.lineWidth = 9;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = selection.lo; i <= selection.hi; i += 1) {
    const x = view.x(points[i].x_pixel);
    const y = view.y(points[i].y_pixel);
    if (i === selection.lo) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

/* Marker for a point on the line being drawn; anchored ones get an outer ring to
   show they are locked to an extracted datapoint rather than free-floating. */
function drawModifyMarker(view, point, anchored) {
  const x = view.x(point.x);
  const y = view.y(point.y);
  ctx.beginPath();
  ctx.arc(x, y, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = cssVar("--series-1");
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = cssVar("--surface-1");
  ctx.stroke();
  if (!anchored) return;
  ctx.beginPath();
  ctx.arc(x, y, 8.5, 0, Math.PI * 2);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = cssVar("--series-1");
  ctx.stroke();
}

/* ----------------------------------------------------------- edit history */
/* The server holds the corrections as one replayable list, so undo/redo just
   re-sends an earlier snapshot of that list instead of inverting an edit. */

function updateEditToolbar() {
  el("edit-undo").disabled = state.editIndex === 0;
  el("edit-redo").disabled = state.editIndex >= state.editHistory.length - 1;
  el("edit-delete").disabled = !state.modify.selection;
}

function resetEditHistory() {
  state.edits = [];
  state.editHistory = [[]];
  state.editIndex = 0;
}

function pushEdits(edits) {
  state.editHistory = state.editHistory.slice(0, state.editIndex + 1);
  state.editHistory.push(edits);
  // MAX_EDIT_HISTORY undoable moves means that many snapshots past the base one.
  while (state.editHistory.length > MAX_EDIT_HISTORY + 1) state.editHistory.shift();
  state.editIndex = state.editHistory.length - 1;
}

async function applyEdit(edit) {
  pushEdits([...state.edits, edit]);
  await sendEdits(state.editHistory[state.editIndex], "Applying the drawn line…");
}

async function undoEdit() {
  if (!state.chart || state.editIndex === 0) return;
  state.editIndex -= 1;
  await sendEdits(state.editHistory[state.editIndex], "Undoing…");
}

async function redoEdit() {
  if (!state.chart || state.editIndex >= state.editHistory.length - 1) return;
  state.editIndex += 1;
  await sendEdits(state.editHistory[state.editIndex], "Redoing…");
}

async function sendEdits(edits, busyMessage) {
  setStatus(busyMessage, "busy");
  try {
    const response = await fetch(`/api/charts/${state.chart.id}/series-edits`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edits }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    state.edits = edits;
    await updateChart(payload);
    setStatus(
      edits.length
        ? `${edits.length} edit${edits.length === 1 ? "" : "s"} applied — Ctrl+Z undoes, Ctrl+Shift+Z redoes.`
        : "No edits — showing the extracted series.",
    );
  } catch (error) {
    setStatus(error.message || "Edit failed.", "error");
  }
}

/* ------------------------------------------------------------------ hover */

function pointIndexAt(clientX) {
  const points = state.chart.series[0].points;
  const rect = canvas.getBoundingClientRect();
  const rectX = clientX - rect.left;
  const zoom = state.zoom;
  const userX = (rectX - zoom.tx) / zoom.scale;
  const imageX = userX / state.viewScale;
  const index = Math.round(imageX - points[0].x_pixel);
  return index >= 0 && index < points.length ? index : null;
}

function showTooltip(index, clientX) {
  const tooltip = el("tooltip");
  const rows = state.chart.series
    .map((series, seriesIndex) => {
      if (state.hiddenSeries.has(seriesIndex)) return "";
      const point = series.points[index];
      const value = point ? point.y_value : null;
      const label = state.chart.series.length > 1 ? `${series.name}: ` : "";
      return `<div class="tooltip-row">
        <span class="dot" style="background:${seriesColor(seriesIndex)}"></span>
        <span>${label}<strong>${value === null ? "no data" : formatNumber(value)}</strong></span>
      </div>`;
    })
    .join("");
  const anchor = state.chart.series[0].points[index];
  tooltip.innerHTML = `<div class="tooltip-x">${formatX(anchor.x_value)}</div>${rows}`;
  tooltip.hidden = false;

  const rect = canvas.getBoundingClientRect();
  const localX = clientX - rect.left;
  const flip = localX > rect.width - tooltip.offsetWidth - 20;
  tooltip.style.left = `${flip ? localX - tooltip.offsetWidth - 14 : localX + 14}px`;
  tooltip.style.top = "12px";
}

function handlePointerMove(event) {
  if (!state.chart) return;
  if (state.modify.active) {
    state.modify.snap = snapCandidateAt(event.clientX, event.clientY);
    if (state.modify.first) {
      state.modify.cursor = imagePointAt(event.clientX, event.clientY);
    }
    updateModifyHint();
    return render();
  }
  if (state.areaSelect.active) {
    if (state.areaSelect.dragging) return updateAreaInteraction(event);
    updateAreaHover(event);
    return handlePointerLeave();
  }
  if (state.isPanning) return panTo(event);
  const index = pointIndexAt(event.clientX);
  if (index === null) return handlePointerLeave();
  state.hoverIndex = index;
  showTooltip(index, event.clientX);
  render();
}

function handlePointerLeave() {
  if (state.modify.active && state.modify.snap) {
    state.modify.snap = null;
    updateModifyHint();
    render();
  }
  if (state.hoverIndex === null) return;
  state.hoverIndex = null;
  el("tooltip").hidden = true;
  render();
}

/* ------------------------------------------------------------------ table */

const MAX_TABLE_ROWS = 300;

function renderTable(chart) {
  const points = chart.series[0].points;
  const step = Math.max(1, Math.ceil(points.length / MAX_TABLE_ROWS));
  const header = ["x", ...chart.series.map((s) => s.name)];

  el("data-thead").innerHTML =
    `<tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>`;

  const rows = [];
  for (let i = 0; i < points.length; i += step) {
    const cells = chart.series.map((s) => {
      const value = s.points[i] ? s.points[i].y_value : null;
      return `<td>${value === null ? "—" : formatNumber(value)}</td>`;
    });
    rows.push(`<tr><td>${formatX(points[i].x_value)}</td>${cells.join("")}</tr>`);
  }
  el("data-tbody").innerHTML = rows.join("");
  el("table-count").textContent =
    step > 1
      ? `(every ${step}${ordinalSuffix(step)} of ${points.length} — CSV has all)`
      : `(${points.length} rows)`;
}

function ordinalSuffix(n) {
  if (n % 100 >= 11 && n % 100 <= 13) return "th";
  return { 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th";
}

/* --------------------------------------------------------------- controls */

function syncTickControls() {
  const usesCounts = el("opt-grid").value === "generated";
  el("tick-x-field").hidden = !usesCounts;
  el("tick-y-field").hidden = !usesCounts;
}

async function handleGridChange() {
  syncTickControls();
  el("opt-xticks-value").textContent = el("opt-xticks").value;
  el("opt-yticks-value").textContent = el("opt-yticks").value;
  await refreshTicks();
  render();
}

function bindControls() {
  el("browse-button").addEventListener("click", () => el("file-input").click());
  el("file-input").addEventListener("change", (event) => {
    uploadImage(event.target.files[0]);
    event.target.value = "";
  });

  const dropzone = el("dropzone");
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    uploadImage(event.dataTransfer.files[0]);
  });

  for (const id of ["opt-series", "opt-area", "opt-legend-area"]) {
    el(id).addEventListener("change", render);
  }
  el("opt-dim").addEventListener("input", () => {
    el("opt-dim-value").textContent = `${el("opt-dim").value}%`;
    render();
  });
  for (const id of ["opt-grid", "opt-xticks", "opt-yticks", "opt-scale"]) {
    el(id).addEventListener("input", handleGridChange);
  }

  el("series-toggles").addEventListener("click", handleSeriesToggleClick);
  el("series-menu-hide").addEventListener("click", () => {
    const index = state.menuIndex;
    closeSeriesMenu();
    if (index !== null) setSeriesHidden(index, !state.hiddenSeries.has(index));
  });
  el("series-menu-modify").addEventListener("click", () => {
    if (state.menuIndex !== null) enterModifyMode(state.menuIndex);
  });
  el("modify-done").addEventListener("click", exitModifyMode);
  el("edit-undo").addEventListener("click", undoEdit);
  el("edit-redo").addEventListener("click", redoEdit);
  el("edit-delete").addEventListener("click", () => applySelection("delete"));
  // Any click that is not on the menu or the chip that opened it dismisses it.
  document.addEventListener("pointerdown", (event) => {
    if (state.menuIndex === null) return;
    if (event.target.closest("#series-menu, .series-toggle")) return;
    closeSeriesMenu();
  });

  for (const kind of Object.keys(AREA_KINDS)) {
    const config = AREA_KINDS[kind];
    el(config.selectButtonId).addEventListener("click", () => {
      const isActive = state.areaSelect.active && state.areaSelect.kind === kind;
      setAreaSelectMode(kind, !isActive);
    });
    el(config.resetButtonId).addEventListener("click", () => resetArea(kind));
  }
  el("area-confirm-accept").addEventListener("click", confirmAreaSelection);
  el("area-confirm-cancel").addEventListener("click", cancelAreaSelection);

  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerleave", handlePointerLeave);
  canvas.addEventListener("pointerdown", startPan);
  canvas.addEventListener("pointerup", endPan);
  canvas.addEventListener("pointercancel", endPan);
  canvas.addEventListener("wheel", handleWheelZoom, { passive: false });
  canvas.addEventListener("dblclick", resetZoomAndRender);
  window.addEventListener("resize", render);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

  el("zoom-in").addEventListener("click", () => {
    if (!state.chart) return;
    const { width, height } = state.lastView;
    zoomAt(width / 2, height / 2, 1.4);
  });
  el("zoom-out").addEventListener("click", () => {
    if (!state.chart) return;
    const { width, height } = state.lastView;
    zoomAt(width / 2, height / 2, 1 / 1.4);
  });
  el("zoom-reset").addEventListener("click", resetZoomAndRender);

  el("fullscreen-toggle").addEventListener("click", toggleFullscreen);
  window.addEventListener("keydown", handleKeyDown);
}

/* Escape backs out of one layer at a time (menu, then the point being placed,
   then the selection, then modify mode, then full screen) so it never closes
   more than the user meant. */
function handleKeyDown(event) {
  if (isTypingTarget(event.target)) return;

  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    return event.shiftKey ? redoEdit() : undoEdit();
  }
  if (state.modify.selection && (event.key === "Delete" || event.key === "Enter")) {
    event.preventDefault();
    return applySelection(event.key === "Delete" ? "delete" : "line");
  }

  if (event.key !== "Escape") return;
  if (state.menuIndex !== null) return closeSeriesMenu();
  if (state.modify.first) {
    state.modify.first = state.modify.cursor = null;
    updateModifyHint();
    return render();
  }
  if (state.modify.selection) return clearSelection();
  if (state.modify.active) return exitModifyMode();
  if (el("result-panel").classList.contains("is-fullscreen")) setFullscreen(false);
}

const isTypingTarget = (target) =>
  ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;

function resetZoomAndRender() {
  resetZoom();
  render();
}

bindControls();
syncTickControls();
