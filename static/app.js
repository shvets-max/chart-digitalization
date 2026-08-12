/* Chart digitalization UI: uploads an image, then draws the extracted series,
   an optional grid and an optional numeric scale on top of the original picture. */

const SERIES_SLOTS = [
  "--series-1", "--series-2", "--series-3", "--series-4",
  "--series-5", "--series-6", "--series-7", "--series-8",
];

// The overlay sits on a foreign image whose own line is usually blue or black,
// so the default starts at the orange slot to stay distinguishable from it.
const DEFAULT_SLOT = 1;

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

const state = {
  chart: null,
  image: null,
  ticks: { x: [], y: [] },
  colorSlot: DEFAULT_SLOT,
  hoverIndex: null,
  hiddenSeries: new Set(),
  zoom: { scale: 1, tx: 0, ty: 0 },
  viewScale: 1,
  lastView: { width: 0, height: 0 },
  isPanning: false,
  panPointerId: null,
  legendSelect: { active: false, dragging: false, pointerId: null, start: null, current: null },
};

const el = (id) => document.getElementById(id);
const canvas = el("overlay-canvas");
const ctx = canvas.getContext("2d");

const cssVar = (name) =>
  getComputedStyle(document.body).getPropertyValue(name).trim();

const seriesColor = (index) =>
  cssVar(SERIES_SLOTS[(state.colorSlot + index) % SERIES_SLOTS.length]);

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
  state.hiddenSeries = new Set();
  resetZoom();

  el("result-panel").hidden = false;
  el("result-file").textContent = chart.filename;
  el("download-csv").href = `/api/charts/${chart.id}/series.csv`;
  renderBadges(chart);
  renderSeriesToggles(chart);
  renderTable(chart);
  updateLegendHint(chart);
  el("scale-hint").textContent = chart.axes.y_is_log
    ? "Values read off a logarithmic y-axis."
    : "Values read off a linear y-axis.";

  await refreshTicks();
  render();
}

/* Refresh from a re-extraction (e.g. after picking a new legend area) without
   reloading the image or resetting zoom/pan -- only the chart data changed. */
async function updateChart(chart) {
  state.chart = chart;
  renderBadges(chart);
  renderSeriesToggles(chart);
  renderTable(chart);
  updateLegendHint(chart);
  el("scale-hint").textContent = chart.axes.y_is_log
    ? "Values read off a logarithmic y-axis."
    : "Values read off a linear y-axis.";

  await refreshTicks();
  render();
}

function updateLegendHint(chart) {
  const hint = el("legend-area-hint");
  if (!chart.legend_matches) {
    hint.textContent = "No legend matched in this area. Drag a rectangle over the legend to search there.";
  } else {
    hint.textContent = `${chart.legend_matches} series name${chart.legend_matches === 1 ? "" : "s"} matched from the legend.`;
  }
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

/* One toggle chip per extracted series; click hides/shows its line, markers and tooltip row. */
function renderSeriesToggles(chart) {
  const container = el("series-toggles");
  container.innerHTML = chart.series
    .map(
      (series, index) => `<button type="button" class="series-toggle" role="switch"
        data-index="${index}" aria-pressed="true" style="--dot: ${seriesColor(index)}">
        <span class="dot"></span>
        <span class="name">${series.name || `Series ${index + 1}`}</span>
      </button>`,
    )
    .join("");
}

/* Bound once in bindControls (not here): renderSeriesToggles rebuilds the
   container's innerHTML on every chart update, but the container element
   itself persists, so a listener added here would accumulate one copy per
   update instead of being replaced. */
function handleSeriesToggleClick(event) {
  const button = event.target.closest(".series-toggle");
  if (!button) return;
  const index = Number(button.dataset.index);
  const pressed = button.getAttribute("aria-pressed") === "true";
  button.setAttribute("aria-pressed", String(!pressed));
  if (pressed) state.hiddenSeries.add(index);
  else state.hiddenSeries.delete(index);
  render();
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
  if (el("opt-scale").checked) drawNumericScale(view);
  if (el("opt-series").checked) drawSeries(view);
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
   user highlighted one), plus a live preview of the rectangle being dragged. */
function drawLegendArea(view) {
  if (state.chart.legend_area) {
    strokeRectArea(view, state.chart.legend_area, cssVar("--series-2"), [6, 3]);
  }
  const drag = state.legendSelect;
  if (drag.dragging && drag.start && drag.current) {
    const area = normalizedDragArea(drag.start, drag.current);
    strokeRectArea(view, area, cssVar("--series-1"), [4, 3], 0.12);
  }
}

function normalizedDragArea(start, current) {
  return {
    x1: Math.min(start.x, current.x),
    y1: Math.min(start.y, current.y),
    x2: Math.max(start.x, current.x),
    y2: Math.max(start.y, current.y),
  };
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
  if (state.legendSelect.active) return startLegendDrag(event);
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
  if (state.legendSelect.dragging) return finishLegendDrag(event);
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
   a legend-area drag into the same pixel space as chart_area/legend_area. */
function imagePointAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const zoom = state.zoom;
  const userX = (clientX - rect.left - zoom.tx) / zoom.scale;
  const userY = (clientY - rect.top - zoom.ty) / zoom.scale;
  return { x: userX / state.viewScale, y: userY / state.viewScale };
}

/* ------------------------------------------------------------- legend area */

function setLegendSelectMode(active) {
  state.legendSelect.active = active;
  canvas.classList.toggle("is-selecting-legend", active);
  el("legend-area-select").setAttribute("aria-pressed", String(active));
  el("legend-area-select").textContent = active ? "Drag over the legend…" : "Select area…";
}

function startLegendDrag(event) {
  const select = state.legendSelect;
  select.dragging = true;
  select.pointerId = event.pointerId;
  select.start = imagePointAt(event.clientX, event.clientY);
  select.current = select.start;
  canvas.setPointerCapture(event.pointerId);
}

function updateLegendDrag(event) {
  state.legendSelect.current = imagePointAt(event.clientX, event.clientY);
  render();
}

async function finishLegendDrag(event) {
  const select = state.legendSelect;
  const { start, current, pointerId } = select;
  select.dragging = false;
  select.start = select.current = null;
  if (pointerId !== null) canvas.releasePointerCapture(pointerId);
  setLegendSelectMode(false);

  // Too small to be a deliberate selection (e.g. a stray click): leave the
  // current legend area untouched instead of searching a sliver.
  if (!start || !current || Math.abs(current.x - start.x) < 6 || Math.abs(current.y - start.y) < 6) {
    render();
    return;
  }
  await applyLegendArea(normalizedDragArea(start, current));
}

async function applyLegendArea(area) {
  await sendLegendArea(area, "Searching the highlighted area for a legend…");
}

async function resetLegendArea() {
  await sendLegendArea(null, "Resetting to the default legend area…");
}

async function sendLegendArea(area, busyMessage) {
  if (!state.chart) return;
  setStatus(busyMessage, "busy");
  try {
    const response = await fetch(`/api/charts/${state.chart.id}/legend-area`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(area),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    await updateChart(payload);
    setStatus(`Extracted ${countValues(payload)} points from ${payload.filename}.`);
  } catch (error) {
    setStatus(error.message || "Legend search failed.", "error");
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
  if (state.legendSelect.dragging) return updateLegendDrag(event);
  if (state.isPanning) return panTo(event);
  const index = pointIndexAt(event.clientX);
  if (index === null) return handlePointerLeave();
  state.hoverIndex = index;
  showTooltip(index, event.clientX);
  render();
}

function handlePointerLeave() {
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

function buildSwatches() {
  const container = el("swatches");
  container.innerHTML = SERIES_SLOTS.map(
    (slot, index) => `<button type="button" class="swatch" role="radio"
      style="--swatch: var(${slot})" data-slot="${index}"
      aria-label="Series colour ${index + 1}"
      aria-checked="${index === DEFAULT_SLOT}"></button>`,
  ).join("");

  container.addEventListener("click", (event) => {
    const button = event.target.closest(".swatch");
    if (!button) return;
    state.colorSlot = Number(button.dataset.slot);
    container.querySelectorAll(".swatch").forEach((swatch) => {
      swatch.setAttribute("aria-checked", swatch === button);
    });
    render();
  });
}

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

  el("legend-area-select").addEventListener("click", () => {
    setLegendSelectMode(!state.legendSelect.active);
  });
  el("legend-area-reset").addEventListener("click", resetLegendArea);

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
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el("result-panel").classList.contains("is-fullscreen")) {
      setFullscreen(false);
    }
  });
}

function resetZoomAndRender() {
  resetZoom();
  render();
}

buildSwatches();
bindControls();
syncTickControls();
