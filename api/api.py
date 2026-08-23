import csv
import io
import logging
import mimetypes
import os
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eval import authoring as eval_authoring
from eval import manifest as eval_manifest
from eval import promote as eval_promote
from src.chart_extraction import (
    ChartExtraction,
    ChartGridData,
    ChartImageData,
    build_axis_ticks,
    compute_chart_grid,
    extract_chart_series,
    prepare_chart_image,
    set_chart_bounds,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_STORED_CHARTS = 20

app = FastAPI(
    title="Chart Digitalization API",
    description="Upload a chart image and get the time series it encodes, "
    "positioned so it can be drawn back on the original picture.",
    version="1.0.0",
)


class AreaBody(BaseModel):
    """A user-highlighted rectangle, in original-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class SeriesEditBody(BaseModel):
    """
    One hand-made correction to a series, spanning the x range between two points
    given in original-image pixel coordinates. `kind` picks what happens to that
    span: "line" overwrites it with the straight line between the two points,
    "delete" drops its values, leaving a gap. Neither point need sit on the
    extracted line.

    `anchor1`/`anchor2` hold the index of the extracted datapoint an endpoint was
    snapped to, if any. An anchored endpoint is resolved from the series itself
    when the edit is replayed, so it keeps meeting its datapoint exactly even
    after earlier edits or a re-extraction moved it.
    """

    series_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    anchor1: Optional[int] = None
    anchor2: Optional[int] = None
    kind: str = Field("line", pattern="^(line|delete)$")


class SeriesEditsBody(BaseModel):
    """The complete set of corrections a chart should carry, replayed in order."""

    edits: list[SeriesEditBody] = []


class SeriesNamesBody(BaseModel):
    """The complete set of series display names, positional by series index."""

    names: list[Optional[str]] = []


class PromoteToTestsetBody(BaseModel):
    """What an annotator provides when saving a chart's current corrected state
    as a candidate ground-truth entry (see `eval.authoring.stage_chart`)."""

    category: str
    notes: str = ""
    annotator: Optional[str] = None


@dataclass
class StoredChart:
    """
    One processed upload: the original bytes plus every extraction stage's result.

    `image_data` and `grid_data` are cached so that re-running extraction after a
    legend-area change reuses both outright (only series extraction is re-run),
    and after a chart-area change reuses `image_data` plus `grid_data`'s fitted
    grid lines/scales -- only `grid_data.chart_bounds` moves (see
    `set_chart_bounds`), so neither OCR/axis-label detection nor grid detection/
    scale fitting are re-run.
    """

    chart_id: str
    filename: str
    media_type: str
    image_bytes: bytes
    image_data: ChartImageData
    grid_data: ChartGridData
    extraction: ChartExtraction
    chart_area_override: Optional[tuple[float, float, float, float]] = None
    legend_area_override: Optional[tuple[float, float, float, float]] = None
    # `extraction.time_series` as it came out of extraction, before any hand-drawn
    # correction. Kept so `series_edits` can be replayed from scratch on every
    # change (including undo/redo) without re-running extraction.
    base_time_series: list = field(default_factory=list)
    series_edits: list[SeriesEditBody] = field(default_factory=list)
    # User-renamed/swapped series display names, positional by series index.
    # Survives re-extraction by index; a name past the new series count is
    # simply unused rather than an error.
    series_names_override: list[Optional[str]] = field(default_factory=list)
    # Indices the user has removed from the sidebar/chart/export. Kept as a set
    # of positions rather than dropped from the data so every other index-based
    # reference (series_edits, series_names_override) stays valid.
    removed_series: set[int] = field(default_factory=set)


class ChartStore:
    """Thread-safe, size-capped in-memory store of processed charts."""

    def __init__(self, max_items: int = MAX_STORED_CHARTS):
        self._items: OrderedDict[str, StoredChart] = OrderedDict()
        self._max_items = max_items
        self._lock = Lock()

    def add(self, chart: StoredChart) -> None:
        with self._lock:
            self._items[chart.chart_id] = chart
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    def get(self, chart_id: str) -> StoredChart:
        with self._lock:
            chart = self._items.get(chart_id)
        if chart is None:
            raise HTTPException(404, f"Unknown chart id: {chart_id}")
        with self._lock:
            self._items.move_to_end(chart_id)
        return chart

    def remove(self, chart_id: str) -> None:
        with self._lock:
            self._items.pop(chart_id, None)


store = ChartStore()


def _x_value(value):
    """JSON-friendly x value: ISO string for datetimes, float otherwise."""
    if isinstance(value, datetime):
        return value.isoformat()
    return None if value is None else float(value)


def _series_payload(
    extraction: ChartExtraction,
    names_override: Optional[list[Optional[str]]] = None,
    removed: Optional[set[int]] = None,
) -> list[dict]:
    """
    Extracted series as drawable point lists. Every point carries both its pixel
    position in the original image and its value on the chart axes.

    `names_override` replaces a series' display name by index (rename/swap), when
    the override at that index is present and non-empty. `removed` flags series
    the user has dropped; they are still returned in full (so indices stay
    stable and callers needing the shared x-axis row, e.g. `series[0]`, keep
    working) but carry `"removed": true` for callers to filter or grey out.
    """
    if not extraction.time_series:
        return []

    n_series = max(len(values) for _, values in extraction.time_series)
    y_pixels = extraction.y_pixels()
    series = []
    for index in range(n_series):
        points = []
        for x_pixel, (x_value, values), pixels in zip(
            extraction.x_pixels, extraction.time_series, y_pixels
        ):
            value = values[index] if index < len(values) else None
            pixel = pixels[index] if index < len(pixels) else None
            points.append(
                {
                    "x_pixel": int(x_pixel),
                    "y_pixel": None if pixel is None else round(float(pixel), 2),
                    "x_value": _x_value(x_value),
                    "y_value": None if value is None else round(float(value), 6),
                }
            )
        name = (
            extraction.series_names[index]
            if index < len(extraction.series_names)
            else None
        )
        if names_override and index < len(names_override) and names_override[index]:
            name = names_override[index]
        is_removed = removed is not None and index in removed
        series.append(
            {
                "name": name or f"series {index + 1}",
                "points": points,
                "removed": is_removed,
            }
        )
    return series


def _http_422(error: Exception, label: str) -> HTTPException:
    logger.exception("Extraction failed for %s", label)
    return HTTPException(
        422,
        "Could not digitalize this image: the axes, their labels or the "
        "plotted line could not be recognised "
        f"({type(error).__name__}: {error}).",
    )


def _prepare_from_bytes(
    image_bytes: bytes, suffix: str, label: str = "upload"
) -> ChartImageData:
    """Stage `image_bytes` to a temp file (OCR reads from disk) and run OCR/axis detection."""
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(image_bytes)
        try:
            return prepare_chart_image(temp_path)
        except Exception as error:  # OCR/axis detection fails on charts it cannot read
            raise _http_422(error, label) from error
    finally:
        os.unlink(temp_path)


def _compute_grid_and_series(
    image_data: ChartImageData, label: str = "upload"
) -> tuple[ChartGridData, ChartExtraction]:
    """Fit the default grid/scales and extract series for a freshly uploaded image."""
    try:
        grid_data = compute_chart_grid(image_data)
        extraction = extract_chart_series(image_data, grid_data, None)
        return grid_data, extraction
    except Exception as error:
        raise _http_422(error, label) from error


def _set_bounds_and_extract(
    image_data: ChartImageData,
    grid_data: ChartGridData,
    chart_area: Optional[tuple[float, float, float, float]],
    legend_area: Optional[tuple[float, float, float, float]],
    label: str = "upload",
) -> tuple[ChartGridData, ChartExtraction]:
    """
    Move `grid_data.chart_bounds` to `chart_area` (or the auto-detected default)
    and re-run series extraction -- reusing the already-fitted grid/scales
    rather than re-detecting them.
    """
    try:
        grid_data = set_chart_bounds(grid_data, chart_area)
        extraction = extract_chart_series(image_data, grid_data, legend_area)
        return grid_data, extraction
    except Exception as error:
        raise _http_422(error, label) from error


def _extract_series(
    image_data: ChartImageData,
    grid_data: ChartGridData,
    legend_area: Optional[tuple[float, float, float, float]],
    label: str = "upload",
) -> ChartExtraction:
    """Run only the series stage, reusing an already-computed `grid_data`."""
    try:
        return extract_chart_series(image_data, grid_data, legend_area)
    except Exception as error:
        raise _http_422(error, label) from error


def _endpoint_pixels(
    extraction: ChartExtraction,
    series: list,
    series_index: int,
    anchor: Optional[int],
    x_pixel: float,
    y_pixel: float,
) -> tuple[float, float]:
    """
    Image-pixel position of one edit endpoint: the anchored datapoint's own
    position when the user snapped to one, else the raw position they clicked.
    Falls back to the raw position if the anchor no longer exists or that column
    has no value, so a stale anchor degrades to a free point instead of failing.
    """
    if anchor is None or not 0 <= anchor < min(len(series), len(extraction.x_pixels)):
        return x_pixel, y_pixel
    value = series[anchor][1][series_index]
    if value is None:
        return x_pixel, y_pixel
    return float(extraction.x_pixels[anchor]), float(extraction.y_pixel_at(value))


def _apply_series_edits(
    extraction: ChartExtraction,
    base_time_series: list,
    edits: list[SeriesEditBody],
) -> list:
    """
    Replay `edits` in order on a copy of `base_time_series` and return the result.
    Each edit rewrites one series at every column whose x pixel falls inside its
    span: a "line" edit gives each column the value the straight line reads there
    -- so the datapoints between the endpoints stop carrying their extracted values
    and the span reads as exactly the drawn line -- while a "delete" edit clears
    them to None, leaving a gap. y pixels are clamped to the plot area, so an
    endpoint placed off the chart still yields a value that is on the axis.

    Anchored endpoints are resolved against the series as it stands *at this point
    in the replay*, so a line can be anchored to the result of an earlier edit.
    """
    series = [(x_value, list(values)) for x_value, values in base_time_series]
    if not series:
        return series

    _, y_top, _, y_bottom = extraction.chart_area
    y_lo, y_hi = min(y_top, y_bottom), max(y_top, y_bottom)
    n_series = len(series[0][1])

    for edit in edits:
        if not 0 <= edit.series_index < n_series:
            continue
        x1, y1 = _endpoint_pixels(
            extraction, series, edit.series_index, edit.anchor1, edit.x1, edit.y1
        )
        x2, y2 = _endpoint_pixels(
            extraction, series, edit.series_index, edit.anchor2, edit.x2, edit.y2
        )
        x_lo, x_hi = sorted((x1, x2))
        if edit.kind == "delete":
            for x_pixel, (_, values) in zip(extraction.x_pixels, series):
                if x_lo <= x_pixel <= x_hi:
                    values[edit.series_index] = None
            continue

        # A zero-width span has no line to read values off; deleting one is fine.
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        for x_pixel, (_, values) in zip(extraction.x_pixels, series):
            if not x_lo <= x_pixel <= x_hi:
                continue
            y_pixel = min(max(y1 + slope * (x_pixel - x1), y_lo), y_hi)
            values[edit.series_index] = float(extraction.y_value_at(y_pixel))
    return series


def _install_extraction(chart: StoredChart, extraction: ChartExtraction) -> None:
    """
    Adopt a freshly computed `extraction` as the chart's pristine base and replay
    any existing corrections on top of it, so re-extraction (e.g. after a plot-area
    change) does not silently discard hand-drawn work.
    """
    chart.extraction = extraction
    chart.base_time_series = [(x, list(v)) for x, v in extraction.time_series]
    if chart.series_edits:
        extraction.time_series = _apply_series_edits(
            extraction, chart.base_time_series, chart.series_edits
        )


def _chart_payload(
    chart: StoredChart, grid_source: str, x_count: int, y_count: int
) -> dict:
    extraction = chart.extraction
    width, height = extraction.image_size
    v_min, v_max = extraction.value_range()
    return {
        "id": chart.chart_id,
        "filename": chart.filename,
        "image": {
            "url": f"/api/charts/{chart.chart_id}/image",
            "width": width,
            "height": height,
        },
        "chart_area": {
            "x1": extraction.chart_area[0],
            "y1": extraction.chart_area[1],
            "x2": extraction.chart_area[2],
            "y2": extraction.chart_area[3],
        },
        "legend_area": (
            {
                "x1": extraction.legend_area[0],
                "y1": extraction.legend_area[1],
                "x2": extraction.legend_area[2],
                "y2": extraction.legend_area[3],
            }
            if extraction.legend_area is not None
            else None
        ),
        "axes": {
            "x_is_datetime": extraction.x_is_datetime,
            "y_is_log": extraction.y_is_log,
            "y_min": v_min,
            "y_max": v_max,
        },
        "series": _series_payload(
            extraction, chart.series_names_override, chart.removed_series
        ),
        "legend_matches": sum(1 for name in extraction.series_names if name),
        "ticks": build_axis_ticks(extraction, grid_source, x_count, y_count),
        "detected_grid": {
            "x": extraction.detected_grid_x,
            "y": extraction.detected_grid_y,
        },
    }


@app.post("/api/charts", summary="Upload a chart image and extract its time series")
async def create_chart(
    file: UploadFile = File(..., description="Chart image (png/jpg/bmp/tiff/webp)"),
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            415,
            f"Unsupported image type '{suffix}'. Use one of: "
            f"{', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )

    image_data = _prepare_from_bytes(image_bytes, suffix, label=file.filename)
    grid_data, extraction = _compute_grid_and_series(image_data, label=file.filename)

    if not extraction.time_series:
        raise HTTPException(422, "No data points could be extracted from this image")

    chart = StoredChart(
        chart_id=uuid.uuid4().hex,
        filename=file.filename or "chart",
        media_type=file.content_type or "image/png",
        image_bytes=image_bytes,
        image_data=image_data,
        grid_data=grid_data,
        extraction=extraction,
        base_time_series=[(x, list(v)) for x, v in extraction.time_series],
    )
    store.add(chart)
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.get("/api/charts/{chart_id}", summary="Fetch a previously extracted chart")
def get_chart(
    chart_id: str,
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    return _chart_payload(store.get(chart_id), grid_source, x_ticks, y_ticks)


@app.get("/api/charts/{chart_id}/ticks", summary="Grid lines and numeric scale ticks")
def get_ticks(
    chart_id: str,
    source: str = Query("generated", pattern="^(generated|detected)$"),
    x_count: int = Query(8, ge=2, le=40),
    y_count: int = Query(6, ge=2, le=40),
) -> dict:
    """Recompute the overlay grid without re-running extraction."""
    return build_axis_ticks(store.get(chart_id).extraction, source, x_count, y_count)


@app.put(
    "/api/charts/{chart_id}/legend-area",
    summary="Search a specific area for the legend and re-run extraction",
)
def set_legend_area(
    chart_id: str,
    area: Optional[AreaBody] = Body(
        None,
        description="Legend region in image pixels, or null to reset to the default",
    ),
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    """
    Re-run extraction with legend detection restricted to `area` (the user-
    highlighted region), or -- if `area` is omitted/null -- back to the default
    top-left corner. Everything downstream of legend detection (series-to-name
    matching, series color separation) is recomputed too. Any chart-area
    override set via PUT .../chart-area is preserved. OCR and grid/scale
    detection are reused unchanged rather than re-run.
    """
    chart = store.get(chart_id)
    legend_area = (area.x1, area.y1, area.x2, area.y2) if area else None
    extraction = _extract_series(
        chart.image_data, chart.grid_data, legend_area, label=chart.filename
    )
    if not extraction.time_series:
        raise HTTPException(422, "No data points could be extracted from this image")
    _install_extraction(chart, extraction)
    chart.legend_area_override = legend_area
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.put(
    "/api/charts/{chart_id}/chart-area",
    summary="Set the plot area to a specific region and re-run extraction",
)
def set_chart_area(
    chart_id: str,
    area: Optional[AreaBody] = Body(
        None,
        description="Plot-area region in image pixels, or null to reset to auto-detection",
    ),
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    """
    Re-run extraction with the plot area fixed to `area` (the user-highlighted
    region), or -- if `area` is omitted/null -- back to auto-detection. Series
    extraction and the legend's default search region (which is itself derived
    from the plot area) are recomputed too. Any legend-area override set via
    PUT .../legend-area is preserved. OCR, axis-label detection and grid
    detection/scale fitting are reused unchanged -- only which sub-region gets
    cropped for series extraction moves.
    """
    chart = store.get(chart_id)
    chart_area = (area.x1, area.y1, area.x2, area.y2) if area else None
    grid_data, extraction = _set_bounds_and_extract(
        chart.image_data,
        chart.grid_data,
        chart_area,
        chart.legend_area_override,
        label=chart.filename,
    )
    if not extraction.time_series:
        raise HTTPException(422, "No data points could be extracted from this image")
    chart.grid_data = grid_data
    _install_extraction(chart, extraction)
    chart.chart_area_override = chart_area
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.put(
    "/api/charts/{chart_id}/series-edits",
    summary="Replace the chart's hand-drawn straight-line corrections",
)
def set_series_edits(
    chart_id: str,
    body: SeriesEditsBody = Body(..., description="The full corrections list to apply"),
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    """
    Overwrite the chart's corrections with `body.edits` and rebuild the series from
    the pristine extraction plus that list. Replace (rather than append) semantics
    let a client drive undo/redo by re-sending an earlier list, so no correction
    ever needs an inverse operation. Edits whose `series_index` no longer exists
    are skipped rather than rejected.
    """
    chart = store.get(chart_id)
    chart.extraction.time_series = _apply_series_edits(
        chart.extraction, chart.base_time_series, body.edits
    )
    chart.series_edits = list(body.edits)
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.put(
    "/api/charts/{chart_id}/series-names",
    summary="Replace the chart's series display names (rename or swap)",
)
def set_series_names(
    chart_id: str,
    body: SeriesNamesBody = Body(
        ..., description="The full display-names list, positional by series index"
    ),
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    """
    Overwrite the chart's series names with `body.names`, positional by series
    index. A single rename sends the current names with one entry changed; a
    swap sends them with two entries exchanged -- both are just a full
    replacement, so neither needs its own representation.
    """
    chart = store.get(chart_id)
    chart.series_names_override = list(body.names)
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.delete(
    "/api/charts/{chart_id}/series/{series_index}",
    summary="Drop one series from the chart, its exports and its sidebar entry",
)
def remove_series(
    chart_id: str,
    series_index: int,
    grid_source: str = Query("generated", pattern="^(generated|detected)$"),
    x_ticks: int = Query(8, ge=2, le=40),
    y_ticks: int = Query(6, ge=2, le=40),
) -> dict:
    """
    Mark `series_index` as removed. The series stays in the underlying data
    (so `series_edits`/`series_names_override`, which reference it by index,
    stay valid) but is left out of the chart, table, sidebar and CSV export.
    """
    chart = store.get(chart_id)
    n_series = (
        len(chart.extraction.time_series[0][1]) if chart.extraction.time_series else 0
    )
    if not 0 <= series_index < n_series:
        raise HTTPException(404, f"Unknown series index: {series_index}")
    chart.removed_series.add(series_index)
    return _chart_payload(chart, grid_source, x_ticks, y_ticks)


@app.post(
    "/api/charts/{chart_id}/promote-to-testset",
    summary="Stage this chart's corrected series as a candidate ground-truth entry",
)
def promote_to_testset(chart_id: str, body: PromoteToTestsetBody = Body(...)) -> dict:
    """
    Write the chart's current series (corrections, renames and removals already
    applied -- the same data the CSV export uses) into
    `eval/ground_truth/staging/` for later review. Never writes straight into
    the canonical dataset (see `eval.promote`): an unreviewed correction must
    not become the baseline every future accuracy run is measured against.
    """
    chart = store.get(chart_id)
    if not chart.extraction.time_series:
        raise HTTPException(422, "This chart has no extracted series to save")

    series = _series_payload(
        chart.extraction, chart.series_names_override, chart.removed_series
    )
    fraction = eval_authoring.correction_fraction(
        chart.base_time_series, chart.extraction.time_series
    )
    return eval_authoring.stage_chart(
        image_bytes=chart.image_bytes,
        image_suffix=os.path.splitext(chart.filename)[1] or ".png",
        series=series,
        category=body.category,
        correction_fraction=fraction,
        annotator=body.annotator,
        notes=body.notes,
    )


# --------------------------------------------------------- testset review


@app.get("/api/testset/categories", summary="Known ground-truth categories")
def get_testset_categories() -> dict:
    """The synthetic categories `tests/data_generation.py` produces, offered as
    suggestions when staging a chart -- a category outside this list is still
    accepted, e.g. for a real-world chart shape none of them cover yet."""
    return {"categories": list(eval_manifest.CATEGORIES)}


@app.get("/api/testset/staged", summary="List charts staged for ground-truth review")
def list_staged_testset() -> dict:
    return {"entries": eval_authoring.list_staged(eval_authoring.STAGING_DIR)}


@app.get(
    "/api/testset/staged/{entry_id}/image",
    summary="A staged entry's image, for review",
)
def get_staged_testset_image(entry_id: str) -> Response:
    path = eval_authoring.staged_image_path(entry_id, eval_authoring.STAGING_DIR)
    if path is None:
        raise HTTPException(404, f"No staged entry: {entry_id}")
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type=media_type)


@app.post(
    "/api/testset/staged/{entry_id}/approve",
    summary="Promote a staged entry into the canonical ground-truth dataset",
)
def approve_staged_testset_entry(entry_id: str) -> dict:
    try:
        dest = eval_promote.approve(
            entry_id,
            staging_dir=eval_authoring.STAGING_DIR,
            canonical_dir=eval_manifest.CANONICAL_DIR,
            version=eval_manifest.CANONICAL_VERSION,
        )
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except FileExistsError as error:
        raise HTTPException(409, str(error)) from error
    return {
        "approved": entry_id,
        "dir": os.path.relpath(dest, eval_manifest.REPO_ROOT).replace(os.sep, "/"),
    }


@app.delete("/api/testset/staged/{entry_id}", summary="Discard a staged entry")
def reject_staged_testset_entry(entry_id: str) -> dict:
    try:
        eval_promote.reject(entry_id, staging_dir=eval_authoring.STAGING_DIR)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return {"rejected": entry_id}


@app.get("/api/charts/{chart_id}/image", summary="Original uploaded image")
def get_image(chart_id: str) -> Response:
    chart = store.get(chart_id)
    return Response(content=chart.image_bytes, media_type=chart.media_type)


@app.get("/api/charts/{chart_id}/series.csv", summary="Extracted series as CSV")
def get_series_csv(chart_id: str) -> Response:
    chart = store.get(chart_id)
    series = [
        s
        for s in _series_payload(
            chart.extraction, chart.series_names_override, chart.removed_series
        )
        if not s["removed"]
    ]
    if not series:
        raise HTTPException(404, "This chart has no extracted series")

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", lineterminator="\n")
    writer.writerow(["x"] + [s["name"] for s in series])
    for row_index in range(len(series[0]["points"])):
        x_value = series[0]["points"][row_index]["x_value"]
        values = [s["points"][row_index]["y_value"] for s in series]
        writer.writerow([x_value] + ["" if v is None else v for v in values])

    stem = os.path.splitext(chart.filename)[0] or "series"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
    )


@app.delete("/api/charts/{chart_id}", summary="Drop a chart from the server")
def delete_chart(chart_id: str) -> dict:
    store.get(chart_id)
    store.remove(chart_id)
    return {"deleted": chart_id}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
