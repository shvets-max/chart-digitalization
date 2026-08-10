import csv
import io
import logging
import os
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.chart_extraction import ChartExtraction, build_axis_ticks, extract_chart

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


@dataclass
class StoredChart:
    """One processed upload: the original bytes plus its extraction result."""

    chart_id: str
    filename: str
    media_type: str
    image_bytes: bytes
    extraction: ChartExtraction


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


def _series_payload(extraction: ChartExtraction) -> list[dict]:
    """
    Extracted series as drawable point lists. Every point carries both its pixel
    position in the original image and its value on the chart axes.
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
        series.append({"name": name or f"series {index + 1}", "points": points})
    return series


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
        "axes": {
            "x_is_datetime": extraction.x_is_datetime,
            "y_is_log": extraction.y_is_log,
            "y_min": v_min,
            "y_max": v_max,
        },
        "series": _series_payload(extraction),
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

    # extract_chart reads from disk, so the upload is staged in a temporary file
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(image_bytes)
        try:
            extraction = extract_chart(temp_path)
        except Exception as error:  # extraction fails on charts it cannot read
            logger.exception("Extraction failed for %s", file.filename)
            raise HTTPException(
                422,
                "Could not digitalize this image: the axes, their labels or the "
                "plotted line could not be recognised "
                f"({type(error).__name__}: {error}).",
            ) from error
    finally:
        os.unlink(temp_path)

    if not extraction.time_series:
        raise HTTPException(422, "No data points could be extracted from this image")

    chart = StoredChart(
        chart_id=uuid.uuid4().hex,
        filename=file.filename or "chart",
        media_type=file.content_type or "image/png",
        image_bytes=image_bytes,
        extraction=extraction,
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


@app.get("/api/charts/{chart_id}/image", summary="Original uploaded image")
def get_image(chart_id: str) -> Response:
    chart = store.get(chart_id)
    return Response(content=chart.image_bytes, media_type=chart.media_type)


@app.get("/api/charts/{chart_id}/series.csv", summary="Extracted series as CSV")
def get_series_csv(chart_id: str) -> Response:
    chart = store.get(chart_id)
    series = _series_payload(chart.extraction)
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
