"""
Turns one chart's corrected extraction into a staged ground-truth candidate
(`eval/ground_truth/staging/<id>/`), for a human to review before it's
promoted into a canonical dataset version (see `eval.promote` and
docs/accuracy-monitoring-design.md §5).

Deliberately independent of `api.api.StoredChart`/FastAPI: callers (the
`/api/charts/{id}/promote-to-testset` endpoint) pass in already-resolved
values -- a chart's own name/removed-series resolution logic
(`api._series_payload`) stays the single source of truth for what "this
chart's current series" means, rather than being re-implemented here.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timezone

from eval.manifest import CSV_SEP, REPO_ROOT

STAGING_DIR = os.path.join(REPO_ROOT, "eval", "ground_truth", "staging")


def correction_fraction(
    base_time_series: list, corrected_time_series: list, tolerance: float = 1e-9
) -> float | None:
    """
    Fraction of (date, series) cells `corrected_time_series` actually changed
    relative to the pristine `base_time_series` extraction they both came from
    (`StoredChart.base_time_series`/`.extraction.time_series` in `api/api.py`).
    `None` if the two aren't comparable (e.g. no series extracted at all).

    A cheap, always-on proxy for extraction quality: a chart that needed no
    correction is presumably one the pipeline got right, and the fraction's
    trend across many charts is useful before any of them are reviewed and
    promoted.
    """
    if not base_time_series or len(base_time_series) != len(corrected_time_series):
        return None
    total = 0
    changed = 0
    for (_, base_values), (_, corrected_values) in zip(
        base_time_series, corrected_time_series
    ):
        for base_value, corrected_value in zip(base_values, corrected_values):
            total += 1
            if base_value is None or corrected_value is None:
                changed += base_value != corrected_value
            elif abs(base_value - corrected_value) > tolerance:
                changed += 1
    return changed / total if total else None


def _date_only(x_value) -> str:
    """Trim a JSON ISO-datetime string ("2023-01-02T00:00:00") down to its date,
    matching the `date;series...` ground-truth CSV convention
    `eval.manifest.load_ground_truth` parses."""
    return x_value.split("T", 1)[0] if isinstance(x_value, str) else x_value


def _csv_rows(series: list[dict]) -> tuple[list[str], list[list]]:
    """Wide-format CSV header + rows from an `api._series_payload`-shaped series
    list, skipping any series flagged `removed`."""
    kept = [s for s in series if not s.get("removed")]
    header = ["date"] + [s["name"] for s in kept]
    rows = []
    for row_index in range(len(kept[0]["points"]) if kept else 0):
        date_str = _date_only(kept[0]["points"][row_index]["x_value"])
        values = [s["points"][row_index]["y_value"] for s in kept]
        rows.append([date_str] + ["" if v is None else v for v in values])
    return header, rows


def stage_chart(
    *,
    image_bytes: bytes,
    image_suffix: str,
    series: list[dict],
    category: str,
    correction_fraction: float | None = None,
    source: str = "production",
    annotator: str | None = None,
    notes: str = "",
    staging_dir: str | None = None,
) -> dict:
    """
    Write `image<image_suffix>`, `series.csv` and `meta.json` into
    `staging_dir/<generated id>/`. Never writes into `canonical/` directly --
    an unreviewed correction must not become the baseline every future
    accuracy run is measured against (see `eval.promote.approve`).

    Returns the written `meta.json` plus its entry directory (relative to the
    repo root, under `dir`).
    """
    staging_dir = staging_dir if staging_dir is not None else STAGING_DIR
    entry_id = f"{category}_{uuid.uuid4().hex[:8]}"
    entry_dir = os.path.join(staging_dir, entry_id)
    os.makedirs(entry_dir, exist_ok=True)

    with open(os.path.join(entry_dir, f"image{image_suffix}"), "wb") as f:
        f.write(image_bytes)

    header, rows = _csv_rows(series)
    with open(os.path.join(entry_dir, "series.csv"), "w", newline="\n") as f:
        writer = csv.writer(f, delimiter=CSV_SEP, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    meta = {
        "id": entry_id,
        "source": source,
        "category": category,
        "n_series": len(header) - 1,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "annotator": annotator,
        "correction_fraction": correction_fraction,
        "notes": notes,
    }
    with open(os.path.join(entry_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {**meta, "dir": os.path.relpath(entry_dir, REPO_ROOT).replace(os.sep, "/")}


def list_staged(staging_dir: str = STAGING_DIR) -> list[dict]:
    """Every staged entry's `meta.json`, oldest first -- for a reviewer to look
    through before approving/rejecting (see `eval.promote`)."""
    if not os.path.isdir(staging_dir):
        return []
    entries = []
    for entry_id in sorted(os.listdir(staging_dir)):
        meta_path = os.path.join(staging_dir, entry_id, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                entries.append(json.load(f))
    return entries


def staged_image_path(entry_id: str, staging_dir: str = STAGING_DIR) -> str | None:
    """Absolute path to a staged entry's image file, whatever its extension.
    `None` if there's no such entry or it has no image."""
    entry_dir = os.path.join(staging_dir, entry_id)
    if not os.path.isdir(entry_dir):
        return None
    for name in sorted(os.listdir(entry_dir)):
        if os.path.splitext(name)[0] == "image":
            return os.path.join(entry_dir, name)
    return None
