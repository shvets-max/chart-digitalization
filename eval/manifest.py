"""
Ground-truth manifest: discovers the paired image/CSV fixtures under
`tests/data/<category>/` plus any charts promoted from staging (see
`eval.authoring`/`eval.promote`) under `eval/ground_truth/canonical/<version>/`,
and records them as one JSONL file, so an eval run works off an explicit,
reproducible chart list rather than re-scanning directories that may have
changed underneath it.

CSVs are the same wide format `tests/test_chart_extraction.py` already reads:
`date;series1;series2;...`, one row per date, semicolon-separated.
"""

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "tests", "data")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "eval", "ground_truth", "v1.jsonl")
CANONICAL_DIR = os.path.join(REPO_ROOT, "eval", "ground_truth", "canonical")
CANONICAL_VERSION = "v1"

# Every category tests/data_generation.py currently produces. A chart lands
# here as soon as it's promoted from a one-off diagnosis fixture into a
# generator category (see docs/accuracy-monitoring-design.md).
CATEGORIES = (
    "linear_scaled",
    "log_scaled",
    "in_area_text",
    "multiline",
    "scrab_style",
    "dense_crossing",
    "crossing_multiline",
)

CSV_SEP = ";"

ExpectedSeries = dict[date, list[float]]


@dataclass
class ChartEntry:
    """One ground-truth chart: a paired image and wide-format CSV, relative to the repo root."""

    id: str
    category: str
    image_path: str
    csv_path: str
    source: str = "synthetic"


def discover_entries(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    categories=CATEGORIES,
    canonical_dir: str = CANONICAL_DIR,
    canonical_version: str = CANONICAL_VERSION,
) -> list[ChartEntry]:
    """Every synthetic fixture under `dataset_dir` plus every chart promoted into
    `canonical_dir/<canonical_version>/`."""
    return _discover_synthetic_entries(
        dataset_dir, categories
    ) + _discover_canonical_entries(canonical_dir, canonical_version)


def _discover_synthetic_entries(
    dataset_dir: str = DEFAULT_DATASET_DIR, categories=CATEGORIES
) -> list[ChartEntry]:
    """Pair up every `<id>.png` / `<id>.csv` found in each category dir under `dataset_dir`."""
    entries = []
    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_dir):
            continue
        for filename in sorted(os.listdir(category_dir)):
            if not filename.endswith(".png"):
                continue
            chart_id = os.path.splitext(filename)[0]
            csv_path = os.path.join(category_dir, f"{chart_id}.csv")
            if not os.path.isfile(csv_path):
                continue
            entries.append(
                ChartEntry(
                    id=chart_id,
                    category=category,
                    image_path=_relpath(os.path.join(category_dir, filename)),
                    csv_path=_relpath(csv_path),
                )
            )
    return entries


def _discover_canonical_entries(
    canonical_dir: str = CANONICAL_DIR, version: str = CANONICAL_VERSION
) -> list[ChartEntry]:
    """Every `eval.authoring.stage_chart`-shaped entry (`image.<ext>` + `series.csv`
    + `meta.json`) promoted into `canonical_dir/<version>/charts/<id>/` (see
    `eval.promote.approve`)."""
    charts_dir = os.path.join(canonical_dir, version, "charts")
    if not os.path.isdir(charts_dir):
        return []
    entries = []
    for chart_id in sorted(os.listdir(charts_dir)):
        entry_dir = os.path.join(charts_dir, chart_id)
        meta_path = os.path.join(entry_dir, "meta.json")
        csv_path = os.path.join(entry_dir, "series.csv")
        if not (os.path.isfile(meta_path) and os.path.isfile(csv_path)):
            continue
        image_path = next(
            (
                os.path.join(entry_dir, f)
                for f in sorted(os.listdir(entry_dir))
                if os.path.splitext(f)[0] == "image"
            ),
            None,
        )
        if image_path is None:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        entries.append(
            ChartEntry(
                id=chart_id,
                category=meta.get("category", "unknown"),
                image_path=_relpath(image_path),
                csv_path=_relpath(csv_path),
                source=meta.get("source", "production"),
            )
        )
    return entries


def _relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def write_manifest(
    entries: list[ChartEntry], path: str = DEFAULT_MANIFEST_PATH
) -> None:
    """Write one JSON object per line, sorted by (category, id) for a stable diff."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = sorted(entries, key=lambda e: (e.category, e.id))
    with open(path, "w", newline="\n") as f:
        f.writelines(json.dumps(asdict(entry)) + "\n" for entry in ordered)


def load_manifest(path: str = DEFAULT_MANIFEST_PATH) -> list[ChartEntry]:
    """Load a manifest written by `write_manifest`."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ChartEntry(**json.loads(line)))
    return entries


def load_ground_truth(csv_path: str) -> tuple[ExpectedSeries, int]:
    """Load one chart's ground-truth CSV (path relative to the repo root) into
    `({date: [value, ...]}, n_series)`."""
    with open(os.path.join(REPO_ROOT, csv_path), newline="") as f:
        reader = csv.reader(f, delimiter=CSV_SEP)
        header = next(reader)
        n_series = len(header) - 1
        expected = {
            datetime.strptime(row[0], "%Y-%m-%d").date(): [
                float(v.replace(",", ".")) for v in row[1:]
            ]
            for row in reader
        }
    return expected, n_series


if __name__ == "__main__":
    found = discover_entries()
    write_manifest(found)
    print(f"Wrote {len(found)} entries to {DEFAULT_MANIFEST_PATH}")
