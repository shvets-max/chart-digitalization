# Accuracy eval (Phases 1-3)

Implements phases 1-3 of `docs/accuracy-monitoring-design.md`: a ground-truth
manifest, a metrics harness, a CI regression gate, a SQLite metrics history, a
static HTML dashboard, and a staging/review flow for turning a chart's
hand-corrected series into a new ground-truth entry. There's no nightly job
that runs the eval and records it automatically — `--record` is a manual/
CI-triggered step for now. Phase 4 (an opt-in "contribute this correction" UI
prompt, wired to the endpoint below) is not built yet.

## Usage

```bash
# Rebuild the manifest after adding/removing fixtures under tests/data/
python -m eval.manifest

# Run the full eval, print a summary, and fail (exit 1) on regression vs the
# committed baseline
python -m eval.run_eval

# Run a subset, write the full report, skip the regression check
python -m eval.run_eval --categories linear_scaled log_scaled --out eval/history/run.json --compare-to ""

# After an intentional accuracy change, replace the committed baseline
python -m eval.run_eval --update-baseline

# Also append this run to the metrics history db (eval/history/eval.db)
python -m eval.run_eval --record

# Regenerate the dashboard (eval/history/report.html) from the history db
python -m eval.report

# List/approve/reject charts staged via POST /api/charts/{id}/promote-to-testset
python -m eval.promote --list
python -m eval.promote --approve <id>
python -m eval.promote --reject <id>
```

## Layout

- `manifest.py` — discovers the paired `<id>.png`/`<id>.csv` fixtures under
  `tests/data/<category>/`, plus any chart promoted into
  `ground_truth/canonical/<version>/charts/<id>/` (see `authoring.py`/
  `promote.py` below), and records them all as `ground_truth/v1.jsonl`, so a
  run works off an explicit chart list rather than a live directory scan.
  Synthetic fixtures stay where they already lived (`tests/data/`) rather than
  being copied elsewhere — same CSVs `tests/test_chart_extraction.py` already
  reads.
- `align.py` — extraction's series order isn't guaranteed to match the CSV's
  column order (see `TestMultilineExtraction` in `tests/test_chart_extraction.py`);
  matches predicted series to ground-truth columns by lowest error, greedily.
- `metrics.py` — Resolved Fraction, MAE, RMSE, normalized MAE, series-count
  accuracy. Pure functions over `{date: [value_or_None, ...]}` pairs.
- `run_eval.py` — the CLI: runs `src.chart_extraction.extract_time_series` over
  the manifest, aggregates per category, and (unless `--compare-to ""`) fails
  if any category regressed past a fixed tolerance vs `eval/history/baseline.json`.
- `history.py` — appends a run's report (`--record`) to `eval/history/eval.db`
  (SQLite, tracked in git): one row per `(run, category)` and one row per
  `(run, chart)`, so a metric's trend over time, or one specific chart's
  history across runs, is a query instead of diffing JSON files by hand.
- `report.py` — renders `eval/history/eval.db` into a self-contained
  `eval/history/report.html`: a trend chart per metric for every category,
  plus its latest snapshot. Not tracked in git — regenerate anytime the db
  changes.
- `authoring.py` — `POST /api/charts/{id}/promote-to-testset`
  (`api/api.py::promote_to_testset`) calls `stage_chart` here to write a
  chart's current (corrected/renamed/removed) series into
  `ground_truth/staging/<id>/` as `image.<ext>` + `series.csv` + `meta.json`.
  Also computes `correction_fraction` — the share of (date, series) cells that
  differ from the chart's pristine, uncorrected extraction — an always-on
  proxy for extraction quality even before a chart is reviewed.
- `promote.py` — CLI to review staged entries and move (`--approve`) or
  discard (`--reject`) them. Promotion is always this explicit, human step;
  the endpoint itself never writes into `canonical/`.

## What "ground truth" means here

Phase 1 covered only the synthetic fixtures `tests/data_generation.py`
produces. Phase 3 adds a second source: a real chart, corrected in the app's
existing modify-mode UI, staged via `promote-to-testset` and promoted by a
human via `eval.promote`. Nothing in the frontend calls the endpoint yet
(that's phase 4) — for now it's reachable directly, e.g. from the browser's
dev console or a script, once a chart has been extracted and corrected.

## Regression tolerances

Defined in `run_eval.py`: a category fails the gate if Resolved Fraction drops
more than 0.02 (absolute) or normalized MAE rises more than 10% (relative) vs
the baseline. Tune there if these prove too tight/loose in practice.
