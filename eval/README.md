# Accuracy eval (Phase 1 MVP)

Implements phase 1 of `docs/accuracy-monitoring-design.md`: a ground-truth
manifest, a metrics harness, and a CI regression gate. Phases 2-4 (metrics
history/dashboard, the ground-truth authoring endpoint, and the production
feedback loop) are not built yet.

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
```

## Layout

- `manifest.py` — discovers the paired `<id>.png`/`<id>.csv` fixtures under
  `tests/data/<category>/` and records them as `ground_truth/v1.jsonl`, so a
  run works off an explicit chart list rather than a live directory scan.
  Ground truth stays where it already lived (`tests/data/`) rather than being
  copied elsewhere — same CSVs `tests/test_chart_extraction.py` already reads.
- `align.py` — extraction's series order isn't guaranteed to match the CSV's
  column order (see `TestMultilineExtraction` in `tests/test_chart_extraction.py`);
  matches predicted series to ground-truth columns by lowest error, greedily.
- `metrics.py` — Resolved Fraction, MAE, RMSE, normalized MAE, series-count
  accuracy. Pure functions over `{date: [value_or_None, ...]}` pairs.
- `run_eval.py` — the CLI: runs `src.chart_extraction.extract_time_series` over
  the manifest, aggregates per category, and (unless `--compare-to ""`) fails
  if any category regressed past a fixed tolerance vs `eval/history/baseline.json`.

## What "ground truth" means here

Phase 1 only covers the synthetic fixtures `tests/data_generation.py` already
produces — there's no real-chart or manually-corrected entry yet. That's phase
3 in the design doc (`promote-to-testset` endpoint + staging/review flow).

## Regression tolerances

Defined in `run_eval.py`: a category fails the gate if Resolved Fraction drops
more than 0.02 (absolute) or normalized MAE rises more than 10% (relative) vs
the baseline. Tune there if these prove too tight/loose in practice.
