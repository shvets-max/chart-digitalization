"""
CLI: run chart extraction against the ground-truth manifest and report accuracy
metrics, optionally gating on a regression against a committed baseline.

    python -m eval.run_eval
    python -m eval.run_eval --categories linear_scaled log_scaled --out eval/history/run.json
    python -m eval.run_eval --update-baseline
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from eval.align import align_extracted_to_expected
from eval.manifest import (
    DEFAULT_MANIFEST_PATH,
    REPO_ROOT,
    ChartEntry,
    load_ground_truth,
    load_manifest,
)
from eval.metrics import (
    mae,
    normalized_mae,
    resolved_fraction,
    rmse,
    series_count_match,
)
from src.chart_extraction import extract_time_series

HISTORY_DIR = os.path.join(REPO_ROOT, "eval", "history")
DEFAULT_BASELINE_PATH = os.path.join(HISTORY_DIR, "baseline.json")

# Regression-gate tolerances: how much worse a category is allowed to get
# relative to the baseline before `run()`'s caller should treat it as a failure.
RESOLVED_FRACTION_DROP_TOLERANCE = 0.02
MAE_NORM_RELATIVE_INCREASE_TOLERANCE = 0.10


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def evaluate_chart(entry: ChartEntry) -> dict:
    """Run extraction on one chart and score it against its ground truth."""
    expected, n_series = load_ground_truth(entry.csv_path)

    start = time.perf_counter()
    extracted_raw = extract_time_series(os.path.join(REPO_ROOT, entry.image_path))
    runtime_s = time.perf_counter() - start

    n_extracted = len(extracted_raw[0][1]) if extracted_raw else 0
    # Extraction order isn't guaranteed to match the CSV's column order (legend
    # or color-separation order can differ) -- realign before scoring, the same
    # way tests/test_chart_extraction.py's TestMultilineExtraction does.
    extracted = align_extracted_to_expected(
        expected, extracted_raw, n_series, n_extracted
    )

    return {
        "id": entry.id,
        "category": entry.category,
        "n_series_expected": n_series,
        "n_series_extracted": n_extracted,
        "series_count_match": series_count_match(n_series, n_extracted),
        "resolved_fraction": resolved_fraction(expected, extracted, n_series),
        "mae": mae(expected, extracted, n_series),
        "rmse": rmse(expected, extracted, n_series),
        "mae_norm": normalized_mae(expected, extracted, n_series),
        "runtime_s": runtime_s,
    }


def _mean(values: list[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict]) -> dict:
    """Mean of every metric across `rows`."""
    return {
        "n_charts": len(rows),
        "resolved_fraction": _mean([r["resolved_fraction"] for r in rows]),
        "mae": _mean([r["mae"] for r in rows]),
        "rmse": _mean([r["rmse"] for r in rows]),
        "mae_norm": _mean([r["mae_norm"] for r in rows]),
        "series_count_accuracy": _mean(
            [1.0 if r["series_count_match"] else 0.0 for r in rows]
        ),
        "runtime_s": _mean([r["runtime_s"] for r in rows]),
    }


def aggregate(per_chart: list[dict]) -> dict:
    """Overall and per-category metric summaries."""
    by_category = defaultdict(list)
    for row in per_chart:
        by_category[row["category"]].append(row)
    return {
        "overall": _summarize(per_chart),
        "by_category": {
            cat: _summarize(rows) for cat, rows in sorted(by_category.items())
        },
    }


def run(
    manifest_path: str = DEFAULT_MANIFEST_PATH, categories: Optional[list[str]] = None
) -> dict:
    """Evaluate every chart in the manifest (optionally filtered to `categories`)."""
    entries = load_manifest(manifest_path)
    if categories:
        entries = [e for e in entries if e.category in categories]
    per_chart = [evaluate_chart(e) for e in entries]
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "manifest": os.path.relpath(manifest_path, REPO_ROOT),
        **aggregate(per_chart),
        "per_chart": per_chart,
    }


def compare(current: dict, baseline: dict) -> list[str]:
    """Human-readable regression messages for any category that got worse than
    tolerance vs `baseline`; empty if nothing regressed. Categories present in
    only one of the two reports are ignored (new/removed categories aren't a
    regression signal)."""
    problems = []
    for category, base_stats in baseline["by_category"].items():
        cur_stats = current["by_category"].get(category)
        if cur_stats is None:
            continue

        base_rf, cur_rf = (
            base_stats["resolved_fraction"],
            cur_stats["resolved_fraction"],
        )
        if base_rf is not None and cur_rf is not None:
            drop = base_rf - cur_rf
            if drop > RESOLVED_FRACTION_DROP_TOLERANCE:
                problems.append(
                    f"{category}: resolved_fraction dropped {base_rf:.3f} -> {cur_rf:.3f}"
                )

        base_mae, cur_mae = base_stats["mae_norm"], cur_stats["mae_norm"]
        if base_mae and cur_mae is not None:
            relative_increase = (cur_mae - base_mae) / base_mae
            if relative_increase > MAE_NORM_RELATIVE_INCREASE_TOLERANCE:
                problems.append(
                    f"{category}: mae_norm rose {base_mae:.4f} -> {cur_mae:.4f} "
                    f"({relative_increase:+.0%})"
                )
    return problems


def _print_report(report: dict) -> None:
    print(f"git_sha={report['git_sha']}  manifest={report['manifest']}")
    print("overall:", json.dumps(report["overall"], indent=2))
    for category, stats in report["by_category"].items():
        print(f"  {category}: {stats}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="Write the full report JSON here")
    parser.add_argument(
        "--compare-to",
        default=DEFAULT_BASELINE_PATH,
        help="Baseline report to regression-check against; pass '' to skip the check",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write this run's report as the new baseline (eval/history/baseline.json)",
    )
    args = parser.parse_args()

    report = run(args.manifest, args.categories)
    _print_report(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report to {args.out}")

    exit_code = 0
    if args.compare_to and os.path.isfile(args.compare_to):
        with open(args.compare_to) as f:
            baseline = json.load(f)
        problems = compare(report, baseline)
        if problems:
            print("\nREGRESSIONS DETECTED vs baseline:")
            for problem in problems:
                print(f"  - {problem}")
            exit_code = 1
        else:
            print("\nNo regression vs baseline.")
    elif args.compare_to:
        print(f"\nNo baseline at {args.compare_to} yet, skipping regression check.")

    if args.update_baseline:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        with open(DEFAULT_BASELINE_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nUpdated baseline: {DEFAULT_BASELINE_PATH}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
