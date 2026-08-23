"""
SQLite-backed history of eval runs, so a metric's trend over time/commits is a
query against `eval/history/eval.db` instead of diffing individual JSON
reports (see `eval.run_eval.run` for how one report is produced).
"""

import os
import sqlite3
from contextlib import contextmanager

from eval.manifest import REPO_ROOT

DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "eval", "history", "eval.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    manifest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    category TEXT NOT NULL,  -- 'overall' or one chart category
    n_charts INTEGER,
    resolved_fraction REAL,
    mae REAL,
    rmse REAL,
    mae_norm REAL,
    series_count_accuracy REAL,
    runtime_s REAL
);

CREATE TABLE IF NOT EXISTS chart_results (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    chart_id TEXT NOT NULL,
    category TEXT NOT NULL,
    resolved_fraction REAL,
    mae REAL,
    rmse REAL,
    mae_norm REAL,
    series_count_match INTEGER,
    runtime_s REAL
);
"""

_CATEGORY_METRIC_COLUMNS = (
    "n_charts",
    "resolved_fraction",
    "mae",
    "rmse",
    "mae_norm",
    "series_count_accuracy",
    "runtime_s",
)
_CHART_RESULT_COLUMNS = (
    "resolved_fraction",
    "mae",
    "rmse",
    "mae_norm",
    "series_count_match",
    "runtime_s",
)


@contextmanager
def _connect(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_run(report: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """Append one eval report (as produced by `eval.run_eval.run`) to the history
    db. Returns the new run's id."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (run_at, git_sha, manifest) VALUES (?, ?, ?)",
            (report["run_at"], report["git_sha"], report["manifest"]),
        )
        run_id = cursor.lastrowid

        scoped_stats = [("overall", report["overall"])] + sorted(
            report["by_category"].items()
        )
        for category, stats in scoped_stats:
            conn.execute(
                f"""INSERT INTO category_metrics
                    (run_id, category, {", ".join(_CATEGORY_METRIC_COLUMNS)})
                    VALUES (?, ?, {", ".join("?" * len(_CATEGORY_METRIC_COLUMNS))})""",
                (run_id, category, *(stats[c] for c in _CATEGORY_METRIC_COLUMNS)),
            )

        for chart in report["per_chart"]:
            conn.execute(
                f"""INSERT INTO chart_results
                    (run_id, chart_id, category, {", ".join(_CHART_RESULT_COLUMNS)})
                    VALUES (?, ?, ?, {", ".join("?" * len(_CHART_RESULT_COLUMNS))})""",
                (
                    run_id,
                    chart["id"],
                    chart["category"],
                    *(
                        int(chart[c]) if c == "series_count_match" else chart[c]
                        for c in _CHART_RESULT_COLUMNS
                    ),
                ),
            )
    return run_id


def list_runs(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Every recorded run, oldest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, run_at, git_sha, manifest FROM runs ORDER BY id"
        ).fetchall()
    return [dict(zip(("id", "run_at", "git_sha", "manifest"), row)) for row in rows]


def categories(db_path: str = DEFAULT_DB_PATH) -> list[str]:
    """Every category with at least one recorded run, including 'overall'."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM category_metrics ORDER BY category"
        ).fetchall()
    return [row[0] for row in rows]


def category_history(category: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Every recorded run's metrics for one category ('overall' or a chart
    category), oldest first."""
    columns = ("run_id", "run_at", "git_sha") + _CATEGORY_METRIC_COLUMNS
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT r.id, r.run_at, r.git_sha, {", ".join("m." + c for c in _CATEGORY_METRIC_COLUMNS)}
                FROM category_metrics m JOIN runs r ON r.id = m.run_id
                WHERE m.category = ?
                ORDER BY r.id""",
            (category,),
        ).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def chart_history(chart_id: str, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Every recorded run's metrics for one chart, oldest first -- for tracking
    down which specific chart regressed rather than just which category."""
    columns = ("run_id", "run_at", "git_sha", "category") + _CHART_RESULT_COLUMNS
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT r.id, r.run_at, r.git_sha, c.category,
                       {", ".join("c." + col for col in _CHART_RESULT_COLUMNS)}
                FROM chart_results c JOIN runs r ON r.id = c.run_id
                WHERE c.chart_id = ?
                ORDER BY r.id""",
            (chart_id,),
        ).fetchall()
    return [dict(zip(columns, row)) for row in rows]
