"""
Static HTML dashboard rendered from `eval/history/eval.db`: one metric-trend
chart per category, plus its latest snapshot. Regenerate anytime the db
changes:

    python -m eval.report
"""

import argparse
import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.history import DEFAULT_DB_PATH, categories, category_history
from eval.manifest import REPO_ROOT

DEFAULT_OUT_PATH = os.path.join(REPO_ROOT, "eval", "history", "report.html")

# One trend chart per metric, in this order, for every category.
PLOTTED_METRICS = ("resolved_fraction", "mae_norm", "series_count_accuracy")


def _plot_metric(history: list[dict], metric: str, title: str) -> str:
    """Render one metric's trend across `history` (oldest first) as a base64 PNG."""
    xs = list(range(len(history)))
    ys = [row[metric] for row in history]
    labels = [row["git_sha"] for row in history]

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.plot(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(metric, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_category_section(category: str, db_path: str) -> str:
    history = category_history(category, db_path)
    if not history:
        return ""

    latest = history[-1]
    images = "".join(
        f'<img src="data:image/png;base64,{_plot_metric(history, metric, f"{category}: {metric}")}">'
        for metric in PLOTTED_METRICS
    )
    return (
        f"<section><h2>{category}</h2>"
        f"<p class='latest'>latest: n_charts={latest['n_charts']} "
        f"resolved_fraction={latest['resolved_fraction']:.3f} "
        f"mae_norm={_fmt(latest['mae_norm'])} "
        f"series_count_accuracy={latest['series_count_accuracy']:.3f} "
        f"&mdash; run {latest['run_id']} ({latest['git_sha']}, {latest['run_at']})</p>"
        f"<div class='charts'>{images}</div></section>"
    )


def _fmt(value) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def build_report(db_path: str = DEFAULT_DB_PATH) -> str:
    """Render the full dashboard as a self-contained HTML string."""
    cats = categories(db_path)
    if not cats:
        return (
            "<html><body><p>No eval runs recorded yet -- run "
            "<code>python -m eval.run_eval --record</code>.</p></body></html>"
        )

    # 'overall' first, then chart categories alphabetically.
    ordered = ["overall"] + sorted(c for c in cats if c != "overall")
    sections = "".join(_render_category_section(c, db_path) for c in ordered)

    return f"""<html>
<head>
<title>Chart extraction accuracy</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  section {{ margin-bottom: 2rem; }}
  .latest {{ color: #444; font-size: 0.9rem; }}
  .charts img {{ margin-right: 8px; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Chart extraction accuracy</h1>
<p>See <code>docs/accuracy-monitoring-design.md</code> for what these metrics mean.</p>
{sections}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    html = build_report(args.db)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
