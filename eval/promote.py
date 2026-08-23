"""
CLI: review staged ground-truth entries (`eval/ground_truth/staging/<id>/`,
written by `POST /api/charts/{id}/promote-to-testset` -- see `eval.authoring`)
and promote or reject them. Promotion is always this explicit, human-triggered
step -- see docs/accuracy-monitoring-design.md §5: an unreviewed correction
must never become the baseline every future accuracy run is measured against.

    python -m eval.promote --list
    python -m eval.promote --approve scrab_style_ab12cd34
    python -m eval.promote --reject scrab_style_ab12cd34
"""

import argparse
import os
import shutil

from eval.authoring import STAGING_DIR, list_staged
from eval.manifest import CANONICAL_DIR, CANONICAL_VERSION


def approve(
    entry_id: str,
    staging_dir: str = STAGING_DIR,
    canonical_dir: str = CANONICAL_DIR,
    version: str = CANONICAL_VERSION,
) -> str:
    """Move `staging_dir/entry_id/` into `canonical_dir/version/charts/entry_id/`.
    Returns the destination directory. Raises if the entry isn't staged, or is
    already promoted to `version`."""
    src = os.path.join(staging_dir, entry_id)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"No staged entry: {entry_id}")
    dest = os.path.join(canonical_dir, version, "charts", entry_id)
    if os.path.exists(dest):
        raise FileExistsError(f"{entry_id} is already promoted to {version}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)
    return dest


def reject(entry_id: str, staging_dir: str = STAGING_DIR) -> None:
    """Permanently discard a staged entry."""
    src = os.path.join(staging_dir, entry_id)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"No staged entry: {entry_id}")
    shutil.rmtree(src)


def _print_staged(staging_dir: str = STAGING_DIR) -> None:
    entries = list_staged(staging_dir)
    if not entries:
        print(f"No staged entries in {staging_dir}")
        return
    for meta in entries:
        print(
            f"{meta['id']:32s} category={meta['category']:18s} "
            f"correction_fraction={meta.get('correction_fraction')} "
            f"source={meta['source']:10s} notes={meta.get('notes') or ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List staged entries")
    parser.add_argument("--approve", metavar="ID", help="Promote a staged entry")
    parser.add_argument("--reject", metavar="ID", help="Discard a staged entry")
    parser.add_argument(
        "--version",
        default=CANONICAL_VERSION,
        help="Canonical dataset version to promote into (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.approve:
        dest = approve(args.approve, version=args.version)
        print(f"Promoted {args.approve} -> {dest}")
        print("Run `python -m eval.manifest` to pick it up in the next eval run.")
    elif args.reject:
        reject(args.reject)
        print(f"Rejected and removed {args.reject}")
    else:
        _print_staged()


if __name__ == "__main__":
    main()
