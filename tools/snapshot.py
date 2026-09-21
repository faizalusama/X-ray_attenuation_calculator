"""Timestamped project snapshots — a safety net while there is no Git repository.

    python tools/snapshot.py save --label before-backend-refactor
    python tools/snapshot.py list
    python tools/snapshot.py diff                 # newest snapshot vs working tree
    python tools/snapshot.py diff 2026-09-20T15-41-05_before-backend-refactor

Snapshots land in ``_snapshots/`` (git-ignored) and contain only authored files,
so they stay small. This is deliberately *not* a version control system: there
is no restore command, because silently overwriting a working tree is exactly
the accident a safety net should not cause. To go back, copy the files you want
out of the snapshot directory yourself, after looking at a ``diff``.

Once Git is available, prefer it: ``git init`` then ordinary commits.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tree import PROJECT_ROOT, digests, project_files

SNAPSHOT_ROOT = PROJECT_ROOT / "_snapshots"
LABEL_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _snapshots() -> list[Path]:
    if not SNAPSHOT_ROOT.is_dir():
        return []
    return sorted((path for path in SNAPSHOT_ROOT.iterdir() if path.is_dir()), key=lambda p: p.name)


def _resolve(name: str | None) -> Path:
    existing = _snapshots()
    if not existing:
        raise SystemExit("No snapshots yet. Create one with: python tools/snapshot.py save")
    if name is None:
        return existing[-1]
    match = SNAPSHOT_ROOT / name
    if not match.is_dir():
        raise SystemExit(f"No snapshot named {name!r}. Run: python tools/snapshot.py list")
    return match


def save(label: str | None) -> int:
    if label and not set(label) <= LABEL_SAFE:
        raise SystemExit("Label may contain only letters, digits, hyphen and underscore.")
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    target = SNAPSHOT_ROOT / (f"{stamp}_{label}" if label else stamp)
    if target.exists():
        raise SystemExit(f"{target} already exists; wait a second and try again.")
    count = 0
    for relative in project_files():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
        count += 1
    size_mb = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) / 1e6
    print(f"Saved {count} files ({size_mb:.1f} MB) to {target.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


def show() -> int:
    existing = _snapshots()
    if not existing:
        print("No snapshots yet. Create one with: python tools/snapshot.py save")
        return 0
    for path in existing:
        # Count by the same rule used to create the snapshot, so a tool that
        # later drops a cache directory inside one cannot inflate the figure.
        contents = list(project_files(path))
        size_mb = sum((path / item).stat().st_size for item in contents) / 1e6
        marker = "  (newest)" if path is existing[-1] else ""
        print(f"{path.name}  {len(contents):>4} files  {size_mb:>6.1f} MB{marker}")
    return 0


def diff(name: str | None) -> int:
    snapshot = _resolve(name)
    before, now = digests(snapshot), digests()
    added = sorted(set(now) - set(before))
    removed = sorted(set(before) - set(now))
    changed = sorted(path for path in set(now) & set(before) if now[path] != before[path])
    print(f"Working tree vs {snapshot.name}\n")
    for label, paths in (("added", added), ("changed", changed), ("removed", removed)):
        for path in paths:
            print(f"{label:>8}: {path}")
    total = len(added) + len(changed) + len(removed)
    print(f"\n{total} difference(s)." if total else "\nIdentical.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    saver = sub.add_parser("save", help="copy the current authored files into a new snapshot")
    saver.add_argument("--label", help="short reminder of why the snapshot was taken")
    sub.add_parser("list", help="show existing snapshots, oldest first")
    differ = sub.add_parser("diff", help="compare the working tree against a snapshot")
    differ.add_argument("snapshot", nargs="?", help="snapshot name (default: the newest)")
    args = parser.parse_args(argv)
    if args.action == "save":
        return save(args.label)
    if args.action == "list":
        return show()
    return diff(args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
