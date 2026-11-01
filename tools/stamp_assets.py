"""Set every local asset reference in static/index.html to ?v=<content hash>.

    python tools/stamp_assets.py           # rewrite the stamps
    python tools/stamp_assets.py --check   # exit 1 if any stamp is stale

The server lets browsers cache ``/static/...?v=...`` for a year as immutable.
That is only safe if the stamp changes whenever the file does, so the stamp is
the file's own hash rather than a date someone has to remember to bump.
tests/test_server.py runs the check, so a stale stamp fails the suite.

Hashes use the same line-ending-normalised digest as MANIFEST.sha256, so a
Windows checkout and a Linux checkout produce identical stamps.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tree import PROJECT_ROOT, digest

INDEX = PROJECT_ROOT / "static" / "index.html"
#: Local stylesheet and script references. Vendor files carry their version in
#: the filename and are excluded.
_REFERENCE = re.compile(r'(?P<attr>href|src)="/static/(?P<path>(?!vendor/)[\w./-]+\.(?:css|js))(?:\?v=(?P<stamp>[\w.-]*))?"')
STAMP_LENGTH = 12


def expected_stamps() -> dict[str, str]:
    text = INDEX.read_text(encoding="utf-8")
    return {m["path"]: digest(PROJECT_ROOT / "static" / m["path"])[:STAMP_LENGTH] for m in _REFERENCE.finditer(text)}


def stale() -> list[str]:
    text = INDEX.read_text(encoding="utf-8")
    wanted = expected_stamps()
    return [f"{m['path']}: has {m['stamp'] or 'no stamp'}, expected {wanted[m['path']]}"
            for m in _REFERENCE.finditer(text) if m["stamp"] != wanted[m["path"]]]


def write() -> int:
    wanted = expected_stamps()
    text = INDEX.read_text(encoding="utf-8")
    updated = _REFERENCE.sub(lambda m: f'{m["attr"]}="/static/{m["path"]}?v={wanted[m["path"]]}"', text)
    if updated != text:
        INDEX.write_text(updated, encoding="utf-8")
    print(f"Stamped {len(wanted)} assets in {INDEX.relative_to(PROJECT_ROOT).as_posix()}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report stale stamps instead of rewriting")
    if not parser.parse_args(argv).check:
        return write()
    problems = stale()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print("Run: python tools/stamp_assets.py", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
