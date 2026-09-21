"""Generate or verify MANIFEST.sha256.

    python tools/manifest.py verify    # exit 1 if the tree no longer matches
    python tools/manifest.py write     # regenerate after an intentional change

The manifest records what a release actually contained, so a reviewer can
confirm that an archive was not altered after the validation run that blessed
it. It is written in ``sha256sum`` format and can be checked with that tool
directly:  ``sha256sum -c MANIFEST.sha256``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tree import PROJECT_ROOT, digests

MANIFEST = PROJECT_ROOT / "MANIFEST.sha256"
#: The manifest cannot record its own digest without a fixed point.
SELF = "MANIFEST.sha256"


def current() -> dict[str, str]:
    return {path: value for path, value in digests().items() if path != SELF}


def recorded() -> dict[str, str]:
    if not MANIFEST.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value, _, path = line.partition("  ")
        if not path:
            raise SystemExit(f"Malformed manifest line: {line!r}")
        entries[path] = value
    return entries


def write() -> int:
    entries = current()
    MANIFEST.write_text("".join(f"{value}  {path}\n" for path, value in entries.items()), encoding="utf-8")
    print(f"Wrote {MANIFEST.name} with {len(entries)} entries.")
    return 0


def verify() -> int:
    now, before = current(), recorded()
    if not before:
        print(f"{MANIFEST.name} is missing. Run: python tools/manifest.py write", file=sys.stderr)
        return 1
    added = sorted(set(now) - set(before))
    removed = sorted(set(before) - set(now))
    changed = sorted(path for path in set(now) & set(before) if now[path] != before[path])
    for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
        for path in paths:
            print(f"{label:>8}: {path}", file=sys.stderr)
    if added or removed or changed:
        print(f"\n{len(added) + len(removed) + len(changed)} difference(s). "
              "If these are intentional, run: python tools/manifest.py write", file=sys.stderr)
        return 1
    print(f"{MANIFEST.name}: {len(now)} files match.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("verify", "write"))
    return verify() if parser.parse_args(argv).action == "verify" else write()


if __name__ == "__main__":
    raise SystemExit(main())
