"""Shared definition of "the project's own files".

Both the integrity manifest and the snapshot helper must agree on exactly which
files belong to the project, otherwise a clean manifest could coexist with an
incomplete snapshot. Keeping the rule in one place makes that impossible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Directories that are regenerated, downloaded or derived, never authored.
EXCLUDED_DIRECTORIES = frozenset({
    ".venv", "venv", "env", "__pycache__", ".git", ".github_cache",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "build", "dist", "htmlcov", "_snapshots", ".idea", ".vscode",
})

#: Suffixes and names that are build output rather than source.
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".log"})
EXCLUDED_NAMES = frozenset({".DS_Store", "Thumbs.db", ".coverage", "coverage.xml"})


def project_files(root: Path = PROJECT_ROOT) -> Iterator[Path]:
    """Yield every authored file under *root*, sorted, as relative paths.

    Sorting makes the manifest and any snapshot listing reproducible across
    platforms, which is what lets two of them be compared at all.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if EXCLUDED_DIRECTORIES.intersection(relative.parts[:-1]):
            continue
        if relative.name in EXCLUDED_NAMES or relative.suffix in EXCLUDED_SUFFIXES:
            continue
        if relative.parts and relative.parts[0] in EXCLUDED_DIRECTORIES:
            continue
        yield relative


#: Byte sequences for line-ending normalisation.
NUL, CR, LF = bytes([0]), bytes([13]), bytes([10])
CRLF = CR + LF


def digest(path: Path) -> str:
    """SHA-256 of a file's content, independent of the platform's line endings.

    Git stores text with LF and may check it out with CRLF (and the reverse for
    ``*.cmd``). Hashing raw bytes made a manifest written on Windows fail on a
    Linux checkout of identical content, so text is hashed with CRLF folded to
    LF. A file containing a NUL byte is treated as binary and hashed verbatim.
    """
    data = path.read_bytes()
    if NUL not in data:
        data = data.replace(CRLF, LF)
    return hashlib.sha256(data).hexdigest()


def digests(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Map POSIX-style relative path -> SHA-256 for every authored file."""
    return {relative.as_posix(): digest(root / relative) for relative in project_files(root)}
