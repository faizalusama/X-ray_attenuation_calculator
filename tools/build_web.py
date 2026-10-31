"""Build the static browser version of the calculator into ``_site/``.

    python tools/build_web.py

The site is the ordinary interface in ``static/`` plus ``web/``, which answers the
interface's API calls from the same Python engine running in the browser through
Pyodide. Nothing is reimplemented: ``xray_workbench`` is packaged as it stands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xray_workbench import __version__  # noqa: E402

PYODIDE_VERSION = "314.0.7"
PYODIDE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
PYODIDE_PACKAGES = ["numpy", "scipy", "sqlalchemy", "typing-extensions", "platformdirs"]
SITE = ROOT / "_site"
#: Exact-match text changes, so an edit to index.html cannot silently break the build.
PAGE_EDITS = (
    ('content="An offline scientific calculator for', 'content="A browser-based scientific calculator for'),
    ("Local workspace", "Runs in your browser"),
    ('<script src="/static/vendor/', '<link rel="stylesheet" href="/static/web/web.css">\n'
                                     '  <script src="/static/web/shim.js"></script>\n'
                                     '  <script src="/static/vendor/'),
)


def locked_version(package: str) -> str:
    """The version verified in requirements-lock.txt, so the site computes with the same data."""
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(package)}==(\S+)", lock, re.MULTILINE | re.IGNORECASE)
    if match is None:
        raise SystemExit(f"{package} is not pinned in requirements-lock.txt.")
    return match.group(1)


def page() -> str:
    text = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for old, new in PAGE_EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"static/index.html no longer contains exactly one {old!r}.")
        text = text.replace(old, new)
    return text.replace('"/static/', '"static/')


def engine_archive(destination: Path) -> str:
    """Zip the package's Python sources and name the file after its content."""
    temporary = destination / "engine.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "xray_workbench").rglob("*.py")):
            archive.write(path, path.relative_to(ROOT).as_posix())
    name = f"engine-{hashlib.sha256(temporary.read_bytes()).hexdigest()[:12]}.zip"
    temporary.rename(destination / name)
    return name


def build(site: Path = SITE, *, fetch_wheel: bool = True) -> dict[str, object]:
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(ROOT / "static", site / "static", ignore=shutil.ignore_patterns("index.html"))
    shutil.copytree(ROOT / "web", site / "static" / "web")
    (site / ".nojekyll").touch()
    (site / "index.html").write_text(page(), encoding="utf-8", newline="\n")
    engine = site / "engine"
    engine.mkdir()
    xraydb = locked_version("xraydb")
    wheel = f"xraydb-{xraydb}-py3-none-any.whl"
    if fetch_wheel:
        subprocess.run([sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "--only-binary", ":all:",
                        "--dest", str(engine), f"xraydb=={xraydb}"], check=True)
        if not (engine / wheel).is_file():
            raise SystemExit(f"Expected {wheel} from pip download.")
    info: dict[str, object] = {
        "version": __version__, "pyodide_version": PYODIDE_VERSION, "pyodide_url": PYODIDE_URL,
        "pyodide_packages": PYODIDE_PACKAGES, "xraydb_wheel": wheel, "engine_zip": engine_archive(engine),
    }
    (engine / "build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SITE, help="site directory (default: _site)")
    args = parser.parse_args()
    info = build(args.output)
    print(f"Built browser version {info['version']} in {args.output}")


if __name__ == "__main__":
    main()
