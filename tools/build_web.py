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
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xray_workbench import __version__  # noqa: E402

PYODIDE_VERSION = "314.0.7"
#: The official release, and the SHA-256 GitHub publishes for it. The site serves its own
#: copy of what it needs, because the jsDelivr CDN was seen to stall and refuse the SciPy wheel.
PYODIDE_RELEASE = f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/pyodide-{PYODIDE_VERSION}.tar.bz2"
PYODIDE_SHA256 = "192b5864e6e6d30ab074861af800cb8b4acb0998ef0f9342c3367448aeb86645"
PYODIDE_CORE = ("pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json")
PYODIDE_CACHE = ROOT / ".pyodide-cache"
PYODIDE_PACKAGES = ["numpy", "scipy", "sqlalchemy", "typing-extensions", "platformdirs"]
SITE = ROOT / "_site"
#: Exact-match text changes, so an edit to index.html cannot silently break the build.
PAGE_EDITS = (
    ('content="An offline scientific calculator for', 'content="A browser-based scientific calculator for'),
    ("Local workspace", "Runs in your browser"),
    ('<script src="/static/vendor/', '<link rel="stylesheet" href="/static/web/web.css?v=WEB_STAMP">\n'
                                     '  <script src="/static/web/shim.js?v=WEB_STAMP"></script>\n'
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
    # The worker takes the same stamp from shim.js, so a browser cannot pair an old engine script with a new build.
    stamp = hashlib.sha256(b"".join(path.read_bytes() for path in sorted((ROOT / "web").iterdir()))).hexdigest()[:12]
    return text.replace("WEB_STAMP", stamp).replace('"/static/', '"static/')


def engine_archive(destination: Path) -> str:
    """Zip the package's Python sources and name the file after its content."""
    temporary = destination / "engine.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "xray_workbench").rglob("*")):
            if path.suffix in (".py", ".json"):  # .json: the materials library
                archive.write(path, path.relative_to(ROOT).as_posix())
    name = f"engine-{hashlib.sha256(temporary.read_bytes()).hexdigest()[:12]}.zip"
    temporary.rename(destination / name)
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pyodide_files(cache: Path = PYODIDE_CACHE, archive: Path | None = None) -> Path:
    """Unpack the runtime and every wheel the engine needs into ``cache``, once, checked against the pin."""
    unpacked = cache / PYODIDE_VERSION
    if (unpacked / PYODIDE_CORE[-1]).is_file():
        return unpacked
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = archive or cache / "release.tar.bz2"
    if archive is None:
        with urlopen(PYODIDE_RELEASE, timeout=120) as response, downloaded.open("wb") as out:  # fixed HTTPS URL
            shutil.copyfileobj(response, out, 1 << 20)
    if sha256(downloaded) != PYODIDE_SHA256:
        raise SystemExit(f"{PYODIDE_RELEASE} does not match the pinned SHA-256.")
    partial = cache / "partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    prefixes = tuple(f"{name.replace('-', '_')}-" for name in PYODIDE_PACKAGES) + tuple(f"{name}-" for name in PYODIDE_PACKAGES)
    with tarfile.open(downloaded, "r:bz2") as release:
        for member in release:
            name = PurePosixPath(member.name).name
            if member.isfile() and (name in PYODIDE_CORE or (name.startswith(prefixes) and name.endswith(".whl"))):
                source = release.extractfile(member)
                assert source is not None  # a regular file, checked above
                with source, (partial / name).open("wb") as out:
                    shutil.copyfileobj(source, out, 1 << 20)
    if archive is None:
        downloaded.unlink()
    partial.rename(unpacked)
    return unpacked


def vendor_pyodide(destination: Path, cache: Path = PYODIDE_CACHE) -> None:
    """Copy the runtime and packages into the site, verifying each wheel against Pyodide's lock file."""
    source = pyodide_files(cache)
    destination.mkdir()
    for name in PYODIDE_CORE:
        shutil.copy2(source / name, destination / name)
    lock = json.loads((source / PYODIDE_CORE[-1]).read_text(encoding="utf-8"))["packages"]
    needed: list[str] = []
    pending = [name.lower() for name in PYODIDE_PACKAGES]
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.append(name)
            pending.extend(depend.lower() for depend in lock[name]["depends"])
    for name in needed:
        wheel = source / lock[name]["file_name"]
        if not wheel.is_file() or sha256(wheel) != lock[name]["sha256"]:
            raise SystemExit(f"{wheel.name} is missing or does not match Pyodide's lock file.")
        shutil.copy2(wheel, destination / wheel.name)


def build(site: Path = SITE, *, fetch_wheel: bool = True, cache: Path = PYODIDE_CACHE) -> dict[str, object]:
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
        vendor_pyodide(site / "pyodide", cache)
    info: dict[str, object] = {
        "version": __version__, "pyodide_version": PYODIDE_VERSION, "pyodide_url": "../pyodide/",
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
