"""Start a local workbench, or calculate a project file without the GUI.

Reachable as ``xray-workbench`` once installed and as ``python launch.py``
inside a checkout. Batch mode deliberately avoids importing the web stack.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from . import MODEL, __version__

DEFAULT_PORT = 8765
MIN_PORT, MAX_PORT = 1024, 65535


def _existing_workbench(url: str) -> bool:
    """Recognize our own loopback server before opening another session.

    Matching version and model as well as status avoids adopting an unrelated
    service, or an older workbench whose results would not be comparable.
    """
    try:
        with urlopen(f"{url}/api/health", timeout=1) as response:
            health = json.loads(response.read(4096))
    except (URLError, OSError, ValueError):
        return False
    return (isinstance(health, dict) and health.get("status") == "ok"
            and health.get("version") == __version__
            and health.get("model") == MODEL)


def _load_configuration(path: Path) -> dict[str, Any]:
    """Read a saved project or a bare configuration object.

    ``utf-8-sig`` tolerates the byte-order mark that Windows editors add.
    Non-finite JSON constants are rejected rather than silently becoming NaN.
    """

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"Non-finite JSON number: {value}")

    document = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise ValueError("Project or configuration must be a JSON object.")
    configuration = document.get("configuration", document)
    if not isinstance(configuration, dict):
        raise ValueError("Project 'configuration' must be a JSON object.")
    return configuration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xray-workbench", description="X-ray Attenuation Workbench")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__} ({MODEL})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port for the graphical application (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--calculate", type=Path, metavar="PROJECT.json",
                        help="calculate a saved project without starting a server")
    parser.add_argument("--output", type=Path, help="write full result JSON (otherwise stdout)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output and not args.calculate:
        parser.error("--output requires --calculate")

    if args.calculate:
        from .results import run_calculation
        try:
            result = run_calculation(_load_configuration(args.calculate))
            output = json.dumps(result, indent=2, allow_nan=False)
        except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
            parser.exit(2, f"Calculation failed: {exc}\n")
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return

    if not MIN_PORT <= args.port <= MAX_PORT:
        parser.error(f"port must be between {MIN_PORT} and {MAX_PORT}")
    import uvicorn
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser and _existing_workbench(url):
        print(f"Opening the running X-ray Attenuation Workbench: {url}")
        webbrowser.open(url)
        return
    print(f"X-ray Attenuation Workbench {__version__}: {url}\nPress Ctrl+C to stop.")
    if not args.no_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run("xray_workbench.server:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
