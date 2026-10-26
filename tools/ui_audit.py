"""Run the interface audit (tools/ui_audit.js) against the real application.

    python tools/ui_audit.py                       # audit, fail on any violation
    python tools/ui_audit.py --report ui-audit.json --report-only

Starts the workbench on a free loopback port, loads it in headless Chromium at
desktop, tablet and phone sizes, audits the main page and the open materials
library dialog, and writes a JSON report. CI runs this on every push.

Requires Playwright: ``pip install playwright && python -m playwright install chromium``.
It is a maintainer and CI tool; the workbench itself never needs a browser engine.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT = (ROOT / "tools" / "ui_audit.js").read_text(encoding="utf-8")
VIEWPORTS = {"desktop": (1440, 900), "tablet": (768, 1024), "phone": (375, 812)}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for(url: str, seconds: float = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/api/health", timeout=2):  # fixed loopback URL
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"Workbench did not start at {url}")


def _count(result: dict[str, Any]) -> int:
    found = sum(len(v) if isinstance(v, list) else int(v) for v in result["violations"].values())
    return found + int(bool(result["stats"]["horizontal_overflow"]))


def audit(url: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    report: dict[str, Any] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, (width, height) in VIEWPORTS.items():
            page = browser.new_page(viewport={"width": width, "height": height})
            errors: list[str] = []
            page.on("pageerror", lambda exc, errors=errors: errors.append(str(exc)))
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector(".js-plotly-plot", timeout=30_000)
            page.wait_for_timeout(1500)  # live recalculation settles
            main = page.evaluate(AUDIT_SCRIPT)
            page.click("#open-materials-library")
            page.wait_for_selector("#materials-dialog[open] .ml-item", timeout=30_000)
            library = page.evaluate(AUDIT_SCRIPT)
            report[name] = {"main": main, "materials_library": library, "page_errors": errors}
            page.close()
        browser.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=ROOT / "ui-audit-report.json")
    parser.add_argument("--report-only", action="store_true", help="write the report but never fail")
    args = parser.parse_args(argv)

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, str(ROOT / "launch.py"), "--no-browser", "--port", str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        _wait_for(url)
        report = audit(url)
    finally:
        server.terminate()
        server.wait(timeout=20)

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = 0
    for viewport, views in report.items():
        for view in ("main", "materials_library"):
            result = views[view]
            count = _count(result)
            total += count + len(views["page_errors"]) * (view == "main")
            flags = " horizontal overflow!" if result["stats"]["horizontal_overflow"] else ""
            print(f"{viewport:<8} {view:<18} {count:>3} violation(s){flags}")
            for kind, items in result["violations"].items():
                details = items if isinstance(items, list) else ([f"{items}"] if items else [])
                for item in details:
                    print(f"           - {kind}: {item}")
        for error in views["page_errors"]:
            print(f"{viewport:<8} page error: {error}")
    print(f"\nReport: {args.report}")
    return 0 if args.report_only or total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
