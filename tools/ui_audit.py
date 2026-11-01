"""Run the interface audit (tools/ui_audit.js) against the real application.

    python tools/ui_audit.py                       # audit, fail on any violation
    python tools/ui_audit.py --report ui-audit.json --report-only

Starts the workbench on a free loopback port, loads it in headless Chromium at
desktop, tablet and phone sizes, audits every analysis tab and the materials library
in the light and dark themes, and writes a JSON report. Tab navigation is exercised with
the keyboard. CI runs this on every push. These automated checks cover only
the criteria listed in ui_audit.js, not a complete WCAG conformance audit.

Requires Playwright: ``pip install playwright && python -m playwright install chromium``.
It is a maintainer and CI tool; the workbench itself never needs a browser engine.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT = (ROOT / "tools" / "ui_audit.js").read_text(encoding="utf-8")
VIEWPORTS = {"desktop": (1440, 900), "tablet": (768, 1024), "phone": (375, 812)}
ANALYSIS_VIEWS = ("materials", "spectrum", "uncertainty", "scattering", "tools", "methods")


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


def _record_error(errors: list[str], exc: Any) -> None:
    errors.append(str(exc))


def audit(url: str, screenshots: Path | None = None) -> dict[str, Any]:
    from playwright.sync_api import Page, sync_playwright

    report: dict[str, Any] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, (width, height) in VIEWPORTS.items():
            page = browser.new_page(viewport={"width": width, "height": height})
            errors: list[str] = []

            def capture(view: str, screenshot_page: Page = page, viewport_name: str = name) -> None:
                if screenshots is not None:
                    screenshot_page.screenshot(path=str(screenshots / f"{viewport_name}-{view}.png"))

            page.on("pageerror", partial(_record_error, errors))
            page.goto(url, wait_until="networkidle")
            # Wait for every default plot and the successful calculation state,
            # rather than hoping a fixed delay covers slower CI machines.
            page.wait_for_function("""() => document.querySelectorAll('.scientific-plot.js-plotly-plot').length === 11 &&
                document.querySelector('#result-status')?.textContent.includes('up to date')""", timeout=30_000)
            views: dict[str, Any] = {"main": page.evaluate(AUDIT_SCRIPT)}
            capture("main")
            reference = page.locator(".plot-reference-details").first
            reference.locator("summary").focus()
            page.keyboard.press("Enter")
            reference.locator("table").wait_for(state="visible")
            views["reference_values"] = page.evaluate(AUDIT_SCRIPT)
            views["reference_values"]["violations"]["reference_keyboard_toggle"] = int(reference.get_attribute("open") is None)
            before = page.locator("#plot-summary-1").inner_text()
            page.locator('[data-field="thickness_mm"]').first.fill("2")
            reference.locator(".table-wrap").focus()
            page.wait_for_function("""before => document.querySelector('#plot-summary-1')?.textContent !== before &&
                document.querySelector('#result-status')?.textContent.includes('up to date')""", arg=before)
            views["reference_values"]["violations"]["reference_live_update_focus"] = int(
                not reference.locator(".table-wrap").evaluate("el => el === document.activeElement") or
                reference.get_attribute("open") is None)
            capture("reference-values")
            reference.locator("summary").focus()
            page.keyboard.press("Enter")
            reference.locator("table").wait_for(state="hidden")
            page.locator('[data-field="thickness_mm"]').first.fill("1")
            page.wait_for_function("""before => document.querySelector('#plot-summary-1')?.textContent === before &&
                document.querySelector('#result-status')?.textContent.includes('up to date')""", arg=before)
            page.locator("#tab-overview").focus()
            for view in ANALYSIS_VIEWS:
                page.keyboard.press("ArrowRight")
                page.locator(f"#view-{view}").wait_for(state="visible")
                if view == "scattering":
                    page.wait_for_function("document.querySelector('#ts-status').textContent.includes('keV')", timeout=60_000)
                if view == "tools":
                    page.wait_for_function("document.querySelector('#st-output').dataset.tool === 'optics'", timeout=60_000)
                result = page.evaluate(AUDIT_SCRIPT)
                tab = page.locator(f"#tab-{view}")
                result["violations"]["tab_keyboard_navigation"] = int(
                    tab.get_attribute("aria-selected") != "true" or not tab.evaluate("el => el === document.activeElement"))
                views[view] = result
                capture(view)
            page.keyboard.press("Home")
            page.locator("#view-overview").wait_for(state="visible")
            # Dark theme: the toggle cycles Auto, Light, Dark, and this browser reports a light preference.
            page.locator("#theme-toggle").click()
            page.locator("#theme-toggle").click()
            page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
            page.wait_for_timeout(500)  # let the 160 ms colour transitions finish before measuring
            for view in ("overview", *ANALYSIS_VIEWS):
                page.locator(f"#tab-{view}").click()
                page.locator(f"#view-{view}").wait_for(state="visible")
                views[f"dark_{view}"] = page.evaluate(AUDIT_SCRIPT)
                capture(f"dark-{view}")
            # Populated tools, not just empty panels. Exercise both themes and every new operation.
            for theme in ("dark", "light"):
                if theme == "light":
                    page.locator("#theme-toggle").click()
                    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
                    page.wait_for_timeout(250)
                page.locator("#tab-tools").click()
                for tool in ("optics", "reflectivity", "contrast", "converter", "powder", "pdf"):
                    page.locator("#st-tool").select_option(tool)
                    if tool == "pdf":
                        page.locator("#st-demo-pdf").wait_for(state="visible")
                        page.locator("#st-demo-pdf").click()
                    page.locator("#st-run").click()
                    page.wait_for_function("tool => document.querySelector('#st-output').dataset.tool === tool", arg=tool, timeout=60_000)
                    views[f"{theme}_tool_{tool}"] = page.evaluate(AUDIT_SCRIPT)
                page.locator("#tab-materials").click()
                page.locator("#batch-run").evaluate("el => el.closest('details').open = true")
                page.locator("#batch-csv").fill("name,basis,density_g_cm3,thickness_mm,composition\ns,mole,2.2,.135,SiO2:1\nbad,mass,2,1,Xx:1")
                page.locator("#batch-run").click()
                page.wait_for_function("document.querySelector('#batch-status').textContent.startsWith('1 of 2')")
                views[f"{theme}_batch_results"] = page.evaluate(AUDIT_SCRIPT)
                for button, dialog in (("share-project", "share-dialog"), ("cite-this", "citation-dialog")):
                    page.locator(f"#{button}").click()
                    page.locator(f"#{dialog}[open]").wait_for()
                    views[f"{theme}_{dialog}"] = page.evaluate(AUDIT_SCRIPT)
                    page.keyboard.press("Escape")
            # Restore the dark state expected by the existing library audit.
            page.locator("#theme-toggle").click()
            page.locator("#theme-toggle").click()
            page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
            page.wait_for_timeout(250)
            # Then the materials library, in the theme the page is showing.
            page.locator("#tab-overview").click()
            page.click("#open-materials-library")
            page.wait_for_selector("#materials-dialog[open] .ml-item", timeout=30_000)
            views["dark_materials_library"] = page.evaluate(AUDIT_SCRIPT)
            capture("dark-materials-library")
            page.locator("#materials-dialog .ml-item").first.click()
            page.wait_for_function("""() => document.querySelector('#ml-curve-summary')?.textContent.startsWith('At 60 keV:')""",
                                   timeout=60_000)
            views["dark_material_detail"] = page.evaluate(AUDIT_SCRIPT)
            page.keyboard.press("Escape")
            page.locator("#theme-toggle").click()
            page.wait_for_function("document.documentElement.dataset.theme === 'light'")
            page.wait_for_timeout(500)
            page.click("#open-materials-library")
            page.wait_for_selector("#materials-dialog[open] .ml-item", timeout=30_000)
            views["materials_library"] = page.evaluate(AUDIT_SCRIPT)
            capture("materials-library")
            # The former audit stopped at the empty library panel, missing
            # composition tables, provenance and the material's plotted curve.
            page.locator("#materials-dialog .ml-item").first.click()
            page.wait_for_function("""() => document.querySelector('#ml-curve-summary')?.textContent.startsWith('At 60 keV:')""",
                                   timeout=60_000)
            views["material_detail"] = page.evaluate(AUDIT_SCRIPT)
            capture("material-detail")
            views["page_errors"] = errors
            report[name] = views
            page.close()
        browser.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=ROOT / "ui-audit-report.json")
    parser.add_argument("--report-only", action="store_true", help="write the report but never fail")
    parser.add_argument("--screenshots-dir", type=Path, help="save viewport screenshots for visual review")
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.screenshots_dir is not None:
        args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    # Scientific units in descriptions must not crash a legacy Windows console.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="backslashreplace")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, str(ROOT / "launch.py"), "--no-browser", "--port", str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        _wait_for(url)
        report = audit(url, args.screenshots_dir)
    finally:
        server.terminate()
        server.wait(timeout=20)

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = 0
    for viewport, views in report.items():
        for view, result in views.items():
            if view == "page_errors":
                continue
            count = _count(result)
            total += count
            flags = " horizontal overflow!" if result["stats"]["horizontal_overflow"] else ""
            print(f"{viewport:<8} {view:<18} {count:>3} violation(s){flags}")
            for kind, items in result["violations"].items():
                details = items if isinstance(items, list) else ([f"{items}"] if items else [])
                for item in details:
                    print(f"           - {kind}: {item}")
            unmeasured = len(result["stats"]["unmeasured_text_contrast"])
            svg = result["stats"]["svg_text_not_audited"]
            print(f"           Coverage limits: {unmeasured} HTML text contrast and {svg} SVG text elements need visual review.")
        for error in views["page_errors"]:
            print(f"{viewport:<8} page error: {error}")
        total += len(views["page_errors"])
    print(f"\nReport: {args.report}")
    return 0 if args.report_only or total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
