"""Compare the browser version with the native engine, number by number.

    python tools/build_web.py
    python tools/web_parity.py

Needs Playwright and network access for the Pyodide runtime. A different maths library
is expected to change the last few bits of some results, so this reports how many
numbers differ and by how much, and fails above TOLERANCE.
"""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xray_workbench.physics import calculate, presets  # noqa: E402

#: Largest relative difference accepted. Measured maximum when written: 5.7e-14.
TOLERANCE = 1e-12
#: Below this a transmission is underflowing, and its relative difference means nothing.
UNDERFLOW = 1e-290
#: Provenance records the version, timestamp and runtime, not calculated numbers.
NOT_COMPARED = frozenset({"provenance", "configuration", "schema_version"})
ENERGY = {"min_keV": 1, "max_keV": 800, "points": 1500, "spacing": "log", "reference_keV": 14}
#: The measured glasses of arXiv:2608.15675: mol%, g/cm³, mm.
GLASSES = {
    "v-SiO2": ({"SiO2": 100.0}, 2.198, 0.1368),
    "NCS-80": ({"SiO2": 81.00, "Na2O": 9.54, "CaO": 9.46}, 2.454, 0.1366),
    "NAS-60": ({"SiO2": 60.62, "Na2O": 20.59, "Al2O3": 18.80}, 2.415, 0.130),
    "CAS-N": ({"SiO2": 32.86, "CaO": 31.52, "Al2O3": 15.73, "AlN": 19.89}, 2.840, 0.1376),
}


def configurations() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for name, (composition, density, thickness) in GLASSES.items():
        configs[name] = {
            "energy": ENERGY,
            "layers": [{"name": name, "basis": "mole", "density_g_cm3": density, "thickness_mm": thickness,
                        "components": [{"formula": f, "fraction": x} for f, x in composition.items()],
                        "density_uncertainty_pct": 1, "thickness_uncertainty_pct": 1}],
            "uncertainty": {"enabled": True, "samples": 2000, "seed": 2608},
            "spectrum": {"energy_keV": [10, 14, 20, 40], "weights": [1, 4, 2, 1], "weighting": "energy"},
        }
    for preset in presets():
        configs[preset["id"]] = {"energy": ENERGY, "layers": [preset["layer"]]}
    configs["stack"] = {"energy": ENERGY, "layers": [preset["layer"] for preset in presets()[:4]]}
    example = json.loads((ROOT / "examples" / "borosilicate-multilayer.json").read_text(encoding="utf-8"))
    configs["example"] = example["configuration"]
    return configs


def differences(browser: Any, native: Any, path: str = "") -> Iterator[tuple[str, float | None]]:
    """Yield ``(path, relative difference)`` per unequal number, ``None`` for a structural mismatch."""
    if isinstance(browser, dict) and isinstance(native, dict):
        for key in (browser.keys() | native.keys()) - NOT_COMPARED:
            if key not in browser or key not in native:
                yield f"{path}.{key}", None
            else:
                yield from differences(browser[key], native[key], f"{path}.{key}")
    elif isinstance(browser, list) and isinstance(native, list):
        if len(browser) != len(native):
            yield path, None
            return
        for index, (left, right) in enumerate(zip(browser, native, strict=True)):
            yield from differences(left, right, f"{path}[{index}]")
    elif isinstance(browser, int | float) and isinstance(native, int | float):
        if browser != native:
            largest = max(abs(browser), abs(native))
            yield path, 0.0 if largest < UNDERFLOW else abs(browser - native) / largest
    elif browser != native:
        yield path, None


def count_numbers(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_numbers(item) for key, item in value.items() if key not in NOT_COMPARED)
    if isinstance(value, list):
        return sum(count_numbers(item) for item in value)
    return int(isinstance(value, int | float) and not isinstance(value, bool))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> int:
    from playwright.sync_api import sync_playwright

    site = ROOT / "_site"
    if not (site / "index.html").is_file():
        raise SystemExit("Build the site first: python tools/build_web.py")
    handler = functools.partial(_QuietHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    page_errors: list[str] = []
    total, unequal, worst, worst_at, structural = 0, 0, 0.0, "", []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        page.evaluate("AttenuationWeb.ready", None)
        print(f"Engine loaded in {page.evaluate('AttenuationWeb.loadSeconds'):.1f} s "
              f"({page.evaluate('AttenuationWeb.runtime')})")
        page.wait_for_selector(".js-plotly-plot", timeout=60_000)
        for name, config in configurations().items():
            answer = page.evaluate("text => AttenuationWeb.request('calculate', text)", json.dumps(config))
            if answer["status"] != 200:
                structural.append(f"{name}: status {answer['status']}: {answer['text'][:200]}")
                continue
            native = calculate(config)
            total += count_numbers(native)
            for path, difference in differences(json.loads(answer["text"]), native, name):
                if difference is None:
                    structural.append(path)
                else:
                    unequal += 1
                    if difference > worst:
                        worst, worst_at = difference, path
        browser.close()
    server.shutdown()
    print(f"{total} numbers compared: {total - unequal} identical, {unequal} differ.")
    print(f"Largest relative difference: {worst:.3g} at {worst_at or 'n/a'} (limit {TOLERANCE:g}).")
    for problem in structural + page_errors:
        print(f"PROBLEM: {problem}")
    return 1 if structural or page_errors or worst > TOLERANCE or total == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
