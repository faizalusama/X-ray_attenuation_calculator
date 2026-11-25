"""Audit real workspace interactions in Chromium, alongside tools/ui_audit.py.

Fresh sessions cover light/dark at 1440, 768, 375 and 320 px, populated results,
keyboard navigation, collapse/reopen, downloads and saved-layout round trips.
This is an interaction/overflow audit, not a complete accessibility assessment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from ui_audit import _free_port, _record_error, _wait_for  # noqa: E402

VIEWPORTS = {"desktop": (1440, 900), "tablet": (768, 1024), "phone": (375, 812), "small_phone": (320, 812)}
VIEWS = ("overview", "materials", "spectrum", "uncertainty", "scattering", "tools", "methods")
PLOTS = "#plot-grid .scientific-plot.js-plotly-plot"
READY = """() => document.querySelector('#result-status')?.textContent.includes('up to date') &&
    document.querySelectorAll('#plot-grid .scientific-plot.js-plotly-plot').length > 0"""
POPULATED = """() => [...document.querySelectorAll('#plot-grid .scientific-plot')].every(
    plot => plot.data?.some(trace => Array.from(trace.z || trace.y || []).flat().some(Number.isFinite)))"""


def record(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def failure_count(report: dict[str, Any]) -> int:
    return sum(sum(not check["passed"] for check in scenario["checks"]) +
               len(scenario["page_errors"]) for scenario in report["scenarios"].values())


def overflow(page: Any, checks: list[dict[str, Any]], name: str) -> None:
    measurement = page.evaluate("""() => {
        const dialog = document.querySelector('dialog[open]');
        const root = document.documentElement, width = root.clientWidth;
        const nodes = dialog ? [dialog, ...dialog.querySelectorAll('*')] : [];
        return {viewport: width, document: root.scrollWidth,
            dialog: dialog ? {client: dialog.clientWidth, scroll: dialog.scrollWidth} : null,
            outside: nodes.filter(el => {
                const r = el.getBoundingClientRect();
                return r.width && r.height && (r.left < -1 || r.right > width + 1);
            }).slice(0, 8).map(el => el.id || el.tagName)};
    }""")
    dialog = measurement["dialog"]
    passed = measurement["document"] <= measurement["viewport"] + 1
    passed = passed and (dialog is None or dialog["scroll"] <= dialog["client"] + 1)
    record(checks, f"overflow:{name}", passed and not measurement["outside"], measurement)


def download_json(page: Any, selector: str) -> dict[str, Any]:
    # Inspect the actual HTTP attachment without replacing the download function.
    with page.expect_download() as pending:
        page.locator(selector).click()
    download = pending.value
    path = download.path()
    if path is None:
        raise AssertionError(f"No downloaded file for {selector}: {download.failure()}")
    value: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return value


def persistent_project(project: dict[str, Any]) -> dict[str, Any]:
    """Compare user settings; restore intentionally renumbers DOM IDs and render revision."""
    saved: dict[str, Any] = json.loads(json.dumps(project))
    dashboard = saved.get("ui", {}).get("dashboard", {})
    dashboard.pop("revision", None)
    for panel in dashboard.get("panels", []):
        panel.pop("id", None)
    return saved


def inputs(page: Any) -> list[dict[str, Any]]:
    return list(page.locator("#config-form input, #config-form select, #view-spectrum input, "
                             "#view-spectrum textarea, #view-spectrum select, #view-uncertainty input").evaluate_all(
        """els => els.map(el => ({id: el.id, field: el.dataset.field, layer: el.dataset.layer,
            value: el.value, checked: el.checked, disabled: el.disabled}))"""))


def ready(page: Any) -> None:
    page.wait_for_function(READY, timeout=60_000)
    page.wait_for_function(POPULATED, timeout=30_000)


def tab_check(page: Any, checks: list[dict[str, Any]], view: str, key: str) -> None:
    page.keyboard.press(key)
    page.locator(f"#view-{view}").wait_for(state="visible")
    selected = page.locator(f"#tab-{view}")
    record(checks, f"keyboard:{key}:{view}", selected.get_attribute("aria-selected") == "true" and
           selected.get_attribute("tabindex") == "0" and selected.evaluate("el => el === document.activeElement"))
    record(checks, f"single-tab:{view}", page.locator('[role="tab"][aria-selected="true"]').count() == 1 and
           page.locator('[role="tab"][tabindex="0"]').count() == 1)


def exercise_technique_guide(page: Any, checks: list[dict[str, Any]]) -> None:
    original_url = page.url
    page.locator("#open-technique-guide").press("Enter")
    guide = page.locator("#technique-guide")
    record(checks, "guide-entry-focus", guide.locator(":scope > summary").evaluate(
        "el => el === document.activeElement"))
    cards = guide.locator("[data-guide-family]")
    topics = guide.locator("[data-guide-stage]")
    total = topics.count()
    record(checks, "guide-catalogs", cards.count() == 6 and total >= 61, total)
    for family in cards.all():
        diagram = family.locator(".guide-diagram")
        diagram.locator("summary").press("Enter")
        before = diagram.locator("svg").get_attribute("aria-label")
        choice = diagram.locator("button").last
        choice.press("Enter")
        record(checks, f"guide-geometry:{family.get_attribute('data-guide-family')}",
               choice.get_attribute("aria-pressed") == "true" and
               diagram.locator("svg").get_attribute("aria-label") != before)
        diagram.locator("summary").press("Enter")
    volume = guide.locator(".guide-volume")
    volume.locator("summary").press("Enter")
    detector = volume.locator(".gv-detector")
    original_face = detector.get_attribute("points")
    original_scene = volume.locator("svg").inner_html()
    volume.locator(".gv-pose").press("ArrowDown")
    record(checks, "guide-rotate-sample", volume.locator("svg").inner_html() != original_scene and
           detector.get_attribute("points") == original_face)
    volume.locator(".gv-camera").press("ArrowDown")
    record(checks, "guide-camera", detector.get_attribute("points") != original_face and
           "Overhead" in volume.locator(".gv-state").inner_text())
    overflow(page, checks, "guide:3d")
    volume.locator("summary").press("Enter")
    guide.locator("#guide-family").select_option("saxs")
    guide.locator("#guide-stage").select_option("Interpret")
    visible = guide.locator("[data-guide-stage]:visible")
    record(checks, "guide-combined-filters", 0 < visible.count() < total and
           visible.evaluate_all("rows => rows.every(r => r.dataset.guideStage === 'Interpret' && "
                                "r.closest('[data-guide-family]').dataset.guideFamily === 'saxs')"))
    headings = guide.locator(".guide-workflow-stage h4:visible")
    record(checks, "guide-stage-heading-follows-filter", headings.all_text_contents() == ["Interpret results"])
    guide.locator("#guide-search").fill("no-such-technique-topic")
    record(checks, "guide-empty-state", guide.locator("#guide-empty").is_visible() and visible.count() == 0)
    guide.locator("#guide-reset").press("Enter")
    record(checks, "guide-reset", visible.count() == total and
           guide.locator("#guide-search").evaluate("el => el === document.activeElement"))
    link = guide.locator("#guide-permalink")
    family = guide.locator("#guide-family")
    family.select_option("saxs")
    record(checks, "guide-shareable-family", link.get_attribute("href") == "#technique-guide/saxs")
    link.press("Enter")
    page.wait_for_function("location.hash === '#technique-guide/saxs'")
    family.select_option("tomo")
    link.press("Enter")
    page.wait_for_function("location.hash === '#technique-guide/tomo'")
    page.go_back()
    page.wait_for_function("document.querySelector('#guide-family').value === 'saxs'")
    page.go_forward()
    page.wait_for_function("document.querySelector('#guide-family').value === 'tomo'")
    record(checks, "guide-link-history-and-focus", guide.locator("[data-guide-family]:visible").count() == 1 and
           guide.locator(":scope > summary").evaluate("el => el === document.activeElement"))
    page.evaluate("location.hash = '#technique-guide/unknown'")
    page.wait_for_function("document.querySelector('#guide-family').value === ''")
    record(checks, "guide-unknown-family-fallback", guide.locator("[data-guide-family]:visible").count() == 6)
    overflow(page, checks, "guide:expanded")
    jump = guide.locator("button[data-guide-target^='tool:']").first
    jump.locator("xpath=ancestor::details[1]/summary").press("Enter")
    target = str(jump.get_attribute("data-guide-target")).split(":", 1)[1]
    jump.press("Enter")
    record(checks, "guide-tool-navigation", page.locator("#st-tool").input_value() == target and
           page.locator("#st-tool").evaluate("el => el === document.activeElement"))
    guide.locator(":scope > summary").press("Enter")
    # The layout reload later in this scenario starts from its original address.
    page.evaluate("(url) => history.replaceState(history.state, '', url)", original_url)


def exercise_live_probes(page: Any, checks: list[dict[str, Any]], capture: Callable[[str], None]) -> None:
    tool = page.locator("#st-tool")
    tool.select_option("guinier")
    guinier = page.locator("#guinier-observation")
    guinier.wait_for(state="visible")
    guinier.press("End")
    fit_readout = page.locator("#st-output .technique-probe [role=status]")
    last = fit_readout.inner_text()
    record(checks, "guinier-probe-end", guinier.input_value() == guinier.get_attribute("max") and bool(last))
    guinier.press("Home")
    record(checks, "guinier-probe-home", guinier.input_value() == guinier.get_attribute("min") and
           fit_readout.inner_text() != last and guinier.evaluate("el => el === document.activeElement"))
    intensity = page.locator('[data-st-tool=guinier] [data-arg=intensity_data]')
    imported = intensity.input_value().replace("0.992", "0.994")
    chooser = page.locator("#st-guinier-file")
    with page.expect_response(lambda r: r.url.endswith("/api/scattering-tools") and r.ok) as from_file:
        chooser.set_input_files({"name": "guinier.csv", "mimeType": "text/csv", "buffer": imported.encode()})
    record(checks, "guinier-file-import", intensity.input_value() == imported and chooser.input_value() == "")
    with page.expect_response(lambda r: r.url.endswith("/api/scattering-tools") and r.ok) as from_paste:
        intensity.evaluate("(el, text) => { el.value = text; el.dispatchEvent(new Event('input', {bubbles:true})); }",
                           imported)
    record(checks, "guinier-import-matches-paste", from_file.value.json() == from_paste.value.json())
    guinier.wait_for(state="visible")
    capture("guinier-probe")
    overflow(page, checks, "guinier-probe")
    maximum = page.locator('[data-st-tool=guinier] [data-arg=qmax_inv_angstrom]')
    old_maximum = maximum.input_value()
    maximum.fill("0.025")
    page.wait_for_function("() => document.querySelector('#st-status').textContent.includes('at least three')")
    record(checks, "guinier-probe-clears-after-error", guinier.count() == 0 and
           page.locator("#st-export").is_disabled())
    maximum.fill(old_maximum)
    guinier.wait_for(state="visible")

    tool.select_option("tomo_sampling")
    view = page.locator("#tomo-view-index")
    view.wait_for(state="visible")
    schedule = page.locator("#tomo-view-schedule")
    schedule.select_option(index=1)
    view.press("End")
    view_readout = page.locator("#st-output .technique-probe [role=status]")
    last = view_readout.inner_text()
    record(checks, "tomo-view-end", view.input_value() == view.get_attribute("max") and bool(last))
    view.press("Home")
    record(checks, "tomo-view-home", view.input_value() == view.get_attribute("min") and
           view_readout.inner_text() != last and view.evaluate("el => el === document.activeElement"))
    capture("tomography-view-probe")
    overflow(page, checks, "tomography-view-probe")
    page.locator('[data-st-tool=tomo_sampling] [data-arg=planned_views]').fill("")
    page.wait_for_function("() => document.querySelector('#st-output')?.dataset.tool === 'tomo_sampling'")
    record(checks, "tomo-view-optional-plan", view.is_visible() and schedule.locator("option").count() <= 1)


def exercise(page: Any, checks: list[dict[str, Any]], capture: Callable[[str], None]) -> None:
    ready(page)
    count = page.locator(PLOTS).count()
    record(checks, "fresh-default-four", count == 4, count)
    record(checks, "no-container-pressed-state", page.locator("html[aria-pressed], #plot-grid[aria-pressed]").count() == 0)
    record(checks, "fresh-simple-mode",
           page.locator('button[data-workspace-mode="simple"]').get_attribute("aria-pressed") == "true")
    record(checks, "simple-hides-inactive-advanced",
           not page.locator('[data-field="angle_deg"]').first.is_visible() and
           not page.locator("#energy-points").is_visible())
    capture("fresh")
    overflow(page, checks, "fresh")
    groups = (("attenuation", "Attenuation", VIEWS[:4]), ("scattering", "Scattering", VIEWS[4:6]),
              ("reference", "Reference", VIEWS[6:]))
    record(checks, "one-tablist", page.locator('[role="tablist"]').count() == 1)
    for group, label, views in groups:
        element = page.locator(f"#{group}-tab-group")
        record(checks, f"group:{group}", element.is_visible() and element.inner_text() == label and
               all(page.locator(f"#tab-{view}").get_attribute("aria-describedby") == f"{group}-tab-group"
                   for view in views))

    geometry = page.evaluate("""() => {
        const a = document.querySelector('#configuration-panel').getBoundingClientRect();
        const b = document.querySelector('#results-workspace').getBoundingClientRect();
        return {left: a.right <= b.left + 1, stacked: a.bottom <= b.top + 1, width: innerWidth};
    }""")
    record(checks, "configuration-placement", geometry["left"] if geometry["width"] > 900 else geometry["stacked"],
           geometry)
    page.locator('button[data-workspace-mode="advanced"]').click()
    record(checks, "advanced-disclosure", page.locator('[data-field="angle_deg"]').first.is_visible() and
           page.locator("#energy-points").is_visible())
    page.locator('[data-field="thickness_mm"]').first.fill("1.25")
    page.locator('[data-field="angle_deg"]').first.fill("17")
    page.locator("#energy-points").fill("321")
    page.locator("#energy-points").press("Tab")

    # Populate analysis panels through their actual controls.
    page.locator("#tab-spectrum").click()
    page.locator("#spectrum-data").fill("energy_keV,weight\n20,100\n40,200\n60,50")
    page.locator("#spectrum-enabled").check()
    page.locator("#tab-uncertainty").click()
    page.locator("#uncertainty-samples").fill("100")
    page.locator("#uncertainty-enabled").check()
    page.locator("#tab-overview").click()
    ready(page)
    baseline_results = download_json(page, "#export-json")
    config = baseline_results["configuration"]
    record(checks, "edited-inputs-reach-results", config["layers"][0]["thickness_mm"] == 1.25 and
           config["layers"][0]["angle_deg"] == 17 and config["energy"]["points"] == 321 and
           config["spectrum"] is not None and config["uncertainty"]["enabled"] and
           config["uncertainty"]["samples"] == 100, config)
    record(checks, "spectrum-and-uncertainty-results",
           bool(baseline_results.get("spectrum")) and bool(baseline_results.get("uncertainty")))
    before = inputs(page)
    record(checks, "project-matches-calculation", download_json(page, "#save-project")["configuration"] == config)
    for index, mode in enumerate(("simple", "advanced", "simple")):
        page.locator(f'button[data-workspace-mode="{mode}"]').click()
        record(checks, f"mode-preserves-inputs:{index}:{mode}", inputs(page) == before)
        record(checks, f"mode-state:{index}:{mode}",
               page.locator(f'button[data-workspace-mode="{mode}"]').get_attribute("aria-pressed") == "true")
    record(checks, "simple-active-settings-discoverable",
           page.locator('[data-field="angle_deg"]').first.is_visible() and page.locator("#energy-points").is_visible()
           and page.locator("#workspace-active-settings").is_visible())
    record(checks, "mode-preserves-configuration", download_json(page, "#save-project")["configuration"] == config)
    overflow(page, checks, "simple-active-settings")

    # A collapsed configuration must remain collapsed while crossing every group.
    page.locator("#close-inputs").click()
    page.locator("#tab-overview").focus()
    for view in VIEWS[1:]:
        tab_check(page, checks, view, "ArrowRight")
        record(checks, f"collapsed-across-view:{view}", not page.locator("#configuration-panel").is_visible() and
               page.locator("#workspace-inputs").get_attribute("aria-expanded") == "false")
        overflow(page, checks, f"{view}:collapsed-navigation")
    page.locator("#workspace-inputs").click()
    page.locator("#tab-methods").focus()
    tab_check(page, checks, "overview", "Home")

    # Hide and reopen from every view; test focus and both expanded states.
    for index, view in enumerate(VIEWS):
        if index:
            tab_check(page, checks, view, "ArrowRight")
        page.locator("#close-inputs").click()
        record(checks, f"collapse:{view}", not page.locator("#configuration-panel").is_visible() and
               page.locator("#workspace-inputs").get_attribute("aria-expanded") == "false" and
               page.locator("#workspace-inputs").evaluate("el => el === document.activeElement"))
        overflow(page, checks, f"{view}:collapsed")
        page.locator("#workspace-inputs").click()
        record(checks, f"reopen:{view}", page.locator("#configuration-panel").is_visible() and
               page.locator("#workspace-inputs").get_attribute("aria-expanded") == "true" and
               page.locator("#close-inputs").evaluate("el => el === document.activeElement"))
        if view == "materials":
            page.locator("#materials-results table").first.wait_for(state="visible")
        if view in ("spectrum", "uncertainty"):
            page.locator(f"#{view}-results .detail-metrics").wait_for(state="visible")
            if view == "spectrum":
                record(checks, "populated-spectrum-bins", page.locator("#spectrum-mini-chart svg").is_visible() and
                       "3 discrete bins" in page.locator("#spectrum-results").inner_text())
            else:
                record(checks, "populated-uncertainty-interval", page.locator("#uncertainty-results .interval-bar").is_visible())
        if view == "scattering":
            page.wait_for_function("() => document.querySelector('#ts-status').textContent.includes('keV')")
            page.locator("#ts-elements table").first.wait_for(state="visible")
        if view == "tools":
            page.wait_for_function("() => document.querySelector('#st-output').dataset.tool === 'optics'")
            page.locator("#st-output .js-plotly-plot").first.wait_for(state="visible")
        capture(view)
        overflow(page, checks, f"{view}:expanded")
        page.locator(f"#tab-{view}").focus()
    tab_check(page, checks, "overview", "ArrowRight")
    tab_check(page, checks, "methods", "ArrowLeft")
    tab_check(page, checks, "overview", "Home")
    tab_check(page, checks, "methods", "End")
    tab_check(page, checks, "overview", "Home")

    exercise_technique_guide(page, checks)
    exercise_live_probes(page, checks, capture)

    # Diagrams use actual tool results; sliders must update their numerical inputs.
    page.locator("#tab-tools").click()
    page.locator("#st-tool").select_option("detector")
    page.locator("#sg-probe").wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    face = page.locator(".sg-detector-face").get_attribute("points")
    original_probe = page.locator("#sg-probe").get_attribute("aria-valuetext")
    page.locator("#sg-azimuth").focus()
    page.locator("#sg-azimuth").press("ArrowRight")
    record(checks, "spatial-camera-preserves-probe",
           page.locator(".sg-detector-face").get_attribute("points") != face and
           page.locator("#sg-probe").get_attribute("aria-valuetext") == original_probe)
    capture("detector-spatial")
    overflow(page, checks, "detector-spatial")
    page.get_by_role("button", name="Side view", exact=True).click()
    record(checks, "detector-side-view-fallback", not page.locator(".sg-orbit").is_visible() and
           page.get_by_role("button", name="Side view", exact=True).get_attribute("aria-pressed") == "true" and
           page.locator("#sg-probe").get_attribute("aria-valuetext") == original_probe)
    before_ray = page.locator(".sg-probe").get_attribute("y2")
    page.locator("#sg-probe").focus()
    page.locator("#sg-probe").press("End")
    record(checks, "schematic-keyboard-probe", page.locator(".sg-probe").get_attribute("y2") != before_ray and
           page.locator("#sg-probe").get_attribute("aria-valuetext").endswith("inverse ångström."))
    distance = page.locator('[data-st-tool="detector"] [data-arg="distance_mm"]')
    distance.fill("2000")
    record(checks, "schematic-clears-stale-results", not page.locator("#st-schematic svg").count() and
           page.locator("[data-sg-export]").is_disabled())
    page.wait_for_function("() => document.querySelector('#sg-distance')?.value === '2000' && document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    record(checks, "schematic-input-to-slider", page.locator("#sg-distance").input_value() == "2000")
    page.locator("#sg-distance").focus()
    page.locator("#sg-distance").press("End")
    page.wait_for_function("() => document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    record(checks, "schematic-slider-to-input", distance.input_value() == "5000")
    page.locator(".sg-equations summary").click()
    exact = json.loads(page.locator(".sg-exact").inner_text())
    record(checks, "schematic-probe-matches-endpoint", exact["distance_mm"] == 5000 and exact["radius_mm"] == 100)
    capture("detector-schematic")
    overflow(page, checks, "detector-schematic")
    with page.expect_download() as pending_diagram:
        page.locator("[data-sg-export]").click()
    diagram = pending_diagram.value
    path = diagram.path()
    source = Path(path).read_text(encoding="utf-8") if path else ""
    record(checks, "schematic-svg-download", diagram.suggested_filename == "technique-diagram.svg" and
           "<svg" in source and "stroke:" in source and "Detector radial coverage" in source)
    page.locator("#st-tool").select_option("projection")
    page.locator("#sg-sample").wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    sample = page.locator('[data-st-tool="projection"] [data-arg="sample_counts"]')
    sample.fill("2050")
    page.wait_for_function("() => document.querySelector('#sg-sample')?.value === '2050' && document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    page.locator(".sg-equations summary").click()
    exact = json.loads(page.locator(".sg-exact").inner_text())
    record(checks, "schematic-above-flat-retained", exact["transmission"] == 2 and exact["line_integral"] < 0)
    capture("projection-schematic")
    overflow(page, checks, "projection-schematic")
    sample.fill("10")
    page.wait_for_function("() => document.querySelector('#st-status').textContent.includes('must exceed dark')")
    record(checks, "schematic-invalid-input", not page.locator("#st-schematic svg").count() and
           page.locator("[data-sg-export]").is_disabled() and not page.locator(".sg-exact").inner_text())
    sample.fill("550")
    page.wait_for_function("() => document.querySelector('#st-schematic').getAttribute('aria-busy') === 'false'")
    record(checks, "schematic-recovers", page.locator("#st-schematic svg").count() == 1)
    page.locator("#st-tool").select_option("optics")
    page.wait_for_function("() => document.querySelector('#st-output').dataset.tool === 'optics'")
    page.locator("#sg-probe").focus()
    page.locator("#sg-probe").press("End")
    page.locator(".sg-equations summary").click()
    optics = json.loads(page.locator(".sg-exact").inner_text())
    optical_result = download_json(page, "#st-export")
    optical_curve = optical_result["plots"][0]
    record(checks, "optics-schematic-probe", optics["grazing_angle_deg"] == optical_curve["x"][-1] and
           optics["intensity_depth_nm"] == optical_curve["traces"][0]["y"][-1])
    capture("optics-schematic")
    overflow(page, checks, "optics-schematic")
    page.locator("#st-tool").select_option("pdf")
    page.locator("#st-demo-pdf").click()
    page.wait_for_function("() => document.querySelector('#st-output').dataset.tool === 'pdf'")
    page.locator("#sg-probe").focus()
    page.locator("#sg-probe").press("End")
    page.locator(".sg-equations summary").click()
    pdf = json.loads(page.locator(".sg-exact").inner_text())
    pdf_result = download_json(page, "#st-export")
    record(checks, "pdf-schematic-probe", pdf["q_inv_angstrom"] == pdf_result["data"]["q"][-1] and
           pdf["modified_integrand_inv_angstrom"] == pdf_result["plots"][1]["traces"][0]["y"][-1])
    capture("pdf-schematic")
    overflow(page, checks, "pdf-schematic")
    page.locator("#st-tool").select_option("converter")
    page.wait_for_function("() => document.querySelector('#st-output').dataset.tool === 'converter'")
    record(checks, "schematic-hidden-for-other-tools", not page.locator("#st-schematic").is_visible())
    page.locator("#tab-overview").click()

    library = page.locator("#workspace-library-slot #open-materials-library")
    record(checks, "prominent-library-entry", library.is_visible())
    library.click()
    page.locator("#materials-dialog[open] .ml-item").first.wait_for(state="visible")
    record(checks, "library-modal-focus", page.locator("#materials-dialog").evaluate(
        "el => el.contains(document.activeElement) && el.matches(':modal')"))
    overflow(page, checks, "library")
    page.locator("#materials-dialog .ml-item").first.click()
    page.wait_for_function("() => document.querySelector('#ml-curve-summary')?.textContent.startsWith('At 60 keV:')")
    capture("library-detail")
    overflow(page, checks, "library-detail")
    page.keyboard.press("Escape")
    page.locator("#materials-dialog").wait_for(state="hidden")
    record(checks, "library-return-focus", library.evaluate("el => el === document.activeElement"))

    # User-selected plot settings must survive Show all and both save mechanisms.
    page.locator("#plot-settings > summary").click()
    page.locator("#log-x").uncheck()
    page.locator("#show-edges").uncheck()
    page.locator('button[data-layout="grid"]').click()
    page.locator('button[data-gesture="pan"]').click()
    customized = download_json(page, "#save-project")["ui"]["dashboard"]
    first = customized["panels"][0]
    record(checks, "custom-settings-applied", not first["logX"] and not first["edges"] and
           customized["layout"] == "grid" and customized["gesture"] == "pan")
    page.locator("#show-all-plots").click()
    page.wait_for_function("() => document.querySelectorAll('#plot-grid .scientific-plot.js-plotly-plot').length === 11")
    ready(page)
    expanded = download_json(page, "#save-project")
    record(checks, "show-all-eleven", len(expanded["ui"]["dashboard"]["panels"]) == 11)
    retained = next(p for p in expanded["ui"]["dashboard"]["panels"] if p["primary"] == first["primary"])
    record(checks, "show-all-preserves-custom-plot", {k: v for k, v in retained.items() if k != "id"} ==
           {k: v for k, v in first.items() if k != "id"})
    overflow(page, checks, "show-all")
    capture("show-all")
    results = download_json(page, "#export-json")
    record(checks, "populated-results-export", results["configuration"] == config and
           len(results["energy_keV"]) >= 321 and len(results["transmission"]) == len(results["energy_keV"]) and
           0 < results["reference"]["transmission"] < 1)
    project = download_json(page, "#save-project")
    record(checks, "export-preserves-settings", project == expanded and project["configuration"] == config)

    # Preserve an explicit non-default layout instead of replacing it with four.
    page.locator("#remove-plot").click()
    page.wait_for_function("() => document.querySelectorAll('#plot-grid .scientific-plot').length === 10")
    saved = download_json(page, "#save-project")
    page.reload(wait_until="networkidle")
    ready(page)
    restored = download_json(page, "#save-project")
    record(checks, "browser-saved-layout", persistent_project(restored) == persistent_project(saved) and
           page.locator(PLOTS).count() == 10, {"expected": saved, "actual": restored})
    page.locator("#show-all-plots").click()
    page.locator("#project-file").set_input_files({
        "name": "audit-project.json", "mimeType": "application/json", "buffer": json.dumps(saved).encode("utf-8")})
    page.wait_for_function("() => document.querySelectorAll('#plot-grid .scientific-plot').length === 10")
    ready(page)
    restored = download_json(page, "#save-project")
    record(checks, "project-file-round-trip", persistent_project(restored) == persistent_project(saved), {"expected": saved, "actual": restored})
    overflow(page, checks, "restored")
    record(checks, "final-inputs-preserved", inputs(page) == before)
    page.goto(page.url.split("#", 1)[0] + "#technique-guide/optics")
    page.reload()
    page.locator("#technique-guide > summary").wait_for(state="visible")
    page.wait_for_function(
        "() => document.querySelector('#guide-family')?.value === 'optics' && "
        "document.querySelector('#technique-guide > summary') === document.activeElement"
    )
    record(checks, "guide-direct-family-entry", page.locator("[data-guide-family]:visible").count() == 1)


def audit(url: str, screenshots: Path | None = None) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    report: dict[str, Any] = {"scenarios": {}, "coverage": "Native Chromium interaction and overflow checks"}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for name, (width, height) in VIEWPORTS.items():
                for theme in ("light", "dark"):
                    key = f"{name}-{theme}"
                    context = browser.new_context(viewport={"width": width, "height": height},
                                                  color_scheme="light", accept_downloads=True)
                    page = context.new_page()
                    page.set_default_timeout(30_000)
                    checks: list[dict[str, Any]] = []
                    errors: list[str] = []
                    page.on("pageerror", partial(_record_error, errors))
                    report["scenarios"][key] = {"checks": checks, "page_errors": errors}

                    def capture(view: str, target: Any = page, label: str = key) -> None:
                        if screenshots is not None:
                            target.screenshot(path=str(screenshots / f"{label}-{view}.png"))

                    try:
                        page.goto(url, wait_until="networkidle")
                        ready(page)
                        # Auto starts with a light media preference; use the real toggle.
                        for _ in range(2 if theme == "dark" else 1):
                            page.locator("#theme-toggle").click()
                        page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)
                        exercise(page, checks, capture)
                        record(checks, "theme-preserved",
                               page.locator("html").get_attribute("data-theme") == theme)
                    except Exception as exc:
                        record(checks, "interaction-completed", False, f"{type(exc).__name__}: {exc}")
                        try:
                            capture("failure")
                        except Exception as screenshot_error:
                            record(checks, "failure-screenshot", False, str(screenshot_error))
                    finally:
                        context.close()
        finally:
            browser.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "workspace-ui-audit-report.json")
    parser.add_argument("--screenshots-dir", type=Path)
    parser.add_argument("--report-only", action="store_true", help="write failures but exit successfully")
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.screenshots_dir is not None:
        args.screenshots_dir.mkdir(parents=True, exist_ok=True)
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
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for name, scenario in report["scenarios"].items():
        failed = [check for check in scenario["checks"] if not check["passed"]]
        print(f"{name}: {len(failed)} failed checks, {len(scenario['page_errors'])} page errors")
        for check in failed:
            print(f"  {check['name']}: {check['details']}")
        for error in scenario["page_errors"]:
            print(f"  page error: {error}")
    print(f"Report: {args.report}")
    return 0 if args.report_only or failure_count(report) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
