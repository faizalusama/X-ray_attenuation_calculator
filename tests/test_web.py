"""The browser version: same answers as the local server, and a site that builds correctly."""

import json
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools.build_web import PYODIDE_VERSION, build, locked_version, pyodide_files
from tools.web_parity import configurations, count_numbers, differences
from xray_workbench import __version__
from xray_workbench.browser import handle
from xray_workbench.server import app

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = {"energy": {"min_keV": 5, "max_keV": 30, "points": 20, "reference_keV": 14},
                 "layers": [{"basis": "mole", "density_g_cm3": 2.454, "thickness_mm": 0.1366,
                             "components": [{"formula": "SiO2", "fraction": 81}, {"formula": "Na2O", "fraction": 9.54},
                                            {"formula": "CaO", "fraction": 9.46}]}]}


def without_timestamp(result):
    result["provenance"].pop("timestamp_utc")
    return result


@pytest.mark.parametrize("route", ["health", "presets", "xray-sources", "materials", "materials?q=alumina",
                                   "materials?category=glass&verified_density=true", "materials/landscape?energy_keV=40",
                                   "materials/stoich-alumina-corundum?thickness_mm=2"])
def test_read_routes_match_the_local_server(route):
    status, text = handle(route)
    assert status == 200
    assert json.loads(text) == TestClient(app).get(f"/api/{route}").json()


def test_calculation_matches_the_local_server():
    status, text = handle("calculate", json.dumps(CONFIGURATION))
    assert status == 200
    served = TestClient(app).post("/api/calculate", json=CONFIGURATION).json()
    assert without_timestamp(json.loads(text)) == without_timestamp(served)


@pytest.mark.parametrize("body,reason", [
    ("[1, 2]", "Configuration must be a JSON object."),
    ('{"layers": [{"thickness_mm": NaN}]}', "Non-finite JSON number: NaN"),
    (json.dumps({"layers": [{"basis": "mass", "components": [{"formula": "Xx2", "fraction": 1}],
                             "density_g_cm3": 1}]}), "Unknown or unsupported element 'Xx'."),
])
def test_rejections_match_the_local_server(body, reason):
    status, text = handle("calculate", body)
    served = TestClient(app).post("/api/calculate", content=body, headers={"Content-Type": "application/json"})
    assert (status, json.loads(text)["detail"]) == (served.status_code, served.json()["detail"]) == (422, reason)


def test_unknown_route_is_refused():
    assert handle("download")[0] == 404


def test_runtime_is_recorded_only_when_given():
    assert "runtime" not in json.loads(handle("calculate", json.dumps(CONFIGURATION))[1])["provenance"]
    result = json.loads(handle("calculate", json.dumps(CONFIGURATION), "Pyodide test")[1])
    assert result["provenance"]["runtime"] == "Pyodide test"


def test_site_build_is_complete_and_relocatable(tmp_path):
    site = tmp_path / "site"
    info = build(site, fetch_wheel=False)
    page = (site / "index.html").read_text(encoding="utf-8")
    assert '"/static/' not in page, "a project page is served below the domain root"
    assert page.index("static/web/shim.js") < page.index("static/app.js"), "the shim must install first"
    assert "Runs in your browser" in page
    assert not (site / "static" / "index.html").exists()
    assert (site / ".nojekyll").is_file()
    assert (site / "static" / "web" / "engine_worker.js").is_file()
    assert info["version"] == __version__
    assert info["pyodide_version"] == PYODIDE_VERSION
    assert info["xraydb_wheel"] == f"xraydb-{locked_version('xraydb')}-py3-none-any.whl"
    assert json.loads((site / "engine" / "build-info.json").read_text(encoding="utf-8")) == info
    with zipfile.ZipFile(site / "engine" / str(info["engine_zip"])) as archive:
        names = archive.namelist()
    assert {"xray_workbench/physics.py", "xray_workbench/browser.py", "xray_workbench/backends/elam.py"} <= set(names)
    packaged = zipfile.ZipFile(site / "engine" / str(info["engine_zip"])).read("xray_workbench/physics.py")
    assert packaged == (ROOT / "xray_workbench" / "physics.py").read_bytes(), "the site must run the same physics"


def test_static_assets_have_no_root_relative_urls():
    for path in (ROOT / "static").glob("*.css"):
        assert not re.search(r"url\(\s*['\"]?/", path.read_text(encoding="utf-8")), path.name


def test_comparison_reports_every_kind_of_difference():
    assert list(differences({"a": [1.0, 2.0]}, {"a": [1.0, 2.0]})) == []
    path, difference = next(differences({"a": 1.0}, {"a": 1.1}))
    assert path == ".a"
    assert difference == pytest.approx(0.1 / 1.1)
    assert list(differences({"a": 1.0}, {})) == [(".a", None)]
    assert list(differences([1.0], [1.0, 2.0])) == [("", None)]
    assert list(differences("x", "y")) == [("", None)]
    assert list(differences({"provenance": {"timestamp_utc": "now"}}, {"provenance": {"timestamp_utc": "then"}})) == []
    assert list(differences(1e-300, 2e-300)) == [("", 0.0)], "underflowing values are not meaningful"


def test_comparison_covers_the_measured_glasses():
    configs = configurations()
    assert {"v-SiO2", "NCS-80", "NAS-60", "CAS-N", "example"} <= set(configs)
    assert len(configs) == 16
    assert count_numbers({"a": [1.0, 2], "b": "text", "provenance": {"c": 3}}) == 2


@pytest.mark.parametrize("route,status", [("materials/not-an-entry", 404), ("materials/landscape?energy_keV=2000", 422),
                                          ("materials/landscape?thickness_mm=0", 422)])
def test_material_errors_match_the_local_server(route, status):
    answer = TestClient(app).get(f"/api/{route}")
    assert answer.status_code == status
    assert handle(route)[0] == status
    assert json.loads(handle(route)[1]) == answer.json()


def test_the_site_packages_the_library_data(tmp_path):
    info = build(tmp_path / "site", fetch_wheel=False)
    with zipfile.ZipFile(tmp_path / "site" / "engine" / info["engine_zip"]) as archive:
        assert "xray_workbench/materials/data/pnnl_15870_rev2.json" in archive.namelist()


def test_the_site_loads_nothing_from_a_third_party(tmp_path):
    info = build(tmp_path / "site", fetch_wheel=False)
    assert info["pyodide_url"] == "../pyodide/"
    for path in [*(tmp_path / "site" / "static" / "web").iterdir(), tmp_path / "site" / "index.html"]:
        assert "jsdelivr" not in path.read_text(encoding="utf-8")


def test_a_release_that_does_not_match_the_pin_is_refused(tmp_path):
    fake = tmp_path / "release.tar.bz2"
    fake.write_bytes(b"not the release")
    with pytest.raises(SystemExit, match="does not match"):
        pyodide_files(tmp_path / "cache", fake)
