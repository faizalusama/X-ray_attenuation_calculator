"""API validation, provenance, and delivered frontend contract."""
import json

from fastapi.testclient import TestClient

from xray_workbench.server import app

client = TestClient(app)


def payload():
    layer = client.get("/api/presets").json()["presets"][0]["layer"]
    return {"energy": {"min_keV": 10, "max_keV": 100, "reference_keV": 30,
                       "points": 50, "spacing": "log"}, "layers": [layer]}


def test_api_calculation_and_provenance():
    request = payload()
    response = client.post("/api/calculate", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["reference"]["transmission"] <= 1
    assert result["configuration"] == request
    assert len(result["provenance"]["configuration_sha256"]) == 64
    assert result["schema_version"] == 1
    json.dumps(result, allow_nan=False)


def test_validation_is_actionable():
    assert client.post("/api/calculate", json=[]).status_code == 422
    assert client.post("/api/calculate", content="NaN", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/calculate", content="{}").status_code == 415
    request = payload()
    request["energy"]["reference_keV"] = 900
    response = client.post("/api/calculate", json=request)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_body_limit_and_host():
    response = client.post("/api/calculate", content=" " * 2_000_001, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert client.get("/api/health", headers={"Host": "unrelated.invalid"}).status_code == 400


def test_app_assets_are_available():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_native_export_preserves_exact_unicode_and_metadata():
    content = '<svg><text>μ/ρ · cm²/g & 0.000000000001</text></svg>\n'
    response = client.post("/api/download", data={"filename": "attenuation-mass.svg", "content": content})
    assert response.status_code == 200
    assert response.content.decode("utf-8") == content
    assert response.headers["content-disposition"] == 'attachment; filename="attenuation-mass.svg"'
    assert response.headers["cache-control"] == "no-store"


def test_export_rejects_unexpected_names_and_payloads():
    assert client.post("/api/download", json={}).status_code == 415
    assert client.post("/api/download", data={"filename": "../secret.txt", "content": "x"}).status_code == 422
    assert client.post("/api/download", data={"content": "x"}).status_code == 422


def test_static_assets_are_compressed_and_cached_by_version():
    plotly = client.get("/static/vendor/plotly-4.0.0.min.js", headers={"Accept-Encoding": "gzip"})
    assert plotly.status_code == 200
    assert plotly.headers["content-encoding"] == "gzip"
    assert "immutable" in plotly.headers["cache-control"]
    versioned = client.get("/static/app.js?v=test", headers={"Accept-Encoding": "gzip"})
    assert "max-age=31536000" in versioned.headers["cache-control"]
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    # API responses must never be cached.
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_page_offers_a_skip_link_to_main():
    page = client.get("/").text
    assert 'class="skip-link" href="#main"' in page
    assert '<main id="main">' in page
    assert "/static/readability.css" in page


def test_every_cached_asset_stamp_matches_its_content():
    # Browsers keep ?v= assets for a year, so a stamp that does not change with
    # the file would serve stale code. Fix with: python tools/stamp_assets.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from stamp_assets import stale
    assert stale() == []


def test_new_export_names_are_accepted():
    for filename in ("total-scattering-curves.csv", "total-scattering-summary.json", "attenuation-all-curves.csv", "attenuation-all-curves-settings.json", "attenuation-spectrum.csv", "attenuation-plot-map.csv"):
        assert client.post("/api/download", data={"filename": filename, "content": "x"}).status_code == 200


def test_dark_theme_is_generated_from_the_current_light_stylesheets():
    # Fix with: python tools/build_dark_css.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from build_dark_css import TARGET, generate
    assert TARGET.read_text(encoding="utf-8").replace("\r\n", "\n") == generate()


def test_no_plot_offers_plotly_cloud_upload():
    # Plotly's default toolbar includes "Share chart...", which uploads the chart.
    # The page promises that nothing entered is uploaded, so every plot hides the toolbar.
    import re
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "static"
    for name in ("materials.js", "scattering.js"):
        options = re.findall(r"\{responsive: true[^}]*\}", (static / name).read_text(encoding="utf-8"))
        assert options, name
        assert all("displayModeBar: false" in option for option in options), name
    assert "displayModeBar: false" in re.search(r"function plotOptions\(\).*", (static / "dashboard.js").read_text(encoding="utf-8")).group()
