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
