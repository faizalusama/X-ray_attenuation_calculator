import json

import pytest
from fastapi.testclient import TestClient

from xray_workbench.browser import handle
from xray_workbench.comparison import compare
from xray_workbench.server import app

CSV = 'name,basis,density_g_cm3,thickness_mm,composition\n"silica, fused",mole,2.2,0.135,SiO2:1\nbad,mass,2,1,NoSuch:1\n'


def test_batch_preserves_quoted_names_and_localizes_errors():
    answer = compare({"energy_keV": 14, "csv": CSV})
    assert answer["successes"] == 1
    assert answer["rows"][0]["name"] == "silica, fused"
    assert answer["rows"][0]["attenuation_length_mm"] == pytest.approx(0.640, abs=0.001)
    assert answer["rows"][1]["error"]
    assert "transmission" not in answer["rows"][1]


def test_compare_route_matches_browser_and_refuses_headers():
    body = {"energy_keV": 14, "csv": CSV}
    status, text = handle("compare", json.dumps(body))
    assert status == 200
    assert json.loads(text) == TestClient(app).post("/api/compare", json=body).json()
    with pytest.raises(ValueError, match="CSV columns"):
        compare({"energy_keV": 14, "csv": "name\nsilica\n"})
    with pytest.raises(ValueError, match="at most 100"):
        compare({"energy_keV": 14, "csv": CSV.splitlines()[0] + "\n" + "s,mole,2.2,1,SiO2:1\n" * 101})
