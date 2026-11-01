import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xray_workbench.browser import handle
from xray_workbench.scattering_tools import scattering_tools
from xray_workbench.server import app

SI = {"name": "Silicon", "basis": "mole", "density_g_cm3": 2.329, "thickness_mm": 0.001,
      "components": [{"formula": "Si", "fraction": 1}]}


def request(tool="optics", **fields):
    return {"tool": tool, "layer": SI, "energy_keV": 8.048, **fields}


def test_silicon_optics_matches_reference_and_normal_absorption_limit():
    answer = scattering_tools(request(angle_max_deg=90))
    data = answer["data"]
    assert data["delta"] == pytest.approx(7.573e-6, rel=1e-4)
    assert data["critical_angle_deg"] == pytest.approx(0.223, abs=0.001)
    depth = answer["plots"][0]["traces"][0]["y"][-1]
    assert depth == pytest.approx(data["wavelength_angstrom"] / (4*np.pi*data["beta"]) / 10, rel=1e-8)


def test_tools_route_matches_browser_and_refuses_unknown_tool():
    payload = request()
    status, text = handle("scattering-tools", json.dumps(payload))
    response = TestClient(app).post("/api/scattering-tools", json=payload)
    assert status == response.status_code == 200
    assert json.loads(text) == response.json()
    with pytest.raises(ValueError, match="Unknown scattering"):
        scattering_tools(request("missing"))
