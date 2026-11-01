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

def test_parratt_reduces_to_fresnel_and_zero_thickness_film():
    from xray_workbench.scattering_tools import parratt

    angles = np.linspace(0.01, 3, 200)
    n = 1 - 7.57e-6 + 1j*1.75e-7
    k0 = np.sin(np.radians(angles))
    k1 = np.sqrt(n*n - np.cos(np.radians(angles))**2 + 0j)
    fresnel = abs((k0-k1)/(k0+k1))**2
    assert parratt([1,n], [0,0], [0], angles, 1.54) == pytest.approx(fresnel, abs=2e-11)
    assert parratt([1,1-3e-6+1j*1e-8,n], [0,0,0], [0,0], angles, 1.54) == pytest.approx(fresnel, abs=2e-11)
    assert np.all(parratt([1,1], [0,0], [0], angles, 1.54) == 0)
    rough = parratt([1,n], [0,0], [3], angles, 1.54)
    assert rough[-1] < fresnel[-1]


def test_reflectivity_route_handles_stack_and_checks_thickness_count():
    answer = scattering_tools(request("reflectivity", layers=[SI,SI], film_thickness_angstrom="100"))
    assert answer["data"]["film_thickness_angstrom"] == [100]
    assert max(answer["plots"][0]["traces"][0]["y"]) <= 1
    with pytest.raises(ValueError, match="one film thickness"):
        scattering_tools(request("reflectivity", layers=[SI], film_thickness_angstrom="100"))

def test_contrast_has_zero_same_phase_and_porod_symmetry():
    water = {**SI, "name": "Water", "density_g_cm3": 1, "components": [{"formula": "H2O", "fraction": 1}]}
    same = scattering_tools(request("contrast", layers=[SI], other_layer=0))["data"]
    assert same["xray_contrast_squared_inv_angstrom4"] == 0
    a = scattering_tools(request("contrast", layers=[SI,water], volume_fraction=0.3))["data"]
    b = scattering_tools(request("contrast", layer=water, layers=[SI,water], other_layer=0, volume_fraction=0.7))["data"]
    assert a == pytest.approx(b, rel=1e-14)
    zero = scattering_tools(request("contrast", layers=[SI,water], volume_fraction=0))["data"]
    assert zero["xray_porod_invariant_inv_angstrom4"] == 0
    assert a["neutron_contrast_squared_inv_angstrom4"] > 0
