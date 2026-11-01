"""Total-scattering planning values, checked against hand results and independent tables."""

import json
from math import exp, pi

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xray_workbench.browser import handle
from xray_workbench.physics import calculate
from xray_workbench.scattering import (
    COMPTON_WAVELENGTH_ANGSTROM,
    FWHM_LORCH,
    FWHM_STEP,
    Q_LIMIT,
    q_from_angle,
    total_scattering,
)
from xray_workbench.server import app
from xray_workbench.sources import HC_KEV_ANGSTROM


def layer(components, density=2.2, basis="mole", **fields):
    return {"name": "sample", "basis": basis, "density_mode": "bulk", "density_g_cm3": density, "thickness_mm": 1, "angle_deg": 0,
            "components": [{"formula": f, "fraction": x} for f, x in components], **fields}


SILICA = layer([("SiO2", 1)])


def request(sample=SILICA, energy=60.0, **fields):
    return {"layer": sample, "energy_keV": energy, **fields}


def test_q_max_follows_bragg_geometry():
    # 4π/λ at backscattering: Cu Kα1 is known to reach 8.157 1/Å, and Mo Kα1 17.72 1/Å.
    assert q_from_angle(180, 1.54056) == pytest.approx(8.157, abs=1e-3)
    assert q_from_angle(180, 0.70930) == pytest.approx(17.72, abs=1e-2)
    assert q_from_angle(60, 1.0) == pytest.approx(2 * pi, rel=1e-14)


def test_resolution_is_the_nyquist_step_and_the_termination_widths():
    result = total_scattering(request(energy=60, two_theta_max_deg=40))["resolution"]
    assert result["q_max"] == pytest.approx(4 * pi * np.sin(np.radians(20)) * 60 / HC_KEV_ANGSTROM, rel=1e-14)
    assert result["r_step_nyquist"] * result["q_max"] == pytest.approx(pi, rel=1e-14)


@pytest.mark.parametrize("window,expected", [(lambda x: np.ones_like(x), FWHM_STEP), (np.sinc, FWHM_LORCH)])
def test_termination_width_constants_are_what_the_integral_gives(window, expected):
    q = np.linspace(0, 1, 20001)
    r = np.linspace(1e-4, 10, 2001)
    peak = np.array([np.trapezoid(window(q) * np.cos(q * x), q) for x in r])
    half = np.trapezoid(window(q), q) / 2
    i = np.argmax(peak < half)
    crossing = r[i - 1] + (half - peak[i - 1]) * (r[i] - r[i - 1]) / (peak[i] - peak[i - 1])
    assert 2 * crossing == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("sample,expected", [(layer([("SiO2", 1)], 2.2), 0.06615), (layer([("H2O", 1)], 0.99705), 0.09999),
                                             (layer([("Si", 1)], 2.329), 0.04994), (layer([("Cu", 1)], 8.96), 0.08491)])
def test_number_density_matches_hand_values(sample, expected):
    # 3 × 2.2 × 0.6022 / 60.08 for silica; 3 × 0.033329 for water at 25 °C; 8/a³ for silicon; 4/a³ for copper.
    assert total_scattering(request(sample))["sample"]["number_density"] == pytest.approx(expected, rel=2e-3)


def test_low_r_slope_is_minus_four_pi_rho():
    sample = total_scattering(request())["sample"]
    assert sample["low_r_slope"] == pytest.approx(-4 * pi * sample["number_density"], rel=1e-14)


def test_silica_pair_weights_at_zero_q_are_the_textbook_values():
    pairs = {"-".join(sorted(p["pair"].split("-"))): p for p in total_scattering(request())["curves"]["pairs"]}
    # Si: c = 1/3, Z = 14. O: c = 2/3, Z = 8. ⟨f⟩ = 10.
    assert pairs["Si-Si"]["at_zero"] == pytest.approx((14 / 3) ** 2 / 100, abs=1e-3)
    assert pairs["O-Si"]["at_zero"] == pytest.approx(2 * (14 / 3) * (16 / 3) / 100, abs=1e-3)
    assert pairs["O-O"]["at_zero"] == pytest.approx((16 / 3) ** 2 / 100, abs=1e-3)


@pytest.mark.parametrize("components", [[("SiO2", 1)], [("SiO2", 60), ("Na2O", 20), ("Al2O3", 20)], [("Cu", 1)], [("HfO2", 1), ("Si3N4", 2)]])
def test_pair_weights_sum_to_one_at_every_q(components):
    pairs = total_scattering(request(layer(components)))["curves"]["pairs"]
    assert np.sum([p["weights"] for p in pairs], axis=0) == pytest.approx(1, abs=1e-12)


def test_laue_term_vanishes_for_one_element_and_not_for_two():
    assert max(map(abs, total_scattering(request(layer([("Cu", 1)])))["curves"]["laue"])) < 1e-12
    assert min(total_scattering(request())["curves"]["laue"]) > 0


def test_form_factors_start_at_the_atomic_number():
    for element in total_scattering(request(layer([("HfO2", 1)])))["elements"]:
        assert element["f0_at_zero"] == pytest.approx(element["atomic_number"], rel=2e-3)


# Limits set just above the differences measured on 1 November 2026, for Q up to 25 and up to 50.7 1/Å.
@pytest.mark.parametrize("symbol,low_q,high_q", [("O", 0.01, 0.06), ("Si", 0.01, 0.04), ("Fe", 0.02, 0.07), ("Hf", 0.04, 0.11), ("Pb", 0.05, 0.12)])
def test_form_factors_agree_with_epdl97_in_xraylib(symbol, low_q, high_q):
    # xraylib's attenuation data are Elam's, but its form factors are an independent tabulation.
    xraylib = pytest.importorskip("xraylib")
    curves = total_scattering(request(layer([(symbol, 1)]), energy=100, two_theta_max_deg=60))["curves"]
    number = xraylib.SymbolToAtomicNumber(symbol)
    expected = np.array([xraylib.FF_Rayl(number, q / (4 * pi)) for q in curves["q"]])
    difference = np.abs(np.array(curves["form_factors"][symbol]) / expected - 1)
    assert difference[np.array(curves["q"]) <= 25].max() < low_q
    assert difference.max() < high_q


@pytest.mark.parametrize("symbol,energy,tolerance", [("Cu", 8.0478, 0.04), ("Hf", 60.0, 0.05), ("Pb", 100.0, 0.06)])
def test_anomalous_terms_agree_with_xraylib(symbol, energy, tolerance):
    xraylib = pytest.importorskip("xraylib")
    element = total_scattering(request(layer([(symbol, 1)]), energy=energy))["elements"][0]
    number = xraylib.SymbolToAtomicNumber(symbol)
    assert element["f_prime"] == pytest.approx(xraylib.Fi(number, energy), rel=tolerance)
    assert element["f_double_prime"] == pytest.approx(-xraylib.Fii(number, energy), rel=tolerance)  # opposite sign convention


def test_attenuation_is_the_calculators_own():
    energy = 30.0
    engine = calculate({"energy": {"min_keV": 5, "max_keV": 100, "points": 5, "reference_keV": energy}, "layers": [SILICA]})["layers"][0]["reference"]
    result = total_scattering(request(energy=energy, capillary_diameter_mm=2))["attenuation"]
    assert result["mu_linear_cm_inv"] == pytest.approx(engine["mu_linear_cm_inv"], rel=1e-12)
    assert result["mu_d"] == pytest.approx(engine["mu_linear_cm_inv"] * 0.2, rel=1e-12)
    assert result["mu_r"] == pytest.approx(result["mu_d"] / 2, rel=1e-14)
    assert result["transmission"] == pytest.approx(exp(-result["mu_d"]), rel=1e-14)
    assert sum(result["shares"].values()) == pytest.approx(1, rel=1e-12)


def test_plate_optimum_maximises_scattering_from_the_transmitted_beam():
    result = total_scattering(request(energy=30))["attenuation"]
    mu = result["mu_linear_cm_inv"] / 10
    thickness = np.linspace(0.001, 20 * result["plate_optimum_mm"], 200001)
    assert thickness[np.argmax(thickness * np.exp(-mu * thickness))] == pytest.approx(result["plate_optimum_mm"], rel=1e-3)


def test_compton_energy_follows_the_kinematics():
    # E' = E / (1 + (E/mₑc²)(1 - cos 2θ)), with mₑc² = 510.999 keV.
    compton = total_scattering(request(energy=100, two_theta_max_deg=90))["compton"]
    assert compton["energy_keV"] == pytest.approx(100 / (1 + 100 / 510.99895), rel=1e-6)
    assert pytest.approx(510.99895, rel=1e-6) == HC_KEV_ANGSTROM / COMPTON_WAVELENGTH_ANGSTROM


def test_fluorescence_lists_only_edges_the_beam_can_excite():
    iron = layer([("Fe2O3", 1)], 5.24)
    excited = {(e["element"], e["edge"]) for e in total_scattering(request(iron, energy=8.0478))["fluorescence"]}
    assert ("Fe", "K") in excited
    assert ("Fe", "K") not in {(e["element"], e["edge"]) for e in total_scattering(request(iron, energy=5.415))["fluorescence"]}
    line = next(e for e in total_scattering(request(iron, energy=8.0478))["fluorescence"] if e["edge"] == "K" and e["element"] == "Fe")
    assert line["line"] == "Ka1"
    assert line["line_keV"] == pytest.approx(6.404, abs=0.005)


def test_a_beam_near_an_edge_is_flagged():
    result = total_scattering(request(layer([("Cu", 1)], 8.96), energy=8.98))
    assert any("Cu K edge" in warning for warning in result["warnings"])
    assert not total_scattering(request(layer([("Cu", 1)], 8.96), energy=60))["warnings"]


def test_curves_stop_at_the_limit_of_the_form_factor_fits():
    result = total_scattering(request(energy=300, two_theta_max_deg=180))
    assert result["resolution"]["q_max"] > Q_LIMIT
    assert result["curves"]["q"][-1] == pytest.approx(Q_LIMIT, rel=1e-12)
    assert any("limit of the form-factor fits" in warning for warning in result["warnings"])


def test_source_table_reports_reach_and_the_angle_for_a_target_q():
    rows = {row["name"]: row for row in total_scattering(request(two_theta_max_deg=120, target_q=25))["sources"]}
    assert rows["Cu Kα"]["two_theta_for_target_deg"] is None, "Cu Kα cannot reach 25 1/Å at any angle"
    assert rows["Mo Kα"]["two_theta_for_target_deg"] is None
    assert rows["60 keV"]["two_theta_for_target_deg"] == pytest.approx(2 * np.degrees(np.arcsin(25 * HC_KEV_ANGSTROM / 60 / (4 * pi))), rel=1e-12)
    assert rows["Ag Kα"]["q_max"] == pytest.approx(4 * pi * np.sin(np.radians(60)) / rows["Ag Kα"]["wavelength_angstrom"], rel=1e-12)


def test_composition_string_gives_the_ratio():
    assert total_scattering(request())["sample"]["composition"] in ("Si1 O2", "O2 Si1")


def test_density_set_by_porosity_is_used_as_the_engine_uses_it():
    powder = layer([("SiO2", 1)], 2.2, density_mode="solid", porosity=0.5)
    assert total_scattering(request(powder))["sample"]["density_g_cm3"] == pytest.approx(1.1, rel=1e-12)


@pytest.mark.parametrize("fields,message", [({"energy_keV": 0.5}, "Energy"), ({"two_theta_max_deg": 0}, "2θ"), ({"two_theta_max_deg": 181}, "2θ"),
                                            ({"capillary_diameter_mm": 0}, "Capillary"), ({"target_q": 0}, "Target Q")])
def test_bad_requests_are_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        total_scattering({**request(), **fields})


def test_server_and_browser_give_the_same_answer():
    body = json.dumps(request(energy=17.44))
    status, text = handle("total-scattering", body)
    answer = TestClient(app).post("/api/total-scattering", content=body, headers={"content-type": "application/json"})
    assert status == answer.status_code == 200
    assert json.loads(text) == answer.json()


def test_server_and_browser_give_the_same_refusal():
    body = json.dumps({**request(), "two_theta_max_deg": 500})
    status, text = handle("total-scattering", body)
    answer = TestClient(app).post("/api/total-scattering", content=body, headers={"content-type": "application/json"})
    assert status == answer.status_code == 422
    assert json.loads(text) == answer.json()


def test_fluorescence_leaves_out_lines_below_one_kev():
    listed = total_scattering(request(layer([("SiO2", 1)]), energy=30))["fluorescence"]
    assert {(e["element"], e["edge"]) for e in listed} == {("Si", "K")}


def test_differential_scattering_limits_and_integral():
    r = total_scattering(request(energy=20))
    d = r["differential"]
    assert d["S_mean"][0] == 0
    assert d["coherent_fraction"][0] == 1
    assert all(v >= 0 for v in d["incoherent_barn_sr_atom"])
    assert d["integrated"]["incoherent_cm2_g"] > 0
    # Independent tabulated Elam comparison: report discrepancy rather than normalize it away.
    for kind in ("coherent", "incoherent"):
        tabulated = r["attenuation"]["mu_mass_cm2_g"] * r["attenuation"]["shares"][kind]
        ratio = d["integrated"][f"{kind}_cm2_g"] / tabulated
        # SiO2 at 20 keV: independent approximations agree within 0.5%; not a fitted normalization.
        assert ratio == pytest.approx(1, abs=0.005)
    assert total_scattering(request(energy=100))["differential"]["integrated"] is None

def test_ionic_electron_counts_and_neutral_attenuation():
    sample = layer([("Fe", 1)], 7.87)
    neutral = total_scattering(request(sample, energy=20))
    ionic = total_scattering(request(sample, energy=20, ions={"Fe": "Fe2+"}))
    assert ionic["elements"][0]["f0_at_zero"] == pytest.approx(24, abs=0.02)
    assert ionic["attenuation"] == neutral["attenuation"]
    assert ionic["curves"]["mean_f_squared"][0] < neutral["curves"]["mean_f_squared"][0]
    with pytest.raises(ValueError, match="Unknown ionic"):
        total_scattering(request(sample, ions={"Fe": "O2-"}))
    with pytest.raises(ValueError, match="selected layer"):
        total_scattering(request(sample, ions={"O": "O2-"}))

@pytest.mark.parametrize("mu_r", [0, 0.2, 1, 5])
def test_disc_integral_against_independent_chord_integrals(mu_r):
    from scipy.integrate import quad

    from xray_workbench.scattering import capillary_absorption

    result = capillary_absorption(mu_r)
    forward = quad(lambda y: 2 * np.sqrt(1 - y*y) * np.exp(-2 * mu_r * np.sqrt(1-y*y)) / pi, -1, 1, epsabs=1e-11)[0]
    backward = 1 if mu_r == 0 else quad(lambda y: -np.expm1(-4*mu_r*np.sqrt(1-y*y)) / (2*mu_r*pi), -1, 1, epsabs=1e-11)[0]
    assert result["absorption"][0] == pytest.approx(forward, abs=2e-6)
    assert result["absorption"][-1] == pytest.approx(backward, abs=2e-6)
    assert result["polarization"]["unpolarized"][45] == pytest.approx(0.5)
    assert result["polarization"]["in_plane"][45] == pytest.approx(0, abs=1e-30)

def test_neutron_negative_weights_and_zero_mean_guard():
    from xray_workbench.scattering import neutron_weights

    water = total_scattering(request(layer([("H2O", 1)], 1)))["neutron"]
    assert sum(p["weight"] for p in water["pairs"]) == pytest.approx(1, abs=1e-12)
    assert next(p["weight"] for p in water["pairs"] if p["pair"] == "H-O") < 0
    zero = neutron_weights(["H", "O"], np.array([5.803, 3.739]) / (5.803 + 3.739), 0.1)
    assert zero["pairs"] == []
    assert "near zero" in zero["note"]
    assert not neutron_weights(["Po"], np.array([1]), 0.1)["available"]

def test_chantler_edge_scan_has_absorption_jump_and_explicit_selection():
    result = total_scattering(request(layer([("Fe", 1)], 7.87), edge_id="Fe:K"))["edge_scan"]
    assert result["edge_keV"] == pytest.approx(7.112)
    assert min(result["f_prime"]) < -5
    assert result["f_double_prime"][-1] > result["f_double_prime"][0] * 3
    assert np.all(np.diff(result["energy_keV"]) > 0)
    with pytest.raises(ValueError, match="absorption edge"):
        total_scattering(request(edge_id="Fe:K"))
