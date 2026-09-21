"""Physics invariants and regressions independent of the browser/server."""

import json
from copy import deepcopy
from math import exp, log

import numpy as np
import pytest
import xraydb

from xray_workbench.physics import EDGE_BRACKET_RELATIVE, calculate, presets


def layer(formula="Al", density=2.699, thickness=1.0, **kwargs):
    return {"name": formula, "basis": "mass", "components": [{"formula": formula, "fraction": 100}],
            "density_g_cm3": density, "thickness_mm": thickness, **kwargs}


def request(layers=None, **kwargs):
    return {"energy": {"min_keV": 10, "max_keV": 100, "points": 23, "spacing": "linear", "reference_keV": 30},
            "layers": layers or [layer()], **kwargs}


def test_pure_element_analytic_units_and_lengths():
    result = calculate(request())
    mu = float(xraydb.mu_elam("Al", 30000))
    reference = result["layers"][0]["reference"]
    assert reference["mu_mass_cm2_g"] == pytest.approx(mu, rel=2e-14)
    assert reference["mu_linear_cm_inv"] == pytest.approx(mu * 2.699)
    assert reference["transmission"] == pytest.approx(exp(-mu * 2.699 * 0.1))
    assert reference["hvl_mm"] == pytest.approx(10 * log(2) / (mu * 2.699))
    assert reference["tvl_mm"] == pytest.approx(10 * log(10) / (mu * 2.699))
    assert reference["attenuation_length_mm"] == pytest.approx(10 / (mu * 2.699))


def test_half_value_layer_yields_half_transmission():
    hvl = calculate(request())["layers"][0]["reference"]["hvl_mm"]
    result = calculate(request([layer(thickness=hvl)]))
    assert result["reference"]["transmission"] == pytest.approx(0.5)


def test_correct_single_mass_weighting_regression():
    mix = layer(density=3)
    mix["components"] = [{"formula": "Al", "fraction": 25}, {"formula": "Cu", "fraction": 75}]
    result = calculate(request([mix]))
    expected = 0.25 * xraydb.mu_elam("Al", 30000) + 0.75 * xraydb.mu_elam("Cu", 30000)
    old_bug = 0.25 ** 2 * xraydb.mu_elam("Al", 30000) + 0.75 ** 2 * xraydb.mu_elam("Cu", 30000)
    assert result["layers"][0]["reference"]["mu_mass_cm2_g"] == pytest.approx(expected)
    assert result["reference"]["transmission"] == pytest.approx(exp(-expected * 3 / 10))
    assert abs(expected - old_bug) / expected > 0.2


def test_duplicate_component_invariance():
    original = layer("SiO2", density=2.2)
    split = deepcopy(original)
    split["components"] = [{"formula": "SiO2", "fraction": 20}, {"formula": "SiO2", "fraction": 80}]
    assert calculate(request([split]))["reference"] == calculate(request([original]))["reference"]


def test_mass_weights_are_relative_not_required_percentages():
    a = layer()
    a["components"] = [{"formula": "SiO2", "fraction": 1}, {"formula": "B2O3", "fraction": 2}]
    b = deepcopy(a)
    for item in b["components"]:
        item["fraction"] *= 1e300
    assert calculate(request([a]))["reference"] == calculate(request([b]))["reference"]


def test_mole_to_mass_conversion_and_stoichiometry():
    mix = layer("SiO2", basis="mole")
    mix["components"] = [{"formula": "SiO2", "fraction": 2}, {"formula": "B2O3", "fraction": 1}]
    result = calculate(request([mix]))["layers"][0]
    m_silica = xraydb.atomic_mass("Si") + 2 * xraydb.atomic_mass("O")
    m_boron = 2 * xraydb.atomic_mass("B") + 3 * xraydb.atomic_mass("O")
    assert result["components"][0]["molar_mass_g_mol"] == pytest.approx(m_silica)
    assert result["components"][0]["mass_fraction"] == pytest.approx(2 * m_silica / (2 * m_silica + m_boron))
    assert sum(result["elemental_mass_fractions"].values()) == pytest.approx(1)


@pytest.mark.parametrize("formula,equivalent", [
    ("Ca5(PO4)3OH", "Ca5P3O13H"),
    ("Pb(Zr0.52Ti0.48)O3", "PbZr0.52Ti0.48O3"),
    ("Ca[Mg(OH)2]2", "CaMg2O4H4"),
    ("CuSO4·5H2O", "CuSO9H10"),
])
def test_chemical_formula_groups_fractional_occupancy_and_hydrates(formula, equivalent):
    a = calculate(request([layer(formula)]))["layers"][0]
    b = calculate(request([layer(equivalent)]))["layers"][0]
    assert a["elemental_mass_fractions"] == pytest.approx(b["elemental_mass_fractions"])
    assert a["reference"] == pytest.approx(b["reference"])


# Every rejection carries the message fragment that identifies *why* it was
# rejected. The engine signals all invalid input with ValueError, so a bare
# `pytest.raises(ValueError)` would also pass if the wrong guard fired.
@pytest.mark.parametrize("formula,message", [
    ("", "1–200 characters"),
    ("Xx2", "Unknown or unsupported element"),
    ("siO2", "Cannot parse formula"),
    ("SiO2 junk", "1–200 characters"),
    ("Ca(OH", "unmatched brackets"),
    ("Ca(OH]2", "unmatched brackets"),
    ("SiO2)", "unmatched brackets"),
    ("Si()2", "empty group"),
    ("Si0", "invalid atom count"),
    ("Si-2", "Cannot parse formula"),
    ("SiO2+", "Cannot parse formula"),
    ("CuSO4··5H2O", "empty hydrate segment"),
    ("Og", "Elam table"),
])
def test_malformed_or_unsupported_formulas_rejected(formula, message):
    with pytest.raises(ValueError, match=message):
        calculate(request([layer(formula)]))


def test_volume_basis_and_additive_density():
    mix = layer(basis="volume", density_mode="ideal", porosity=0.25)
    mix["components"] = [{"formula": "Al", "fraction": 1, "density_g_cm3": 2},
                         {"formula": "Cu", "fraction": 3, "density_g_cm3": 8}]
    result = calculate(request([mix]))["layers"][0]
    assert result["density_g_cm3"] == pytest.approx((0.25 * 2 + 0.75 * 8) * 0.75)
    assert result["components"][0]["mass_fraction"] == pytest.approx(2 / 26)


def test_mass_basis_ideal_density_uses_harmonic_rule():
    mix = layer(density_mode="ideal")
    mix["components"] = [{"formula": "Al", "fraction": 1, "density_g_cm3": 2},
                         {"formula": "Cu", "fraction": 1, "density_g_cm3": 8}]
    assert calculate(request([mix]))["layers"][0]["density_g_cm3"] == pytest.approx(1 / (0.5 / 2 + 0.5 / 8))


def test_bulk_vs_solid_density_prevents_porosity_double_counting():
    solid = calculate(request([layer(density=4, density_mode="solid", porosity=0.25)]))
    bulk = calculate(request([layer(density=3, density_mode="bulk", porosity=0)]))
    assert solid["reference"] == bulk["reference"]
    with pytest.raises(ValueError, match="already includes pores"):
        calculate(request([layer(density=3, density_mode="bulk", porosity=0.25)]))


def test_multilayer_optical_depth_additive_order_independent():
    layers = [layer("Al", 2.699, 0.4), layer("Cu", 8.96, 0.1), layer("SiO2", 2.2, 1.2)]
    stack = calculate(request(layers))
    reverse = calculate(request(list(reversed(layers))))
    singles = [calculate(request([item]))["reference"] for item in layers]
    assert stack["reference"]["optical_depth"] == pytest.approx(sum(s["optical_depth"] for s in singles))
    assert stack["reference"]["transmission"] == pytest.approx(np.prod([s["transmission"] for s in singles]))
    assert stack["reference"] == pytest.approx(reverse["reference"])


def test_oblique_path_measured_from_normal():
    result = calculate(request([layer(angle_deg=60)]))
    normal = calculate(request([layer(thickness=2)]))
    assert result["layers"][0]["path_length_mm"] == pytest.approx(2)
    assert result["reference"] == pytest.approx(normal["reference"])


def test_target_scales_all_stack_thicknesses():
    layers = [layer("Al", 2.699, 0.1), layer("Cu", 8.96, 0.05, angle_deg=35)]
    result = calculate(request(layers, target_transmission=0.0123))
    for item in layers:
        item["thickness_mm"] *= result["reference"]["target_thickness_scale"]
    assert calculate(request(layers))["reference"]["transmission"] == pytest.approx(0.0123)


def test_zero_thickness_transmits_everything_and_target_is_null():
    result = calculate(request([layer(thickness=0)]))
    assert result["transmission"] == [1.0] * len(result["energy_keV"])
    assert result["reference"]["target_thickness_scale"] is None
    assert result["reference"]["removed_fraction"] == 0
    json.dumps(result, allow_nan=False)


def test_edge_brackets_and_direct_reference_evaluation():
    payload = request([layer("Ba", 3.5, 0.1)])
    payload["energy"].update(min_keV=30, max_keV=45, points=2, reference_keV=37.445)
    result = calculate(payload)
    edge = next(e for e in result["edges"] if e["element"] == "Ba" and e["shell"] == "K")
    energy = np.array(result["energy_keV"])
    assert np.any(energy == edge["energy_keV"] * (1 - EDGE_BRACKET_RELATIVE))
    assert np.any(energy == edge["energy_keV"] * (1 + EDGE_BRACKET_RELATIVE))
    assert np.all(np.diff(energy) > 0)
    direct = xraydb.mu_elam("Ba", 37445)
    assert result["layers"][0]["reference"]["mu_mass_cm2_g"] == pytest.approx(direct)


def test_edge_brackets_cover_actual_elam_knots_for_all_supported_elements():
    # Independent inspection of backend data guards against a grid that
    # straddles nominal metadata while missing the actual tabulated jump.
    # Repeated log-energy knots represent the two sides of each Elam edge.
    database = xraydb.get_xraydb()
    checked = 0
    for atomic_number in range(1, 99):
        symbol = xraydb.atomic_symbol(atomic_number)
        row = database.get_cache("photoabsorption", column="element", value=symbol)[0]
        log_energy = np.array(json.loads(row.log_energy))
        steps = np.exp(log_energy[np.flatnonzero(np.diff(log_energy) == 0)])
        for edge in xraydb.xray_edges(symbol).values():
            if not 1000 <= edge.energy <= 800000:
                continue
            nearest_step = min(steps, key=lambda value: abs(value - edge.energy))
            assert edge.energy * (1 - EDGE_BRACKET_RELATIVE) < nearest_step
            assert nearest_step < edge.energy * (1 + EDGE_BRACKET_RELATIVE)
            checked += 1
    assert checked > 500


@pytest.mark.parametrize("symbol,edge_energy", [("Fe", 7.112), ("Ba", 37.441),
                                                ("W", 69.525), ("Pb", 88.005)])
def test_sweep_captures_both_sides_of_k_edge(symbol, edge_energy):
    payload = request([layer(symbol)])
    payload["energy"].update(min_keV=edge_energy * 0.99, max_keV=edge_energy * 1.01, points=2)
    result = calculate(payload)
    edge = next(item for item in result["edges"] if item["shell"] == "K")
    energies = np.asarray(result["energy_keV"])
    coefficients = np.asarray(result["layers"][0]["mu_mass_cm2_g"])
    below = coefficients[energies == edge["energy_keV"] * (1 - EDGE_BRACKET_RELATIVE)][0]
    above = coefficients[energies == edge["energy_keV"] * (1 + EDGE_BRACKET_RELATIVE)][0]
    assert above > 2 * below


def test_direct_reference_values_are_not_shifted_to_nominal_edge_side():
    payload = request([layer("Fe")])
    payload["energy"]["reference_keV"] = 7.112 * (1 + 1e-8)
    result = calculate(payload)
    exact_backend = xraydb.mu_elam("Fe", payload["energy"]["reference_keV"] * 1000)
    assert result["layers"][0]["reference"]["mu_mass_cm2_g"] == pytest.approx(exact_backend, rel=1e-14)


def test_channels_sum_to_total_and_sweep_is_json_safe():
    result = calculate(request([layer("PbO", 9.5, 0.4)]))
    data = result["layers"][0]
    channel_sum = np.array(data["photoelectric_cm2_g"]) + data["coherent_cm2_g"] + data["incoherent_cm2_g"]
    np.testing.assert_allclose(data["mu_mass_cm2_g"], channel_sum, rtol=1e-14)
    np.testing.assert_allclose(np.array(result["transmission"]) + result["removed_fraction"], 1)
    json.dumps(result, allow_nan=False)


def test_range_boundaries_and_high_energy_warning():
    payload = request()
    payload["energy"].update(min_keV=1, max_keV=800, reference_keV=800)
    result = calculate(payload)
    assert result["energy_keV"][0] == 1
    assert result["energy_keV"][-1] == 800
    assert any("250 keV" in warning for warning in result["warnings"])


@pytest.mark.parametrize("density,thickness,porosity", [
    (1e-12, 1e-300, np.nextafter(1.0, 0.0)),
    (1e-12, 1e9, 0),
    (1e4, 1e9, 0),
    (1e4, 0, np.nextafter(1.0, 0.0)),
])
def test_extreme_valid_inputs_keep_all_outputs_finite(density, thickness, porosity):
    payload = request([layer("Pb", density, thickness, density_mode="solid", porosity=porosity,
                             angle_deg=89.899999, density_uncertainty_pct=100,
                             thickness_uncertainty_pct=100)],
                      spectrum={"energy_keV": [1, 100, 800], "weights": [5e-324, 1e308, 1e308]},
                      uncertainty={"enabled": True, "samples": 100, "seed": 1},
                      target_transmission=5e-324)
    payload["energy"].update(min_keV=1, max_keV=800, reference_keV=800)
    result = calculate(payload)
    json.dumps(result, allow_nan=False)
    assert np.all(np.isfinite(result["optical_depth"]))
    assert all(0 <= value <= 1 for value in result["transmission"])
    assert result["spectrum"]["mean_energy_out_keV"] >= 1
    assert result["spectrum"]["mean_energy_out_keV"] <= 800


def test_unrepresentable_json_integer_rejected_as_validation_error():
    payload = request()
    payload["energy"]["reference_keV"] = 10 ** 400
    with pytest.raises(ValueError, match="finite"):
        calculate(payload)


@pytest.mark.parametrize("key,value,message", [
    ("min_keV", 0.99, "Minimum energy"),
    ("max_keV", 800.01, "Maximum energy"),
    ("reference_keV", 0, "Reference energy"),
    ("points", 1, "Energy points must be at least 2"),
    ("points", 2.5, "Energy points must be an integer"),
    ("points", True, "Energy points must be a finite number"),
    ("spacing", "cubic", "spacing must be log or linear"),
])
def test_invalid_energy_requests(key, value, message):
    payload = request()
    payload["energy"][key] = value
    with pytest.raises(ValueError, match=message):
        calculate(payload)


@pytest.mark.parametrize("key,value,message", [
    ("density_g_cm3", 0, "density"),
    ("density_g_cm3", float("inf"), "must be finite"),
    ("thickness_mm", -1, "thickness"),
    ("thickness_mm", float("nan"), "must be finite"),
    ("angle_deg", 90, "angle"),
    ("angle_deg", -90, "angle"),
    ("porosity", 1, "porosity must be less than 1"),
    ("density_mode", "guess", "density_mode must be bulk, solid, or ideal"),
    ("basis", "atom", "basis must be mass, mole, or volume"),
    ("density_uncertainty_pct", -2, "density uncertainty"),
])
def test_invalid_layer_inputs(key, value, message):
    item = layer()
    item[key] = value
    with pytest.raises(ValueError, match=message):
        calculate(request([item]))


@pytest.mark.parametrize("fraction,message", [
    (0, "at least one positive component fraction"),
    (-1, "must be at least 0"),
    (float("nan"), "must be finite"),
    (float("inf"), "must be finite"),
    (True, "must be a finite number"),
    ("100", "must be a finite number"),
])
def test_invalid_fractions(fraction, message):
    item = layer()
    item["components"][0]["fraction"] = fraction
    with pytest.raises(ValueError, match=message):
        calculate(request([item]))


@pytest.mark.parametrize("payload,message", [
    (None, "Request must be a JSON object"),
    ([], "Request must be a JSON object"),
    ({"layers": []}, "needs 1–24 layers"),
    ({"layers": [None]}, "Layer 1 must be a JSON object"),
    ({"layers": [layer()], "uncertainty": []}, "Uncertainty settings must be a JSON object"),
    ({"layers": [layer()], "target_transmission": 1}, "strictly between 0 and 1"),
    ({"layers": [layer()], "target_transmission": 0}, "must be greater than zero"),
    ({"layers": [layer()], "uncertainty": {"enabled": "yes"}}, "must be true or false"),
])
def test_invalid_request_shapes(payload, message):
    with pytest.raises(ValueError, match=message):
        calculate(payload)


@pytest.mark.parametrize("basis,mode", [("volume", "bulk"), ("mass", "ideal"), ("mole", "ideal")])
def test_component_densities_required(basis, mode):
    with pytest.raises(ValueError, match="component density"):
        calculate(request([layer(basis=basis, density_mode=mode)]))


def test_discrete_spectrum_exact_evaluation_and_energy_weighting():
    bins = [17, 29, 73]
    weights = [1, 3, 2]
    spectrum = {"energy_keV": bins, "weights": weights, "weighting": "photon", "label": "Test bins"}
    payload = request(spectrum=spectrum)
    result = calculate(payload)
    direct_t = np.exp(-np.asarray(xraydb.mu_elam("Al", np.array(bins) * 1000)) * 2.699 / 10)
    expected = np.dot(weights, direct_t) / sum(weights)
    assert result["spectrum"]["transmission"] == pytest.approx(expected)
    np.testing.assert_allclose(result["spectrum"]["transmitted_weights"], np.array(weights) * direct_t)
    assert result["spectrum"]["mean_energy_in_keV"] == pytest.approx(np.average(bins, weights=weights))
    assert result["spectrum"]["mean_energy_out_keV"] == pytest.approx(np.average(bins, weights=weights * direct_t))
    assert result["spectrum"]["mean_energy_out_keV"] > result["spectrum"]["mean_energy_in_keV"]
    spectrum["weighting"] = "energy"
    result_energy = calculate(payload)
    expected_energy = np.dot(np.array(weights) * bins, direct_t) / np.dot(weights, bins)
    assert result_energy["spectrum"]["transmission"] == pytest.approx(expected_energy)
    assert result_energy["spectrum"]["mean_energy_out_keV"] == result["spectrum"]["mean_energy_out_keV"]


def test_monochromatic_spectrum_matches_reference():
    result = calculate(request(spectrum={"energy_keV": [30], "weights": [123.4]}))
    assert result["spectrum"]["transmission"] == result["reference"]["transmission"]


def test_spectrum_underflow_and_extreme_finite_weights_remain_json_safe():
    result = calculate(request([layer("Pb", 11.35, 10000)],
                       spectrum={"energy_keV": [10, 20, 100], "weights": [1e308, 1e308, 1e307]}))
    assert result["spectrum"]["transmission"] == 0
    assert result["spectrum"]["mean_energy_out_keV"] == pytest.approx(100)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("spectrum,message", [
    ([], "Spectrum must be a JSON object"),
    ({}, "equally sized energy_keV and weights"),
    ({"energy_keV": [10], "weights": []}, "equally sized energy_keV and weights"),
    ({"energy_keV": [10], "weights": [0]}, "at least one positive photon-fluence weight"),
    ({"energy_keV": [900], "weights": [1]}, "Spectrum energy 1"),
    ({"energy_keV": [10], "weights": [-1]}, "Spectrum weight 1"),
    ({"energy_keV": [10], "weights": [1], "weighting": "dose"}, "weighting must be photon or energy"),
])
def test_invalid_spectrum(spectrum, message):
    with pytest.raises(ValueError, match=message):
        calculate(request(spectrum=spectrum))


def test_uncertainty_reproducibility_bounds_and_seed_sensitivity():
    payload = request([layer(density_uncertainty_pct=8, thickness_uncertainty_pct=5)],
                      uncertainty={"enabled": True, "samples": 1000, "seed": 123})
    first = calculate(payload)
    repeat = calculate(payload)
    assert first["uncertainty"] == repeat["uncertainty"]
    bands = first["uncertainty"]
    low, high = np.array(bands["transmission_p025"]), np.array(bands["transmission_p975"])
    assert np.all(low >= 0)
    assert np.all(low <= high)
    assert np.all(high <= 1)
    assert bands["reference"]["p025"] < first["reference"]["transmission"] < bands["reference"]["p975"]
    assert bands["reference"]["std"] > 0
    payload["uncertainty"]["seed"] = 124
    assert calculate(payload)["uncertainty"]["reference"] != bands["reference"]


def test_uncertainty_log_normal_quantiles_match_analytic_density_case():
    from scipy.stats import norm
    payload = request([layer(density_uncertainty_pct=20)],
                      uncertainty={"enabled": True, "samples": 20000, "seed": 14})
    result = calculate(payload)
    tau = result["reference"]["optical_depth"]
    sigma = np.sqrt(np.log1p(0.2 ** 2))
    expected_lower = np.exp(-tau * np.exp(-0.5 * sigma ** 2 + sigma * norm.ppf(0.975)))
    expected_upper = np.exp(-tau * np.exp(-0.5 * sigma ** 2 + sigma * norm.ppf(0.025)))
    assert result["uncertainty"]["reference"]["p025"] == pytest.approx(expected_lower, abs=0.004)
    assert result["uncertainty"]["reference"]["p975"] == pytest.approx(expected_upper, abs=0.004)


def test_zero_uncertainty_reduces_to_deterministic_result():
    result = calculate(request(uncertainty={"enabled": True, "samples": 100, "seed": 1}))
    np.testing.assert_allclose(result["uncertainty"]["transmission_p025"], result["transmission"], rtol=1e-15)
    np.testing.assert_allclose(result["uncertainty"]["transmission_p975"], result["transmission"], rtol=1e-15)
    assert result["uncertainty"]["reference"]["std"] < 1e-14


def test_presets_are_independent_fresh_objects_and_all_calculate():
    first = presets()
    first[0]["layer"]["components"][0]["fraction"] = -2
    assert presets()[0]["layer"]["components"][0]["fraction"] > 0
    for preset in presets():
        result = calculate(request([preset["layer"]]))
        json.dumps(result, allow_nan=False)
        assert 0 <= result["reference"]["transmission"] <= 1
