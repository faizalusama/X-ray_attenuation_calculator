"""Reproduce the bounded calculation review: python -m paper.validation_check.

Run from the repository root after installing the Workbench dependencies.
Uses manual atom dictionaries and scalar algebra, but the same XrayDB data.
"""
import copy
import json
import math
import platform
from pathlib import Path
from statistics import NormalDist

import numpy as np
import xraydb

from xray_workbench.physics import calculate

ROOT = Path(__file__).resolve().parents[1]

ATOMS = {"Al": {"Al": 1}, "Fe": {"Fe": 1}, "Pb": {"Pb": 1},
         "H2O": {"H": 2, "O": 1}, "SiO2": {"Si": 1, "O": 2},
         "B2O3": {"B": 2, "O": 3}, "Na2O": {"Na": 2, "O": 1},
         "Al2O3": {"Al": 2, "O": 3}, "Ca(OH)2": {"Ca": 1, "O": 2, "H": 2},
         "CuSO4·5H2O": {"Cu": 1, "S": 1, "O": 9, "H": 10}}


def prepared(layer):
    components = layer["components"]
    molar = [math.fsum(xraydb.atomic_mass(s) * n for s, n in ATOMS[c["formula"]].items()) for c in components]
    parts = [c["fraction"] for c in components]
    if layer.get("basis", "mass") == "mole":
        parts = [v * mass for v, mass in zip(parts, molar, strict=True)]
    if layer.get("basis", "mass") == "volume":
        parts = [v * c["density_g_cm3"] for v, c in zip(parts, components, strict=True)]
    weights = [v / math.fsum(parts) for v in parts]
    elements = {}
    for component, mass, weight in zip(components, molar, weights, strict=True):
        for symbol, count in ATOMS[component["formula"]].items():
            elements[symbol] = elements.get(symbol, 0) + weight * count * xraydb.atomic_mass(symbol) / mass
    mode = layer.get("density_mode", "bulk")
    density = layer.get("density_g_cm3")
    if mode == "ideal":
        density = 1 / math.fsum(w / c["density_g_cm3"] for w, c in zip(weights, components, strict=True))
    if mode in ("solid", "ideal"):
        density *= 1 - layer.get("porosity", 0)
    path_cm = (layer.get("thickness_mm", 1) / 10) / math.cos(math.radians(layer.get("angle_deg", 0)))
    return elements, density, path_cm, weights


def scalar(layer, energy):
    elements, density, path_cm, weights = prepared(layer)
    channels = [math.fsum(w * float(xraydb.mu_elam(s, energy * 1000, kind=k)) for s, w in elements.items()) for k in ("photo", "coh", "incoh")]
    mass_mu = math.fsum(channels)
    return {"elements": elements, "density": density, "path_cm": path_cm,
            "weights": weights, "mass_mu": mass_mu, "mu": mass_mu * density,
            "tau": mass_mu * density * path_cm, "channels": channels}


def request(layers, reference=30):
    return {"layers": copy.deepcopy(layers), "energy": {"min_keV": 5, "max_keV": 800,
            "points": 17, "spacing": "log", "reference_keV": reference}, "target_transmission": .1}


def layer(formula, density=1, thickness=1, **kw):
    return {"basis": "mass", "components": [{"formula": formula, "fraction": 1}],
            "density_g_cm3": density, "thickness_mm": thickness, **kw}


def near(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-14), (actual, expected)


oxide = {"name": "Constructed mole mixture", "basis": "mole", "components": [
    {"formula": "SiO2", "fraction": 80}, {"formula": "B2O3", "fraction": 15},
    {"formula": "Na2O", "fraction": 5}], "density_g_cm3": 2.23, "thickness_mm": 1}
equal = {"basis": "mass", "components": [{"formula": "Al", "fraction": .5},
         {"formula": "Fe", "fraction": .5}], "density_g_cm3": 4, "thickness_mm": .4}
mass_ideal = {"basis": "mass", "density_mode": "ideal", "porosity": .1, "components": [
    {"formula": "SiO2", "fraction": .7, "density_g_cm3": 2.2},
    {"formula": "Al2O3", "fraction": .3, "density_g_cm3": 3.95}], "thickness_mm": .5}
volume_ideal = {"basis": "volume", "density_mode": "ideal", "porosity": .12, "components": [
    {"formula": "SiO2", "fraction": .6, "density_g_cm3": 2.2},
    {"formula": "Al2O3", "fraction": .4, "density_g_cm3": 3.95}], "thickness_mm": .8}
cases = {"pure_aluminium": [layer("Al", 2.699, .3)], "equal_mass_Al_Fe": [equal],
         "oxide_mole_parts": [oxide], "mass_ideal_density": [mass_ideal],
         "volume_ideal_density": [volume_ideal],
         "solid_porosity": [layer("Al2O3", 3.95, .75, density_mode="solid", porosity=.25, angle_deg=15)],
         "parenthesized_formula": [layer("Ca(OH)2", 2.2, .2)],
         "hydrate_formula": [layer("CuSO4·5H2O", 2.2, .2)],
         "zero_thickness": [layer("Al", 2.699, 0)],
         "oblique_two_layer": [layer("Al", 2.7, .2, angle_deg=45), layer("Fe", 7.8, .05, angle_deg=45)]}
case_results = []
max_mu_relative = max_tau_relative = max_transmission_absolute = 0
evaluations = 0
for name, layers in cases.items():
    result = calculate(request(layers))
    for index, energy in enumerate(result["energy_keV"]):
        expected_layers = [scalar(specimen, energy) for specimen in layers]
        expected_tau = math.fsum(e["tau"] for e in expected_layers)
        near(result["optical_depth"][index], expected_tau)
        expected_t = math.exp(-expected_tau)
        near(result["transmission"][index], expected_t)
        max_transmission_absolute = max(max_transmission_absolute, abs(result["transmission"][index] - expected_t))
        if expected_tau:
            max_tau_relative = max(max_tau_relative, abs(result["optical_depth"][index] - expected_tau) / expected_tau)
        for found, expected in zip(result["layers"], expected_layers, strict=True):
            near(found["mu_mass_cm2_g"][index], expected["mass_mu"])
            max_mu_relative = max(max_mu_relative, abs(found["mu_mass_cm2_g"][index] - expected["mass_mu"]) / expected["mass_mu"])
            near(found["density_g_cm3"], expected["density"])
            near(found["path_length_mm"], expected["path_cm"] * 10)
            for e, w in expected["elements"].items():
                near(found["elemental_mass_fractions"][e], w)
            for component, weight in zip(found["components"], expected["weights"], strict=True):
                near(component["mass_fraction"], weight)
        evaluations += 1
    for found, specimen in zip(result["layers"], layers, strict=True):
        mu = scalar(specimen, 30)["mu"]
        near(found["reference"]["hvl_mm"], 10 * math.log(2) / mu)
        near(found["reference"]["tvl_mm"], 10 * math.log(10) / mu)
        near(found["reference"]["attenuation_length_mm"], 10 / mu)
        near(math.exp(-mu * found["reference"]["hvl_mm"] / 10), .5)
    if result["reference"]["target_thickness_scale"] is not None:
        scaled = copy.deepcopy(layers)
        for specimen in scaled:
            specimen["thickness_mm"] *= result["reference"]["target_thickness_scale"]
        near(calculate(request(scaled))["reference"]["transmission"], .1)
    case_results.append({"name": name, "sweep_samples": len(result["energy_keV"]), "reference_transmission": result["reference"]["transmission"], "effective_densities": [specimen["density_g_cm3"] for specimen in result["layers"]]})

# Independent tabulated values, read from the NIST total attenuation column.
nist = {"Al": [3.441, .3681, .1704], "Fe": [25.68, 1.958, .3717],
        "Pb": [86.36, 8.041, 5.549], "H2O": [.8096, .2269, .1707]}
nist_results = []
for formula, values in nist.items():
    for energy, tabulated in zip((20, 50, 100), values, strict=True):
        found = calculate(request([layer(formula)], energy))["layers"][0]["reference"]["mu_mass_cm2_g"]
        relative = (found - tabulated) / tabulated
        assert abs(relative) < .01
        nist_results.append({"formula": formula, "energy_keV": energy, "nist_cm2_g": tabulated, "calculated_cm2_g": found, "relative_difference_percent": 100 * relative})

edge_results = []
for direction, tabulated in ((-1, 53.19), (1, 407.6)):
    energy = 7.112 * (1 + direction * 1e-4)
    found = calculate(request([layer("Fe")], energy))["layers"][0]["reference"]["mu_mass_cm2_g"]
    relative = (found - tabulated) / tabulated
    assert abs(relative) < .01
    edge_results.append({"energy_keV": energy, "nist_limit_cm2_g": tabulated,
                         "calculated_cm2_g": found, "relative_difference_percent": 100 * relative})

base = calculate(request([equal], 50))
old_mass_mu = .25 * sum(float(xraydb.mu_elam(s, 50_000, kind="total")) for s in ("Al", "Fe"))
correct_mass_mu = scalar(equal, 50)["mass_mu"]
near(old_mass_mu / correct_mass_mu, .5)
legacy_case = {"energy_keV": 50, "correct_mass_mu": correct_mass_mu, "old_double_weighted_mass_mu": old_mass_mu,
               "correct_transmission": base["reference"]["transmission"], "old_transmission": math.exp(-old_mass_mu * 4 * .4 / 10)}

# Layer ordering and subdivision identities, and oblique/normal path equality.
two = [layer("Al", 2.7, .2), layer("Fe", 7.8, .05)]
near(calculate(request(two))["reference"]["transmission"], calculate(request(list(reversed(two))))["reference"]["transmission"])
split = [layer("Al", 2.699, .15), layer("Al", 2.699, .15)]
near(calculate(request(split))["reference"]["transmission"], calculate(request([layer("Al", 2.699, .3)]))["reference"]["transmission"])
near(calculate(request([layer("Fe", 7.8, .4, angle_deg=60)]))["reference"]["transmission"], calculate(request([layer("Fe", 7.8, .8)]))["reference"]["transmission"])

example = json.loads((ROOT / "examples/borosilicate-multilayer.json").read_text(encoding="utf-8"))["configuration"]
result = calculate(example)
spectrum = example["spectrum"]
transmitted = [math.exp(-math.fsum(scalar(specimen, e)["tau"] for specimen in example["layers"])) for e in spectrum["energy_keV"]]
photon_t = math.fsum(w * t for w, t in zip(spectrum["weights"], transmitted, strict=True)) / math.fsum(spectrum["weights"])
energy_t = math.fsum(w * e * t for w, e, t in zip(spectrum["weights"], spectrum["energy_keV"], transmitted, strict=True)) / math.fsum(w * e for w, e in zip(spectrum["weights"], spectrum["energy_keV"], strict=True))
mean_in = math.fsum(w * e for w, e in zip(spectrum["weights"], spectrum["energy_keV"], strict=True)) / math.fsum(spectrum["weights"])
mean_out = math.fsum(w * e * t for w, e, t in zip(spectrum["weights"], spectrum["energy_keV"], transmitted, strict=True)) / math.fsum(w * t for w, t in zip(spectrum["weights"], transmitted, strict=True))
near(result["spectrum"]["transmission"], photon_t)
near(result["spectrum"]["mean_energy_in_keV"], mean_in)
near(result["spectrum"]["mean_energy_out_keV"], mean_out)
energy_config = copy.deepcopy(example)
energy_config["spectrum"]["weighting"] = "energy"
energy_result = calculate(energy_config)
near(energy_result["spectrum"]["transmission"], energy_t)

# An analytic quantile check for a single independent density uncertainty.
uncertain_layer = layer("Al", 2.699, 1, density_uncertainty_pct=10)
uncertain = request([uncertain_layer])
uncertain["uncertainty"] = {"enabled": True, "samples": 20000, "seed": 14}
uncertain_result = calculate(uncertain)
tau = scalar(uncertain_layer, 30)["tau"]
sigma = math.sqrt(math.log(1.01))
analytic = [math.exp(-tau * math.exp(-.5 * sigma * sigma + sigma * NormalDist().inv_cdf(1-q))) for q in (.025, .975)]
sampled = [uncertain_result["uncertainty"]["reference"][key] for key in ("p025", "p975")]
assert max(abs(a-b) for a,b in zip(analytic,sampled, strict=True)) < .005

report = {"python": platform.python_version(), "numpy": np.__version__, "xraydb": xraydb.__version__,
          "cases": case_results, "sweep_energy_evaluations": evaluations,
          "max_mass_coefficient_relative_error": max_mu_relative,
          "max_optical_depth_relative_error": max_tau_relative,
          "max_transmission_absolute_error": max_transmission_absolute,
          "nist_comparisons": nist_results, "fe_k_edge_brackets": edge_results, "legacy_constructed_case": legacy_case,
          "example": {"engine_photon_weighted_transmission": result["spectrum"]["transmission"],
                      "scalar_photon_weighted_transmission": photon_t, "energy_weighted_transmission": energy_t,
                      "incident_mean_energy_keV": mean_in, "transmitted_mean_energy_keV": mean_out,
                      "reference_transmission": result["reference"]["transmission"],
                      "target_scale": result["reference"]["target_thickness_scale"]},
          "uncertainty_quantiles": {"analytic": analytic, "sampled": sampled,
                                  "max_absolute_difference": max(abs(a-b) for a,b in zip(analytic,sampled, strict=True))}}
Path(__file__).with_name("calculation-review-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
