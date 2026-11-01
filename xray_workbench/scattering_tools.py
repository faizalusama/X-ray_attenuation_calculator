"""Optics and scattering utilities, separate from narrow-beam attenuation.

Angles here are explicitly grazing angles from the surface when so labelled.
SLD uses rho_x = r_e sum N_j (Z_j + f'_j - i f''_j).
The wave convention is exp(i kz z), n = 1 - delta + i beta.
"""
from __future__ import annotations

from math import asin, cos, pi, radians, sin, sqrt
from typing import Any

import numpy as np
import xraydb

from . import __version__
from ._types import JsonObject
from .physics import _number, _object, _prepare_layer
from .scattering import AVOGADRO, _table
from .sources import HC_KEV_ANGSTROM

R_E_ANGSTROM = 2.8179403262e-5


def optical_constants(raw: Any, energy: float) -> JsonObject:
    layer = _prepare_layer(raw, 0)
    real, imag = 0.0, 0.0
    for symbol, weight in layer.elements.items():
        if int(xraydb.atomic_number(symbol)) > 92:
            raise ValueError("Chantler optical constants are available only through uranium.")
        atoms = layer.density * AVOGADRO * weight / float(xraydb.atomic_mass(symbol)) * 1e-24
        real += atoms * (int(xraydb.atomic_number(symbol)) + float(xraydb.f1_chantler(symbol, energy * 1000)))
        imag += atoms * float(xraydb.f2_chantler(symbol, energy * 1000))
    wavelength = HC_KEV_ANGSTROM / energy
    factor = R_E_ANGSTROM * wavelength ** 2 / (2 * pi)
    delta, beta = real * factor, imag * factor
    return {"name": layer.name, "delta": delta, "beta": beta,
            "sld_real_inv_angstrom2": real * R_E_ANGSTROM,
            "sld_absorptive_inv_angstrom2": imag * R_E_ANGSTROM,
            "critical_angle_deg": sqrt(2 * delta) * 180 / pi if delta > 0 else None,
            "wavelength_angstrom": wavelength}


def plot(title: str, xlabel: str, ylabel: str, x: np.ndarray, traces: dict[str, np.ndarray], log_y: bool = False) -> JsonObject:
    return {"title": title, "xlabel": xlabel, "ylabel": ylabel, "x": x.tolist(),
            "traces": [{"name": name, "y": values.tolist()} for name, values in traces.items()], "log_y": log_y}


def result(tool: str, data: JsonObject, values: list[tuple[str, Any, str]], plots: list[JsonObject],
           notes: list[str], tables: list[JsonObject] | None = None) -> JsonObject:
    return {"tool": tool, "data": data, "values": [{"label": label, "value": value, "unit": unit} for label, value, unit in values],
            "plots": plots, "notes": notes, "tables": tables or [],
            "provenance": {"calculator_version": __version__, "optical_data": "Chantler through XrayDB " + xraydb.__version__}}


def optics(payload: JsonObject) -> JsonObject:
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    constants = optical_constants(payload.get("layer"), energy)
    maximum = _number(payload.get("angle_max_deg", 5), "Maximum grazing angle (degrees from surface)", 0.01, 90)
    angles = np.linspace(0, maximum, 401)
    n = 1 - constants["delta"] + 1j * constants["beta"]
    kz = 2 * pi / constants["wavelength_angstrom"] * np.sqrt(n*n - np.cos(np.radians(angles))**2 + 0j)
    depth = np.divide(1, 20 * kz.imag, out=np.full_like(angles, np.nan), where=kz.imag > 0)  # intensity 1/e depth, Å -> nm
    if not np.all(np.isfinite(depth)):
        raise ValueError("Penetration depth is undefined for nonabsorbing or nonphysical optical constants.")
    values = [("δ", constants["delta"], ""), ("β", constants["beta"], ""),
              ("Real X-ray SLD", constants["sld_real_inv_angstrom2"], "Å⁻²"),
              ("Absorptive X-ray SLD magnitude", constants["sld_absorptive_inv_angstrom2"], "Å⁻²"),
              ("Approximate critical grazing angle", constants["critical_angle_deg"], "° from surface")]
    return result("optics", constants, values,
                  [plot("Intensity penetration depth", "Grazing angle (° from surface)", "Normal 1/e depth (nm)", angles, {"Depth": depth}, True)],
                  ["Chantler independent-atom optical constants; no chemical-state near-edge structure.",
                   "Intensity depth = 1 / [2 Im(kz)]. A semi-infinite homogeneous material, with angles from the SURFACE.",
                   "This optical model is separate from the Elam narrow-beam attenuation engine."])


def parratt(indices: list[complex], thickness: list[float], roughness: list[float],
            angles: np.ndarray, wavelength: float, polarization: str = "s") -> np.ndarray:
    """Vacuum incident medium; first/last media semi-infinite. Thickness and sigma in Å."""
    n = np.asarray(indices, dtype=complex)
    kz = 2*pi/wavelength * np.sqrt(n[:, None]**2 - np.cos(np.radians(angles))[None, :]**2 + 0j)
    amplitude = np.zeros_like(angles, dtype=complex)
    for j in range(len(n)-2, -1, -1):
        a, b = (kz[j], kz[j+1]) if polarization == "s" else (n[j+1]**2*kz[j], n[j]**2*kz[j+1])
        interface = np.divide(a-b, a+b, out=np.zeros_like(a), where=abs(a+b) > 0)
        interface *= np.exp(-2 * kz[j] * kz[j+1] * roughness[j]**2)
        phase = np.exp(2j * kz[j+1] * thickness[j+1])
        amplitude = (interface + amplitude * phase) / (1 + interface * amplitude * phase)
    reflectance: np.ndarray = np.abs(amplitude)**2
    if not np.all(np.isfinite(reflectance)) or np.any(reflectance > 1 + 1e-8):
        raise ValueError("Roughness model produced nonphysical reflectivity; reduce roughness or use a graded-interface model.")
    return reflectance


def reflectivity(payload: JsonObject) -> JsonObject:
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not 1 <= len(raw_layers) <= 24:
        raise ValueError("Reflectivity needs 1–24 layers, with the last as semi-infinite substrate.")
    constants = [optical_constants(raw, energy) for raw in raw_layers]
    overrides = payload.get("film_thickness_angstrom", "")
    if not isinstance(overrides, str) or len(overrides) > 1000:
        raise ValueError("Film thicknesses must be a comma-separated list in Å.")
    films = ([_number(float(x), "Film thickness (Å)", 0, 1e7) for x in overrides.split(",")] if overrides.strip() else
             [_number(_prepare_layer(raw, i).thickness * 1e7, "Film thickness (Å)", 0, 1e7) for i, raw in enumerate(raw_layers[:-1])])
    if len(films) != len(constants)-1:
        raise ValueError("Supply one film thickness for each layer before the substrate.")
    sigma = _number(payload.get("roughness_angstrom", 0), "Interface RMS roughness (Å)", 0, 100)
    maximum = _number(payload.get("angle_max_deg", 5), "Maximum grazing angle (° from surface)", 0.01, 45)
    polarization = payload.get("polarization", "s")
    if polarization not in ("s", "p"):
        raise ValueError("Polarization must be s or p.")
    angles = np.linspace(0.001, maximum, 1001)
    wavelength = HC_KEV_ANGSTROM / energy
    indices = [1+0j, *[complex(1-c["delta"], c["beta"]) for c in constants]]
    curve = parratt(indices, [0, *films, 0], [sigma]*len(constants), angles, wavelength, polarization)
    notes = ["Incident medium is vacuum. Stack order runs from the top film to the LAST layer as a semi-infinite substrate.",
             "Parratt recursion, " + polarization + " polarization; Névot–Croce RMS roughness is the same at every interface.",
             "No resolution convolution, footprint, diffuse scattering or fitting. Sample fringes more finely before interpreting thick films."]
    if any(sigma > d/5 for d in films if d > 0):
        notes.append("Roughness exceeds one fifth of a film thickness: interfaces may overlap; a graded-interface model is preferable.")
    return result("reflectivity", {"layers": constants, "film_thickness_angstrom": films, "roughness_angstrom": sigma},
                  [("Number of films", len(films), ""), ("Substrate", constants[-1]["name"], ""),
                   ("Energy", energy, "keV")],
                  [plot("Specular multilayer reflectivity", "Grazing angle (° from surface)", "Reflectivity", angles, {"R": curve}, True)],
                  notes)


def neutron_sld(raw: Any) -> complex | None:
    layer = _prepare_layer(raw, 0)
    values = _table("neutron_sears1992.json")["values"]
    if any(s not in values for s in layer.elements):
        return None
    return sum((layer.density * AVOGADRO * weight / float(xraydb.atomic_mass(symbol)) * 1e-29 *
                complex(values[symbol]["b_coherent_fm"], values[symbol]["b_coherent_imaginary_fm"])
                for symbol, weight in layer.elements.items()), start=0j)


def contrast(payload: JsonObject) -> JsonObject:
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    layers = payload.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 24:
        raise ValueError("Contrast needs 1–24 layer compositions.")
    index = _number(payload.get("other_layer", min(1, len(layers)-1)), "Second phase index", 0, len(layers)-1)
    if not index.is_integer():
        raise ValueError("Second phase index must be an integer.")
    first_raw, second_raw = payload.get("layer"), layers[int(index)]
    first, second = optical_constants(first_raw, energy), optical_constants(second_raw, energy)
    phi = _number(payload.get("volume_fraction", 0.5), "First phase volume fraction", 0, 1)
    x1 = complex(first["sld_real_inv_angstrom2"], -first["sld_absorptive_inv_angstrom2"])
    x2 = complex(second["sld_real_inv_angstrom2"], -second["sld_absorptive_inv_angstrom2"])
    dx = abs(x1-x2)**2
    n1, n2 = neutron_sld(first_raw), neutron_sld(second_raw)
    dn = abs(n1-n2)**2 if n1 is not None and n2 is not None else None
    fraction = np.linspace(0, 1, 201)
    scale = 2*pi*pi*fraction*(1-fraction)
    data = {"xray_contrast_squared_inv_angstrom4": dx, "neutron_contrast_squared_inv_angstrom4": dn,
            "xray_porod_invariant_inv_angstrom4": 2*pi*pi*phi*(1-phi)*dx,
            "neutron_porod_invariant_inv_angstrom4": 2*pi*pi*phi*(1-phi)*dn if dn is not None else None}
    traces = {"X-ray": scale*dx}
    if dn is not None:
        traces["Neutron"] = scale*dn
    return result("contrast", data,
                  [("X-ray |ΔSLD|²", dx, "Å⁻⁴"), ("Neutron |ΔSLD|²", dn, "Å⁻⁴"),
                   ("X-ray Porod invariant", data["xray_porod_invariant_inv_angstrom4"], "Å⁻⁴"),
                   ("Neutron Porod invariant", data["neutron_porod_invariant_inv_angstrom4"], "Å⁻⁴")],
                  [plot("Two-phase Porod invariant", "First phase volume fraction", "Integral q² I(q) dq (Å⁻⁴)", fraction, traces)],
                  ["Phases: " + first["name"] + " / " + second["name"] + ". Phase fractions are independent of stack thickness.",
                   "Q_P = 2π² φ(1−φ)|ΔSLD|² for a statistically isotropic, homogeneous, sharp two-phase system in absolute scattering units.",
                   "Natural-isotope nuclear neutron contrast only; magnetic scattering, polydispersity and finite-q truncation are not modeled."])


def converter(payload: JsonObject) -> JsonObject:
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    wavelength = HC_KEV_ANGSTROM / energy
    quantity = payload.get("quantity", "two_theta_deg")
    value = _number(payload.get("value", 28.44), "Conversion input", 0, 1e9)
    if quantity == "two_theta_deg":
        if value > 180:
            raise ValueError("2θ must lie between 0 and 180 degrees.")
        q = 4*pi*sin(radians(value)/2)/wavelength
    elif quantity == "q":
        q = value
    elif quantity == "d":
        if value <= 0:
            raise ValueError("d spacing must be positive.")
        q = 2*pi/value
    elif quantity == "s":
        q = 4*pi*value
    else:
        raise ValueError("Quantity must be two_theta_deg, q, d or s.")
    def angle_at(e: float) -> float | None:
        reach = q*HC_KEV_ANGSTROM/e/(4*pi)
        return 2*asin(reach)*180/pi if reach <= 1 else None
    angle = angle_at(energy)
    target = _number(payload.get("target_energy_keV", 17.479), "Target energy (keV)", 1, 800)
    width = _number(payload.get("fwhm_deg", 0), "Observed FWHM (° in 2θ)", 0, 20)
    instrument = _number(payload.get("instrument_fwhm_deg", 0), "Instrument FWHM (° in 2θ)", 0, 20)
    shape = _number(payload.get("shape_factor", 0.9), "Scherrer shape factor", 0.01, 2)
    if instrument > width:
        raise ValueError("Instrument FWHM exceeds observed FWHM.")
    breadth = radians(sqrt(width**2-instrument**2))
    denominator = breadth*cos(radians(angle or 0)/2)
    size = shape*wavelength/denominator/10 if angle is not None and 0 < angle < 180 and denominator > 1e-15 else None
    data = {"q_inv_angstrom": q, "s_inv_angstrom": q/(4*pi), "d_angstrom": 2*pi/q if q > 0 else None,
            "two_theta_deg": angle, "target_two_theta_deg": angle_at(target), "scherrer_size_nm": size}
    return result("converter", data,
                  [("Q", q, "Å⁻¹"), ("s = sinθ/λ = Q/(4π)", q/(4*pi), "Å⁻¹"), ("d spacing", data["d_angstrom"], "Å"),
                   ("2θ at reference energy", angle, "°"), ("2θ at target energy", data["target_two_theta_deg"], "°"),
                   ("Scherrer coherent-domain size", size, "nm")], [],
                  ["Q = 4πsinθ/λ = 2π/d. Bragg order n=1. Unreachable angles and d at Q=0 are undefined.",
                   "Scherrer L = Kλ/(β cosθ); β is FWHM in 2θ radians after Gaussian quadrature subtraction of instrumental width.",
                   "Size is a coherent-domain estimate, not particle size. Strain, size distributions and profile-shape effects are excluded."])


def scattering_tools(payload: Any) -> JsonObject:
    payload = _object(payload, "Scattering tools request")
    if payload.get("tool") == "converter":
        return converter(payload)
    if payload.get("tool") == "contrast":
        return contrast(payload)
    if payload.get("tool") == "reflectivity":
        return reflectivity(payload)
    if payload.get("tool", "optics") == "optics":
        return optics(payload)
    raise ValueError("Unknown scattering tool.")
