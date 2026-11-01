"""Optics and scattering utilities, separate from narrow-beam attenuation.

Angles here are explicitly grazing angles from the surface when so labelled.
SLD uses rho_x = r_e sum N_j (Z_j + f'_j - i f''_j).
The wave convention is exp(i kz z), n = 1 - delta + i beta.
"""
from __future__ import annotations

from math import pi, sqrt
from typing import Any

import numpy as np
import xraydb

from . import __version__
from ._types import JsonObject
from .physics import _number, _object, _prepare_layer
from .scattering import AVOGADRO
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


def scattering_tools(payload: Any) -> JsonObject:
    payload = _object(payload, "Scattering tools request")
    if payload.get("tool", "optics") == "optics":
        return optics(payload)
    raise ValueError("Unknown scattering tool.")
