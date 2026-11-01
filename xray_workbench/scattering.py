"""Planning quantities for an X-ray total-scattering (pair distribution function) experiment.

Nothing here reduces measured data. For a material and an incident energy it reports the
accessible Q range, the number density, the composition-weighted form factors and the
Faber-Ziman pair weights, the absorption of a capillary, and likely fluorescence.

Composition, density and attenuation come from the same code as the attenuation calculator.
Form factors are the neutral-atom fits of Waasmaier & Kirfel (doi:10.1107/S0108767394013292),
valid for sinθ/λ from 0 to 6 Å⁻¹, and f′ and f″ are from Chantler (doi:10.1063/1.1321055),
both through XrayDB. Pair weights use f₀ alone, so they do not apply close to an absorption edge.
"""

from __future__ import annotations

import json
from functools import lru_cache
from math import asin, cos, exp, pi, radians, sin
from pathlib import Path
from typing import Any

import numpy as np
import xraydb

from . import __version__
from ._types import JsonObject
from .backends import get_backend
from .physics import _number, _object, _prepare_layer
from .sources import HC_KEV_ANGSTROM, xray_sources

#: Exact SI value, per mole.
AVOGADRO = 6.02214076e23
#: h/(mₑc) in Å, CODATA 2018.
COMPTON_WAVELENGTH_ANGSTROM = 0.0242631023867
#: Upper limit of the form-factor fits: 4π × 6 Å⁻¹.
Q_LIMIT = 4 * pi * 6.0
#: Full width at half maximum of the peak that a finite Qmax imposes, times Qmax. Without a
#: window it is that of sin(x)/x; with a Lorch window it is computed in tests/test_scattering.py.
FWHM_STEP, FWHM_LORCH = 3.7910, 5.4366
#: Energies shown beside the laboratory lines. Examples only, not any particular beamline.
EXAMPLE_ENERGIES_KEV = (60.0, 80.0, 100.0)
#: How close to an edge, as a fraction of its energy, counts as a resonance warning.
NEAR_EDGE = 0.02
#: Fluorescence lines below the 1 keV floor of the calculator's energy range are not listed.
MIN_LINE_KEV = 1.0
CURVE_POINTS = 301
LIMITS = ("Independent-atom form factors; optional ionic fits affect f0 and pair weights only.",
          "Pair weights use f0(Q) without f' or f''.",
          "Compton curves use the incoherent scattering-function approximation with Klein-Nishina recoil.",
          "Capillary model assumes a uniformly illuminated disc; no container or multiple scattering.")


def q_from_angle(two_theta_deg: float, wavelength: float) -> float:
    """Q = 4π sinθ / λ, in Å⁻¹."""
    return 4 * pi * sin(radians(two_theta_deg) / 2) / wavelength


def _mass_attenuation(elements: dict[str, float], energies_keV: Any) -> dict[str, np.ndarray]:
    backend = get_backend()
    energies = np.atleast_1d(np.asarray(energies_keV, dtype=float))
    total: dict[str, np.ndarray] = {}
    for symbol, weight in elements.items():
        for channel, values in backend.mass_attenuation(symbol, energies).items():
            total[channel] = total.get(channel, 0) + weight * values
    return total


def _fluorescence(symbols: list[str], energy_keV: float) -> tuple[list[JsonObject], list[str]]:
    """K and L3 edges the beam can excite, with each shell's strongest line, and any close resonance."""
    excited: list[JsonObject] = []
    warnings = []
    for symbol in symbols:
        for shell, edge in get_backend().edges(symbol):
            if shell not in ("K", "L3"):
                continue
            if abs(energy_keV - edge) / edge < NEAR_EDGE:
                warnings.append(f"The beam is within {NEAR_EDGE:.0%} of the {symbol} {shell} edge at {edge:.3f} keV, "
                                "where f' changes rapidly and the pair weights here do not apply.")
            lines = xraydb.xray_lines(symbol, initial_level=shell)
            if edge < energy_keV and lines:
                name, line = max(lines.items(), key=lambda item: item[1].intensity)
                if line.energy / 1000 >= MIN_LINE_KEV:
                    excited.append({"element": symbol, "edge": shell, "edge_keV": edge, "line": name,
                                    "line_keV": line.energy / 1000})
    return excited, warnings


def _anomalous(symbol: str, energy_keV: float) -> tuple[float | None, float | None]:
    try:  # Chantler's tables stop at uranium.
        return (float(xraydb.f1_chantler(symbol, energy_keV * 1000)),
                float(xraydb.f2_chantler(symbol, energy_keV * 1000)))
    except (ValueError, TypeError, KeyError):
        return None, None


def _pairs(symbols: list[str], fractions: np.ndarray, factors: dict[str, np.ndarray], mean: np.ndarray) -> list[JsonObject]:
    pairs = []
    for i, first in enumerate(symbols):
        for j, second in enumerate(symbols[i:], start=i):
            multiplicity = 1 if i == j else 2
            weights = multiplicity * fractions[i] * fractions[j] * factors[first] * factors[second] / mean ** 2
            pairs.append({"pair": f"{first}-{second}", "weights": weights.tolist(),
                          "at_zero": float(weights[0]), "at_q_max": float(weights[-1])})
    return sorted(pairs, key=lambda pair: -pair["at_zero"])


def _source_table(elements: dict[str, float], density: float, two_theta: float, diameter: float, target: float) -> list[JsonObject]:
    entries = [(s["name"], s["energy_keV"]) for s in xray_sources() if s["line"] == "Ka"]
    entries += [(f"{energy:g} keV", energy) for energy in EXAMPLE_ENERGIES_KEV]
    mass = np.sum(list(_mass_attenuation(elements, [energy for _, energy in entries]).values()), axis=0)
    table = []
    for index, (name, energy) in enumerate(entries):
        wavelength = HC_KEV_ANGSTROM / energy
        linear = float(mass[index]) * density
        reach = target * wavelength / (4 * pi)
        table.append({"name": name, "energy_keV": energy, "wavelength_angstrom": wavelength,
                      "q_max": q_from_angle(two_theta, wavelength),
                      "two_theta_for_target_deg": 2 * asin(reach) * 180 / pi if reach <= 1 else None,
                      "mu_d": linear * diameter / 10, "transmission": exp(-linear * diameter / 10)})
    return table


@lru_cache(maxsize=4)
def _quadrature(points: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(points)


@lru_cache(maxsize=2)
def _table(name: str) -> JsonObject:
    return json.loads((Path(__file__).parent / "scattering_data" / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def differential_scattering(symbols: list[str], fractions: np.ndarray, energy: float,
                            q: np.ndarray) -> JsonObject:
    """Unpolarized independent-atom scattering, barns/sr/atom (r_e squared = 0.0794079 barn)."""
    table = _table("incoherent_hubbell1975.json")
    grid = np.asarray(table["grid_sin_theta_over_lambda"])
    wavelength = HC_KEV_ANGSTROM / energy

    def evaluate(qq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        argument = qq / (4 * pi)
        if np.any(argument > grid[-1]):
            raise ValueError("Compton curve exceeds the tabulated sin(theta)/lambda range.")
        cosine = 1 - 2 * (qq * wavelength / (4 * pi)) ** 2
        recoil = 1 / (1 + energy / 510.99895 * (1 - cosine))
        kn = 0.0794079 / 2 * recoil ** 2 * (recoil + 1 / recoil - (1 - cosine ** 2))
        incoherent = sum(c * np.interp(argument, grid, table["values"][s])
                         for s, c in zip(symbols, fractions, strict=True))
        coherent = sum(c * np.asarray(xraydb.f0(s, argument)) ** 2
                       for s, c in zip(symbols, fractions, strict=True))
        return incoherent, kn * incoherent, 0.0794079 * (1 + cosine ** 2) / 2 * coherent

    sf, inc, coh = evaluate(q)
    # Integrate over solid angle using u=cos(2theta), Gauss-Legendre quadrature.
    # Do not extrapolate f0 or S past their respective domains.
    integral = None
    if 1 / wavelength <= min(6.0, float(grid[-1])):
        u, weights = _quadrature(512)
        _, ii, cc = evaluate(4 * pi / wavelength * np.sqrt((1 - u) / 2))
        mean_mass = sum(c * float(xraydb.atomic_mass(s)) for s, c in zip(symbols, fractions, strict=True))
        factor = 2 * pi * AVOGADRO * 1e-24 / mean_mass
        integral = {"incoherent_cm2_g": float(np.dot(weights, ii) * factor),
                    "coherent_cm2_g": float(np.dot(weights, cc) * factor)}
    return {"S_mean": sf.tolist(), "incoherent_barn_sr_atom": inc.tolist(),
            "coherent_barn_sr_atom": coh.tolist(),
            "coherent_fraction": np.divide(coh, coh + inc, out=np.zeros_like(coh), where=coh + inc > 0).tolist(),
            "integrated": integral, "source": table["source"],
            "note": "Neutral-atom Compton/coherent comparison, independent of ionic choices. Independent-atom form-factor approximation; compare integrals with Elam, never rescale to force agreement."}


@lru_cache(maxsize=128)
def capillary_absorption(mu_r: float) -> JsonObject:
    """Area average of exp[-mu(l_in+l_out)] over a full, uniformly lit disc.
    Capillary axis perpendicular to the scattering plane. No wall or multiple scattering.
    Unit disc coordinates: incident direction +x; outgoing direction (cos a, sin a).
    """
    angles = np.linspace(0, 180, 91)
    cosine = np.cos(np.radians(angles))
    factors = {"unpolarized": ((1 + cosine ** 2) / 2).tolist(),
               "in_plane": (cosine ** 2).tolist(), "perpendicular": np.ones_like(angles).tolist()}
    if mu_r > 30:
        return {"two_theta_deg": angles.tolist(), "absorption": None, "polarization": factors,
                "note": "Disc integration withheld for muR > 30; optically thick surface sampling needs a dedicated method."}
    y, wy = _quadrature(128)
    u, wu = _quadrature(80)
    half = np.sqrt(1 - y ** 2)
    x = half[:, None] * u[None, :]
    yy = np.broadcast_to(y[:, None], x.shape)
    weight = wy[:, None] * wu[None, :] * half[:, None]
    weight /= weight.sum()
    incoming = x + half[:, None]
    average = []
    for angle in np.radians(angles):
        projection = x * np.cos(angle) + yy * np.sin(angle)
        transverse = x * np.sin(angle) - yy * np.cos(angle)
        outgoing = -projection + np.sqrt(np.maximum(0, 1 - transverse ** 2))
        average.append(float(np.sum(weight * np.exp(-mu_r * (incoming + outgoing)))))
    return {"two_theta_deg": angles.tolist(), "absorption": average, "polarization": factors,
            "note": "Uniform full-disc illumination, no walls; divide measured single scattering by this factor only if these assumptions apply."}


def neutron_weights(symbols: list[str], fractions: np.ndarray, number_density: float) -> JsonObject:
    """Natural-abundance, bound coherent lengths; complex b convention from Sears."""
    data = _table("neutron_sears1992.json")
    missing = [s for s in symbols if s not in data["values"]]
    if missing:
        return {"available": False, "pairs": [], "note": "Missing natural-abundance neutron data: " + ", ".join(missing)}
    entries = [data["values"][s] for s in symbols]
    b = np.array([complex(e["b_coherent_fm"], e["b_coherent_imaginary_fm"]) for e in entries])
    mean = complex(np.dot(fractions, b))
    scale = float(np.dot(fractions, abs(b)))
    near_zero = abs(mean) < 1e-6 * max(scale, 1)
    pairs = []
    if not near_zero:
        for i, first in enumerate(symbols):
            for j in range(i, len(symbols)):
                value = (1 if i == j else 2) * fractions[i] * fractions[j] * (b[i] * b[j].conjugate()).real / abs(mean) ** 2
                pairs.append({"pair": f"{first}-{symbols[j]}", "weight": float(value)})
    return {"available": True, "pairs": pairs, "mean_b_fm": mean.real, "mean_b_imag_fm": mean.imag,
            "sld_real_inv_angstrom2": number_density * mean.real * 1e-5,
            "sld_imag_inv_angstrom2": number_density * mean.imag * 1e-5,
            "incoherent_barn": float(np.dot(fractions, [e["sigma_incoherent_barn"] for e in entries])),
            "absorption_barn_2200_m_s": float(np.dot(fractions, [e["sigma_absorption_barn"] for e in entries])),
            "source": data["source"],
            "note": ("Normalized pair weights undefined: mean b is near zero." if near_zero else
                     "Natural isotopes; Re(bi bj*) / |mean b|². Negative weights are physical; absorption is at 2200 m/s.")}


@lru_cache(maxsize=128)
def edge_scan(symbol: str, shell: str, edge_keV: float) -> JsonObject:
    """Independent-atom Chantler factors, not chemical-state XANES or EXAFS."""
    energies = np.unique(np.r_[np.linspace(max(1, edge_keV * 0.95), min(800, edge_keV * 1.05), 301),
                               edge_keV * (1 - 1e-7), edge_keV, edge_keV * (1 + 1e-7)])
    return {"symbol": symbol, "shell": shell, "edge_keV": edge_keV, "energy_keV": energies.tolist(),
            "f_prime": np.asarray(xraydb.f1_chantler(symbol, energies * 1000)).tolist(),
            "f_double_prime": np.asarray(xraydb.f2_chantler(symbol, energies * 1000)).tolist()}


def total_scattering(payload: Any) -> JsonObject:
    payload = _object(payload, "Total scattering request")
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    two_theta = _number(payload.get("two_theta_max_deg", 120), "Maximum 2θ (°)", 1, 180)
    diameter = _number(payload.get("capillary_diameter_mm", 1), "Capillary inner diameter (mm)", 0.001, 1000)
    target = _number(payload.get("target_q", 25), "Target Q (1/Å)", 0.1, 200)
    layer = _prepare_layer(payload.get("layer"), 0)
    if layer.density <= 0:
        raise ValueError("Total scattering needs a positive density.")

    symbols = list(layer.elements)
    masses = np.array([float(xraydb.atomic_mass(symbol)) for symbol in symbols])
    numbers = np.array([int(xraydb.atomic_number(symbol)) for symbol in symbols])
    mass_fractions = np.array([layer.elements[symbol] for symbol in symbols])
    moles_per_gram = mass_fractions / masses
    fractions = moles_per_gram / moles_per_gram.sum()
    number_density = layer.density * AVOGADRO * float(moles_per_gram.sum()) * 1e-24  # atoms/Å³

    wavelength = HC_KEV_ANGSTROM / energy
    q_max = q_from_angle(two_theta, wavelength)
    warnings = []
    curve_max = min(q_max, Q_LIMIT)
    if q_max > Q_LIMIT:
        warnings.append(f"Qmax is beyond the {Q_LIMIT:.1f} 1/Å limit of the form-factor fits, so the curves stop there.")
    q = np.linspace(0, curve_max, CURVE_POINTS)
    ions = _object(payload.get("ions", {}), "Ionic form-factor choices")
    if any(s not in symbols for s in ions):
        raise ValueError("Ionic choices must refer to elements in the selected layer.")
    choices = {s: xraydb.f0_ions(s) for s in symbols}
    selected = {s: ions.get(s, s) for s in symbols}
    for symbol, ion in selected.items():
        if not isinstance(ion, str) or ion not in choices[symbol]:
            raise ValueError(f"Unknown ionic form factor for {symbol}: {ion}.")
    factors = {s: np.asarray(xraydb.f0(selected[s], q / (4 * pi)), dtype=float) for s in symbols}
    mean = sum(c * factors[s] for c, s in zip(fractions, symbols, strict=True))
    mean_squares = sum(c * factors[s] ** 2 for c, s in zip(fractions, symbols, strict=True))

    channels = _mass_attenuation(layer.elements, energy)
    mass_mu = float(sum(values[0] for values in channels.values()))
    linear_mu = mass_mu * layer.density
    excited, near = _fluorescence(symbols, energy)
    warnings += near
    compton = energy / (1 + COMPTON_WAVELENGTH_ANGSTROM * (1 - cos(radians(two_theta))) / wavelength)

    elements = []
    for index, symbol in enumerate(symbols):
        f_prime, f_double_prime = _anomalous(symbol, energy)
        elements.append({"symbol": symbol, "atomic_number": int(numbers[index]), "atomic_fraction": float(fractions[index]),
                         "ion": selected[symbol], "ion_choices": choices[symbol],
                         "mass_fraction": float(mass_fractions[index]), "f0_at_zero": float(factors[symbol][0]),
                         "f_prime": f_prime, "f_double_prime": f_double_prime})
    edges: list[JsonObject] = [{"id": f"{symbol}:{shell}", "symbol": symbol, "shell": shell, "energy_keV": edge}
             for symbol in symbols for shell, edge in get_backend().edges(symbol)
             if shell in ("K", "L3") and 1.01 < edge < 790 and int(xraydb.atomic_number(symbol)) <= 92]
    chosen = payload.get("edge_id")
    selected_edge = next((edge for edge in edges if edge["id"] == chosen), None) if chosen else (
        min(edges, key=lambda edge: abs(edge["energy_keV"] - energy)) if edges else None)
    if chosen and selected_edge is None:
        raise ValueError("Choose an absorption edge of an element in this layer.")
    anomalous = edge_scan(selected_edge["symbol"], selected_edge["shell"], selected_edge["energy_keV"]) if selected_edge else None
    differential = differential_scattering(symbols, fractions, energy, q)
    if differential["integrated"] is not None:
        differential["comparison"] = [
            {"channel": kind, "integrated_cm2_g": differential["integrated"][f"{kind}_cm2_g"],
             "elam_cm2_g": float(channels[kind][0]),
             "relative_difference": differential["integrated"][f"{kind}_cm2_g"] / float(channels[kind][0]) - 1}
            for kind in ("coherent", "incoherent")]
    else:
        differential["comparison"] = []
    smallest = float(fractions.min())
    return {
        "energy_keV": energy, "wavelength_angstrom": wavelength, "two_theta_max_deg": two_theta,
        "capillary_diameter_mm": diameter, "target_q": target,
        "resolution": {"q_max": q_max, "r_step_nyquist": pi / q_max, "fwhm_step": FWHM_STEP / q_max, "fwhm_lorch": FWHM_LORCH / q_max},
        "sample": {"name": layer.name, "density_g_cm3": layer.density, "number_density": number_density,
                   "low_r_slope": -4 * pi * number_density, "mean_atomic_mass": float(np.dot(fractions, masses)),
                   "mean_atomic_number": float(np.dot(fractions, numbers)),
                   "composition": " ".join(f"{s}{c / smallest:.4g}" for s, c in zip(symbols, fractions, strict=True))},
        "elements": elements,
        "attenuation": {"mu_mass_cm2_g": mass_mu, "mu_linear_cm_inv": linear_mu, "mu_d": linear_mu * diameter / 10,
                        "mu_r": linear_mu * diameter / 20, "transmission": exp(-linear_mu * diameter / 10),
                        "plate_optimum_mm": 10 / linear_mu,
                        "shares": {channel: float(values[0]) / mass_mu for channel, values in channels.items()}},
        "edge_options": edges, "edge_scan": anomalous,
        "neutron": neutron_weights(symbols, fractions, number_density),
        "capillary": capillary_absorption(linear_mu * diameter / 20),
        "compton": {"energy_keV": compton, "shift_keV": energy - compton},
        "differential": differential,
        "fluorescence": excited,
        "sources": _source_table(layer.elements, layer.density, two_theta, diameter, target),
        "curves": {"q": q.tolist(), "form_factors": {s: values.tolist() for s, values in factors.items()},
                   "mean_f_squared": (mean ** 2).tolist(), "mean_of_f_squared": mean_squares.tolist(),
                   "laue": (mean_squares - mean ** 2).tolist(), "pairs": _pairs(symbols, fractions, factors, mean)},
        "warnings": warnings, "limits": list(LIMITS),
        "provenance": {"calculator_version": __version__, "xraydb_version": xraydb.__version__,
                       "form_factors": "Waasmaier & Kirfel (1995), doi:10.1107/S0108767394013292",
                       "anomalous": "Chantler (2000), doi:10.1063/1.1321055",
                       "attenuation": get_backend().info.as_provenance()},
    }
