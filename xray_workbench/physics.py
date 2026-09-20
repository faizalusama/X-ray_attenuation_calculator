"""Validated, unit-explicit independent-atom X-ray attenuation calculations.

All public input energies are keV, lengths are normal thicknesses in mm,
densities are g/cm³, and angles are degrees from the surface normal. XrayDB's
Elam backend is called with eV and returns elemental coefficients in cm²/g.
No interpolation of transmission, empirical buildup, or dose model is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from math import cos, isfinite, log, radians, sqrt
from numbers import Real
from typing import Any

import numpy as np
import xraydb

from . import MODEL
from ._types import Array, JsonObject
from .backends import COHERENT, INCOHERENT, PHOTOELECTRIC, AttenuationBackend, get_backend

ENERGY_RANGE_KEV = (1.0, 800.0)
# Elam log-energy knots have limited decimal precision. Their coefficient
# discontinuities can differ from xray_edges() metadata by up to ~5e-5 relative
# in XrayDB 4.5.8. Bracket the tabulated step without shifting any coefficients.
EDGE_BRACKET_RELATIVE = 1e-4
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
_ELEMENT = re.compile(r"[A-Z][a-z]?")


def _number(value: Any, label: str, minimum: float | None = None,
            maximum: float | None = None, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite representable number.") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite; NaN and infinity are not allowed.")
    if positive and result <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum:g}.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    result = _number(value, label, minimum, maximum)
    if not result.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(result)


def _object(value: Any, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _text(value: Any, label: str, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError(f"{label} must be text with at most 200 characters.")
    return value.strip() or default


def _energy(value: Any, label: str) -> float:
    return _number(value, label, *ENERGY_RANGE_KEV)


@lru_cache(maxsize=128)
def _atomic_mass(symbol: str) -> float:
    try:
        z = xraydb.atomic_number(symbol)
        mass = float(xraydb.atomic_mass(symbol))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError(f"Unknown or unsupported element '{symbol}'.") from exc
    if not 1 <= z <= 98 or not isfinite(mass) or mass <= 0:
        raise ValueError(f"Element '{symbol}' is outside the Elam table (H–Cf).")
    return mass


@lru_cache(maxsize=512)
def _parse_formula(formula: str) -> tuple[tuple[str, float], ...]:
    """Strict formula parser: nested ()/[], decimal occupancies and · hydrates.

    ASCII periods are decimal points, never hydration separators. Explicit
    parsing prevents permissive chemistry parsers from ignoring trailing text.
    """
    if not formula or len(formula) > 200 or any(c.isspace() for c in formula):
        raise ValueError("Chemical formulas must contain 1–200 characters without spaces.")

    def parse_segment(segment: str) -> dict[str, float]:
        pos = 0

        def multiplier() -> float:
            nonlocal pos
            match = _NUMBER.match(segment, pos)
            if match is None:
                return 1.0
            pos = match.end()
            count = float(match.group())
            if not isfinite(count) or not 0 < count <= 1e6:
                raise ValueError(f"Formula '{formula}' has an invalid atom count.")
            return count

        def group(closing: str | None = None, depth: int = 0) -> dict[str, float]:
            nonlocal pos
            if depth > 12:
                raise ValueError("Chemical formulas may have at most 12 nested groups.")
            counts: dict[str, float] = {}
            while pos < len(segment) and segment[pos] not in ")]":
                token = segment[pos]
                if token in "([":
                    pos += 1
                    nested = group(")" if token == "(" else "]", depth + 1)
                    multiple = multiplier()
                    for symbol, count in nested.items():
                        counts[symbol] = counts.get(symbol, 0.0) + multiple * count
                else:
                    match = _ELEMENT.match(segment, pos)
                    if match is None:
                        raise ValueError(f"Cannot parse formula '{formula}' near '{segment[pos:]}'.")
                    symbol = match.group()
                    _atomic_mass(symbol)
                    pos = match.end()
                    counts[symbol] = counts.get(symbol, 0.0) + multiplier()
            if not counts:
                raise ValueError(f"Formula '{formula}' contains an empty group.")
            if closing is not None:
                if pos >= len(segment) or segment[pos] != closing:
                    raise ValueError(f"Formula '{formula}' has unmatched brackets.")
                pos += 1
            return counts

        leading = multiplier()
        result = group()
        if pos != len(segment):
            raise ValueError(f"Formula '{formula}' has unmatched brackets.")
        return {symbol: count * leading for symbol, count in result.items()}

    counts: dict[str, float] = {}
    for segment in formula.split("·"):
        if not segment:
            raise ValueError(f"Formula '{formula}' contains an empty hydrate segment.")
        for symbol, count in parse_segment(segment).items():
            counts[symbol] = counts.get(symbol, 0.0) + count
    if any(not isfinite(count) or not 0 < count <= 1e12 for count in counts.values()):
        raise ValueError(f"Formula '{formula}' has atom counts outside the supported numeric range.")
    return tuple(sorted(counts.items(), key=lambda item: xraydb.atomic_number(item[0])))


@dataclass(frozen=True)
class _Layer:
    name: str
    density: float
    thickness: float
    path_length: float
    components: list[JsonObject]
    elements: dict[str, float]
    density_sd: float
    thickness_sd: float


def _prepare_layer(raw: Any, index: int) -> _Layer:
    raw = _object(raw, f"Layer {index + 1}")
    label = f"Layer {index + 1}"
    name = _text(raw.get("name"), f"{label} name", label)
    basis = raw.get("basis", "mass")
    if basis not in ("mass", "mole", "volume"):
        raise ValueError(f"{label} basis must be mass, mole, or volume.")
    mode = raw.get("density_mode", "bulk")
    if mode not in ("bulk", "solid", "ideal"):
        raise ValueError(f"{label} density_mode must be bulk, solid, or ideal.")
    components = raw.get("components")
    if not isinstance(components, list) or not 1 <= len(components) <= 40:
        raise ValueError(f"{label} must contain 1–40 components.")
    formulas, amounts, masses, elemental, densities = [], [], [], [], []
    for i, item in enumerate(components):
        item = _object(item, f"{label}, component {i + 1}")
        formula = _text(item.get("formula"), f"{label} formula")
        atoms = dict(_parse_formula(formula))
        molar_mass = sum(_atomic_mass(symbol) * count for symbol, count in atoms.items())
        amount = _number(item.get("fraction"), f"{label}, {formula} fraction", 0)
        if basis == "volume" or mode == "ideal":
            density = _number(item.get("density_g_cm3"),
                              f"{label}, {formula} component density (g/cm³)", 1e-12, 1e4)
        elif item.get("density_g_cm3") is not None:
            density = _number(item["density_g_cm3"],
                              f"{label}, {formula} component density (g/cm³)", 1e-12, 1e4)
        else:
            density = 1.0  # Not used unless all component densities are explicitly supplied.
        formulas.append(formula)
        amounts.append(amount)
        masses.append(molar_mass)
        elemental.append({s: _atomic_mass(s) * n / molar_mass for s, n in atoms.items()})
        densities.append(density)
    fractions = np.asarray(amounts, dtype=float)
    if fractions.max() == 0:
        raise ValueError(f"{label} needs at least one positive component fraction.")
    fractions /= fractions.max()  # Avoid overflow for large but finite relative weights.
    if basis == "mole":
        fractions *= masses
    elif basis == "volume":
        fractions *= densities
    fractions /= fractions.sum()
    porosity = _number(raw.get("porosity", 0), f"{label} porosity", 0, 1)
    if porosity >= 1:
        raise ValueError(f"{label} porosity must be less than 1.")
    if mode == "bulk":
        if porosity != 0:
            raise ValueError(f"{label}: measured bulk density already includes pores; set porosity to 0 or choose solid density.")
        density = _number(raw.get("density_g_cm3"), f"{label} density (g/cm³)", 1e-12, 1e4)
    elif mode == "solid":
        density = _number(raw.get("density_g_cm3"), f"{label} solid density (g/cm³)", 1e-12, 1e4) * (1 - porosity)
    else:
        density = float(1 / np.sum(fractions / np.asarray(densities))) * (1 - porosity)
    thickness = _number(raw.get("thickness_mm", 1), f"{label} thickness (mm)", 0, 1e9)
    angle = _number(raw.get("angle_deg", 0), f"{label} angle (degrees)", -89.9, 89.9)
    if abs(angle) >= 89.9:
        raise ValueError(f"{label} angle must lie strictly between −89.9° and 89.9°.")
    path = thickness / cos(radians(angle))
    density_sd = _number(raw.get("density_uncertainty_pct", 0), f"{label} density uncertainty (%)", 0, 100) / 100
    thickness_sd = _number(raw.get("thickness_uncertainty_pct", 0), f"{label} thickness uncertainty (%)", 0, 100) / 100
    elements: dict[str, float] = {}
    output_components = []
    # strict=True: these four sequences are filled in one pass over the
    # components, so a length mismatch is a bug, not a shorter result.
    for formula, mass, fraction, by_element in zip(formulas, masses, fractions, elemental, strict=True):
        output_components.append({"formula": formula, "mass_fraction": float(fraction), "molar_mass_g_mol": mass})
        if fraction > 0:
            for symbol, weight in by_element.items():
                elements[symbol] = elements.get(symbol, 0.0) + float(fraction * weight)
    return _Layer(name, density, thickness, path, output_components, elements, density_sd, thickness_sd)


def _edge_grid(layers: list[_Layer], low: float, high: float, points: int,
               spacing: str, backend: AttenuationBackend) -> tuple[Array, list[JsonObject]]:
    base = np.geomspace(low, high, points) if spacing == "log" else np.linspace(low, high, points)
    edges: list[JsonObject] = []
    extra: list[float] = []
    for element in sorted({s for layer in layers for s in layer.elements}, key=xraydb.atomic_number):
        for shell, energy in backend.edges(element):
            if low <= energy <= high:
                edges.append({"element": element, "shell": shell, "energy_keV": energy})
                extra.extend(x for x in (energy * (1 - EDGE_BRACKET_RELATIVE), energy,
                                        energy * (1 + EDGE_BRACKET_RELATIVE)) if low <= x <= high)
    edges.sort(key=lambda item: (item["energy_keV"], item["element"], item["shell"]))
    return np.unique(np.concatenate((base, np.asarray(extra)))), edges


def _spectrum_input(raw: Any) -> tuple[Array, Array, str, str] | None:
    if raw is None:
        return None
    raw = _object(raw, "Spectrum")
    energies = raw.get("energy_keV")
    weights = raw.get("weights")
    if not isinstance(energies, list) or not isinstance(weights, list) or not 1 <= len(energies) <= 10000 or len(energies) != len(weights):
        raise ValueError("Spectrum needs equally sized energy_keV and weights lists with 1–10000 discrete bins.")
    energy = np.asarray([_energy(e, f"Spectrum energy {i + 1} (keV)") for i, e in enumerate(energies)])
    weight = np.asarray([_number(w, f"Spectrum weight {i + 1}", 0) for i, w in enumerate(weights)])
    if not np.any(weight > 0):
        raise ValueError("Spectrum needs at least one positive photon-fluence weight.")
    weighting = raw.get("weighting", "photon")
    if weighting not in ("photon", "energy"):
        raise ValueError("Spectrum weighting must be photon or energy.")
    return energy, weight, weighting, _text(raw.get("label"), "Spectrum label", "Discrete incident spectrum")


def _uncertainty(layers: list[_Layer], layer_tau: Array,
                 samples: int, seed: int, sweep_count: int) -> JsonObject:
    rng = np.random.default_rng(seed)
    factors = np.ones((samples, len(layers)))
    for index, layer in enumerate(layers):
        for relative_sd in (layer.density_sd, layer.thickness_sd):
            if relative_sd:
                sigma = sqrt(np.log1p(relative_sd ** 2))
                factors[:, index] *= rng.lognormal(-0.5 * sigma ** 2, sigma, size=samples)
    lower, upper = np.empty(sweep_count), np.empty(sweep_count)
    # Chunk by energy: never allocate samples × full energy sweep × layers.
    for start in range(0, sweep_count, 128):
        stop = min(start + 128, sweep_count)
        sample_transmission = factors @ layer_tau[:, start:stop]
        np.negative(sample_transmission, out=sample_transmission)
        np.exp(sample_transmission, out=sample_transmission)
        lower[start:stop], upper[start:stop] = np.quantile(sample_transmission, [0.025, 0.975], axis=0)
    reference = np.exp(-(factors @ layer_tau[:, sweep_count]))
    lo, hi = np.quantile(reference, [0.025, 0.975])
    return {
        "samples": samples, "seed": seed,
        "transmission_p025": lower.tolist(), "transmission_p975": upper.tolist(),
        "reference": {"mean": float(reference.mean()), "std": float(reference.std(ddof=1)),
                      "p025": float(lo), "p975": float(hi)},
        "assumptions": "Independent positive lognormal effective densities and normal thicknesses; supplied values are arithmetic means and percentages are relative one-sigma standard deviations. Composition, incidence, coefficient uncertainty and correlations are excluded. Bands are pointwise 95% intervals, not simultaneous confidence bands.",
    }


def calculate(payload: JsonObject) -> JsonObject:
    """Evaluate a validated material stack; raise ValueError for invalid inputs."""
    payload = _object(payload, "Request")
    raw_energy = _object(payload.get("energy", {}), "Energy settings")
    low = _energy(raw_energy.get("min_keV", 5), "Minimum energy (keV)")
    high = _energy(raw_energy.get("max_keV", 120), "Maximum energy (keV)")
    if high <= low:
        raise ValueError("Maximum energy must exceed minimum energy.")
    reference_energy = _energy(raw_energy.get("reference_keV", 30), "Reference energy (keV)")
    points = _integer(raw_energy.get("points", 500), "Energy points", 2, 4000)
    spacing = raw_energy.get("spacing", "log")
    if spacing not in ("log", "linear"):
        raise ValueError("Energy spacing must be log or linear.")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not 1 <= len(raw_layers) <= 24:
        raise ValueError("A calculation needs 1–24 layers.")
    layers = [_prepare_layer(layer, i) for i, layer in enumerate(raw_layers)]
    target = _number(payload.get("target_transmission", 0.1), "Target transmission", positive=True)
    if target >= 1:
        raise ValueError("Target transmission must lie strictly between 0 and 1.")
    spectrum = _spectrum_input(payload.get("spectrum"))
    raw_uncertainty = payload.get("uncertainty")
    settings = _object({} if raw_uncertainty is None else raw_uncertainty, "Uncertainty settings")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Uncertainty enabled must be true or false.")
    samples = _integer(settings.get("samples", 1000), "Uncertainty samples", 100, 20000)
    seed = _integer(settings.get("seed", 42), "Uncertainty seed", 0, 2 ** 32 - 1)
    requested_backend = payload.get("backend")
    if requested_backend is not None and not isinstance(requested_backend, str):
        raise ValueError("Backend must be the text identifier of an attenuation data source.")
    backend = get_backend(requested_backend)
    energies, edges = _edge_grid(layers, low, high, points, spacing, backend)
    count = len(energies)
    evaluation = np.concatenate((energies, [reference_energy], spectrum[0] if spectrum is not None else []))
    warnings = [
        "Narrow-beam primary transmission: photons scattered out of the beam are removed. Removed fraction is not absorbed energy or dose; buildup, multiple scattering, secondary fluorescence and detector response are not modeled.",
        "Independent-atom mixture coefficients omit chemical-state near-edge structure, diffraction, coherent interfaces and microstructure-dependent transport. Edge brackets locate tabulated steps; they do not predict XANES/EXAFS.",
    ]
    warnings.extend(backend.domain_warnings(float(evaluation.min()), float(evaluation.max())))
    if not low <= reference_energy <= high:
        warnings.append("Reference energy lies outside the plotted sweep and was evaluated directly.")
    if any(item.get("density_mode") == "ideal" for item in raw_layers):
        warnings.append("Ideal density assumes additive constituent volumes. Real glasses, solid solutions and sintered ceramics can deviate; use measured bulk density for quantitative work.")
    # Elemental coefficients are shared across layers: fetch each element once.
    # The backend validates its own output, so an invalid coefficient raises
    # here rather than propagating into a plotted curve.
    coefficient_cache: dict[str, tuple[Array, Array, Array]] = {}
    for symbol in {symbol for layer in layers for symbol in layer.elements}:
        element_channels = backend.mass_attenuation(symbol, evaluation)
        coefficient_cache[symbol] = (element_channels[PHOTOELECTRIC],
                                     element_channels[COHERENT],
                                     element_channels[INCOHERENT])
    results, depths = [], []
    for layer in layers:
        # An explicit zero start keeps each mixture channel a float64 array even
        # before any component is added, rather than the integer 0 sum() defaults to.
        channels = [sum((weight * coefficient_cache[symbol][channel]
                         for symbol, weight in layer.elements.items()),
                        start=np.zeros(len(evaluation)))
                    for channel in range(3)]
        mass_mu = channels[0] + channels[1] + channels[2]
        linear_mu = layer.density * mass_mu
        tau = linear_mu * (layer.path_length / 10)  # mm → cm, exactly once.
        depths.append(tau)
        ref_linear = float(linear_mu[count])
        results.append({
            "name": layer.name, "density_g_cm3": layer.density,
            "thickness_mm": layer.thickness, "path_length_mm": layer.path_length,
            "components": layer.components, "elemental_mass_fractions": layer.elements,
            "mu_mass_cm2_g": mass_mu[:count].tolist(), "mu_linear_cm_inv": linear_mu[:count].tolist(),
            "photoelectric_cm2_g": channels[0][:count].tolist(),
            "coherent_cm2_g": channels[1][:count].tolist(), "incoherent_cm2_g": channels[2][:count].tolist(),
            "reference": {
                "mu_mass_cm2_g": float(mass_mu[count]), "mu_linear_cm_inv": ref_linear,
                "transmission": float(np.exp(-tau[count])), "hvl_mm": 10 * log(2) / ref_linear,
                "tvl_mm": 10 * log(10) / ref_linear, "attenuation_length_mm": 10 / ref_linear,
            },
        })
    layer_tau = np.stack(depths)
    optical_depth = np.sum(layer_tau, axis=0)
    if not np.all(np.isfinite(optical_depth)):
        raise ValueError("Inputs produce non-finite optical depths; reduce extreme values.")
    transmission = np.exp(-optical_depth)
    ref_tau = float(optical_depth[count])
    target_scale = -log(target) / ref_tau if ref_tau > 0 else None
    if target_scale is None:
        warnings.append("All layer thicknesses are zero: no finite common thickness scaling can reach the requested target transmission.")
    elif not isfinite(target_scale):
        target_scale = None
        warnings.append("Target thickness scale exceeds finite numeric range.")
    spectrum_result = None
    if spectrum is not None:
        bins, weights, weighting, label = spectrum
        spectral_tau = optical_depth[count + 1:]
        spectral_transmission = transmission[count + 1:]
        relative = weights / weights.max()
        incident = relative / relative.sum()
        measure = incident * bins if weighting == "energy" else incident
        weighted_transmission = float(np.dot(measure, spectral_transmission) / measure.sum())
        # Log-domain normalization gives meaningful transmitted mean energies
        # even if the absolute transmitted fluence underflows to zero.
        supported = weights > 0
        log_out = np.log(weights[supported]) - spectral_tau[supported]
        relative_out = np.exp(log_out - log_out.max())
        mean_out = float(np.dot(relative_out, bins[supported]) / relative_out.sum())
        spectrum_result = {
            "energy_keV": bins.tolist(), "input_weights": weights.tolist(),
            "transmitted_weights": (weights * spectral_transmission).tolist(),
            "transmission": weighted_transmission, "weighting": weighting,
            "mean_energy_in_keV": float(np.dot(incident, bins)),
            "mean_energy_out_keV": mean_out, "label": label,
        }
        warnings.append("Spectrum weights are integrated photon fluence per discrete bin, not a spectral density. Energy weighting uses photon weight × energy. Spectra with gaps are not interpolated.")
    return {
        "energy_keV": energies.tolist(), "transmission": transmission[:count].tolist(),
        "removed_fraction": (-np.expm1(-optical_depth[:count])).tolist(),
        "optical_depth": optical_depth[:count].tolist(), "layers": results,
        "reference": {"energy_keV": reference_energy, "transmission": float(transmission[count]),
                      "removed_fraction": float(-np.expm1(-ref_tau)), "optical_depth": ref_tau,
                      "target_transmission": target, "target_thickness_scale": target_scale},
        "edges": edges, "spectrum": spectrum_result,
        "uncertainty": _uncertainty(layers, layer_tau, samples, seed, count) if enabled else None,
        "warnings": warnings,
        # "engine" and "xraydb_version" are kept under their original names for
        # schema-version-1 consumers; "backend" carries the full source identity.
        "provenance": {"engine": backend.info.name, "xraydb_version": xraydb.__version__,
                       "energy_range_keV": list(ENERGY_RANGE_KEV), "model": MODEL,
                       "edge_bracket_relative": EDGE_BRACKET_RELATIVE,
                       "backend": backend.info.as_provenance(),
                       "timestamp_utc": datetime.now(UTC).isoformat()},
    }


def presets() -> list[JsonObject]:
    """Illustrative compositions, deliberately not certified reference materials."""
    def entry(identifier: str, name: str, basis: str, components: list[tuple[str, float]], density: float,
              thickness: float = 1.0) -> JsonObject:
        return {"id": identifier, "name": name,
                "description": "Illustrative nominal composition and density; replace with measured sample values.",
                "layer": {"name": name, "basis": basis,
                          "components": [{"formula": formula, "fraction": fraction} for formula, fraction in components],
                          "density_g_cm3": density, "thickness_mm": thickness, "angle_deg": 0,
                          "porosity": 0, "density_mode": "bulk",
                          "density_uncertainty_pct": 0, "thickness_uncertainty_pct": 0}}
    return [
        entry("borosilicate", "Borosilicate glass", "mole", [("SiO2", 80), ("B2O3", 15), ("Na2O", 5)], 2.23),
        entry("soda_lime", "Soda-lime silicate glass", "mass", [("SiO2", 72), ("Na2O", 14), ("CaO", 10), ("MgO", 4)], 2.5),
        entry("bismuth_glass", "Bismuth borate glass", "mole", [("Bi2O3", 30), ("B2O3", 70)], 5.5),
        entry("alumina", "Alumina ceramic", "mass", [("Al2O3", 100)], 3.95),
        entry("zirconia", "3 mol% yttria-stabilized zirconia", "mole", [("ZrO2", 97), ("Y2O3", 3)], 6.05),
        entry("silica", "Fused silica", "mass", [("SiO2", 100)], 2.2),
        entry("aluminium", "Aluminium foil", "mass", [("Al", 100)], 2.699, 0.1),
        entry("copper", "Copper filter", "mass", [("Cu", 100)], 8.96, 0.05),
        entry("tungsten", "Tungsten layer", "mass", [("W", 100)], 19.3, 0.025),
        entry("water", "Water reference", "mass", [("H2O", 100)], 1.0),
    ]
