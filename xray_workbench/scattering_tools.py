"""Optics and scattering utilities, separate from narrow-beam attenuation.

Angles here are explicitly grazing angles from the surface when so labelled.
SLD uses rho_x = r_e sum N_j (Z_j + f'_j - i f''_j).
The wave convention is exp(i kz z), n = 1 - delta + i beta.
"""
from __future__ import annotations

import csv
import io
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


DIAMOND_SI_ATOMS = "Si,0,0,0,1,0\nSi,0,0.5,0.5,1,0\nSi,0.5,0,0.5,1,0\nSi,0.5,0.5,0,1,0\nSi,0.25,0.25,0.25,1,0\nSi,0.25,0.75,0.75,1,0\nSi,0.75,0.25,0.75,1,0\nSi,0.75,0.75,0.25,1,0"


def crystal_atoms(text: Any) -> list[JsonObject]:
    if not isinstance(text, str) or len(text) > 30000:
        raise ValueError("Atom list must be text smaller than 30,000 characters.")
    atoms: list[JsonObject] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [v.strip() for v in line.split(",")]
        if len(values) != 6:
            raise ValueError("Each atom row needs element,x,y,z,occupancy,B_angstrom2.")
        symbol = values[0]
        if not xraydb.f0_ions(symbol) or xraydb.atomic_symbol(xraydb.atomic_number(symbol)) != symbol:
            raise ValueError("Atom list requires valid neutral element symbols.")
        position = [_number(float(x), "Fractional coordinate", -100, 100) % 1 for x in values[1:4]]
        atoms.append({"symbol": symbol, "position": position,
                      "occupancy": _number(float(values[4]), "Occupancy", 0, 1),
                      "B": _number(float(values[5]), "Isotropic B (Å²)", 0, 100)})
        if len(atoms) > 200:
            raise ValueError("At most 200 atoms per unit cell.")
    if not atoms or not any(a["occupancy"] > 0 for a in atoms):
        raise ValueError("Unit cell needs at least one occupied atom.")
    return atoms


def structure_factors(hkl: np.ndarray, s: np.ndarray, atoms: list[JsonObject]) -> np.ndarray:
    amplitude = np.zeros(len(hkl), dtype=complex)
    factors = {a["symbol"]: np.asarray(xraydb.f0(a["symbol"], s)) for a in atoms}
    for atom in atoms:
        amplitude += atom["occupancy"] * factors[atom["symbol"]] * np.exp(-atom["B"]*s*s) * np.exp(2j*pi*(hkl @ np.array(atom["position"])))
    return amplitude


def powder(payload: JsonObject) -> JsonObject:
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    text = payload.get("lattice", "5.431,5.431,5.431,90,90,90")
    if not isinstance(text, str) or len(text) > 300:
        raise ValueError("Lattice must be a,b,c,alpha,beta,gamma.")
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 6:
        raise ValueError("Lattice must have six parameters a,b,c,alpha,beta,gamma.")
    a,b,c = [_number(v, "Cell length (Å)", 0.5, 1000) for v in parts[:3]]
    alpha,beta,gamma = [radians(_number(v, "Cell angle (°)", 1, 179)) for v in parts[3:]]
    metric = np.array([[a*a, a*b*cos(gamma), a*c*cos(beta)],
                       [a*b*cos(gamma), b*b, b*c*cos(alpha)],
                       [a*c*cos(beta), b*c*cos(alpha), c*c]])
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues[0] <= eigenvalues[-1]*1e-12:
        raise ValueError("Lattice metric is degenerate or not positive definite.")
    limit = _number(payload.get("hkl_max", 4), "Maximum absolute Miller index", 1, 8)
    if not limit.is_integer():
        raise ValueError("Maximum Miller index must be an integer.")
    maximum = _number(payload.get("two_theta_max_deg", 120), "Maximum 2θ (°)", 1, 179)
    atoms = crystal_atoms(payload.get("atoms", DIAMOND_SI_ATOMS))
    indices = np.arange(-int(limit), int(limit)+1)
    hkl = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"), axis=-1).reshape(-1,3)
    inverse_d_squared = np.einsum("ni,ij,nj->n", hkl, np.linalg.inv(metric), hkl)
    wavelength = HC_KEV_ANGSTROM/energy
    s = np.sqrt(np.maximum(inverse_d_squared, 0))/2
    valid = (inverse_d_squared > 0) & (s <= min(6, sin(radians(maximum)/2)/wavelength))
    hkl, s = hkl[valid], s[valid]
    theta = np.arcsin(s*wavelength)
    f_squared = abs(structure_factors(hkl, s, atoms))**2
    lp = (1 + np.cos(2*theta)**2)/(2*np.sin(theta)**2*np.cos(theta))
    groups: dict[float, JsonObject] = {}
    for h, ss, tt, ff, factor in zip(hkl, s, theta, f_squared, lp, strict=True):
        key = round(float(4*ss*ss), 12)
        row = groups.setdefault(key, {"hkl": tuple(int(v) for v in h), "two_theta_deg": float(tt*360/pi),
                                     "d_angstrom": float(1/(2*ss)), "multiplicity": 0, "intensity": 0.0})
        row["hkl"] = max(row["hkl"], tuple(int(v) for v in h))
        row["multiplicity"] += 1
        row["intensity"] += float(ff*factor)
    rows = sorted(groups.values(), key=lambda r: r["two_theta_deg"])
    largest = max((r["intensity"] for r in rows), default=0)
    peaks = []
    for row in rows:
        row["relative_intensity"] = row["intensity"]/largest*100 if largest > 0 else 0
        if row["relative_intensity"] > 1e-8:
            peaks.append(row)
    x = np.array([r["two_theta_deg"] for r in peaks for _ in range(3)])
    y = np.array([v for r in peaks for v in (0, r["relative_intensity"], 0)])
    return result("powder", {"peaks": peaks, "lattice": parts, "atoms": atoms},
                  [("Predicted peak groups", len(peaks), ""), ("Wavelength", wavelength, "Å"),
                   ("Maximum absolute h, k, l", int(limit), "")],
                  [plot("Kinematic powder peak positions", "2θ (°)", "Relative integrated intensity", x, {"Peaks": y})],
                  ["Complete unit-cell atoms are required; no space-group symmetry expansion is performed.",
                   "Neutral f₀, occupancy and exp(−B s²), with s=sinθ/λ. Intensities sum all enumerated hkl, including Friedel pairs.",
                   "LP = (1+cos²2θ)/(2 sin²θ cosθ), unpolarized Bragg–Brentano powder geometry. No texture, absorption or peak broadening.",
                   "Only indices within the stated bound are searched. Equal d groups use 1e-12 Å⁻² reciprocal-metric bins; table shows up to 500 groups.",
                   "Default cell is an editable diamond-Si demonstration, independent of the selected attenuation layer."],
                  [{"caption": "Predicted powder peaks", "headers": ["h k l", "2θ (°)", "d (Å)", "Multiplicity", "Relative intensity"],
                    "rows": [[" ".join(map(str,r["hkl"])), r["two_theta_deg"], r["d_angstrom"], r["multiplicity"], r["relative_intensity"]] for r in peaks[:500]]}])


def pdf_transform(payload: JsonObject) -> JsonObject:
    """Finite-range sine transform of F(Q)=Q[S(Q)-1], with no data reduction."""
    if "q" in payload:
        q = np.asarray(payload["q"], dtype=float)
        sq = np.asarray(payload.get("sq"), dtype=float)
    else:
        text = payload.get("sq_data", "")
        if not isinstance(text, str) or len(text) > 800000:
            raise ValueError("S(Q) data must be CSV text smaller than 800,000 characters.")
        pairs = []
        try:
            for row in csv.reader(io.StringIO(text), strict=True):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if [v.strip().lower() for v in row] == ["q", "s"]:
                    continue
                if len(row) != 2:
                    raise ValueError("S(Q) CSV rows must contain exactly q,S.")
                pairs.append([float(row[0]), float(row[1])])
                if len(pairs) > 20000:
                    raise ValueError("At most 20,000 S(Q) samples.")
        except csv.Error as exc:
            raise ValueError(f"Malformed S(Q) CSV: {exc}") from exc
        array = np.asarray(pairs, dtype=float)
        if array.ndim != 2 or array.shape[0] < 3:
            raise ValueError("S(Q) needs at least three q,S samples.")
        q, sq = array[:,0], array[:,1]
    if q.ndim != 1 or sq.shape != q.shape or not 3 <= len(q) <= 20000:
        raise ValueError("q and S must be equal-length arrays with 3–20,000 samples.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(sq)) or np.any(q < 0) or q[-1] > 200:
        raise ValueError("S(Q) data must be finite; Q must lie in 0–200 Å⁻¹.")
    if np.any(np.diff(q) <= 0):
        raise ValueError("Q must increase strictly; sort and resolve duplicates before transforming.")
    r_max = _number(payload.get("r_max_angstrom", 20), "Maximum r (Å)", 0.01, 1000)
    points = _number(payload.get("r_points", 501), "r grid points", 3, 4000)
    if not points.is_integer() or points*len(q) > 20_000_000:
        raise ValueError("Use integer r grid points and at most 20 million Q×r samples.")
    nyquist = pi/float(np.max(np.diff(q)))
    if r_max > nyquist:
        raise ValueError(f"Maximum r exceeds the conservative sampling limit {nyquist:.6g} Å from the largest Q interval.")
    if r_max/(points-1) > pi/float(q[-1]):
        raise ValueError("r grid is too coarse for Qmax; increase r grid points.")
    window = payload.get("window", "none")
    if window not in ("none", "lorch"):
        raise ValueError("Window must be none or lorch.")
    r = np.linspace(0, r_max, int(points))
    f = q*(sq-1)
    modified = f * (np.sinc(q/q[-1]) if window == "lorch" else 1)
    g = np.empty_like(r)
    for start in range(0, len(r), 64):
        values = np.sin(r[start:start+64,None]*q[None,:]) * modified[None,:]
        g[start:start+64] = 2/pi*np.trapezoid(values, q, axis=1)
    return result("pdf", {"q": q.tolist(), "sq": sq.tolist(), "r_angstrom": r.tolist(), "g_inv_angstrom2": g.tolist(), "window": window},
                  [("Qmin", float(q[0]), "Å⁻¹"), ("Qmax", float(q[-1]), "Å⁻¹"), ("Input samples", len(q), ""),
                   ("Conservative sampling limit for r", nyquist, "Å")],
                  [plot("Reduced pair distribution function", "r (Å)", "G(r) (Å⁻²)", r, {"G(r)": g}),
                   plot("Transform integrand before sine kernel", "Q (Å⁻¹)", "Q[S(Q)−1] M(Q) (Å⁻¹)", q, {"F(Q) M(Q)": modified})],
                  ["G(r) = (2/π) ∫ Q[S(Q)−1] M(Q) sin(Qr) dQ, integrated only over the supplied Q interval.",
                   "Lorch M(Q)=sin(πQ/Qmax)/(πQ/Qmax), with its continuous limit at Q=0. No extrapolation or baseline fitting.",
                   "Input must already be normalized S(Q), corrected for instrumental and sample effects. This tool does not reduce raw intensities.",
                   "Finite-Q termination, Qmin omissions, noise and uneven Q sampling affect G(r). No uncertainty propagation is performed."])


def scattering_tools(payload: Any) -> JsonObject:
    payload = _object(payload, "Scattering tools request")
    if payload.get("tool") == "pdf":
        return pdf_transform(payload)
    if payload.get("tool") == "powder":
        return powder(payload)
    if payload.get("tool") == "converter":
        return converter(payload)
    if payload.get("tool") == "contrast":
        return contrast(payload)
    if payload.get("tool") == "reflectivity":
        return reflectivity(payload)
    if payload.get("tool", "optics") == "optics":
        return optics(payload)
    raise ValueError("Unknown scattering tool.")
