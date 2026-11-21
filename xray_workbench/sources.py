"""Laboratory X-ray sources, for choosing a reference energy by anode and emission line.

Line energies are from Table 1-2 of the X-ray Data Booklet (Lawrence Berkeley National
Laboratory; https://xdb.lbl.gov/Section1/Table_1-2.pdf). The Elam tables behind the
attenuation coefficients round these lines more coarsely (Cu Kα1 would read 1.5409 Å
instead of 1.5406 Å), so they serve only as a cross-check in ``tests/test_sources.py``.
"""

from __future__ import annotations

from ._types import JsonObject

#: hc in keV·Å, from the exact SI values of h, c and e.
HC_KEV_ANGSTROM = 12.398419843320026
#: (Kα1, Kα2, Kβ1) in eV, copied from the table. Ga and In are liquid-metal-jet anodes.
LINES_EV = {
    "Cr": (5414.72, 5405.509, 5946.71),
    "Fe": (6403.84, 6390.84, 7057.98),
    "Co": (6930.32, 6915.30, 7649.43),
    "Cu": (8047.78, 8027.83, 8905.29),
    "Ga": (9251.74, 9224.82, 10264.2),
    "Mo": (17479.34, 17374.3, 19608.3),
    "Rh": (20216.1, 20073.7, 22723.6),
    "Ag": (22162.92, 21990.3, 24942.4),
    "In": (24209.7, 24002.0, 27275.9),
    "W": (59318.24, 57981.7, 67244.3),
}
#: The usual convention for unresolved Kα: wavelengths weighted 2:1 for Kα1:Kα2.
KA1_WEIGHT, KA2_WEIGHT = 2, 1


def _entry(anode: str, line: str, name: str, wavelength: float) -> JsonObject:
    return {"id": f"{anode}-{line}", "anode": anode, "line": line, "name": f"{anode} {name}",
            "energy_keV": HC_KEV_ANGSTROM / wavelength, "wavelength_angstrom": wavelength}


def xray_sources() -> list[JsonObject]:
    """Kα (weighted mean of Kα1 and Kα2), Kα1, Kα2 and Kβ1 for each anode."""
    sources = []
    for anode, (ka1_ev, ka2_ev, kb1_ev) in LINES_EV.items():
        ka1, ka2, kb1 = (HC_KEV_ANGSTROM * 1000 / energy for energy in (ka1_ev, ka2_ev, kb1_ev))
        mean = (KA1_WEIGHT * ka1 + KA2_WEIGHT * ka2) / (KA1_WEIGHT + KA2_WEIGHT)
        sources += [_entry(anode, "Ka", "Kα", mean), _entry(anode, "Ka1", "Kα1", ka1),
                    _entry(anode, "Ka2", "Kα2", ka2), _entry(anode, "Kb1", "Kβ1", kb1)]
    return sources
