"""Radial q reach of a flat detector perpendicular to the incident beam."""
from __future__ import annotations

from math import pi

import numpy as np

from ._types import JsonObject
from .physics import _number
from .sources import HC_KEV_ANGSTROM


def detector_coverage(payload: JsonObject) -> JsonObject:
    """Map beam-centred radii on the detector plane to elastic scattering angles and q."""
    from .scattering_tools import plot, result

    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    distance = _number(payload.get("distance_mm", 1000), "Sample-to-detector distance (mm)", positive=True)
    inner = _number(payload.get("inner_radius_mm", 2), "Inner radius (mm)", minimum=0)
    outer = _number(payload.get("outer_radius_mm", 100), "Outer radius (mm)", positive=True)
    if outer <= inner:
        raise ValueError("Outer radius must be greater than inner radius.")

    wavelength = HC_KEV_ANGSTROM / energy
    radius = np.linspace(inner, outer, 201)
    two_theta = np.arctan2(radius, distance)  # Full scattering angle in radians.
    q = (4 * pi / wavelength) * np.sin(two_theta / 2)
    two_theta_deg = np.degrees(two_theta)
    q_min, q_max = float(q[0]), float(q[-1])
    angle_min, angle_max = float(two_theta_deg[0]), float(two_theta_deg[-1])
    data = {
        "energy_keV": energy, "wavelength_angstrom": wavelength, "distance_mm": distance,
        "inner_radius_mm": inner, "outer_radius_mm": outer,
        "two_theta_min_deg": angle_min, "two_theta_max_deg": angle_max,
        "q_min_inv_angstrom": q_min, "q_max_inv_angstrom": q_max,
        "radius_mm": radius.tolist(), "two_theta_deg": two_theta_deg.tolist(), "q_inv_angstrom": q.tolist(),
    }
    answer = result(
        "detector", data,
        [("Minimum q", q_min, "Å⁻¹"), ("Maximum q", q_max, "Å⁻¹"),
         ("Minimum 2θ", angle_min, "°"), ("Maximum 2θ", angle_max, "°")],
        [plot("Radial q reach", "Radius from beam centre (mm)", "q (Å⁻¹)", radius, {"q": q})],
        ["Assumes elastic scattering from a point sample onto a flat detector perpendicular to the incident beam.",
         "Both radii are measured in the detector plane from the direct beam. Use the innermost usable radius "
         "outside any beamstop shadow and the outermost usable radius.",
         "The bounds describe radial reach only. Detector edges, module gaps and masks can leave partial or "
         "missing rings. Full azimuthal coverage is not implied.",
         "The plotted points sample the ideal radius-to-q relation. Pixel size, beam divergence and "
         "wavelength spread are omitted from the model."],
    )
    answer["provenance"].update(
        optical_data="None (geometry only)",
        geometry_reference="L. Foucar (2016), §2.5, Eq. (1), doi:10.1107/S1600576716009201",
    )
    return answer
