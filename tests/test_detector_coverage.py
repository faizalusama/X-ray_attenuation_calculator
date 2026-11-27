"""Independent reference values from pyFAI 2026.9.0, pyFAI.geometry.Geometry.

Reproduction (dist in m, cols are the three radii in mm in each test case):

    from scipy.constants import h, c, e
    from pyFAI.detectors import Detector
    from pyFAI.geometry import Geometry
    g = Geometry(dist=dist, poni1=.0005, poni2=.0005, detector=Detector(pixel1=.001, pixel2=.001),
                 wavelength=h*c/(energy_keV*1000*e))
    row, col = np.zeros(3), np.array(cols)
    q_A_inv = g.qFunction(row, col) / 10
    two_theta_deg = np.rad2deg(g.tth(row, col))

Zero rotations put the beam at pixel (0, 0)'s centre. The reference values are
pyFAI predictions at the specified pixels. Geometry reference: Kieffer et al. (2020),
doi:10.1107/S1600577520000776. Flat-detector equation: Foucar (2016), §2.5,
Eq. (1), doi:10.1107/S1600576716009201.
"""
import numpy as np
import pytest

from xray_workbench.detector_coverage import detector_coverage


@pytest.mark.parametrize("payload,distance,radii,wavelength,angles,q", [
    ({"energy_keV": 8}, 1000, [2, 51, 100], 1.5498024804150032,
     [0.11459140623778596, 2.919555254052032, 5.710593137499643],
     [0.008108356983327934, 0.20656207946292043, 0.4039078833656692]),
    ({"energy_keV": 12.398419843320026, "inner_radius_mm": 0, "outer_radius_mm": 200, "distance_mm": 100},
     100, [0, 100, 200], 1.0, [0, 45, 63.43494882292201], [0, 4.8089418390747705, 6.606531998388247]),
])
def test_detector_bounds_and_curve_against_pyfai(payload, distance, radii, wavelength, angles, q):
    answer = detector_coverage(payload)
    data = answer["data"]
    assert answer["tool"] == "detector"
    assert data["energy_keV"] == payload["energy_keV"]
    assert data["distance_mm"] == distance
    assert (data["inner_radius_mm"], data["outer_radius_mm"]) == (radii[0], radii[-1])
    assert data["wavelength_angstrom"] == pytest.approx(wavelength, rel=1e-13)
    assert [data["two_theta_min_deg"], data["two_theta_max_deg"]] == pytest.approx(angles[::2], rel=1e-13)
    assert [data["q_min_inv_angstrom"], data["q_max_inv_angstrom"]] == pytest.approx(q[::2], rel=1e-13)

    radius = data["radius_mm"]
    assert np.all(np.diff(radius) > 0)
    assert np.all(np.diff(data["q_inv_angstrom"]) > 0)
    indices = [0, len(radius) // 2, -1]
    assert [radius[i] for i in indices] == pytest.approx(radii)
    assert [data["two_theta_deg"][i] for i in indices] == pytest.approx(angles, rel=1e-13)
    assert [data["q_inv_angstrom"][i] for i in indices] == pytest.approx(q, rel=1e-13)
    curve, = answer["plots"]
    assert curve["x"] == radius
    assert curve["traces"][0]["y"] == data["q_inv_angstrom"]


@pytest.mark.parametrize("fields,reason", [
    ({"distance_mm": 0}, "distance .*greater than zero"),
    ({"distance_mm": -10}, "distance .*greater than zero"),
    ({"distance_mm": float("nan")}, "distance .*finite"),
    ({"inner_radius_mm": -1}, "Inner radius .*at least 0"),
    ({"outer_radius_mm": 0}, "Outer radius .*greater than zero"),
    ({"outer_radius_mm": float("inf")}, "Outer radius .*finite"),
    ({"outer_radius_mm": True}, "Outer radius .*finite number"),
    ({"outer_radius_mm": 1}, "Outer radius must be greater than inner radius"),
    ({"inner_radius_mm": 100}, "Outer radius must be greater than inner radius"),
    ({"energy_keV": None}, "Energy .*finite number"),
    ({"energy_keV": 0}, "Energy .*at least 1"),
    ({"energy_keV": 801}, "Energy .*at most 800"),
])
def test_detector_rejects_invalid_inputs(fields, reason):
    with pytest.raises(ValueError, match=reason):
        detector_coverage({"energy_keV": 8, **fields})
