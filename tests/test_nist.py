"""Off-edge NIST reference comparisons through the public calculation API.

Reference data transcribed from NIST on 2026-09-19, energies in keV and
total mass attenuation coefficients in cm^2/g:
https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html
https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html
https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html
https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html

The 1% criterion is a regression agreement threshold for selected rounded
off-edge reference values. It is not a claim of 1% physical accuracy.
"""

import math
import unittest

from xray_workbench.physics import calculate

BENCHMARKS = {
    "Al": {20.0: 3.441, 50.0: 0.3681, 100.0: 0.1704},
    "Fe": {20.0: 25.68, 50.0: 1.958, 100.0: 0.3717},
    "Pb": {20.0: 86.36, 50.0: 8.041, 100.0: 5.549},
    "H2O": {20.0: 0.8096, 50.0: 0.2269, 100.0: 0.1707},
}


def request(components, energy_keV, density=1.0, thickness_mm=1.0):
    """Use explicit inputs so the check cannot depend on preset defaults."""
    return {
        "energy": {
            "min_keV": 10.0,
            "max_keV": 110.0,
            "points": 100,
            "spacing": "log",
            "reference_keV": energy_keV,
        },
        "layers": [{
            "name": "Independent NIST benchmark",
            "basis": "mass",
            "components": components,
            "density_g_cm3": density,
            "thickness_mm": thickness_mm,
            "angle_deg": 0,
            "porosity": 0,
            "density_mode": "bulk",
            "density_uncertainty_pct": 0,
            "thickness_uncertainty_pct": 0,
        }],
        "target_transmission": 0.1,
        "spectrum": None,
        "uncertainty": {"enabled": False, "samples": 1000, "seed": 42},
    }


class NistBenchmarks(unittest.TestCase):
    def test_independent_pure_material_coefficients(self):
        for formula, values in BENCHMARKS.items():
            for energy_keV, expected in values.items():
                with self.subTest(formula=formula, energy_keV=energy_keV):
                    result = calculate(request(
                        [{"formula": formula, "fraction": 1.0}], energy_keV
                    ))
                    observed = result["layers"][0]["reference"]["mu_mass_cm2_g"]
                    self.assertAlmostEqual(observed / expected, 1.0, delta=0.01)

    def test_fe_k_edge_one_sided_nist_limits(self):
        # NIST gives two limiting values at 7.112 keV. XrayDB 4.5.8 stores
        # the repeated Elam log-energy knot at 8.86954, i.e. 7112.0086966 eV,
        # while atomic-edge metadata say 7112 eV. A 1e-8 relative offset can
        # still be BELOW the coefficient step. Explicit 1e-4 offsets bracket
        # the rounding discrepancy without shifting backend coefficients;
        # the smooth-region change remains far below the 1% tolerance.
        for offset, expected in ((-1e-4, 53.19), (1e-4, 407.6)):
            with self.subTest(edge_side="below" if offset < 0 else "above"):
                payload = request([{"formula": "Fe", "fraction": 1.0}], 7.112 * (1 + offset))
                payload["energy"].update(min_keV=6.0, max_keV=9.0, points=2)
                result = calculate(payload)
                observed = result["layers"][0]["reference"]["mu_mass_cm2_g"]
                self.assertAlmostEqual(observed / expected, 1.0, delta=0.01)

    def test_nist_equal_mass_mixture_prevents_squared_fraction_regression(self):
        # With two 50% components, the original program accidentally returned
        # half the expected linear attenuation by multiplying each w_i twice.
        density = 4.0
        thickness_mm = 0.4
        expected_mass = 0.5 * BENCHMARKS["Al"][50.0] + 0.5 * BENCHMARKS["Fe"][50.0]
        expected_tau = expected_mass * density * thickness_mm / 10.0
        result = calculate(request([
            {"formula": "Al", "fraction": 50.0},
            {"formula": "Fe", "fraction": 50.0},
        ], 50.0, density, thickness_mm))
        observed_mass = result["layers"][0]["reference"]["mu_mass_cm2_g"]
        self.assertAlmostEqual(observed_mass / expected_mass, 1.0, delta=0.01)
        # Work in optical depth so rounding tolerance does not become an
        # arbitrary transmission tolerance for highly absorbing samples.
        observed_tau = -math.log(result["reference"]["transmission"])
        self.assertAlmostEqual(observed_tau / expected_tau, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
