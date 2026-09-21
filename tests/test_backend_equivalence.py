"""XrayDB/Elam and xraylib are the same data, and the suite must keep saying so.

Measured 20 September 2026 (xraylib 4.3.0, XrayDB 4.5.8): on Windows the two
packages return bit-identical mass attenuation coefficients for every element,
energy and channel sampled. CI later showed that on some Linux and macOS
builds they differ, so equality is platform-dependent; the tests bound the
difference instead and print its measured size. They expose one evaluated
dataset through two interfaces.

These tests exist for two reasons. First, so nobody can quote "validated
against xraylib" as cross-database validation — the workbench declares the
relationship in metadata and refuses to list the pair as a cross-check. Second,
so that if either upstream package ever changes its tables, the change is
reported here instead of silently moving published numbers.

If `test_elam_and_xraylib_evaluate_the_same_data` starts failing, that is real news:
investigate which package changed and why before touching the tolerance.
"""
import numpy as np
import pytest

xraylib = pytest.importorskip("xraylib", reason="optional backend")

from xray_workbench.backends import available_backends, get_backend  # noqa: E402
from xray_workbench.backends.compare import (  # noqa: E402
    compare_element,
    independent_pairs,
)

ELEMENTS = ("H", "C", "O", "Al", "Si", "Fe", "Cu", "Ba", "W", "Pb", "U")
ENERGIES_KEV = np.array([5.0, 10.0, 17.3, 30.0, 60.0, 100.0, 200.0, 500.0, 700.0])


def test_xraylib_is_registered_and_optional():
    assert "xraylib" in available_backends()
    assert get_backend("xraylib").info.identifier == "xraylib"


#: Largest relative difference still read as "the same data evaluated by a
#: different C maths library". Both packages evaluate the same log-log splines;
#: on Windows the results match bit for bit, while other platforms' compiled
#: wheels can differ in the last few units of a float64 (~1e-16 relative).
#: Genuinely different evaluated data would differ by ~1e-3 or more, many
#: orders of magnitude above this bound, so the bound cannot hide a real change.
SAME_DATA_RTOL = 1e-9


@pytest.mark.parametrize("element", ELEMENTS)
def test_elam_and_xraylib_evaluate_the_same_data(element):
    elam = get_backend("elam").mass_attenuation(element, ENERGIES_KEV)
    xlib = get_backend("xraylib").mass_attenuation(element, ENERGIES_KEV)
    for channel in ("photoelectric", "coherent", "incoherent"):
        worst = float(np.max(np.abs(elam[channel] - xlib[channel]) / xlib[channel]))
        # Reported on every platform so the measured magnitude is on record.
        print(f"{element:>2} {channel:<13} max relative difference {worst:.3e}")
        assert worst <= SAME_DATA_RTOL, (
            f"{element} {channel} differs between XrayDB and xraylib by {worst:.3e} relative. "
            f"That is far above floating-point rounding, so the two packages no longer "
            f"evaluate the same data. Investigate the upstream change; do not widen the bound.")


def test_xraylib_declares_that_it_shares_data_with_elam():
    info = get_backend("xraylib").info
    assert "elam" in info.shares_data_with
    assert not info.is_independent_of(get_backend("elam").info)


def test_the_two_backends_are_not_offered_as_a_cross_check():
    # The whole point: agreement between these two proves nothing, so the pair
    # must never appear in the list of legitimate cross-checks.
    assert ("elam", "xraylib") not in independent_pairs()
    assert independent_pairs() == []


def test_comparison_reports_identity_and_refuses_to_claim_validation():
    comparison = compare_element("Pb", "elam", "xraylib", points=60)
    summary = comparison.summary()
    assert summary["independent"] is False
    # Bit-identical on Windows; within floating-point rounding elsewhere.
    assert summary["max_relative_difference"] <= SAME_DATA_RTOL
    assert "NOT independent" in summary["interpretation"]


def test_comparison_restricts_itself_to_the_overlapping_domain():
    comparison = compare_element("Cu", "elam", "xraylib", points=40)
    low, high = comparison.overlap_keV
    # Elam is declared 1-800 keV, xraylib 0.1-800.026 keV: the overlap starts
    # at the higher floor and stops at the lower ceiling.
    assert low == 1.0
    assert high == pytest.approx(800.0)
    assert comparison.energies_keV.min() >= low
    assert comparison.energies_keV.max() <= high


def test_comparing_a_backend_with_itself_is_refused():
    with pytest.raises(ValueError, match="itself measures nothing"):
        compare_element("Si", "elam", "elam")


def test_xraylib_refuses_to_extrapolate_outside_its_domain():
    with pytest.raises(ValueError, match="outside that domain"):
        get_backend("xraylib").mass_attenuation("Si", np.array([0.05]))


def test_edge_labels_differ_even_though_coefficients_do_not():
    # A concrete reason the two backends are not interchangeable for edge work.
    elam_k = dict(get_backend("elam").edges("Ba"))["K"]
    xlib_k = dict(get_backend("xraylib").edges("Ba"))["K"]
    assert elam_k != xlib_k
    assert abs(elam_k - xlib_k) * 1000 == pytest.approx(0.4, abs=0.05)  # ~0.4 eV
