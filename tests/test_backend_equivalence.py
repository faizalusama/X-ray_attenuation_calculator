"""XrayDB/Elam and xraylib are the same data, and the suite must keep saying so.

Measured 20 September 2026 (xraylib 4.3.0, XrayDB 4.5.8): the two packages
return **bit-identical** mass attenuation coefficients for every element,
energy and channel sampled. They expose one evaluated dataset through two
interfaces.

These tests exist for two reasons. First, so nobody can quote "validated
against xraylib" as cross-database validation — the workbench declares the
relationship in metadata and refuses to list the pair as a cross-check. Second,
so that if either upstream package ever changes its tables, the change is
reported here instead of silently moving published numbers.

If `test_elam_and_xraylib_are_bit_identical` starts failing, that is real news:
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


@pytest.mark.parametrize("element", ELEMENTS)
def test_elam_and_xraylib_are_bit_identical(element):
    elam = get_backend("elam").mass_attenuation(element, ENERGIES_KEV)
    xlib = get_backend("xraylib").mass_attenuation(element, ENERGIES_KEV)
    for channel in ("photoelectric", "coherent", "incoherent"):
        # Deliberately exact. These are not two evaluations that happen to
        # agree; they are one dataset read twice.
        assert np.array_equal(elam[channel], xlib[channel]), (
            f"{element} {channel} diverged between XrayDB and xraylib. This is a real "
            f"upstream change, not a tolerance problem.")


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
    assert summary["bit_identical"] is True
    assert summary["independent"] is False
    assert summary["max_relative_difference"] == 0.0
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
