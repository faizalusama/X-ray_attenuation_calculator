"""The attenuation backend contract.

The point of these tests is that a second data source cannot be added quietly:
it must declare a domain, the channels it really provides, and its identity,
and it must not be able to shadow an existing one.
"""
import numpy as np
import pytest
import xraydb

from xray_workbench.backends import (
    COHERENT,
    DEFAULT_BACKEND,
    INCOHERENT,
    PHOTOELECTRIC,
    BackendInfo,
    available_backends,
    get_backend,
    register_backend,
)
from xray_workbench.physics import calculate


def layer(formula="SiO2", **extra):
    return {"components": [{"formula": formula, "fraction": 1}],
            "density_g_cm3": 2.2, "thickness_mm": 1, **extra}


def request(**extra):
    return {"energy": {"min_keV": 5, "max_keV": 120, "points": 60, "reference_keV": 30},
            "layers": [layer()], **extra}


def test_default_backend_is_elam_and_is_listed_first():
    assert DEFAULT_BACKEND == "elam"
    assert available_backends()[0] == "elam"


def test_backend_reports_its_identity_and_installed_dataset_version():
    info = get_backend().info
    assert info.identifier == "elam"
    assert info.dataset_version == xraydb.__version__
    assert "Elam" in info.dataset
    assert set(info.channels) == {PHOTOELECTRIC, COHERENT, INCOHERENT}


def test_unknown_backend_names_the_alternatives_instead_of_falling_back():
    # A silent fallback to the default would let a result claim provenance it
    # does not have, so the failure must be loud.
    with pytest.raises(ValueError, match="Unknown attenuation backend"):
        get_backend("xcom")
    with pytest.raises(ValueError, match="elam"):
        get_backend("xcom")


def test_registering_over_an_existing_backend_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_backend("elam", get_backend)


@pytest.mark.parametrize("channels,message", [
    ((PHOTOELECTRIC, COHERENT), "cannot serve a narrow-beam model"),
    ((PHOTOELECTRIC, COHERENT, INCOHERENT, "compton"), "unknown channels"),
])
def test_backend_metadata_rejects_impossible_channel_sets(channels, message):
    with pytest.raises(ValueError, match=message):
        BackendInfo(identifier="t", name="T", dataset="d", dataset_version="1",
                    energy_range_keV=(1.0, 800.0), recommended_range_keV=(1.0, 250.0),
                    channels=channels, atomic_number_range=(1, 98),
                    edge_convention="none", reference_url="https://example.invalid")


def test_backend_metadata_rejects_a_recommended_range_outside_its_own_domain():
    with pytest.raises(ValueError, match="recommends a range outside its own domain"):
        BackendInfo(identifier="t", name="T", dataset="d", dataset_version="1",
                    energy_range_keV=(10.0, 100.0), recommended_range_keV=(1.0, 250.0),
                    channels=(PHOTOELECTRIC, COHERENT, INCOHERENT), atomic_number_range=(1, 98),
                    edge_convention="none", reference_url="https://example.invalid")


def test_coefficients_are_finite_nonnegative_and_in_declared_channels():
    energies = np.geomspace(1.0, 800.0, 200)
    channels = get_backend().mass_attenuation("Pb", energies)
    assert set(channels) == set(get_backend().info.channels)
    for values in channels.values():
        assert values.shape == energies.shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)


def test_edges_are_sorted_and_within_the_declared_atomic_range():
    edges = get_backend().edges("Ba")
    assert edges == sorted(edges, key=lambda item: (item[1], item[0]))
    assert all(energy > 0 for _, energy in edges)
    assert any(shell == "K" and 37 < energy < 38 for shell, energy in edges)


def test_selecting_the_default_explicitly_changes_nothing():
    # The refactor to a pluggable backend must not have moved any number.
    implicit = calculate(request())
    explicit = calculate(request(backend="elam"))
    assert implicit["transmission"] == explicit["transmission"]
    assert implicit["reference"] == explicit["reference"]
    assert implicit["layers"][0]["mu_mass_cm2_g"] == explicit["layers"][0]["mu_mass_cm2_g"]


def test_result_provenance_identifies_the_data_source():
    provenance = calculate(request())["provenance"]
    assert provenance["backend"]["identifier"] == "elam"
    assert provenance["backend"]["dataset_version"] == xraydb.__version__
    assert provenance["backend"]["recommended_range_keV"] == [1.0, 250.0]
    # Schema-version-1 consumers still read these two names.
    assert provenance["engine"] == "XrayDB / Elam"
    assert provenance["xraydb_version"] == xraydb.__version__


def test_unknown_backend_is_rejected_through_the_public_request():
    with pytest.raises(ValueError, match="Unknown attenuation backend"):
        calculate(request(backend="xraylib"))
    with pytest.raises(ValueError, match="text identifier"):
        calculate(request(backend=7))


def test_out_of_recommended_range_warning_comes_from_the_backend():
    result = calculate({"energy": {"min_keV": 5, "max_keV": 400, "points": 40, "reference_keV": 30},
                        "layers": [layer()]})
    assert any("250 keV" in warning for warning in result["warnings"])
