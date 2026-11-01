"""X-ray sources: published line wavelengths, consistent units, and agreement with the Elam data."""

import re
from pathlib import Path

import pytest
import xraydb

from xray_workbench.sources import HC_KEV_ANGSTROM, LINES_EV, xray_sources

ROOT = Path(__file__).resolve().parents[1]
BY_ID = {s["id"]: s for s in xray_sources()}
#: Wavelengths in Å that X-ray laboratories use, typed independently of the source table.
KNOWN_ANGSTROM = {"Cu-Ka1": 1.5406, "Cu-Ka": 1.5418, "Mo-Ka1": 0.7093, "Mo-Ka": 0.7107, "Co-Ka1": 1.7890, "Cr-Ka1": 2.2897,
                  "Fe-Ka1": 1.9360, "Ag-Ka1": 0.5594}


@pytest.mark.parametrize("source,expected", KNOWN_ANGSTROM.items())
def test_wavelengths_match_laboratory_values(source, expected):
    assert BY_ID[source]["wavelength_angstrom"] == pytest.approx(expected, abs=1.5e-4)


def test_every_anode_has_each_line_and_consistent_units():
    sources = xray_sources()
    assert len(sources) == 4 * len(LINES_EV)
    assert all(1 <= s["energy_keV"] <= 800 for s in sources), "every line must be a valid reference energy"
    for s in sources:
        assert s["energy_keV"] * s["wavelength_angstrom"] == pytest.approx(HC_KEV_ANGSTROM, rel=1e-14)


@pytest.mark.parametrize("anode", LINES_EV)
def test_lines_agree_with_the_elam_tables(anode):
    elam = xraydb.xray_lines(anode)
    for line in ("Ka1", "Ka2", "Kb1"):
        assert BY_ID[f"{anode}-{line}"]["energy_keV"] * 1000 == pytest.approx(elam[line].energy, rel=2e-3)


def test_ka_mean_lies_between_its_components_and_nearer_ka1():
    wavelength = {line: BY_ID[f"Cu-{line}"]["wavelength_angstrom"] for line in ("Ka", "Ka1", "Ka2")}
    assert wavelength["Ka1"] < wavelength["Ka"] < wavelength["Ka2"]
    assert wavelength["Ka"] - wavelength["Ka1"] < wavelength["Ka2"] - wavelength["Ka"]


def test_the_interface_uses_the_same_constant():
    script = (ROOT / "static" / "dashboard.js").read_text(encoding="utf-8")
    assert float(re.search(r"HC_KEV_ANGSTROM = ([0-9.]+)", script).group(1)) == HC_KEV_ANGSTROM
