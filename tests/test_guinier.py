"""Analytic checks and fixed-window regressions on the SASDUE4 consensus SAXS curve.

Trewhella, Vachette and Larsen (2024), IUCrJ 11, 762–779, §2.2 and Table 2.
https://doi.org/10.1107/S205225252400486X
Source record https://www.sasbdb.org/data/SASDUE4/
Unmodified data https://www.sasbdb.org/media/intensities_files/SASDUE4.dat
SASBDB permits reuse with author attribution (https://www.sasbdb.org/about/).
"""

import json
from hashlib import sha256
from math import exp
from pathlib import Path

import pytest

from xray_workbench._types import JsonObject
from xray_workbench.guinier import guinier

_WINDOW: JsonObject = {"qmin_inv_angstrom": 0.01, "qmax_inv_angstrom": 0.03}


@pytest.mark.parametrize("weighted, expected_rg, expected_i0, expected_slope, expected_rms, expected_chi2", [
    (False, 14.444679810730591, 1.0043843705307729, -69.549591611509314, 0.008709117253566344, None),
    (True, 14.499421700112727, 1.0060907514343314, -70.077743212566602, 0.008797655407313301, 1.230366676552294),
])
def test_sasdue4_fixed_window(weighted: bool, expected_rg: float, expected_i0: float,
                             expected_slope: float, expected_rms: float, expected_chi2: float | None) -> None:
    raw = Path(__file__).with_name("data").joinpath("sasdue4.dat").read_bytes()
    assert sha256(raw).hexdigest() == "a977858b32cc338fdc0e17410ad2056fa41421581a3a46bce298c34c9697a601"
    # The DAT file is in Å⁻¹, though the SASBDB web plot uses nm⁻¹.
    # Preserve all samples and change only the delimiters and sigma inclusion.
    width = 3 if weighted else 2
    csv_text = "\n".join(",".join(line.split()[:width]) for line in raw.decode("ascii").splitlines()
                         if line.strip() and not line.startswith("#"))
    # Independent references from scipy.linalg.lstsq with a two-column design
    # matrix, scaling each row and target by I/sigma in the weighted case.
    # This window differs from Table 2's AUTORG window (qRg 0.17–1.30,
    # Rg 14.54 Å). Tolerances here cover numerical rounding on the fixed fit.
    output = guinier({"intensity_data": csv_text, "qmin_inv_angstrom": 0.0, "qmax_inv_angstrom": 0.083})
    data = output["data"]
    assert data["input_rows"] == 223
    assert data["selected_rows"] == 101
    assert data["qmin_used_inv_angstrom"] == 0.0103382
    assert data["qmax_used_inv_angstrom"] == 0.08249249
    assert data["weighting"] == ("1/(sigma/I)^2" if weighted else "equal")
    assert data["rg_angstrom"] == pytest.approx(expected_rg, rel=1e-12)
    assert data["i0"] == pytest.approx(expected_i0, rel=1e-12)
    assert data["slope_angstrom2"] == pytest.approx(expected_slope, rel=1e-12)
    assert data["rms_ln_intensity_residual"] == pytest.approx(expected_rms, rel=1e-12)
    if expected_chi2 is None:
        assert data["reduced_chi_squared_log"] is None
    else:
        assert data["reduced_chi_squared_log"] == pytest.approx(expected_chi2, rel=1e-12)


def test_analytic_fit_selection_and_export() -> None:
    # Synthetic curve from the Guinier equation with Rg = 15 Å and I(0) = 2.
    expected_q = [0.01, 0.02, 0.03]
    rows = [f'{q},{2 * exp(-75 * q * q):.17g}' for q in reversed(expected_q)]
    csv_text = "\ufeff# Background-corrected SAXS\n\nq,I\n0.005,nan\n" + "\n".join(rows) + "\n0.04,-1"
    answer = json.loads(json.dumps(guinier({**_WINDOW, "intensity_data": csv_text}), allow_nan=False))
    assert answer["tool"] == "guinier"
    data = answer["data"]
    assert data["q_inv_angstrom"] == expected_q
    assert data["input_rows"] == 5
    assert data["selected_rows"] == 3
    assert data["rg_angstrom"] == pytest.approx(15.0, rel=1e-12)
    assert data["i0"] == pytest.approx(2.0, rel=1e-12)
    assert data["qmax_rg"] == pytest.approx(0.45)
    assert data["ln_intensity_residuals"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)
    assert answer["plots"][0]["x"] == pytest.approx([q * q for q in expected_q])
    assert answer["plots"][1]["traces"][0]["y"] == data["ln_intensity_residuals"]
    assert "10.1107/S205225252400486X" in answer["provenance"]["method_reference"]
    assert "background-corrected" in answer["provenance"]["input_data"]
    assert all("above the usual" not in note for note in answer["notes"])


def test_qrg_check_uses_last_fitted_point() -> None:
    csv_text = "\n".join(f"{q},{exp(-3072 * q * q):.17g}" for q in (0.01, 0.02, 0.03))
    answer = guinier({**_WINDOW, "intensity_data": csv_text})
    assert answer["data"]["qmax_rg"] == pytest.approx(2.88)
    assert any("above the usual globular guide of 1.3" in note for note in answer["notes"])

    # A requested upper bound beyond the last sample should not itself trigger a warning.
    short_range = "\n".join(f"{q},{exp(-75 * q * q):.17g}" for q in (0.01, 0.02, 0.03))
    answer = guinier({**_WINDOW, "qmax_inv_angstrom": 1.0, "intensity_data": short_range})
    assert answer["data"]["qmax_rg"] == pytest.approx(0.45)
    assert all("above the usual" not in note for note in answer["notes"])


@pytest.mark.parametrize("csv_text, reason", [
    ("0.01,9\n0.02,8", "at least three"),
    ("0.02,9\n0.02,8\n0.02,7", "nonzero range"),
    ("0.01,9\n0.02,9\n0.03,9", "slope must be negative"),
    ("0.01,7\n0.02,8\n0.03,9", "slope must be negative"),
    ("0.01,9\n0.02,0\n0.03,7", "finite and positive"),
    ("0.01,9\n0.02,nan\n0.03,7", "finite and positive"),
    ("0.01,9,1\n0.02,8,0\n0.03,7,1", "finite and positive"),
    ("0.01,9,1\n0.02,8,inf\n0.03,7,1", "finite and positive"),
    ("nan,9\n0.02,8\n0.03,7", "q on CSV line 1 must be finite"),
    ("q,I,sigma\n0.01,9,1\n0.02,8\n0.03,7,1", "column layout"),
    ('0.01,"9\n0.02,8', "Malformed intensity CSV"),
    pytest.param("#" + "x" * 800_000, "at most 800,000 characters", id="character-limit"),
    pytest.param("0.01,9\n" * 20_001, "at most 20,000 data rows", id="row-limit"),
])
def test_invalid_csv(csv_text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        guinier({**_WINDOW, "intensity_data": csv_text})


@pytest.mark.parametrize("bounds, reason", [
    ({}, "Minimum q"),
    ({"qmin_inv_angstrom": 0.0}, "Maximum q"),
    ({"qmin_inv_angstrom": 0.03, "qmax_inv_angstrom": 0.03}, "Maximum q must exceed minimum q"),
    ({"qmin_inv_angstrom": -1.0, "qmax_inv_angstrom": 0.03}, "Minimum q"),
    ({"qmin_inv_angstrom": 0.0, "qmax_inv_angstrom": float("inf")}, "Maximum q"),
])
def test_invalid_bounds(bounds: JsonObject, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        guinier({**bounds, "intensity_data": "0.01,9\n0.02,8\n0.03,7"})
