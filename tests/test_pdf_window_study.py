import csv
import io
import json
from math import pi

import numpy as np
import pytest

from xray_workbench._types import Array, JsonObject
from xray_workbench.pdf_window_study import pdf_window_study


def _analytic_pdf(r: Array, q_min: float, q_max: float, window: str) -> Array:
    """Closed form of the finite sine transform for F(Q) = Q."""
    if window == "lorch":
        k = pi / q_max
        # Q times the Lorch window is sin(kQ)/k. Integrate the product of sines.
        low, high = k - r, k + r
        at_max = q_max * (np.sinc(low * q_max / pi) - np.sinc(high * q_max / pi))
        at_min = q_min * (np.sinc(low * q_min / pi) - np.sinc(high * q_min / pi))
        return (at_max - at_min) / (pi * k)

    nonzero = r[1:]
    at_max = np.sin(q_max * nonzero) / nonzero**2 - q_max * np.cos(q_max * nonzero) / nonzero
    at_min = np.sin(q_min * nonzero) / nonzero**2 - q_min * np.cos(q_min * nonzero) / nonzero
    return np.concatenate(([0.0], 2 / pi * (at_max - at_min)))


@pytest.mark.parametrize("window, use_csv", [("none", False), ("lorch", True)])
def test_both_pdf_curves_match_the_analytic_finite_integral(window: str, use_csv: bool) -> None:
    q = np.linspace(0.5, 5.5, 2501)
    payload: JsonObject = {
        "qmax_a_inv_angstrom": 2.501, "qmax_b_inv_angstrom": 4.5,
        "r_max_angstrom": 3, "r_points": 101, "window": window,
    }
    if use_csv:
        payload["sq_data"] = "# normalized S(Q)\nq,S\n" + "".join(f"{sample:.3f},2\n" for sample in q)
    else:
        payload.update(q=q.tolist(), sq=[2.0] * len(q))

    answer = pdf_window_study(payload)
    data = answer["data"]
    assert answer["tool"] == "pdf_window"
    assert data["window"] == window
    assert data["q_min_inv_angstrom"] == 0.5
    assert data["qmax_a_inv_angstrom"] == 2.501
    assert data["qmax_b_inv_angstrom"] == 4.5
    assert data["qmax_a_effective_inv_angstrom"] == 2.5
    assert data["qmax_b_effective_inv_angstrom"] == 4.5
    assert (data["samples_a"], data["samples_b"]) == (1001, 2001)
    assert any("snaps down" in note for note in answer["notes"])
    assert "10.1107/S1600576725004340" in answer["provenance"]["method"]

    r = np.linspace(0, 3, 101)
    np.testing.assert_array_equal(data["r_angstrom"], r)
    for key, cutoff in (("g_a_inv_angstrom2", 2.5), ("g_b_inv_angstrom2", 4.5)):
        expected = _analytic_pdf(r, q_min=0.5, q_max=cutoff, window=window)
        assert data[key][0] == 0.0
        # At dQ = 0.002, the composite trapezoid error bound is below 4e-5.
        np.testing.assert_allclose(data[key], expected, rtol=0, atol=4e-5)

    graph, = answer["plots"]
    assert graph["x"] == data["r_angstrom"]
    assert [trace["y"] for trace in graph["traces"]] == [data["g_a_inv_angstrom2"], data["g_b_inv_angstrom2"]]
    reader = csv.DictReader(io.StringIO(data["csv"]))
    fields = ["r_angstrom", "g_a_inv_angstrom2", "g_b_inv_angstrom2"]
    assert reader.fieldnames == fields
    assert [[float(row[key]) for key in fields] for row in reader] == [
        list(values) for values in zip(*(data[key] for key in fields), strict=True)
    ]
    json.dumps(answer, allow_nan=False)


def _request() -> JsonObject:
    return {
        "q": [0, 1, 2, 3, 4, 5], "sq": [1.0] * 6,
        "qmax_a_inv_angstrom": 2.1, "qmax_b_inv_angstrom": 4.1, "r_max_angstrom": 2,
    }


@pytest.mark.parametrize("overrides, reason", [
    ({"q": [0, 1, 2, 3, 4, 4]}, "increase strictly"),
    ({"q": [0, 1, 2, 3, 4, 3.5]}, "increase strictly"),
    ({"q": [0, 1, 2, 3, 4, 201]}, "Q must lie in"),
    ({"sq": [1, 1, 1, 1, 1, float("nan")]}, "must be finite"),
    ({"sq": [1, 1, 1, 1, 1, None]}, "must be finite"),
    ({"sq": [1, 1, 1, 1, 1, {}]}, "numeric arrays"),
    ({"sq": [1] * 5}, "equal-length arrays"),
    ({"qmax_a_inv_angstrom": False}, "must be a finite number"),
    ({"qmax_b_inv_angstrom": None}, "Qmax B.*must be a finite number"),
    ({"qmax_b_inv_angstrom": float("inf")}, "must be finite"),
    ({"qmax_b_inv_angstrom": 5.01}, "Qmax B.*must be at most 5"),
    ({"qmax_a_inv_angstrom": 4.2}, "strictly less"),
    ({"qmax_a_inv_angstrom": 4.1}, "strictly less"),
    ({"qmax_a_inv_angstrom": 1.9}, "at least three Q samples"),
    ({"qmax_b_inv_angstrom": 2.2}, "same last Q sample"),
    ({"window": "hann"}, "Window must be none or lorch"),
    ({"r_points": 3.5}, "r grid points must be an integer"),
    ({"r_points": 3}, "r grid is too coarse for Qmax"),
    ({"r_max_angstrom": 4}, "conservative sampling limit"),
    ({"sq": [1, 1, 1, 1, 1e308, 1]}, "exceeds the finite numerical range"),
])
def test_rejects_invalid_study_inputs(overrides: JsonObject, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        pdf_window_study({**_request(), **overrides})


def test_validates_csv_rows_past_both_cutoffs() -> None:
    with pytest.raises(ValueError, match="exactly q,S"):
        pdf_window_study({
            "sq_data": "q,S\n0,1\n1,1\n2,1\n3,1\n4,1\n5,1,unexpected",
            "qmax_a_inv_angstrom": 2, "qmax_b_inv_angstrom": 4, "r_max_angstrom": 2,
        })


def test_combined_work_budget_applies_to_both_curves() -> None:
    q = np.linspace(0, 30, 3001).tolist()
    with pytest.raises(ValueError, match="20 million Qxr samples across both PDF curves"):
        pdf_window_study({
            "q": q, "sq": [1.0] * len(q), "qmax_a_inv_angstrom": 25, "qmax_b_inv_angstrom": 29,
            "r_max_angstrom": 2, "r_points": 4000,
        })
