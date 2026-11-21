"""Single-ray numerical references from TomoPy, independent of the workbench.

Gürsoy et al. (2014), §4.1, Eq. (1), DOI 10.1107/S1600577514013939.
Implementation: TomoPy revision 3377fed68103959e37a849309a096a9919b0bc97,
source/tomopy/prep/normalize.py, functions normalize and minus_log.
https://tomopy.readthedocs.io/en/stable/_modules/tomopy/prep/normalize.html

For the reference values below, the published NumExpr expressions were run
on one float32 projection with three test pixels and one flat and dark frame.
The sample/flat/dark triples were (550,1050,50), (14.5,14.5,1.5) and
(16.5,14.5,6.5). We used mean flat/dark reduction, the standard denominator
floor and no cutoff, then evaluated minus_log on the normalized array.
These are arithmetic inputs. The logarithm tolerance allows float32 rounding.
"""
import json

import pytest

from xray_workbench.tomo_projection import projection


@pytest.mark.parametrize("payload, expected_t, expected_log", [
    ({}, 0.5, 0.6931471824645996),
    ({"sample_counts": 14.5, "flat_counts": 14.5, "dark_counts": 1.5}, 1.0, 0.0),
    ({"sample_counts": 16.5, "flat_counts": 14.5, "dark_counts": 6.5}, 1.25, -0.2231435477733612),
])
def test_single_ray_against_tomopy(payload, expected_t, expected_log):
    answer = projection(payload)
    assert answer["tool"] == "projection"
    assert answer["data"]["transmission"] == expected_t
    assert answer["data"]["line_integral"] == pytest.approx(expected_log, rel=6e-8, abs=0)
    assert any("T > 1" in note for note in answer["notes"]) == (expected_t > 1)


def test_default_ray_export():
    exported = json.loads(json.dumps(projection({}), allow_nan=False))
    data = exported["data"]
    assert (data["sample_counts"], data["flat_counts"], data["dark_counts"]) == (550, 1050, 50)
    assert [v["value"] for v in exported["values"]] == [data["transmission"], data["line_integral"]]
    assert exported["plots"] == []


@pytest.mark.parametrize("payload, reason", [
    ({"sample_counts": -1}, "Sample counts must be at least 0"),
    ({"flat_counts": None}, "Flat counts must be a finite number"),
    ({"dark_counts": True}, "Dark counts must be a finite number"),
    ({"dark_counts": float("nan")}, "Dark counts must be finite"),
    ({"flat_counts": 50}, "Flat counts must exceed dark counts"),
    ({"flat_counts": 49}, "Flat counts must exceed dark counts"),
    ({"sample_counts": 50}, "Sample counts must exceed dark counts"),
    ({"sample_counts": 49}, "Sample counts must exceed dark counts"),
    ({"sample_counts": 1e300, "flat_counts": 1e-300, "dark_counts": 0}, "Transmission must be representable"),
    ({"sample_counts": 1e-300, "flat_counts": 1e300, "dark_counts": 0}, "Transmission must be representable"),
])
def test_invalid_counts_explain_the_domain(payload, reason):
    with pytest.raises(ValueError, match=reason):
        projection(payload)
