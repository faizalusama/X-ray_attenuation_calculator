"""Parallel-beam angular sampling and the object-space Nyquist floor."""
import json

import pytest

from xray_workbench._types import JsonObject
from xray_workbench.tomo_sampling import tomo_sampling

# Rogalinski et al. (2026), §2.2.2–2.3, Eq. (1), doi:10.1107/S1600577525011038:
# about 1600 views for 1024 pixels across the support. At their 4 µm effective
# pitch that width is 4096 µm and the two-pixel period is 8 µm.
_FORMAX: JsonObject = {"diameter_um": 4096, "resolution_um": 8, "pixel_pitch_um": 4}


def test_formax_crowther_example() -> None:
    # Evaluating 512*pi in decimal gives 1608.495438637974..., whose ceiling is 1609.
    exported: JsonObject = json.loads(json.dumps(tomo_sampling({**_FORMAX, "planned_views": 1600}), allow_nan=False))
    assert exported["tool"] == "tomo_sampling"
    assert exported["plots"] == []
    assert "10.1107/S1600576721000194" in exported["provenance"]["method"]
    assert "10.1098/rspa.1970.0119" in exported["provenance"]["original_reference"]
    data = exported["data"]
    assert data["minimum_views"] == 1609
    assert data["rotation_range_deg"] == 180
    assert data["angular_step_deg"] == pytest.approx(0.11187072715972654, rel=1e-12)
    assert data["nyquist_resolution_um"] == 8
    assert data["pitch_supports_resolution"] is True
    assert data["planned_views"] == 1600
    assert data["planned_angular_step_deg"] == pytest.approx(0.1125, rel=1e-12)
    assert data["planned_view_shortfall"] == 9
    assert data["planned_views_meet_minimum"] is False


def test_sufficient_views_cannot_fix_coarse_pixel_sampling() -> None:
    answer = tomo_sampling({**_FORMAX, "planned_views": 1609.0, "pixel_pitch_um": 5})
    data = answer["data"]
    assert data["minimum_views"] == 1609
    assert data["planned_views_meet_minimum"] is True
    assert data["planned_view_shortfall"] == 0
    assert data["nyquist_resolution_um"] == 10
    assert data["pitch_supports_resolution"] is False
    assert any("below the pixel sampling floor" in note for note in answer["notes"])


@pytest.mark.parametrize("optional_fields", [
    {}, {"planned_views": None}, {"planned_views": ""}, {"planned_views": "   "},
])
def test_blank_planned_views_omit_comparison(optional_fields: JsonObject) -> None:
    answer = tomo_sampling({**_FORMAX, **optional_fields})
    assert answer["data"] == tomo_sampling(_FORMAX)["data"]
    assert "planned_views" not in answer["data"]


@pytest.mark.parametrize("fields, reason", [
    ({"diameter_um": None}, "Diameter .* must be a finite number"),
    ({"diameter_um": -1}, "Diameter .* must be greater than zero"),
    ({"resolution_um": 0}, "Resolution .* must be greater than zero"),
    ({"resolution_um": float("nan")}, "Resolution .* must be finite"),
    ({"pixel_pitch_um": True}, "Effective pixel pitch .* must be a finite number"),
    ({"pixel_pitch_um": float("inf")}, "Effective pixel pitch .* must be finite"),
    ({"planned_views": False}, "Planned views must be a finite number"),
    ({"planned_views": "1600"}, "Planned views must be a finite number"),
    ({"planned_views": float("inf")}, "Planned views must be finite"),
    ({"planned_views": 0}, "Planned views must be at least 1"),
    ({"planned_views": 1.5}, "Planned views must be an integer"),
    ({"diameter_um": 1e308, "resolution_um": 1e-300}, "Crowther view requirement must be finite"),
    ({"pixel_pitch_um": 1e308}, "pixel sampling resolution floor must be finite"),
])
def test_invalid_inputs_explain_the_domain(fields: JsonObject, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        tomo_sampling({**_FORMAX, **fields})
