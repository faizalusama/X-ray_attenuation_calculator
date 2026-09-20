"""Compare two attenuation backends without ever blending them.

Two evaluated databases that disagree are saying something real. Averaging them
produces a number neither source supports and no experiment can be compared
against, so nothing here returns a combined coefficient. The output is the
disagreement itself, plus enough context to judge what it means.

The comparison also reports whether the two sources are **independent**. A pair
that shares an underlying evaluation can agree perfectly while demonstrating
nothing, and that distinction is the difference between validation and
self-confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._types import Array, JsonObject
from . import get_backend
from .base import AttenuationBackend


@dataclass(frozen=True)
class Comparison:
    """Channel-by-channel disagreement between two backends over one grid."""

    element: str
    energies_keV: Array
    reference: str
    candidate: str
    independent: bool
    relative_difference: dict[str, Array]
    """Per channel, ``|reference - candidate| / candidate``, plus ``total``."""

    overlap_keV: tuple[float, float]
    """The energy interval both backends actually cover."""

    def worst(self, channel: str = "total") -> tuple[float, float]:
        """Largest relative difference in *channel*, and the energy it occurs at."""
        values = self.relative_difference[channel]
        index = int(np.argmax(values))
        return float(values[index]), float(self.energies_keV[index])

    def summary(self) -> JsonObject:
        worst, energy = self.worst()
        return {
            "element": self.element,
            "reference": self.reference,
            "candidate": self.candidate,
            "independent": self.independent,
            "overlap_keV": list(self.overlap_keV),
            "samples": int(self.energies_keV.size),
            "max_relative_difference": worst,
            "max_at_keV": energy,
            "median_relative_difference": float(np.median(self.relative_difference["total"])),
            "bit_identical": bool(np.all(self.relative_difference["total"] == 0.0)),
            "interpretation": (
                "Independent evaluations: a difference here is a real disagreement between "
                "databases and an upper bound on data-driven uncertainty."
                if self.independent else
                "NOT independent: these backends declare shared underlying data, so agreement "
                "demonstrates nothing and must not be reported as cross-database validation."),
        }


def compare_element(element: str, reference: str = "elam", candidate: str = "xraylib",
                    points: int = 200) -> Comparison:
    """Compare one element across the energies both backends cover.

    The grid is restricted to the overlap of the two declared domains, because
    comparing a value against an extrapolation is not a comparison.
    """
    first: AttenuationBackend = get_backend(reference)
    second: AttenuationBackend = get_backend(candidate)
    if first.info.identifier == second.info.identifier:
        raise ValueError("Comparing a backend with itself measures nothing.")

    low = max(first.info.energy_range_keV[0], second.info.energy_range_keV[0])
    high = min(first.info.energy_range_keV[1], second.info.energy_range_keV[1])
    if not low < high:
        raise ValueError(
            f"{reference} ({first.info.energy_range_keV}) and {candidate} "
            f"({second.info.energy_range_keV}) have no overlapping energy domain.")
    # Nudge inside the closed interval: both backends refuse to extrapolate and
    # a boundary energy can land a hair outside after floating-point rounding.
    energies = np.geomspace(low * (1 + 1e-9), high * (1 - 1e-9), points)

    a = first.mass_attenuation(element, energies)
    b = second.mass_attenuation(element, energies)
    shared = [channel for channel in first.info.channels if channel in b]

    differences: dict[str, Array] = {}
    for channel in shared:
        denominator = np.where(b[channel] == 0, np.nan, b[channel])
        differences[channel] = np.abs(a[channel] - b[channel]) / denominator
    total_a = sum(a[channel] for channel in shared)
    total_b = sum(b[channel] for channel in shared)
    differences["total"] = np.abs(total_a - total_b) / total_b

    return Comparison(
        element=element, energies_keV=energies,
        reference=first.info.identifier, candidate=second.info.identifier,
        independent=first.info.is_independent_of(second.info),
        relative_difference=differences, overlap_keV=(low, high),
    )


def independent_pairs(identifiers: list[str] | None = None) -> list[tuple[str, str]]:
    """Backend pairs that can legitimately cross-check one another.

    A pair sharing an underlying evaluation is excluded, however different the
    two interfaces look.
    """
    from . import available_backends
    names = available_backends() if identifiers is None else identifiers
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            try:
                a, b = get_backend(left).info, get_backend(right).info
            except ValueError:
                continue  # An optional backend that is not installed here.
            if a.is_independent_of(b):
                pairs.append((left, right))
    return pairs
