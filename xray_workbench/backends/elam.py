"""XrayDB's Elam backend — the reference implementation and current default.

This wraps the calls the engine has always made, unchanged. Its numbers are the
ones recorded in docs/VALIDATION.md, so a change here changes published results.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xraydb

from .._types import Array
from .base import COHERENT, INCOHERENT, PHOTOELECTRIC, AttenuationBackend, BackendInfo

#: XrayDB names the channels differently from this project's vocabulary.
_XRAYDB_KIND = {PHOTOELECTRIC: "photo", COHERENT: "coh", INCOHERENT: "incoh"}


class ElamBackend(AttenuationBackend):
    """Elam, Ravel & Sieber (2002) atomic data, as distributed inside XrayDB."""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            identifier="elam",
            name="XrayDB / Elam",
            dataset="Elam, Ravel & Sieber (2002), doi:10.1016/S0969-806X(01)00227-4",
            dataset_version=str(xraydb.__version__),
            # The engine's own guarded domain is narrower still; this is the
            # range over which the backend itself returns unclamped values.
            energy_range_keV=(1.0, 800.0),
            # XrayDB describes the cross-sections as typically most reliable
            # over roughly 0.25-250 keV.
            recommended_range_keV=(1.0, 250.0),
            channels=(PHOTOELECTRIC, COHERENT, INCOHERENT),
            atomic_number_range=(1, 98),
            edge_convention=(
                "Tabulated log-energy knots carry limited decimal precision, so a coefficient "
                "discontinuity can sit up to ~5e-5 relative away from the xray_edges() metadata "
                "energy. Coefficients are never shifted to match the reported edge label."),
            reference_url="https://xraypy.github.io/XrayDB/",
        )

    def mass_attenuation(self, symbol: str, energies_keV: Array) -> dict[str, Array]:
        channels = {}
        for channel, kind in _XRAYDB_KIND.items():
            # XrayDB takes eV and returns cm^2/g.
            values = np.asarray(xraydb.mu_elam(symbol, energies_keV * 1000, kind=kind), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"The attenuation database returned invalid coefficients for {symbol}.")
            channels[channel] = values
        return channels

    def edges(self, symbol: str) -> list[tuple[str, float]]:
        found = [(shell, float(edge.energy) / 1000) for shell, edge in xraydb.xray_edges(symbol).items()]
        return sorted(found, key=lambda item: (item[1], item[0]))


@lru_cache(maxsize=1)
def elam_backend() -> ElamBackend:
    return ElamBackend()
