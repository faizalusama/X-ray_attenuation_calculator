"""xraylib backend — **not** an independent check on the Elam data.

Measured on 20 September 2026 with xraylib 4.3.0 and XrayDB 4.5.8, on 99
element/energy samples spanning H to U and 5-700 keV, xraylib's ``CS_Photo``,
``CS_Rayl`` and ``CS_Compt`` returned values **bit-identical** to XrayDB's
``mu_elam`` for every sample and every channel — equal to the last bit of a
float64, not merely close. The two packages expose the same evaluated data
through different interfaces.

That is why :attr:`BackendInfo.shares_data_with` names ``elam`` here. Running
the workbench "against xraylib" and finding agreement would demonstrate
nothing, and quoting it as cross-database validation would be false confidence.
An independent check needs a genuinely separate evaluation such as NIST XCOM;
see docs/ROADMAP.md.

This backend is still worth having. It is a second implementation path, and
``tests/test_backend_equivalence.py`` asserts the equality above so that a
future divergence in either upstream package is reported loudly instead of
silently changing results. xraylib also exposes quantities XrayDB does not,
notably the mass energy-absorption coefficient, which later work may need.

Optional dependency: ``pip install xray-attenuation-workbench[xraylib]``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .._types import Array
from .base import COHERENT, INCOHERENT, PHOTOELECTRIC, AttenuationBackend, BackendInfo

#: Empirically pinned by bisection against xraylib 4.3.0: below and above these
#: the library raises "Spline extrapolation is not allowed" rather than
#: extrapolating. Declared honestly instead of assuming a round number.
_DOMAIN_KEV = (0.1, 800.026475)


def _require_xraylib() -> Any:
    """Import xraylib, turning a missing optional dependency into clear guidance.

    Typed as Any: xraylib ships no type information, being a SWIG wrapper
    around a C library.
    """
    try:
        import xraylib
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise ValueError(
            "The xraylib backend needs the 'xraylib' package. Install it with "
            "'pip install xraylib', or use the default 'elam' backend."
        ) from exc
    return xraylib


class XraylibBackend(AttenuationBackend):
    """Photon cross-sections via xraylib's scalar C API."""

    def __init__(self) -> None:
        self._xraylib = _require_xraylib()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            identifier="xraylib",
            name="xraylib",
            dataset="xraylib photon cross-sections (same evaluated data as the Elam tables)",
            dataset_version=str(getattr(self._xraylib, "__version__", "unknown")),
            energy_range_keV=_DOMAIN_KEV,
            recommended_range_keV=(1.0, 250.0),
            # xraylib 4.3.0 exposes no pair-production cross-section, so this
            # backend must not be used above 1.022 MeV even if a future version
            # widens the spline domain. The declared domain already excludes it.
            channels=(PHOTOELECTRIC, COHERENT, INCOHERENT),
            atomic_number_range=(1, 98),
            edge_convention=(
                "EdgeEnergy() returns tabulated edge energies in keV. These differ slightly from "
                "the XrayDB metadata values (Ba K: 37.4406 vs 37.4410 keV, 0.4 eV) even though the "
                "coefficients themselves are identical, so edge labels are not interchangeable."),
            reference_url="https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions",
            shares_data_with=("elam",),
        )

    def mass_attenuation(self, symbol: str, energies_keV: Array) -> dict[str, Array]:
        xraylib = self._xraylib
        z = xraylib.SymbolToAtomicNumber(symbol)
        low, high = _DOMAIN_KEV
        outside = energies_keV[(energies_keV < low) | (energies_keV > high)]
        if outside.size:
            raise ValueError(
                f"xraylib evaluates {low:g}-{high:g} keV; {outside.min():g} keV is outside that "
                f"domain. It refuses to extrapolate rather than returning an invented value.")
        # xraylib's C API is scalar, so this loops. Elemental coefficients are
        # cached once per calculation upstream, which keeps the cost bounded.
        channels = {}
        for channel, function in ((PHOTOELECTRIC, xraylib.CS_Photo),
                                  (COHERENT, xraylib.CS_Rayl),
                                  (INCOHERENT, xraylib.CS_Compt)):
            values = np.fromiter((function(z, float(e)) for e in energies_keV),
                                 dtype=float, count=energies_keV.size)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"xraylib returned invalid {channel} coefficients for {symbol}.")
            channels[channel] = values
        return channels

    def edges(self, symbol: str) -> list[tuple[str, float]]:
        xraylib = self._xraylib
        z = xraylib.SymbolToAtomicNumber(symbol)
        found = []
        for name in dir(xraylib):
            if not name.endswith("_SHELL"):
                continue
            try:
                energy = float(xraylib.EdgeEnergy(z, getattr(xraylib, name)))
            except (ValueError, RuntimeError):
                continue
            if energy > 0:
                found.append((name.removesuffix("_SHELL"), energy))
        return sorted(found, key=lambda item: (item[1], item[0]))


@lru_cache(maxsize=1)
def xraylib_backend() -> XraylibBackend:
    return XraylibBackend()
