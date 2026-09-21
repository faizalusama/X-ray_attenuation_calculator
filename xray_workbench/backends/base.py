"""The contract every attenuation data source must satisfy.

Adding a second source is a scientific act, not a configuration switch. Two
databases that disagree are reporting something real — differing evaluations,
edge conventions, or domains of validity — and this contract is built so that
such a disagreement stays visible:

* A backend declares the **channels it actually provides**. Nothing is ever
  zero-filled to fit a fixed schema, because a zero coherent cross-section and
  an unavailable one are different statements.
* A backend declares **two energy ranges**: the hard domain it will evaluate at
  all, and the narrower range its authors consider reliable. Being inside the
  table is not the same as being trustworthy.
* A backend declares its **edge convention**, because tabulated coefficient
  discontinuities and reported atomic edge energies need not coincide.
* Results carry the backend's identity and version, so a number can always be
  traced to the data that produced it.

There is deliberately no facility for combining backends. Averaging two
evaluated databases produces a number that neither source supports and that no
experiment can be compared against. Compare them; do not blend them.

**Known limitation.** Atomic masses and atomic numbers are currently taken from
XrayDB for every backend, so a mixture's mass fractions are computed from one
source even when its cross-sections come from another. Published atomic weights
agree far more closely than the cross-sections do, so the effect is small, but
it is real and must be revisited when a second backend lands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._types import Array, JsonObject

#: Interaction channels this project knows how to name, in reporting order.
PHOTOELECTRIC = "photoelectric"
COHERENT = "coherent"
INCOHERENT = "incoherent"
PAIR_NUCLEAR = "pair_nuclear"
PAIR_ELECTRON = "pair_electron"

#: Pair production is included because XCOM provides it above 1.022 MeV. No
#: backend is required to supply it, and Elam does not.
KNOWN_CHANNELS: tuple[str, ...] = (PHOTOELECTRIC, COHERENT, INCOHERENT, PAIR_NUCLEAR, PAIR_ELECTRON)

#: Channels that make up narrow-beam removal from the primary photon beam.
#: Every backend must provide all three or it cannot serve this model.
REQUIRED_CHANNELS: tuple[str, ...] = (PHOTOELECTRIC, COHERENT, INCOHERENT)


@dataclass(frozen=True)
class BackendInfo:
    """Everything a reader needs to judge a number this backend produced."""

    identifier: str
    """Stable machine name used in configurations and results, e.g. ``elam``."""

    name: str
    """Human-readable name for the interface."""

    dataset: str
    """The underlying evaluated dataset, with its primary citation."""

    dataset_version: str
    """Version of the package or table actually installed."""

    energy_range_keV: tuple[float, float]
    """Hard domain. Requests outside it are refused, never clamped."""

    recommended_range_keV: tuple[float, float]
    """Where the source's authors consider the data reliable."""

    channels: tuple[str, ...]
    """Interaction channels this backend provides, a subset of KNOWN_CHANNELS."""

    atomic_number_range: tuple[int, int]
    """Inclusive Z range the dataset covers."""

    edge_convention: str
    """How tabulated discontinuities relate to reported edge energies."""

    reference_url: str
    """Where a reader can check the data for themselves."""

    shares_data_with: tuple[str, ...] = ()
    """Identifiers of backends drawing on the **same** underlying evaluation.

    Two packages can expose one dataset through different interfaces. Comparing
    them then measures nothing, however different the APIs look, and reporting
    it as agreement between databases would be false confidence. A backend that
    is not independent must say so here, and
    :func:`xray_workbench.backends.compare.independent_pairs` refuses to treat
    such a pair as a cross-check.
    """

    def is_independent_of(self, other: BackendInfo) -> bool:
        """True only when neither backend declares shared data with the other."""
        return (other.identifier not in self.shares_data_with
                and self.identifier not in other.shares_data_with)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.channels) - set(KNOWN_CHANNELS))
        if unknown:
            raise ValueError(f"Backend '{self.identifier}' declares unknown channels: {unknown}.")
        missing = sorted(set(REQUIRED_CHANNELS) - set(self.channels))
        if missing:
            raise ValueError(
                f"Backend '{self.identifier}' cannot serve a narrow-beam model without {missing}.")
        low, high = self.energy_range_keV
        if not 0 < low < high:
            raise ValueError(f"Backend '{self.identifier}' has an invalid energy range.")
        recommended_low, recommended_high = self.recommended_range_keV
        if not low <= recommended_low < recommended_high <= high:
            raise ValueError(
                f"Backend '{self.identifier}' recommends a range outside its own domain.")

    def as_provenance(self) -> JsonObject:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "energy_range_keV": list(self.energy_range_keV),
            "recommended_range_keV": list(self.recommended_range_keV),
            "channels": list(self.channels),
            "atomic_number_range": list(self.atomic_number_range),
            "edge_convention": self.edge_convention,
            "reference_url": self.reference_url,
            "shares_data_with": list(self.shares_data_with),
        }


class AttenuationBackend(ABC):
    """A source of elemental mass attenuation coefficients."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Identity, domain and provenance of this data source."""

    @abstractmethod
    def mass_attenuation(self, symbol: str, energies_keV: Array) -> dict[str, Array]:
        """Elemental mass attenuation coefficients in **cm²/g**.

        Keys are exactly ``self.info.channels``. Energies are keV, and every
        returned value must be finite and nonnegative — a backend raises
        ``ValueError`` rather than returning a sentinel a caller might plot.
        """

    @abstractmethod
    def edges(self, symbol: str) -> list[tuple[str, float]]:
        """Absorption edges as ``(shell, energy_keV)``, ascending in energy."""

    def domain_warnings(self, lowest_keV: float, highest_keV: float) -> list[str]:
        """Warn where the request leaves the source's reliable range.

        Inside the table but outside the recommended range is the case worth
        stating: the number is produced, and it needs independent checking.
        """
        low, high = self.info.recommended_range_keV
        messages = []
        if highest_keV > high:
            messages.append(
                f"Energies above {high:g} keV are within the {self.info.name} table range but "
                f"outside the commonly recommended accuracy range "
                f"(~{low:g}–{high:g} keV); validate high-energy results against NIST XCOM "
                f"or a transport code.")
        if lowest_keV < low:
            messages.append(
                f"Energies below {low:g} keV are within the {self.info.name} table range but "
                f"outside the commonly recommended accuracy range; near-edge chemistry and "
                f"solid-state effects dominate there.")
        return messages
