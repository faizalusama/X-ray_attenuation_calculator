"""Materials library: compositions with explicit, checkable provenance.

Every entry belongs to exactly one tier, and the tier decides what evidence it
must carry. The loader refuses an entry whose claims exceed its evidence, so a
number in this library can always be traced to where it came from, or is
visibly marked as untraced.

``reference_data``
    Parsed by a script from a published reference dataset (for example NIST SRD
    126). The dataset block records the URL, DOI, retrieval time and a hash of
    the exact page parsed.
``stoichiometric``
    Composition follows exactly from the chemical formula, or from a designation
    such as 3Y-TZP. The formula is checked by the engine's own parser.
``literature``
    Taken from a publication. Requires a DOI, a locator (table, figure or page),
    and a verification record: the DOI's resolved metadata plus the name of the
    person who checked the values against the paper, and when. An entry that
    has not been checked by a person is refused, however plausible it looks.

Densities carry their own ``status``: ``sourced`` when they come from the
entry's source, ``unverified`` when they do not. Unverified densities are
usable, but the interface must tell the user to enter a measured value.

Nothing here averages or reconciles sources. Two entries for the same material
from different sources stay two entries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

from .._types import JsonObject

TIERS = ("reference_data", "stoichiometric", "literature")
DENSITY_STATUSES = ("sourced", "unverified")
DENSITY_KINDS = ("measured", "theoretical", "nominal", "typical")
CATEGORIES = (
    "oxide_ceramic", "non_oxide_ceramic", "nuclear_ceramic", "silicate", "bioceramic",
    "electroceramic", "magnetic_ceramic", "scintillator", "halide", "semiconductor",
    "glass", "glass_ceramic", "borate", "sulfate", "concrete", "cement_phase",
    "biological_reference", "polymer", "liquid", "gas", "detector_medium",
    "alloy", "geological", "explosive", "nuclear_material", "engineering_reference",
)
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


@dataclass(frozen=True)
class Material:
    """One library entry, validated against its tier's evidence rules."""

    id: str
    name: str
    category: str
    tier: str
    basis: str
    components: tuple[tuple[str, float], ...]
    density_g_cm3: float
    density_kind: str
    density_status: str
    density_note: str
    dataset_id: str
    notes: str = ""
    literature: JsonObject | None = None
    reference_constants: JsonObject = field(default_factory=dict)
    formula: str = ""
    """Stoichiometric formula, when an elemental entry corresponds to one."""

    @property
    def density_verified(self) -> bool:
        return self.density_status == "sourced"

    def summary(self) -> JsonObject:
        return {
            "id": self.id, "name": self.name, "category": self.category, "tier": self.tier,
            "density_g_cm3": self.density_g_cm3, "density_status": self.density_status,
            "dataset": self.dataset_id, "formulas": [formula for formula, _ in self.components] + ([self.formula] if self.formula else []),
        }

    def detail(self) -> JsonObject:
        dataset = datasets()[self.dataset_id]
        return {
            **self.summary(),
            "composition": {"basis": self.basis,
                            "components": [{"formula": f, "fraction": x} for f, x in self.components]},
            "density": {"value_g_cm3": self.density_g_cm3, "kind": self.density_kind,
                        "status": self.density_status, "note": self.density_note},
            "notes": self.notes,
            "reference_constants": self.reference_constants,
            "literature": self.literature,
            "source": {key: dataset[key] for key in dataset if key != "entries"},
        }

    def to_layer(self, thickness_mm: float = 1.0) -> JsonObject:
        """A calculation layer using this entry's composition and density.

        The density is passed through as a bulk density. An unverified density
        is still used, which is why the caller must surface ``density_status``.
        """
        return {
            "name": self.name, "basis": self.basis,
            "components": [{"formula": f, "fraction": x} for f, x in self.components],
            "density_g_cm3": self.density_g_cm3, "density_mode": "bulk", "porosity": 0,
            "thickness_mm": thickness_mm, "angle_deg": 0,
            "density_uncertainty_pct": 0, "thickness_uncertainty_pct": 0,
        }


class LibraryError(ValueError):
    """A data file breaks the provenance rules. Raised at load, never ignored."""


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise LibraryError(f"{where}: {message}")


def _validate_literature(where: str, literature: Any) -> JsonObject:
    _require(isinstance(literature, dict), where, "literature entries need a 'literature' object.")
    doi = literature.get("doi", "")
    _require(isinstance(doi, str) and bool(_DOI.match(doi)), where, f"invalid DOI {doi!r}.")
    _require(bool(literature.get("locator")), where,
             "a locator (table, figure or page) is required so the value can be found in the paper.")
    check = literature.get("verification")
    if not isinstance(check, dict):
        raise LibraryError(f"{where}: a verification record is required.")
    resolved = check.get("doi_resolved")
    _require(isinstance(resolved, dict) and bool(resolved.get("title")), where,
             "doi_resolved must hold the title returned by Crossref/DataCite (run tools/verify_doi.py).")
    _require(bool(check.get("values_checked_by")) and bool(check.get("values_checked_on")), where,
             "values_checked_by and values_checked_on are required: a person must have compared "
             "the values with the paper.")
    return dict(literature)


def _parse_entry(raw: Any, dataset_id: str, index: int) -> Material:
    where = f"{dataset_id} entry {index + 1}"
    _require(isinstance(raw, dict), where, "must be an object.")
    ident = raw.get("id", "")
    where = f"{dataset_id}/{ident or index + 1}"
    _require(isinstance(ident, str) and bool(_ID.match(ident)), where, "id must be lowercase-hyphenated.")
    _require(bool(raw.get("name")), where, "name is required.")
    _require(raw.get("category") in CATEGORIES, where, f"unknown category {raw.get('category')!r}.")
    tier = raw.get("tier")
    _require(tier in TIERS, where, f"unknown tier {tier!r}.")

    composition = raw.get("composition") or {}
    basis = composition.get("basis")
    _require(basis in ("mass", "mole", "volume"), where, "composition basis must be mass, mole or volume.")
    components = composition.get("components")
    if not isinstance(components, list) or not components:
        raise LibraryError(f"{where}: at least one component is required.")
    parsed = []
    for component in components:
        formula, fraction = component.get("formula"), component.get("fraction")
        _require(isinstance(formula, str) and bool(formula), where, "every component needs a formula.")
        _require(isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and fraction > 0,
                 where, f"fraction for {formula} must be a positive number.")
        parsed.append((formula, float(fraction)))

    density = raw.get("density") or {}
    value = density.get("value_g_cm3")
    _require(isinstance(value, (int, float)) and 0 < value < 30, where, "density must be in (0, 30) g/cm³.")
    status, kind = density.get("status"), density.get("kind")
    _require(status in DENSITY_STATUSES, where, f"density status must be one of {DENSITY_STATUSES}.")
    _require(kind in DENSITY_KINDS, where, f"density kind must be one of {DENSITY_KINDS}.")
    if status == "unverified":
        _require(bool(density.get("note")), where, "an unverified density must say so in its note.")
    if tier == "stoichiometric":
        # A formula carries no density, so a stoichiometric entry cannot claim a sourced one
        # unless it names a literature source for it.
        _require(status == "unverified" or raw.get("literature") is not None, where,
                 "a stoichiometric density is unverified unless a literature source is attached.")

    literature = None
    if tier == "literature" or raw.get("literature") is not None:
        literature = _validate_literature(where, raw.get("literature"))

    return Material(
        id=ident, name=raw["name"], category=raw["category"], tier=str(tier), basis=str(basis),
        components=tuple(parsed), density_g_cm3=float(value), density_kind=str(kind),  # type: ignore[arg-type]
        density_status=str(status), density_note=density.get("note", ""), dataset_id=dataset_id,
        notes=raw.get("notes", ""), literature=literature,
        reference_constants=raw.get("reference_constants", {}),
        formula=raw.get("formula", ""),
    )


@lru_cache(maxsize=1)
def datasets() -> dict[str, JsonObject]:
    """Every data file, keyed by dataset id, in a stable order."""
    found: dict[str, JsonObject] = {}
    folder = resources.files(__package__) / "data"
    for item in sorted(folder.iterdir(), key=lambda p: p.name):
        if not item.name.endswith(".json"):
            continue
        document = json.loads(item.read_text(encoding="utf-8"))
        dataset = document.get("dataset") or {}
        ident = dataset.get("id")
        if document.get("schema_version") != 1 or not ident:
            raise LibraryError(f"{item.name}: needs schema_version 1 and a dataset id.")
        if ident in found:
            raise LibraryError(f"{item.name}: dataset id {ident!r} is used twice.")
        found[ident] = {**dataset, "entries": document.get("entries", [])}
    return found


@lru_cache(maxsize=1)
def load() -> tuple[Material, ...]:
    """All entries across all datasets. Raises LibraryError on any rule breach."""
    materials: list[Material] = []
    seen: set[str] = set()
    for dataset_id, dataset in datasets().items():
        for index, raw in enumerate(dataset["entries"]):
            material = _parse_entry(raw, dataset_id, index)
            if material.id in seen:
                raise LibraryError(f"{dataset_id}/{material.id}: id is already used by another entry.")
            seen.add(material.id)
            materials.append(material)
    return tuple(materials)


def get(identifier: str) -> Material:
    for material in load():
        if material.id == identifier:
            return material
    raise ValueError(f"No material with id {identifier!r}.")


def search(query: str = "", category: str | None = None, tier: str | None = None,
           verified_density_only: bool = False) -> list[Material]:
    """Case-insensitive search over names, ids and formulas."""
    words = query.lower().split()
    results = []
    for material in load():
        if category and material.category != category:
            continue
        if tier and material.tier != tier:
            continue
        if verified_density_only and not material.density_verified:
            continue
        haystack = " ".join([material.id, material.name.lower(),
                             *(f.lower() for f, _ in material.components)])
        if all(word in haystack for word in words):
            results.append(material)
    return results


def landscape(energy_keV: float, thickness_mm: float = 1.0) -> list[JsonObject]:
    """Evaluate every entry at one energy, for a library-wide overview plot.

    At a single energy the narrow-beam mixture rule is just the mass-fraction
    weighted sum of elemental coefficients, so each element is fetched once
    for the whole library instead of running the full engine per entry.
    ``tests/test_materials.py`` checks this against the full engine.

    Each point uses the entry's own density, so ``density_status`` travels with
    it: a point computed from an unverified density is not a sourced result.
    """
    from math import exp, log

    import numpy as np

    from ..backends import get_backend
    from ..physics import _prepare_layer  # the engine's own normalisation, not a copy

    backend = get_backend()
    energy = np.array([float(energy_keV)])
    prepared = [(material, _prepare_layer(material.to_layer(thickness_mm), 0)) for material in load()]
    coefficient: dict[str, float] = {}
    for symbol in sorted({s for _, layer in prepared for s in layer.elements}):
        channels = backend.mass_attenuation(symbol, energy)
        coefficient[symbol] = float(sum(values[0] for values in channels.values()))
    points = []
    for material, layer in prepared:
        mu_mass = sum(weight * coefficient[symbol] for symbol, weight in layer.elements.items())
        mu_linear = mu_mass * layer.density
        points.append({**material.summary(),
                       "mu_mass_cm2_g": mu_mass,
                       "mu_linear_cm_inv": mu_linear,
                       "hvl_mm": 10 * log(2) / mu_linear,
                       "transmission": exp(-mu_linear * layer.path_length / 10)})
    return points


__all__ = ["CATEGORIES", "TIERS", "LibraryError", "Material", "datasets", "get", "landscape", "load", "search"]
