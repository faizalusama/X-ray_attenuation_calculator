"""Citation metadata tested against CITATION.cff; no DOI is minted here."""
from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from ._types import JsonObject


def citation() -> JsonObject:
    data: JsonObject = json.loads((Path(__file__).parent / "citation_data.json").read_text(encoding="utf-8"))
    note = "Unreleased development version. Concept DOI covers all versions; cite the version actually used."
    fields = {**data, "version": __version__, "note": note}
    bibtex = "@software{xray_attenuation_calculator,\n"
    bibtex += ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items())
    fields["bibtex"] = bibtex + "\n}\n"
    return fields
