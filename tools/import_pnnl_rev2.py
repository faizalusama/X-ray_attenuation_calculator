"""Import multielement materials from PNNL-15870 Revision 2.

The report is the authoritative source.  This importer reads the elemental
weight-fraction table printed for each of its 411 records, combines isotopes
back to elements (X-ray attenuation is isotope independent in this model), and
omits single-element records.  The generated JSON is committed so installing
the application never requires a PDF parser or network access.

Usage:
    python tools/import_pnnl_rev2.py path/to/PNNL-15870Rev2.pdf
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_URL = "https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf"
SOURCE_DOI = "10.2172/1782721"
OUTPUT = Path(__file__).parents[1] / "xray_workbench/materials/data/pnnl_15870_rev2.json"

_HEADING = re.compile(
    r"(?m)^[ \t]*(\d{1,3})\. ([^\n]+(?:\n(?![ \t]*\d{1,3}\. )[^\n]+){0,2})(?=\nFormula\s*=)"
)
_DENSITY = re.compile(r"Density \(g/cm3\)\s*=\s*([0-9.Ee+-]+)")
_ELEMENT_ROW = re.compile(
    r"(?m)^\s*([A-Z][a-z]?)\s+\d{4,5}\s+-([0-9.]+)\s+\d{4,5}\s+[0-9.]+\s+\d{4,5}\s+[0-9.]+\s*$"
)


def _extract_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only by the maintainer tool
        raise SystemExit("pypdf is required to rebuild this dataset: python -m pip install pypdf") from exc
    reader = PdfReader(pdf_path)
    # Printed material records start on report page 16, PDF page 33.
    return "\n".join(page.extract_text() or "" for page in reader.pages[32:])


def _clean_name(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _slug(number: int, name: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"pnnl-r2-{number:03d}-{words}"[:118].rstrip("-")


def _category(name: str, density: float) -> str:
    value = name.lower()
    if any(word in value for word in ("tissue", "blood", "bone", "brain", "lung", "muscle", "skin", "eye lens")):
        return "biological_reference"
    if any(word in value for word in ("concrete", "cement", "mortar", "grout")):
        return "concrete"
    if any(word in value for word in ("glass", "fiberglass", "borosilicate", "frit")):
        return "glass"
    if any(word in value for word in ("alloy", "steel", "stainless", "brass", "bronze", "inconel", "zircaloy")):
        return "alloy"
    if any(word in value for word in ("plastic", "poly", "nylon", "rubber", "resin", "lucite", "teflon", "kapton", "mylar")):
        return "polymer"
    if any(word in value for word in ("explosive", " tnt", " rdx", " hmx", "petn")):
        return "explosive"
    if any(word in value for word in ("soil", "earth", "clay", "brick", "sand", "rock", "granite", "limestone", "gypsum")):
        return "geological"
    if any(word in value for word in ("uranium", "plutonium", "thorium", "fuel", "uo2", "mox")):
        return "nuclear_material"
    if any(word in value for word in ("scint", "detector", "doped", "bgo", "cwo", "czt", "gagg", "lyso", "phosphor")):
        return "detector_medium"
    if any(word in value for word in ("water", "solution", "liquid", "oil", "gasoline", "diesel")):
        return "liquid"
    if density < 0.02:
        return "gas"
    if any(word in value for word in ("carbide", "nitride", "boride", "sulfide", "selenide", "telluride")):
        return "non_oxide_ceramic"
    if any(word in value for word in ("oxide", "alumina", "ferrite", "tungstate", "silicate", "carbonate", "sulfate")):
        return "oxide_ceramic"
    return "engineering_reference"


def parse(pdf_path: Path) -> list[dict[str, Any]]:
    text = _extract_text(pdf_path)
    headings = list(_HEADING.finditer(text))
    numbers = [int(match.group(1)) for match in headings]
    if numbers != list(range(1, 412)):
        missing = sorted(set(range(1, 412)) - set(numbers))
        raise ValueError(f"Expected material records 1..411 exactly once; missing {missing}, parsed {len(numbers)}.")

    entries: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        number = int(heading.group(1))
        name = _clean_name(heading.group(2))
        stop = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.start():stop]
        density_match = _DENSITY.search(block)
        if not density_match:
            raise ValueError(f"PNNL material {number} ({name}) has no parsed density.")
        density = float(density_match.group(1))
        if "Elemental" not in block or "Comments and References" not in block:
            raise ValueError(f"PNNL material {number} ({name}) is missing a table delimiter.")
        elemental = block.split("Elemental", 1)[1].split("Comments and References", 1)[0]
        fractions: dict[str, float] = defaultdict(float)
        for symbol, fraction in _ELEMENT_ROW.findall(elemental):
            fractions[symbol] += float(fraction)
        if not fractions:
            raise ValueError(f"PNNL material {number} ({name}) has no parsed elemental rows.")
        total = sum(fractions.values())
        # Record 321 is printed with a 0.999672 total; retain the report's
        # numbers and let the engine perform its documented normalization.
        if abs(total - 1) > 5e-4:
            raise ValueError(f"PNNL material {number} ({name}) mass fractions sum to {total:.8f}.")
        if len(fractions) < 2:
            continue
        components = [{"formula": symbol, "fraction": round(fraction, 9)}
                      for symbol, fraction in sorted(fractions.items())]
        entries.append({
            "id": _slug(number, name),
            "name": name,
            "category": _category(name, density),
            "tier": "reference_data",
            "composition": {"basis": "mass", "components": components},
            "density": {
                "value_g_cm3": density,
                "kind": "nominal",
                "status": "sourced",
                "note": ("Value tabulated by PNNL-15870 Rev. 2. The report states that its values are "
                         "representative and that real material composition and density can vary."),
            },
            "reference_constants": {"pnnl_material_number": number},
        })
    return entries


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/import_pnnl_rev2.py path/to/PNNL-15870Rev2.pdf")
    source = Path(sys.argv[1])
    raw = source.read_bytes()
    entries = parse(source)
    document = {
        "schema_version": 1,
        "dataset": {
            "id": "pnnl-15870-rev2",
            "title": "Compendium of Material Composition Data for Radiation Transport Modeling, Revision 2",
            "authors": "R. S. Detwiler, R. J. McConn, T. F. Grimes, S. A. Upton, and E. J. Engel",
            "publisher": "Pacific Northwest National Laboratory",
            "report_number": "PNNL-15870 Rev. 2 / 200-DMAMC-128170",
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": "Publicly available U.S. Government-sponsored technical report; cite PNNL and the report.",
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "retrieved_sha256": hashlib.sha256(raw).hexdigest(),
            "method": ("Generated from the report's printed elemental weight-fraction and density tables. "
                       "Isotopic rows are combined by element; single-element records are omitted."),
            "caveats": [
                "PNNL describes the values as reasonably representative and warns that composition and density vary in practice.",
                "Source weight fractions are printed to finite precision and are normalized by the calculation engine.",
                "Isotopic enrichments are collapsed to elements because this workbench models photon attenuation with element-level coefficients.",
                "Categories are this project's editorial classification, not part of the PNNL report.",
            ],
        },
        "entries": entries,
    }
    OUTPUT.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} multielement materials to {OUTPUT}")


if __name__ == "__main__":
    main()
