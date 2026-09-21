"""Import NIST SRD 126, Table 2 (material compositions) into the materials library.

    python tools/import_nist_table2.py            # fetch from NIST and write
    python tools/import_nist_table2.py --html F   # parse a saved copy instead

Source: J. H. Hubbell and S. M. Seltzer, "Tables of X-Ray Mass Attenuation
Coefficients and Mass Energy-Absorption Coefficients", NIST Standard Reference
Database 126, Table 2. US Government work, not subject to copyright in the US.

The output records the retrieval time, URL and SHA-256 of the exact page that
was parsed, so the data file can be regenerated and checked against its source.
Nothing is edited by hand: if NIST's table and this file disagree, rerun this.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xraydb

URL = "https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html"
OUTPUT = Path(__file__).resolve().parent.parent / "xray_workbench" / "materials" / "data" / "nist_srd126_table2.json"

#: Category is our editorial classification; NIST gives only names.
CATEGORIES = {
    "Cadmium Telluride": "semiconductor", "Gallium Arsenide": "semiconductor",
    "Calcium Fluoride": "halide", "Lithium Fluride": "halide", "Cesium Iodide": "halide",
    "Mercuric Iodide": "halide",
    "Calcium Sulfate": "sulfate",
    "Concrete, Ordinary": "concrete", "Concrete, Barite (TYPE BA)": "concrete",
    "Gadolinium Oxysulfide": "oxide_ceramic",
    "Glass, Borosilicate (Pyrex)": "glass", "Glass, Lead": "glass",
    "Lithium Tetraborate": "borate", "Magnesium Tetroborate": "borate",
}
#: NIST's own spellings are kept verbatim as names; these are the corrections.
SPELLING = {"Lithium Fluride": "Lithium Fluoride", "Magnesium Tetroborate": "Magnesium Tetraborate"}


def _category(name: str) -> str:
    if name in CATEGORIES:
        return CATEGORIES[name]
    lowered = name.lower()
    if "icru" in lowered or "bone" in lowered or "blood" in lowered:
        return "biological_reference"
    if "gas" in lowered or lowered.startswith("air"):
        return "gas"
    if lowered.startswith("water") or "ferrous sulfate" in lowered or "solution" in lowered:
        return "liquid"
    if "emulsion" in lowered or "film" in lowered or "gafchromic" in lowered:
        return "detector_medium"
    return "polymer"


def _slug(name: str) -> str:
    return "nist-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse(page: str) -> list[dict[str, Any]]:
    rows = re.findall(r"<TR VALIGN=\"top\">(.*?)</TR>", page, re.S | re.I)
    entries = []
    for row in rows:
        cells = [html.unescape(re.sub(r"<[^>]+>", "\n", c)).strip()
                 for c in re.findall(r"<TD[^>]*>(.*?)</TD>", row, re.S | re.I)]
        if len(cells) != 5:
            continue
        name, z_over_a, mean_excitation, density, composition = cells
        pairs = re.findall(r"(\d+)\s*:\s*([0-9.]+)", composition)
        if not pairs:
            continue
        components = [{"formula": xraydb.atomic_symbol(int(z)), "fraction": float(w)} for z, w in pairs]
        corrected = SPELLING.get(name, name)
        entries.append({
            "id": _slug(corrected),
            "name": corrected,
            "category": _category(name),
            "tier": "reference_data",
            "composition": {"basis": "mass", "components": components},
            "density": {"value_g_cm3": float(density), "kind": "nominal",
                        "status": "sourced",
                        "note": "NIST states that some density values in this table are only nominal."},
            "reference_constants": {"z_over_a": float(z_over_a), "mean_excitation_eV": float(mean_excitation)},
            "source_name": name,
        })
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", type=Path, help="parse a saved copy of the page instead of fetching")
    args = parser.parse_args(argv)
    if args.html:
        raw = args.html.read_bytes()
    else:
        request = Request(URL, headers={"User-Agent": "xray-attenuation-workbench importer"})
        with urlopen(request, timeout=60) as response:  # fixed, trusted government URL
            raw = response.read()
    page = raw.decode("latin-1")
    entries = parse(page)
    if len(entries) < 40:
        raise SystemExit(f"Parsed only {len(entries)} materials; the page layout may have changed.")
    document = {
        "schema_version": 1,
        "dataset": {
            "id": "nist-srd126-table2",
            "title": "Tables of X-Ray Mass Attenuation Coefficients and Mass Energy-Absorption "
                     "Coefficients, Table 2: material constants and composition",
            "authors": "J. H. Hubbell and S. M. Seltzer",
            "publisher": "National Institute of Standards and Technology (NIST SRD 126)",
            "url": URL,
            "doi": "10.18434/T4D01F",
            "license": "US Government work; not subject to copyright in the United States.",
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "retrieved_sha256": hashlib.sha256(raw).hexdigest(),
            "caveats": [
                "Some density values in the source table are only nominal; use a measured density "
                "for quantitative work.",
                "Human-tissue compositions in the source were taken from ICRU Report 44 (1989).",
                "Categories are this project's editorial classification, not part of the source.",
            ],
        },
        "entries": entries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} materials to {OUTPUT.relative_to(OUTPUT.parents[3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
