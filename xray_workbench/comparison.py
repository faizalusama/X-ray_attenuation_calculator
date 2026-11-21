"""CSV comparison evaluated by the unchanged attenuation engine, row by row."""
from __future__ import annotations

import csv
import io
from typing import Any

from ._types import JsonObject
from .physics import _number, _object, calculate

COLUMNS = ("name", "basis", "density_g_cm3", "thickness_mm", "composition")
BASES = {"mass": "mass", "wt%": "mass", "mole": "mole", "mol%": "mole", "atomic": "atomic", "at%": "atomic"}


def compare(payload: Any) -> JsonObject:
    payload = _object(payload, "Comparison request")
    energy = _number(payload.get("energy_keV"), "Energy (keV)", 1, 800)
    source = payload.get("csv")
    if not isinstance(source, str) or len(source.encode("utf-8")) > 1_000_000:
        raise ValueError("CSV must be text smaller than 1 MB.")
    reader = csv.DictReader(io.StringIO(source.lstrip("\ufeff")), strict=True)
    if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS) or len(reader.fieldnames) != len(COLUMNS):
        raise ValueError("CSV columns must be: " + ",".join(COLUMNS))
    rows: list[JsonObject] = []
    try:
        for index, row in enumerate(reader, 2):
            if len(rows) >= 100:
                raise ValueError("A comparison accepts at most 100 rows.")
            name = row.get("name", "")
            item: JsonObject = {"row": index, "name": name, "error": None}
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Row has the wrong number of columns.")
                if row["basis"].strip() not in BASES:
                    raise ValueError("Basis must be mass/wt%, mole/mol%, or atomic/at%.")
                components = []
                for term in row["composition"].split(";"):
                    formula, fraction = term.rsplit(":", 1)
                    components.append({"formula": formula.strip(), "fraction": float(fraction)})
                sample = {"name": name, "basis": BASES[row["basis"].strip()], "density_g_cm3": float(row["density_g_cm3"]),
                          "thickness_mm": float(row["thickness_mm"]), "components": components}
                answer = calculate({"energy": {"min_keV": 1, "max_keV": 800, "points": 2, "reference_keV": energy},
                                    "layers": [sample]})
                item.update(answer["layers"][0]["reference"])
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                item["error"] = str(exc)
            rows.append(item)
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV: {exc}") from exc
    if not rows:
        raise ValueError("CSV needs at least one data row.")
    return {"energy_keV": energy, "rows": rows, "successes": sum(r["error"] is None for r in rows)}
