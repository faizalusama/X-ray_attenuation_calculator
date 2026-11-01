"""Regenerate the data tables in xray_workbench/scattering_data/. Maintainers only.

    pip install xraylib
    python tools/build_scattering_tables.py [--neutron-page saved.html]

Incoherent scattering functions S(sinθ/λ, Z) are read from xraylib, whose documentation
gives Hubbell et al. (1975), doi:10.1063/1.555523, as their source. Neutron values are
parsed from NIST's listing of Sears (1992), doi:10.1080/10448639208218770.

Neither is needed to run the calculator: the tables are stored in the package.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "xray_workbench" / "scattering_data"
NEUTRON_PAGE = "https://www.ncnr.nist.gov/resources/n-lengths/list.html"
#: sinθ/λ in 1/Å. Close spacing where S rises fastest, up to Q = 4π × 10.
GRID = np.concatenate([np.arange(0, 0.2, 0.005), np.arange(0.2, 1, 0.02), np.arange(1, 4, 0.1), np.arange(4, 10.01, 0.5)]).round(4)


def incoherent_table() -> dict[str, Any]:
    import xraydb  # noqa: PLC0415
    import xraylib  # noqa: PLC0415

    values = {xraydb.atomic_symbol(z): [float(f"{xraylib.SF_Compt(z, float(s)) if s > 0 else 0.0:.6g}") for s in GRID] for z in range(1, 99)}
    return {"description": "Incoherent scattering function S(sinθ/λ) per atom, electrons.",
            "source": "Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975), doi:10.1063/1.555523",
            "obtained_through": f"xraylib {xraylib.__version__}, tools/build_scattering_tables.py",
            "grid_sin_theta_over_lambda": GRID.tolist(), "values": values}


def _number(cell: str) -> float | None:
    """The real part of a table entry such as ``5.30-0.213i``, ``3.26(3)``, ``767.(8.)`` or ``---``."""
    match = re.match(r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", cell.replace("(", " ("))
    return float(match.group(1)) if match else None


def _imaginary(cell: str) -> float:
    match = re.search(r"([+-]\d*\.?\d+)\s*i\s*$", cell)
    return float(match.group(1)) if match else 0.0


def neutron_table(page: str) -> dict[str, Any]:
    rows = re.findall(r"<td>\s*([A-Z][a-z]?)\s*<td>(.*?)<td>(.*?)<td>(.*?)<td>(.*?)<td>(.*?)<td>(.*?)<td>(.*?)<tr>", page, flags=re.S)
    values = {}
    for symbol, _abundance, coherent, _incoherent, _coh_xs, inc_xs, _scatt_xs, abs_xs in rows:
        clean = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip() for cell in (coherent, inc_xs, abs_xs)]
        length = _number(clean[0])
        if length is not None:
            values[symbol] = {"b_coherent_fm": length, "b_coherent_imaginary_fm": _imaginary(clean[0]),
                              "sigma_incoherent_barn": _number(clean[1]), "sigma_absorption_barn": _number(clean[2])}
    if len(values) < 80:
        raise SystemExit(f"Expected about 90 natural elements in the NIST table; found {len(values)}.")
    return {"description": "Natural-abundance bound coherent scattering lengths and cross sections. Absorption at 2200 m/s.",
            "source": "Sears, Neutron News 3(3), 26-37 (1992), doi:10.1080/10448639208218770",
            "obtained_through": NEUTRON_PAGE, "values": values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neutron-page", type=Path, help="a saved copy of the NIST page, instead of downloading it")
    args = parser.parse_args()
    if args.neutron_page:
        page = args.neutron_page.read_text(encoding="latin-1")
    else:
        with urlopen(NEUTRON_PAGE, timeout=60) as response:  # fixed HTTPS URL
            page = response.read().decode("latin-1")
    TABLES.mkdir(exist_ok=True)
    for name, table in (("incoherent_hubbell1975.json", incoherent_table()), ("neutron_sears1992.json", neutron_table(page))):
        (TABLES / name).write_text(json.dumps(table, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {name} with {len(table['values'])} elements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
