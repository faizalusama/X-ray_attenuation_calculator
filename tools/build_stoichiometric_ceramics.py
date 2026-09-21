"""Generate the stoichiometric ceramics dataset of the materials library.

    python tools/build_stoichiometric_ceramics.py

This is the single editable source for that dataset: change the table below and
rerun, rather than editing the JSON by hand.

Composition follows exactly from each chemical formula, or for designated solid
solutions such as 3Y-TZP from the designation, and is checked by the engine's
own formula parser in the test suite. **Densities are typical full-density
values entered without a checked citation.** Every one is marked ``unverified``
so the interface tells the user to replace it with a measured bulk density.
Promote an entry by adding a verified source, not by deleting that flag.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = (Path(__file__).resolve().parent.parent
          / "xray_workbench" / "materials" / "data" / "stoichiometric_ceramics.json")


def pure(formula: str) -> list[tuple[str, float]]:
    return [(formula, 1.0)]


# (identifier, name, category, mole-basis components, typical density g/cm3, note)
ROWS: list[tuple[str, str, str, list[tuple[str, float]], float, str]] = [
    # Oxide ceramics
    ("alumina-corundum", "Alumina (α-Al2O3, corundum)", "oxide_ceramic", pure("Al2O3"), 3.987, ""),
    ("magnesia", "Magnesia (MgO, periclase)", "oxide_ceramic", pure("MgO"), 3.58, ""),
    ("calcia", "Calcia (CaO)", "oxide_ceramic", pure("CaO"), 3.34, ""),
    ("silica-fused", "Silica, fused (amorphous SiO2)", "oxide_ceramic", pure("SiO2"), 2.20, ""),
    ("silica-quartz", "Quartz (α-SiO2)", "oxide_ceramic", pure("SiO2"), 2.65, ""),
    ("titania-rutile", "Titania, rutile (TiO2)", "oxide_ceramic", pure("TiO2"), 4.25, ""),
    ("titania-anatase", "Titania, anatase (TiO2)", "oxide_ceramic", pure("TiO2"), 3.89, ""),
    ("zirconia-monoclinic", "Zirconia, monoclinic (ZrO2)", "oxide_ceramic", pure("ZrO2"), 5.68, ""),
    ("zirconia-3y", "3 mol% yttria-stabilised zirconia (3Y-TZP)", "oxide_ceramic",
     [("ZrO2", 97), ("Y2O3", 3)], 6.05, "Composition follows from the 3 mol% Y2O3 designation."),
    ("zirconia-8y", "8 mol% yttria-stabilised zirconia (8YSZ)", "oxide_ceramic",
     [("ZrO2", 92), ("Y2O3", 8)], 5.90, "Composition follows from the 8 mol% Y2O3 designation."),
    ("hafnia", "Hafnia (HfO2)", "oxide_ceramic", pure("HfO2"), 9.68, ""),
    ("yttria", "Yttria (Y2O3)", "oxide_ceramic", pure("Y2O3"), 5.01, ""),
    ("ceria", "Ceria (CeO2)", "oxide_ceramic", pure("CeO2"), 7.22, ""),
    ("zinc-oxide", "Zinc oxide (ZnO)", "oxide_ceramic", pure("ZnO"), 5.61, ""),
    ("beryllia", "Beryllia (BeO)", "oxide_ceramic", pure("BeO"), 3.01, ""),
    ("spinel", "Magnesium aluminate spinel (MgAl2O4)", "oxide_ceramic", pure("MgAl2O4"), 3.58, ""),
    ("mullite", "Mullite (3Al2O3·2SiO2)", "oxide_ceramic", pure("Al6Si2O13"), 3.16,
     "Stoichiometric 3:2 mullite; commercial mullites vary."),
    ("zircon", "Zircon (ZrSiO4)", "oxide_ceramic", pure("ZrSiO4"), 4.65, ""),
    ("lanthana", "Lanthana (La2O3)", "oxide_ceramic", pure("La2O3"), 6.51, ""),
    ("gadolinia", "Gadolinia, cubic (Gd2O3)", "oxide_ceramic", pure("Gd2O3"), 7.41, ""),
    ("hematite", "Hematite (α-Fe2O3)", "oxide_ceramic", pure("Fe2O3"), 5.26, ""),
    ("magnetite", "Magnetite (Fe3O4)", "oxide_ceramic", pure("Fe3O4"), 5.18, ""),
    ("nickel-oxide", "Nickel oxide (NiO)", "oxide_ceramic", pure("NiO"), 6.67, ""),
    ("chromia", "Chromia (Cr2O3)", "oxide_ceramic", pure("Cr2O3"), 5.22, ""),
    ("tin-oxide", "Tin dioxide (SnO2, cassiterite)", "oxide_ceramic", pure("SnO2"), 6.95, ""),
    ("tungsten-trioxide", "Tungsten trioxide (WO3)", "oxide_ceramic", pure("WO3"), 7.16, ""),
    ("tantala", "Tantalum pentoxide (Ta2O5)", "oxide_ceramic", pure("Ta2O5"), 8.2, ""),
    ("niobia", "Niobium pentoxide (Nb2O5)", "oxide_ceramic", pure("Nb2O5"), 4.6, ""),
    ("bismuth-oxide", "Bismuth oxide (α-Bi2O3)", "oxide_ceramic", pure("Bi2O3"), 8.9, ""),
    ("aluminium-titanate", "Aluminium titanate (Al2TiO5)", "oxide_ceramic", pure("Al2TiO5"), 3.70, ""),
    ("lanthanum-aluminate", "Lanthanum aluminate (LaAlO3)", "oxide_ceramic", pure("LaAlO3"), 6.52, ""),
    ("boron-oxide", "Boron oxide, vitreous (B2O3)", "glass", pure("B2O3"), 1.84, ""),
    # Nuclear ceramics
    ("urania", "Uranium dioxide (UO2)", "nuclear_ceramic", pure("UO2"), 10.97, ""),
    ("thoria", "Thorium dioxide (ThO2)", "nuclear_ceramic", pure("ThO2"), 10.0, ""),
    # Silicates, refractories and bioceramics
    ("cordierite", "Cordierite (Mg2Al4Si5O18)", "silicate", pure("Mg2Al4Si5O18"), 2.51, ""),
    ("forsterite", "Forsterite (Mg2SiO4)", "silicate", pure("Mg2SiO4"), 3.22, ""),
    ("enstatite", "Enstatite, steatite phase (MgSiO3)", "silicate", pure("MgSiO3"), 3.2, ""),
    ("wollastonite", "Wollastonite (CaSiO3)", "silicate", pure("CaSiO3"), 2.9, ""),
    ("anorthite", "Anorthite (CaAl2Si2O8)", "silicate", pure("CaAl2Si2O8"), 2.76, ""),
    ("kaolinite", "Kaolinite (Al2Si2O5(OH)4)", "silicate", pure("Al2Si2O5(OH)4"), 2.6, ""),
    ("hydroxyapatite", "Hydroxyapatite (Ca10(PO4)6(OH)2)", "bioceramic", pure("Ca10(PO4)6(OH)2"), 3.16, ""),
    ("tricalcium-phosphate-beta", "β-Tricalcium phosphate (Ca3(PO4)2)", "bioceramic",
     pure("Ca3(PO4)2"), 3.07, ""),
    # Electroceramics and magnetic ceramics
    ("barium-titanate", "Barium titanate (BaTiO3)", "electroceramic", pure("BaTiO3"), 6.02, ""),
    ("strontium-titanate", "Strontium titanate (SrTiO3)", "electroceramic", pure("SrTiO3"), 5.12, ""),
    ("lead-titanate", "Lead titanate (PbTiO3)", "electroceramic", pure("PbTiO3"), 7.97, ""),
    ("lithium-niobate", "Lithium niobate (LiNbO3)", "electroceramic", pure("LiNbO3"), 4.64, ""),
    ("lithium-tantalate", "Lithium tantalate (LiTaO3)", "electroceramic", pure("LiTaO3"), 7.46, ""),
    ("yig", "Yttrium iron garnet (Y3Fe5O12)", "magnetic_ceramic", pure("Y3Fe5O12"), 5.17, ""),
    ("barium-hexaferrite", "Barium hexaferrite (BaFe12O19)", "magnetic_ceramic", pure("BaFe12O19"), 5.28, ""),
    ("strontium-hexaferrite", "Strontium hexaferrite (SrFe12O19)", "magnetic_ceramic", pure("SrFe12O19"), 5.1, ""),
    # Scintillator and detector crystals (undoped hosts)
    ("yag", "Yttrium aluminium garnet (Y3Al5O12)", "scintillator", pure("Y3Al5O12"), 4.56,
     "Undoped host; activators such as Ce are not included."),
    ("luag", "Lutetium aluminium garnet (Lu3Al5O12)", "scintillator", pure("Lu3Al5O12"), 6.73, "Undoped host."),
    ("ggg", "Gadolinium gallium garnet (Gd3Ga5O12)", "scintillator", pure("Gd3Ga5O12"), 7.09, "Undoped host."),
    ("lso", "Lutetium oxyorthosilicate (Lu2SiO5)", "scintillator", pure("Lu2SiO5"), 7.4,
     "Undoped host; LYSO compositions vary and are not included."),
    ("yap", "Yttrium aluminium perovskite (YAlO3)", "scintillator", pure("YAlO3"), 5.37, "Undoped host."),
    ("bgo", "Bismuth germanate (Bi4Ge3O12)", "scintillator", pure("Bi4Ge3O12"), 7.13, ""),
    ("cadmium-tungstate", "Cadmium tungstate (CdWO4)", "scintillator", pure("CdWO4"), 7.9, ""),
    ("lead-tungstate", "Lead tungstate (PbWO4)", "scintillator", pure("PbWO4"), 8.28, ""),
    ("calcium-tungstate", "Calcium tungstate (CaWO4)", "scintillator", pure("CaWO4"), 6.06, ""),
    ("barium-fluoride", "Barium fluoride (BaF2)", "halide", pure("BaF2"), 4.89, ""),
    ("magnesium-fluoride", "Magnesium fluoride (MgF2)", "halide", pure("MgF2"), 3.15, ""),
    ("strontium-fluoride", "Strontium fluoride (SrF2)", "halide", pure("SrF2"), 4.24, ""),
    ("lead-fluoride", "Lead fluoride (PbF2)", "halide", pure("PbF2"), 7.77, ""),
    ("sodium-iodide", "Sodium iodide (NaI)", "halide", pure("NaI"), 3.67, "Undoped host; Tl activator not included."),
    ("lanthanum-bromide", "Lanthanum bromide (LaBr3)", "halide", pure("LaBr3"), 5.08,
     "Undoped host; Ce activator not included."),
    # Non-oxide ceramics
    ("silicon-carbide", "Silicon carbide (SiC)", "non_oxide_ceramic", pure("SiC"), 3.21, ""),
    ("silicon-nitride", "Silicon nitride (β-Si3N4)", "non_oxide_ceramic", pure("Si3N4"), 3.19, ""),
    ("aluminium-nitride", "Aluminium nitride (AlN)", "non_oxide_ceramic", pure("AlN"), 3.26, ""),
    ("boron-nitride-hex", "Boron nitride, hexagonal (h-BN)", "non_oxide_ceramic", pure("BN"), 2.27, ""),
    ("boron-nitride-cubic", "Boron nitride, cubic (c-BN)", "non_oxide_ceramic", pure("BN"), 3.48, ""),
    ("boron-carbide", "Boron carbide (B4C)", "non_oxide_ceramic", pure("B4C"), 2.52, ""),
    ("titanium-carbide", "Titanium carbide (TiC)", "non_oxide_ceramic", pure("TiC"), 4.93, ""),
    ("titanium-nitride", "Titanium nitride (TiN)", "non_oxide_ceramic", pure("TiN"), 5.40, ""),
    ("titanium-diboride", "Titanium diboride (TiB2)", "non_oxide_ceramic", pure("TiB2"), 4.52, ""),
    ("zirconium-carbide", "Zirconium carbide (ZrC)", "non_oxide_ceramic", pure("ZrC"), 6.73, ""),
    ("zirconium-nitride", "Zirconium nitride (ZrN)", "non_oxide_ceramic", pure("ZrN"), 7.09, ""),
    ("zirconium-diboride", "Zirconium diboride (ZrB2)", "non_oxide_ceramic", pure("ZrB2"), 6.09, ""),
    ("hafnium-carbide", "Hafnium carbide (HfC)", "non_oxide_ceramic", pure("HfC"), 12.2, ""),
    ("hafnium-diboride", "Hafnium diboride (HfB2)", "non_oxide_ceramic", pure("HfB2"), 11.2, ""),
    ("tantalum-carbide", "Tantalum carbide (TaC)", "non_oxide_ceramic", pure("TaC"), 14.5, ""),
    ("niobium-carbide", "Niobium carbide (NbC)", "non_oxide_ceramic", pure("NbC"), 7.82, ""),
    ("tungsten-carbide", "Tungsten carbide (WC)", "non_oxide_ceramic", pure("WC"), 15.63,
     "Binder-free WC; cemented carbides contain Co or Ni."),
    ("chromium-carbide", "Chromium carbide (Cr3C2)", "non_oxide_ceramic", pure("Cr3C2"), 6.68, ""),
    ("molybdenum-disilicide", "Molybdenum disilicide (MoSi2)", "non_oxide_ceramic", pure("MoSi2"), 6.26, ""),
    ("lanthanum-hexaboride", "Lanthanum hexaboride (LaB6)", "non_oxide_ceramic", pure("LaB6"), 4.72, ""),
    ("ti3sic2", "MAX phase Ti3SiC2", "non_oxide_ceramic", pure("Ti3SiC2"), 4.53, ""),
    ("ti2alc", "MAX phase Ti2AlC", "non_oxide_ceramic", pure("Ti2AlC"), 4.11, ""),
    ("ti3alc2", "MAX phase Ti3AlC2", "non_oxide_ceramic", pure("Ti3AlC2"), 4.25, ""),
    # Semiconductors
    ("zinc-sulfide", "Zinc sulfide (ZnS, sphalerite)", "semiconductor", pure("ZnS"), 4.09, ""),
    ("zinc-selenide", "Zinc selenide (ZnSe)", "semiconductor", pure("ZnSe"), 5.27, ""),
    ("gallium-nitride", "Gallium nitride (GaN)", "semiconductor", pure("GaN"), 6.15, ""),
    # Cement and binder phases
    ("alite", "Tricalcium silicate, alite (Ca3SiO5)", "cement_phase", pure("Ca3SiO5"), 3.15, ""),
    ("tricalcium-aluminate", "Tricalcium aluminate (Ca3Al2O6)", "cement_phase", pure("Ca3Al2O6"), 3.04, ""),
    ("portlandite", "Portlandite (Ca(OH)2)", "cement_phase", pure("Ca(OH)2"), 2.24, ""),
    ("gypsum", "Gypsum (CaSO4·2H2O)", "cement_phase", pure("CaSO4·2H2O"), 2.32, ""),
]

UNVERIFIED_DENSITY_NOTE = ("Typical full-density value entered without a checked citation. "
                           "Replace with the measured bulk density of your specimen.")


def main() -> int:
    entries = []
    for identifier, name, category, components, density, note in ROWS:
        entry = {
            "id": f"stoich-{identifier}", "name": name, "category": category, "tier": "stoichiometric",
            "composition": {"basis": "mole",
                            "components": [{"formula": f, "fraction": x} for f, x in components]},
            "density": {"value_g_cm3": density, "kind": "typical", "status": "unverified",
                        "note": UNVERIFIED_DENSITY_NOTE},
        }
        if note:
            entry["notes"] = note
        entries.append(entry)
    document = {"schema_version": 1, "dataset": {
        "id": "stoichiometry",
        "title": "Stoichiometric ceramic, crystal and binder phases",
        "publisher": "X-ray Attenuation Workbench project",
        "license": "MIT (this project)",
        "method": ("Composition follows exactly from the chemical formula, or for designated solid "
                   "solutions such as 3Y-TZP from the designation. Densities are typical values "
                   "without a checked citation and are marked unverified entry by entry."),
        "caveats": [
            "Real specimens contain dopants, impurities, secondary phases, porosity and sintering aids.",
            "Densities are not traceable to a source; use measured bulk density for quantitative work.",
        ]}, "entries": entries}
    OUTPUT.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} stoichiometric entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
