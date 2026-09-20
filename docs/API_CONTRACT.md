# Implementation contract (local API v1)

The application is a local Python FastAPI server, with static HTML/CSS/JS in `static/`.
Scientific engine: `xray_workbench/physics.py`, exposing `calculate(payload: dict) -> dict` and `presets() -> list`.
HTTP: GET `/api/presets` returns `{presets: [...]}`; POST `/api/calculate` accepts the request below and returns the result. Validation failures use HTTP 422 `{detail: string}`.

## Request
```json
{
  "energy": {"min_keV": 5, "max_keV": 120, "points": 500, "spacing": "log", "reference_keV": 30},
  "layers": [{
    "name": "Borosilicate glass", "basis": "mole",
    "components": [{"formula": "SiO2", "fraction": 80}, {"formula": "B2O3", "fraction": 15}, {"formula": "Na2O", "fraction": 5}],
    "density_g_cm3": 2.23, "thickness_mm": 1, "angle_deg": 0,
    "porosity": 0, "density_mode": "bulk",
    "density_uncertainty_pct": 0, "thickness_uncertainty_pct": 0
  }],
  "target_transmission": 0.1,
  "spectrum": null,
  "uncertainty": {"enabled": false, "samples": 1000, "seed": 42}
}
```
Basis is `mass`, `mole`, or `volume`. Components can optionally provide `density_g_cm3` (required for volume fractions or ideal density). Density mode `bulk` means supplied measured density already includes pores and porosity must be 0; `solid` multiplies supplied density by (1-porosity); `ideal` infers solid density via additive specific volume and then applies porosity. Angles measured from surface normal. Uncertainties are independent relative one-sigma percentages of effective density and normal thickness; seeded positive lognormal sampling. Density and thickness uncertainties allowed in all modes, applied to final effective density/thickness.

Spectrum, if present: `{energy_keV: [10,20,30], weights: [1,2,1], weighting: "photon", label: "Measured spectrum"}`. Each weight is integrated fluence in its discrete bin; **not** spectral density. Weighting `photon` or `energy` controls weighted transmission. Evaluate each input energy directly. No invented X-ray tube spectrum.

## Result
```json
{
  "energy_keV": [], "transmission": [], "removed_fraction": [], "optical_depth": [],
  "layers": [{"name":"...", "density_g_cm3":2.23, "thickness_mm":1, "path_length_mm":1,
    "components":[{"formula":"SiO2","mass_fraction":0.8,"molar_mass_g_mol":60}],
    "elemental_mass_fractions":{"Si":0.3,"O":0.7},
    "mu_mass_cm2_g":[], "mu_linear_cm_inv":[], "photoelectric_cm2_g":[],
    "coherent_cm2_g":[], "incoherent_cm2_g":[],
    "reference":{"mu_mass_cm2_g":1,"mu_linear_cm_inv":2.23,"transmission":0.8,"hvl_mm":3,"tvl_mm":10,"attenuation_length_mm":4}
  }],
  "reference": {"energy_keV":30,"transmission":0.8,"removed_fraction":0.2,"optical_depth":0.223,
    "target_transmission":0.1,"target_thickness_scale":10.32},
  "edges": [{"element":"Ba", "shell":"K", "energy_keV":37.44}],
  "spectrum": null,
  "uncertainty": null,
  "warnings": [],
  "provenance": {"engine":"XrayDB / Elam", "xraydb_version":"...", "energy_range_keV":[1,800],"model":"narrow-beam Beer-Lambert", "edge_bracket_relative":0.0001, "timestamp_utc":"..."}
}
```
Spectrum result: `{energy_keV:[], input_weights:[], transmitted_weights:[], transmission:0.5, weighting:"photon", mean_energy_in_keV:30, mean_energy_out_keV:40, label:"..."}`. Mean energies always photon-fluence weighted. Uncertainty result: `{samples:1000, seed:42, transmission_p025:[], transmission_p975:[], reference:{mean:0.8, std:0.01,p025:0.78,p975:0.82}, assumptions:"..."}`.

HVL/TVL/attenuation length describe homogeneous material at the reference energy (path length); stack target scaling multiplies every layer's normal thickness. Reference values are direct backend evaluations, never interpolation across a plotted edge. The sweep adds nominal edge energies and relative ±0.0001 brackets within range; `points` is the baseline grid count. This bracket exceeds the current Elam log-energy-knot rounding offsets. Nominal metadata energies can differ slightly from the backend's coefficient discontinuities; coefficients are not shifted to match the labels. See SCIENTIFIC_METHODS.md for the exact-edge convention and validation.

Preset list entries: `{id:"borosilicate", name:"Borosilicate glass", description:"Illustrative nominal composition; replace density with measured value.", layer:{...}}`.
