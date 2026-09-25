# X-ray Attenuation Workbench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602370.svg)](https://doi.org/10.5281/zenodo.17602370)
[![CI](https://github.com/faizalusama/X-ray_attenuation_calculator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/faizalusama/X-ray_attenuation_calculator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local scientific application for multicomponent glasses, ceramics, composites, filters, and planar multilayers. Version 1.0.0 is the first stable release of Mohamed Faizal Ussama Jalaludeen's X-ray attenuation work, superseding the 0.1.x calculator preserved under `legacy/`. Released 21 September 2026; cite it via the Zenodo concept DOI [10.5281/zenodo.17602370](https://doi.org/10.5281/zenodo.17602370), which covers all versions. The 1.0.0 release itself is [10.5281/zenodo.22878495](https://doi.org/10.5281/zenodo.22878495).

> **Results differ from 0.1.x.** The original multicomponent script applied each component's mass fraction twice. This version implements the mixture rule correctly, so multi-component results change. Work based on the original script should be rechecked.

## Start the graphical application

Use **Python 3.12 or later**. The verified environment uses Python 3.12 on Windows.

On Windows, extract the complete project folder and double-click **START_WINDOWS.cmd**. The launcher prefers Python 3.12 through the Windows Python launcher, then tries `python` (3.12 or later). On first launch it creates an isolated `.venv` and installs the dependencies; later launches check the direct dependency versions. Internet access is needed for installation; calculations and the interface then run locally without an external service.

Manual setup, Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python launch.py
```

macOS / Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py
```

The application opens at **http://127.0.0.1:8765**. Keep the terminal open while using it; Ctrl+C stops the server. Use `python launch.py --port 8766` if the default port is occupied. The server binds to this computer only.

The workbench can also be installed as a package, which provides an `xray-workbench` command equivalent to `python launch.py`:

```powershell
python -m pip install .
xray-workbench --version
```

The installed wheel carries its own browser assets, so the graphical application works from any working directory.

In the application, start with an illustrative material preset or import `examples/borosilicate-multilayer.json`, enter measured composition and density, and define each layer's thickness and incidence angle. Set the energy range and reference energy, then calculate. Review the reported assumptions alongside the curves before exporting a project or result. The input project can be reopened for editing; the result export preserves the exact configuration and its fingerprint.

## Live interaction

Plots and results recalculate automatically as you edit any calculation parameter. A 300 ms debounce coalesces typing and slider movement; only one scientific request runs at a time. The thickness, angle, and reference-energy sliders support quick exploration, while numeric fields retain exact input control. `Refresh now` is an optional immediate refresh, not a required trigger.

The previous plot remains visible with an explicit pending label during calculation or incomplete input. Exports are disabled until the latest configuration has been evaluated. Older responses cannot overwrite newer inputs. Spectrum and enabled uncertainty calculations follow the same live workflow; large Monte Carlo runs can take longer.

## Interactive dashboard

All eleven plots open initially in a responsive grid. Each plot has its own relevant log-scale, edge-marker, layer-curve, grid, bold-label and font-size controls. Plot ticks default to 18 px with larger bold axis titles; plot text can be adjusted from 11–24 px. Important site headings use 24 px (18 pt), with a consistent smaller hierarchy for controls and supporting text. Logarithmic y-axis labels use sparse powers of ten; legends sit above the plotting area to reduce bottom whitespace. Axes & quantities opens additional configuration, including a second y-axis. Linked zoom converts energy ranges correctly between linear and logarithmic plots. Use Box zoom, Pan, Lock and Reset view to control mouse interaction; linked energy axes are optional. Rows and Grid change the arrangement, and Show all plots restores the complete set after removing panels.

Graphs and active controls lead the page. Reference measurements and expandable scientific limits sit below the plots; suggestions open from the bottom-right Ideas button. The system editor is visible on the right by default, with a calculation-flow schematic and the current layer stack beneath its inputs. On narrow screens the editor stacks above the plots. Hide inputs frees the full width; that choice is remembered. Length units can be changed between nm, µm, mm, cm and m without changing physical dimensions; project/API thicknesses remain in mm. Preferences, plot visibility and the last valid inputs are remembered locally. Existing saved layouts expand once to introduce all eleven plots, then subsequent custom layouts are preserved.

SVG export includes the selected plot, its axes and configuration/provenance metadata. Heatmaps embed a raster colour field within the SVG.

## What is included

- **Composition:** chemical formula parsing, automatic molecular masses, normalized mass / mole / volume fractions, elemental mass breakdown, illustrative presets, and a 501-entry provenance-aware materials library spanning NIST reference media, PNNL engineering materials, and stoichiometric phases.
- **Density:** measured bulk density, solid density with explicit void porosity, or ideal additive-specific-volume density. Each density assumption is reported. A measured bulk density is never reduced by porosity a second time.
- **Geometry:** multiple homogeneous planar layers, individual thicknesses in mm, and incidence angles measured from the surface normal.
- **Energy:** 1–800 keV guarded domain, linear / logarithmic sweeps, additional points at absorption edges, and direct evaluation at the reference energy.
- **Outputs:** total mass and linear attenuation, photoelectric / coherent / incoherent contributions, primary transmission, removed-primary fraction, optical depth, individual material HVL / TVL / 1/e length, and the thickness multiplier to meet a monoenergetic target transmission.
- **Spectra:** import a discrete-bin photon-fluence spectrum; calculate photon- or energy-weighted transmission, transmitted spectrum, and mean photon energy before / after the stack.
- **Uncertainty:** reproducible Monte Carlo propagation of independent density and thickness uncertainties through positive lognormal distributions. The interval excludes composition, database, geometry-angle, and model uncertainty.
- **Reproducibility:** project JSON, complete calculation JSON with input and provenance, full-precision CSV, SVG charts, a batch command, an example notebook, and automated scientific/API tests.

## Scientific scope

This is a **narrow-beam Beer–Lambert calculation** using XrayDB's Elam atomic data. Removed primary photons include scattering; **1 − T is not absorbed energy or dose**. The model does not propagate fluorescent or scattered photons, simulate broad-beam buildup, model diffraction / XANES / EXAFS, or predict detector response. Layer order does not change primary transmission in this model.

The allowed 1–800 keV interval avoids the backend's out-of-range clamping. XrayDB describes cross-sections as typically most reliable around 0.25–250 keV; energies above 250 keV produce a warning. A denser grid is not a guarantee of physical accuracy. Measured composition, density, thickness, geometry, and application-specific validation remain essential, especially around absorption edges.

The presets are editable illustrations, not certified material specifications. For glasses, use measured bulk density where available; ideal volume additivity need not hold after mixing or melting.

See [scientific methods and references](docs/SCIENTIFIC_METHODS.md), [validation record](docs/VALIDATION.md), and the [research roadmap](docs/ROADMAP.md).

## Attenuation data sources

Cross-sections come from a selectable backend, named in the configuration as `"backend": "elam"` and recorded in every result's provenance. The default is `elam` (XrayDB's Elam tables). Each backend declares its hard energy domain, the narrower range its authors consider reliable, the interaction channels it actually provides, its atomic-number range and its edge convention. Backends are never averaged together: where sources disagree, the disagreement is the information.

An optional `xraylib` backend is available via `pip install "xray-attenuation-workbench[xraylib]"`. **It is not an independent check.** Measured on 20 September 2026, xraylib 4.3.0 returns the same coefficients as XrayDB 4.5.8 to within floating-point rounding (bit-identical on Windows, at most 1.9 × 10⁻¹⁵ relative on Linux and macOS) across all 99 element/energy samples tested — the two packages expose the same evaluated data. It therefore declares `shares_data_with = ("elam",)`, and the comparison tools refuse to treat the pair as cross-database validation. The two do disagree on edge *labels* (Ba K: 37.4410 vs 37.4406 keV), so they are not interchangeable for edge work.

This project currently has **no independent cross-database validation**. The NIST reference comparison in the [validation record](docs/VALIDATION.md) covers a small off-edge subset only. See the [roadmap](docs/ROADMAP.md) for the planned NIST XCOM adapter.

## Input conventions

Fractions are nonnegative relative proportions, normalized explicitly by the engine. For example, `80, 15, 5` and `0.8, 0.15, 0.05` represent the same composition. Mole fractions refer to **formula units of the entered components**. Component density is required for volume fractions and for the ideal-density mode. Formulas are case sensitive: use ordinary digits rather than Unicode subscripts, such as `SiO2`, `Al2O3`, or `Ca(OH)2`. Nested parentheses/brackets and decimal occupancies are supported. Hydrates use a middle dot, for example `CuSO4·5H2O`; an ASCII period is always a decimal point.

Angles are from the surface normal, so normal incidence is 0°. The effective path is thickness / cos(angle). HVL, TVL, and attenuation length are distances **along the ray** in an individual homogeneous material, not the thickness of the complete multilayer stack. Target scaling multiplies all layer thicknesses at their existing angles and composition.

Spectrum CSV files use two columns: `energy_keV,weight`. Each row is **integrated photon fluence in a discrete bin**, not counts per keV. If your measurement is a density per keV, multiply by bin widths before importing. Energy weighting changes the output observable; it does not change the meaning of the input weights. The sample CSV is illustrative and is not a calibrated tube-spectrum model.

## Batch and Python use

```powershell
python launch.py --calculate examples/borosilicate-multilayer.json --output result.json
```

```python
import json
from xray_workbench.physics import calculate

with open("examples/borosilicate-multilayer.json") as handle:
    project = json.load(handle)
result = calculate(project["configuration"])
print(result["reference"]["transmission"])
```

The API route additionally attaches the complete input and its SHA-256 digest. The command uses the same route-independent wrapper. To inspect the local API, open `/docs` on the running server.

Batch input may be a saved project with a `configuration` object or a bare configuration object. Omit `--output` to emit JSON to standard output. Invalid inputs exit with status 2 and a message on standard error. The example notebook is `notebooks/materials_workbench.ipynb`; open it with Jupyter using the same environment (install Jupyter separately if needed).

## Tests and checks

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python tools/manifest.py verify
```

Tests cover the original mass-weighting regression, unit conversion, composition bases, multilayer identities, reference evaluation, spectra, uncertainty, validation errors, and independent off-edge NIST reference values. Agreement between databases at selected energies is a regression benchmark, not proof of uniform experimental accuracy. Exact tested dependency versions are also recorded in `requirements-lock.txt`.

Every rejection test asserts the *reason* for the rejection, not merely that a `ValueError` was raised: the engine signals all invalid input with that one exception type, so an unqualified check could pass because the wrong guard fired.

The browser assets have their own suite, which needs Node.js:

```powershell
node --check static/app.js
node tests/test_frontend.cjs
```

`tools/manifest.py` maintains `MANIFEST.sha256`, the record of what a release actually contained. Run `python tools/manifest.py write` after an intentional change and commit the result. `tools/snapshot.py` takes timestamped copies of the authored files into a git-ignored `_snapshots/` directory; it is a safety net for working without version control, not a replacement for it. All of these run in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) across Python 3.12–3.14 on Linux, Windows and macOS.

## Original project and publication

`legacy/` preserves the supplied archive for attribution and reproducibility. **Do not use its old script or stored notebook results for new calculations.** The original script applied each component's mass fraction twice; this version fixes the mixture rule. The archived notebook also contains exploratory density and attenuation-length averaging that must not be used as the new calculation method.

The existing DOI identifies the original software, not this unpublished extension. Authorship is preserved in `CITATION.cff`; the original citation is retained under `legacy/`. No new DOI, release, repository commit, or publication has been made.

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE).

The published repository of the original calculator states "Free to use and modify" and its README refers to a `LICENSE` file, but no such file exists there. That phrasing grants no defined rights, so the original work should be given the same explicit licence before anything is distributed.

Third-party components keep their own terms, including the Plotly.js bundled under `static/vendor/`. They are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The Elam atomic data underlying every reported number carry their own citation expectations that the MIT licence does not address; cite the data source, not only this tool.
